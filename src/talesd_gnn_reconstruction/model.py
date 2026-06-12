from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


WAVEFORM_SHAPE_DIM = 20


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _scatter_mean(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.zeros(num_graphs, values.shape[1], dtype=values.dtype, device=values.device)
    out.index_add_(0, batch, values)
    counts = torch.zeros(num_graphs, 1, dtype=values.dtype, device=values.device)
    counts.index_add_(0, batch, torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device))
    return out / counts.clamp_min(1.0)


def _scatter_max(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.full((num_graphs, values.shape[1]), -torch.inf, dtype=values.dtype, device=values.device)
    if hasattr(out, "scatter_reduce_"):
        index = batch[:, None].expand(-1, values.shape[1])
        out.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    rows = []
    for graph_index in range(num_graphs):
        graph_values = values[batch == graph_index]
        rows.append(torch.max(graph_values, dim=0).values if graph_values.numel() else torch.zeros(values.shape[1], dtype=values.dtype, device=values.device))
    return torch.stack(rows, dim=0)


def _graph_context(
    batch: torch.Tensor,
    src: torch.Tensor | None,
    num_graphs: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    node_counts = torch.zeros(num_graphs, 1, dtype=dtype, device=device)
    node_counts.index_add_(0, batch, torch.ones(batch.shape[0], 1, dtype=dtype, device=device))
    edge_counts = torch.zeros(num_graphs, 1, dtype=dtype, device=device)
    if src is not None and src.numel() > 0:
        edge_graph = batch[src]
        edge_counts.index_add_(0, edge_graph, torch.ones(edge_graph.shape[0], 1, dtype=dtype, device=device))
    edge_density = edge_counts / node_counts.clamp_min(1.0)
    return torch.cat([torch.log1p(node_counts), torch.log1p(edge_counts), edge_density], dim=-1)


def _waveform_shape_summary(
    waveform_x: torch.Tensor | None,
    num_nodes: int,
    *,
    enabled: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not enabled:
        return torch.zeros(num_nodes, 0, dtype=dtype, device=device)
    if waveform_x is None:
        return torch.zeros(num_nodes, WAVEFORM_SHAPE_DIM, dtype=dtype, device=device)
    waveform = waveform_x.to(device=device, dtype=dtype)
    if waveform.ndim != 3 or waveform.shape[0] != num_nodes or waveform.shape[1] == 0 or waveform.shape[2] == 0:
        return torch.zeros(num_nodes, WAVEFORM_SHAPE_DIM, dtype=dtype, device=device)
    if waveform.shape[1] < 4:
        padded = torch.zeros(num_nodes, 4, waveform.shape[2], dtype=dtype, device=device)
        padded[:, : waveform.shape[1], :] = waveform
        waveform = padded
    else:
        waveform = waveform[:, :4, :]

    upper_raw = waveform[:, 0, :].clamp_min(0.0)
    lower_raw = waveform[:, 1, :].clamp_min(0.0)
    upper_mask = waveform[:, 2, :].clamp(0.0, 1.0)
    lower_mask = waveform[:, 3, :].clamp(0.0, 1.0)
    positive = torch.stack(
        [
            upper_raw,
            lower_raw,
            upper_raw * upper_mask,
            lower_raw * lower_mask,
        ],
        dim=1,
    )
    sums = positive.sum(dim=-1)
    peaks = positive.amax(dim=-1)
    length = int(positive.shape[-1])
    positions = torch.linspace(0.0, 1.0, length, dtype=dtype, device=device)
    centroids = (positive * positions[None, None, :]).sum(dim=-1) / sums.clamp_min(1.0e-6)
    tail = positive[:, :, length // 2 :].sum(dim=-1) / sums.clamp_min(1.0e-6)

    raw_sum = sums[:, 0] + sums[:, 1]
    accepted_sum = sums[:, 2] + sums[:, 3]
    raw_asym = (sums[:, 0] - sums[:, 1]) / raw_sum.clamp_min(1.0e-6)
    accepted_asym = (sums[:, 2] - sums[:, 3]) / accepted_sum.clamp_min(1.0e-6)
    upper_accepted_fraction = sums[:, 2] / sums[:, 0].clamp_min(1.0e-6)
    lower_accepted_fraction = sums[:, 3] / sums[:, 1].clamp_min(1.0e-6)

    summary = torch.cat(
        [
            torch.log1p(sums),
            torch.log1p(peaks),
            centroids,
            tail,
            raw_asym[:, None],
            accepted_asym[:, None],
            upper_accepted_fraction[:, None],
            lower_accepted_fraction[:, None],
        ],
        dim=-1,
    )
    return torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


def _waveform_mass_readout(
    waveform_embedding: torch.Tensor,
    waveform_x: torch.Tensor | None,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    summary_enabled: bool,
) -> torch.Tensor:
    parts = []
    if waveform_embedding.shape[1] > 0:
        parts.append(_scatter_mean(waveform_embedding, batch, num_graphs))
        parts.append(_scatter_max(waveform_embedding, batch, num_graphs))
    summary = _waveform_shape_summary(
        waveform_x,
        int(batch.shape[0]),
        enabled=summary_enabled,
        dtype=waveform_embedding.dtype,
        device=waveform_embedding.device,
    )
    if summary.shape[1] > 0:
        parts.append(_scatter_mean(summary, batch, num_graphs))
        parts.append(_scatter_max(summary, batch, num_graphs))
    if not parts:
        return torch.zeros(num_graphs, 0, dtype=waveform_embedding.dtype, device=waveform_embedding.device)
    return torch.cat(parts, dim=-1)


def _scatter_softmax(scores: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if scores.ndim == 1:
        scores = scores[:, None]
    max_values = _scatter_max(scores, batch, num_graphs)
    stable = scores - max_values[batch]
    weights = torch.exp(torch.clamp(stable, min=-80.0, max=40.0)).to(dtype=scores.dtype)
    denom = torch.zeros(num_graphs, scores.shape[1], dtype=scores.dtype, device=scores.device)
    denom.index_add_(0, batch, weights)
    return weights / denom[batch].clamp_min(1.0e-12)


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.05):
        super().__init__()
        self.message = _make_mlp(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim, dropout)
        self.update = _make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node: torch.Tensor,
        src: torch.Tensor | None,
        dst: torch.Tensor | None,
        edge_attr: torch.Tensor,
        degree: torch.Tensor | None,
    ) -> torch.Tensor:
        if src is None or dst is None or degree is None:
            aggregate = torch.zeros_like(node)
        else:
            message_input = torch.cat([node[src], node[dst], edge_attr], dim=-1)
            messages = self.message(message_input).to(dtype=node.dtype)
            aggregate = torch.zeros_like(node)
            aggregate.index_add_(0, dst, messages)
            aggregate = aggregate / degree.clamp_min(1.0)
        return self.norm(node + self.update(torch.cat([node, aggregate], dim=-1)))


class GatedEdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.05):
        super().__init__()
        self.message = _make_mlp(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node: torch.Tensor,
        src: torch.Tensor | None,
        dst: torch.Tensor | None,
        edge_attr: torch.Tensor,
        degree: torch.Tensor | None,
    ) -> torch.Tensor:
        if src is None or dst is None or degree is None:
            mean_aggregate = torch.zeros_like(node)
            max_aggregate = torch.zeros_like(node)
        else:
            message_input = torch.cat([node[src], node[dst], edge_attr], dim=-1)
            messages = (self.message(message_input) * torch.sigmoid(self.gate(message_input))).to(dtype=node.dtype)
            mean_aggregate = torch.zeros_like(node)
            mean_aggregate.index_add_(0, dst, messages)
            mean_aggregate = mean_aggregate / degree.clamp_min(1.0)
            max_aggregate = torch.full_like(node, -torch.inf)
            if hasattr(max_aggregate, "scatter_reduce_"):
                index = dst[:, None].expand(-1, messages.shape[1])
                max_aggregate.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)
                max_aggregate = torch.where(torch.isfinite(max_aggregate), max_aggregate, torch.zeros_like(max_aggregate))
            else:
                max_aggregate = torch.zeros_like(node)
                for node_index in torch.unique(dst).tolist():
                    mask = dst == int(node_index)
                    max_aggregate[int(node_index)] = torch.max(messages[mask], dim=0).values
        node = self.norm(node + self.update(torch.cat([node, mean_aggregate, max_aggregate], dim=-1)))
        return node + self.ffn(node)


class EdgeTimeDeltaEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.05,
        *,
        enabled: bool = True,
        start_index: int = 4,
        width: int = 3,
    ):
        super().__init__()
        self.start_index = int(start_index)
        self.width = int(width)
        self.enabled = bool(enabled) and int(edge_dim) >= self.start_index + self.width
        self.encoder = _make_mlp(self.width, hidden_dim, hidden_dim, dropout) if self.enabled else None
        self.norm = nn.LayerNorm(hidden_dim) if self.enabled else None

    def forward(
        self,
        node: torch.Tensor,
        edge_attr: torch.Tensor,
        dst: torch.Tensor | None,
        degree: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.encoder is None or self.norm is None or dst is None or edge_attr.numel() == 0:
            return node
        time_attr = edge_attr[:, self.start_index : self.start_index + self.width]
        messages = self.encoder(time_attr).to(dtype=node.dtype)
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, dst, messages)
        if degree is None:
            degree = torch.zeros(node.shape[0], 1, dtype=node.dtype, device=node.device)
            degree.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=node.dtype, device=node.device))
        aggregate = aggregate / degree.to(dtype=node.dtype).clamp_min(1.0)
        return self.norm(node + aggregate)


class AttentiveReadout(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.heads = max(int(heads), 1)
        self.score = nn.Linear(hidden_dim, self.heads)

    def forward(self, node: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        scores = self.score(node)
        weights = _scatter_softmax(scores, batch, num_graphs)
        outputs = []
        for head in range(self.heads):
            weighted = node * weights[:, head : head + 1]
            pooled = torch.zeros(num_graphs, node.shape[1], dtype=node.dtype, device=node.device)
            pooled.index_add_(0, batch, weighted.to(dtype=pooled.dtype))
            outputs.append(pooled)
        return torch.cat(outputs, dim=-1)


def _make_classification_head(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    *,
    enhanced: bool,
) -> nn.Sequential:
    if not enhanced:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim * 2),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class DetectorIdEmbedding(nn.Module):
    def __init__(self, detector_lids: Sequence[int] | None = None, embedding_dim: int = 0):
        super().__init__()
        self.embedding_dim = max(int(embedding_dim), 0)
        raw_detector_lids = [] if detector_lids is None else detector_lids
        lids = sorted({int(lid) for lid in raw_detector_lids if int(lid) >= 0})
        self.register_buffer("detector_lid_values", torch.as_tensor(lids, dtype=torch.long), persistent=True)
        self.unknown_index = len(lids)
        self.embedding = nn.Embedding(len(lids) + 1, self.embedding_dim) if self.embedding_dim > 0 else None

    @property
    def output_dim(self) -> int:
        return self.embedding_dim

    def forward(
        self,
        detector_lids: torch.Tensor | None,
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.embedding is None:
            return torch.zeros(num_nodes, 0, dtype=dtype, device=device)
        if detector_lids is None:
            indices = torch.full((num_nodes,), self.unknown_index, dtype=torch.long, device=device)
            return self.embedding(indices).to(dtype=dtype)

        lids = detector_lids.to(device=device, dtype=torch.long).reshape(-1)
        if lids.numel() != num_nodes:
            padded = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            n_copy = min(int(lids.numel()), int(num_nodes))
            if n_copy > 0:
                padded[:n_copy] = lids[:n_copy]
            lids = padded

        values = self.detector_lid_values.to(device=device)
        if values.numel() == 0:
            indices = torch.full_like(lids, self.unknown_index)
        else:
            positions = torch.searchsorted(values, lids)
            safe_positions = torch.clamp(positions, max=values.numel() - 1)
            matched = (positions < values.numel()) & (values[safe_positions] == lids)
            unknown = torch.full_like(positions, self.unknown_index)
            indices = torch.where(matched, positions, unknown)
        return self.embedding(indices).to(dtype=dtype)


class WaveformEncoder(nn.Module):
    def __init__(
        self,
        waveform_channels: int,
        waveform_length: int,
        embedding_dim: int,
        mode: str = "cnn-gru",
        dropout: float = 0.05,
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        transformer_max_tokens: int = 128,
        transformer_downsample: str = "adaptive_avg",
    ):
        super().__init__()
        self.mode = str(mode)
        self.waveform_channels = max(int(waveform_channels), 0)
        self.waveform_length = max(int(waveform_length), 0)
        self.embedding_dim = max(int(embedding_dim), 0)
        self.transformer_max_tokens = max(int(transformer_max_tokens), 1)
        self.transformer_downsample = str(transformer_downsample)
        if self.transformer_downsample not in {"adaptive_avg", "stride_conv"}:
            raise ValueError("transformer_downsample must be 'adaptive_avg' or 'stride_conv'")
        if self.mode == "none" or self.embedding_dim == 0:
            self.mode = "none"
        if self.mode != "none" and (self.waveform_channels <= 0 or self.waveform_length <= 0):
            raise ValueError("waveform encoder requires waveform_channels > 0 and waveform_length > 0")

        if self.mode == "none":
            self.encoder = None
            self.gru = None
            self.proj = None
            self.positional = None
            self.transformer = None
            self.transformer_downsample_conv = None
            return

        conv_hidden = max(self.embedding_dim // 2, 16)
        waveform_norm: nn.Module
        if self.mode == "transformer":
            waveform_norm = nn.GroupNorm(1, conv_hidden)
        else:
            waveform_norm = nn.LayerNorm([conv_hidden, self.waveform_length])
        self.encoder = nn.Sequential(
            nn.Conv1d(self.waveform_channels, conv_hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            waveform_norm,
            nn.Conv1d(conv_hidden, self.embedding_dim, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        if self.mode == "cnn-gru":
            gru_hidden = max(self.embedding_dim // 2, 1)
            self.gru = nn.GRU(
                input_size=self.embedding_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.proj = nn.Sequential(
                nn.Linear(gru_hidden * 2 + self.embedding_dim * 2, self.embedding_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.positional = None
            self.transformer = None
            self.transformer_downsample_conv = None
        elif self.mode == "transformer":
            heads = max(int(transformer_heads), 1)
            if self.embedding_dim % heads != 0:
                heads = 1
            layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=heads,
                dim_feedforward=self.embedding_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=max(int(transformer_layers), 1))
            self.positional = nn.Parameter(torch.zeros(1, self.transformer_max_tokens, self.embedding_dim))
            stride = max((self.waveform_length + self.transformer_max_tokens - 1) // self.transformer_max_tokens, 1)
            self.transformer_downsample_conv = (
                nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=3, padding=1, stride=stride)
                if self.transformer_downsample == "stride_conv"
                else None
            )
            self.proj = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.gru = None
        elif self.mode == "cnn":
            self.proj = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.gru = None
            self.positional = None
            self.transformer = None
            self.transformer_downsample_conv = None
        else:
            raise ValueError("waveform_encoder must be 'none', 'cnn', 'cnn-gru', or 'transformer'")

    @property
    def output_dim(self) -> int:
        return 0 if self.mode == "none" else self.embedding_dim

    def _encode_valid_waveforms(self, waveform: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        encoded = self.encoder(waveform)
        if self.mode == "transformer":
            if encoded.shape[-1] > self.transformer_max_tokens:
                if self.transformer_downsample_conv is not None:
                    encoded = self.transformer_downsample_conv(encoded)
                    if encoded.shape[-1] > self.transformer_max_tokens:
                        encoded = F.adaptive_avg_pool1d(encoded, self.transformer_max_tokens)
                else:
                    encoded = F.adaptive_avg_pool1d(encoded, self.transformer_max_tokens)
            encoded = encoded.transpose(1, 2)
            positional = self.positional[:, : encoded.shape[1]].to(device=device, dtype=dtype)
            transformed = self.transformer(encoded + positional)
            return self.proj(torch.cat([transformed.mean(dim=1), transformed.amax(dim=1)], dim=-1))
        encoded = encoded.transpose(1, 2)
        if self.mode == "cnn-gru":
            recurrent, _hidden = self.gru(encoded)
            pooled = torch.cat([recurrent[:, -1], encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)
            return self.proj(pooled)
        return self.proj(torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1))

    def forward(
        self,
        waveform_x: torch.Tensor | None,
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(num_nodes, 0, dtype=dtype, device=device)
        if waveform_x is None:
            raise ValueError("waveform encoder is enabled, but batch has no waveform_x")
        waveform = waveform_x.to(device=device, dtype=dtype)
        if waveform.ndim != 3 or waveform.shape[0] != num_nodes or waveform.shape[1] != self.waveform_channels:
            raise ValueError("waveform_x shape does not match model waveform configuration")
        if valid_mask is None:
            return self._encode_valid_waveforms(waveform, device=device, dtype=dtype)
        valid = valid_mask.to(device=device).reshape(-1) > 0.5
        if valid.shape[0] != num_nodes:
            raise ValueError("valid_mask length does not match waveform_x")
        output = torch.zeros(num_nodes, self.output_dim, dtype=dtype, device=device)
        if bool(valid.any()):
            encoded = self._encode_valid_waveforms(waveform[valid], device=device, dtype=dtype)
            output[valid] = encoded.to(dtype=output.dtype)
        return output


class TaleSdGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        pulse_dim: int = 0,
        target_dim: int = 6,
        classification_dim: int = 0,
        quality_dim: int = 0,
        error_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.05,
        detector_lids: Sequence[int] | None = None,
        detector_embedding_dim: int = 0,
        time_edge_encoder: bool = True,
        waveform_channels: int = 0,
        waveform_length: int = 0,
        waveform_encoder: str = "none",
        waveform_embedding_dim: int = 64,
        waveform_transformer_heads: int = 4,
        waveform_transformer_layers: int = 1,
        waveform_transformer_max_tokens: int = 128,
        waveform_transformer_downsample: str = "adaptive_avg",
        classification_arch: str = "legacy",
    ):
        super().__init__()
        classification_arch = str(classification_arch).lower()
        if classification_arch not in {"legacy", "enhanced"}:
            raise ValueError("classification_arch must be 'legacy' or 'enhanced'")
        raw_detector_lids = [] if detector_lids is None else detector_lids
        detector_lids_list = sorted({int(lid) for lid in raw_detector_lids if int(lid) >= 0})
        self.config = {
            "architecture": "baseline",
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "pulse_dim": pulse_dim,
            "target_dim": target_dim,
            "classification_dim": classification_dim,
            "quality_dim": quality_dim,
            "error_dim": error_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "detector_lids": detector_lids_list,
            "detector_embedding_dim": int(detector_embedding_dim),
            "time_edge_encoder": bool(time_edge_encoder),
            "waveform_channels": int(waveform_channels),
            "waveform_length": int(waveform_length),
            "waveform_encoder": str(waveform_encoder),
            "waveform_embedding_dim": int(waveform_embedding_dim),
            "waveform_transformer_heads": int(waveform_transformer_heads),
            "waveform_transformer_layers": int(waveform_transformer_layers),
            "waveform_transformer_max_tokens": int(waveform_transformer_max_tokens),
            "waveform_transformer_downsample": str(waveform_transformer_downsample),
            "classification_arch": classification_arch,
        }
        self.pulse_dim = int(pulse_dim)
        self.hidden_dim = int(hidden_dim)
        self.target_dim = max(int(target_dim), 0)
        self.classification_dim = int(classification_dim)
        self.quality_dim = int(quality_dim)
        self.error_dim = int(error_dim)
        self.classification_arch = classification_arch
        self.waveform_shape_summary_enabled = int(waveform_channels) > 0 and int(waveform_length) > 0
        self.detector_encoder = DetectorIdEmbedding(detector_lids_list, detector_embedding_dim)
        self.waveform_encoder = WaveformEncoder(
            waveform_channels=waveform_channels,
            waveform_length=waveform_length,
            embedding_dim=waveform_embedding_dim,
            mode=waveform_encoder,
            dropout=dropout,
            transformer_heads=waveform_transformer_heads,
            transformer_layers=waveform_transformer_layers,
            transformer_max_tokens=waveform_transformer_max_tokens,
            transformer_downsample=waveform_transformer_downsample,
        )
        if self.pulse_dim > 0:
            self.pulse_encoder = _make_mlp(self.pulse_dim, hidden_dim, hidden_dim, dropout)
            node_input_dim = node_dim + hidden_dim * 2
        else:
            self.pulse_encoder = None
            node_input_dim = node_dim
        node_input_dim += self.detector_encoder.output_dim
        node_input_dim += self.waveform_encoder.output_dim
        self.node_encoder = nn.Sequential(nn.Linear(node_input_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.time_edge_encoder = EdgeTimeDeltaEncoder(hidden_dim, edge_dim, dropout, enabled=time_edge_encoder)
        self.layers = nn.ModuleList(
            [EdgeMessageLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = None
        if self.target_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, self.target_dim),
            )
        self.class_head = None
        if self.classification_dim > 0:
            class_input_dim = hidden_dim * 2
            if self.classification_arch == "enhanced":
                class_input_dim += hidden_dim * 2 + 3
                class_input_dim += self.waveform_encoder.output_dim * 2
                if self.waveform_shape_summary_enabled:
                    class_input_dim += WAVEFORM_SHAPE_DIM * 2
            self.class_head = _make_classification_head(
                class_input_dim,
                hidden_dim,
                self.classification_dim,
                dropout,
                enhanced=self.classification_arch == "enhanced",
            )
        self.quality_head = None
        if self.quality_dim > 0:
            self.quality_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.quality_dim),
            )
        self.error_head = None
        if self.error_dim > 0:
            self.error_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.error_dim),
            )

    def _pool(self, node: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        mean = torch.zeros(num_graphs, node.shape[1], dtype=node.dtype, device=node.device)
        mean.index_add_(0, batch, node)
        counts = torch.zeros(num_graphs, 1, dtype=node.dtype, device=node.device)
        counts.index_add_(0, batch, torch.ones(node.shape[0], 1, dtype=node.dtype, device=node.device))
        mean = mean / counts.clamp_min(1.0)

        max_pool = torch.full_like(mean, -torch.inf)
        if hasattr(max_pool, "scatter_reduce_"):
            index = batch[:, None].expand(-1, node.shape[1])
            max_pool.scatter_reduce_(0, index, node, reduce="amax", include_self=True)
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        else:
            max_values = []
            for graph_index in range(num_graphs):
                graph_nodes = node[batch == graph_index]
                if graph_nodes.numel() == 0:
                    max_values.append(torch.zeros(node.shape[1], dtype=node.dtype, device=node.device))
                else:
                    max_values.append(torch.max(graph_nodes, dim=0).values)
            max_pool = torch.stack(max_values, dim=0)
        return torch.cat([mean, max_pool], dim=-1)

    def _pulse_pool(self, pulse_x: torch.Tensor, pulse_node_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self.pulse_encoder is None:
            return torch.zeros(num_nodes, 0, dtype=pulse_x.dtype, device=pulse_x.device)
        if pulse_x.numel() == 0:
            return torch.zeros(num_nodes, self.hidden_dim * 2, dtype=pulse_x.dtype, device=pulse_x.device)

        encoded = self.pulse_encoder(pulse_x)
        mean = torch.zeros(num_nodes, encoded.shape[1], dtype=encoded.dtype, device=encoded.device)
        mean.index_add_(0, pulse_node_index, encoded)
        counts = torch.zeros(num_nodes, 1, dtype=encoded.dtype, device=encoded.device)
        counts.index_add_(0, pulse_node_index, torch.ones(encoded.shape[0], 1, dtype=encoded.dtype, device=encoded.device))
        mean = mean / counts.clamp_min(1.0)

        max_pool = torch.full_like(mean, -torch.inf)
        if hasattr(max_pool, "scatter_reduce_"):
            index = pulse_node_index[:, None].expand(-1, encoded.shape[1])
            max_pool.scatter_reduce_(0, index, encoded, reduce="amax", include_self=True)
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        else:
            for node_index in torch.unique(pulse_node_index).tolist():
                mask = pulse_node_index == int(node_index)
                max_pool[int(node_index)] = torch.max(encoded[mask], dim=0).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean, max_pool], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        detector_embedding = self.detector_encoder(
            batch.get("detector_lids"),
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        waveform_embedding = self.waveform_encoder(
            batch.get("waveform_x"),
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        if self.pulse_encoder is not None:
            pulse_summary = self._pulse_pool(
                batch.get("pulse_x", x.new_zeros((0, self.pulse_dim))),
                batch.get("pulse_node_index", torch.zeros(0, dtype=torch.long, device=x.device)),
                x.shape[0],
            )
            x = torch.cat([x, pulse_summary], dim=-1)
        if detector_embedding.shape[1] > 0:
            x = torch.cat([x, detector_embedding], dim=-1)
        if waveform_embedding.shape[1] > 0:
            x = torch.cat([x, waveform_embedding], dim=-1)

        node = self.node_encoder(x)
        node_initial = node
        edge_index = batch["edge_index"]
        if edge_index.numel() == 0:
            src = None
            dst = None
            degree = None
        else:
            src = edge_index[0]
            dst = edge_index[1]
            degree = batch.get("edge_dst_degree")
            if degree is None:
                degree = torch.zeros(node.shape[0], 1, dtype=node.dtype, device=node.device)
                degree.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=node.dtype, device=node.device))
            else:
                degree = degree.to(dtype=node.dtype)
        node = self.time_edge_encoder(node, batch["edge_attr"], dst, degree)
        for layer in self.layers:
            node = layer(node, src, dst, batch["edge_attr"], degree)
        num_graphs = int(batch["num_graphs"])
        pooled = self._pool(node, batch["batch"], num_graphs)
        if self.head is None:
            reconstruction = pooled.new_zeros((num_graphs, 0))
        else:
            reconstruction = self.head(pooled)
        outputs = [reconstruction]
        if self.class_head is not None:
            class_input = pooled
            if self.classification_arch == "enhanced":
                initial_pooled = self._pool(node_initial, batch["batch"], num_graphs)
                context = _graph_context(batch["batch"], src, num_graphs, dtype=node.dtype, device=node.device)
                waveform_direct = _waveform_mass_readout(
                    waveform_embedding,
                    batch.get("waveform_x"),
                    batch["batch"],
                    num_graphs,
                    summary_enabled=self.waveform_shape_summary_enabled,
                )
                class_input = torch.cat([pooled, initial_pooled, context, waveform_direct], dim=-1)
            outputs.append(self.class_head(class_input))
        if self.quality_head is not None:
            outputs.append(self.quality_head(pooled))
        if self.error_head is not None:
            outputs.append(self.error_head(pooled))
        return torch.cat(outputs, dim=-1)


class PhysicsTaleSdGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        pulse_dim: int = 0,
        target_dim: int = 6,
        classification_dim: int = 0,
        quality_dim: int = 0,
        error_dim: int = 0,
        hidden_dim: int = 160,
        num_layers: int = 5,
        dropout: float = 0.05,
        readout_heads: int = 4,
        detector_lids: Sequence[int] | None = None,
        detector_embedding_dim: int = 0,
        time_edge_encoder: bool = True,
        waveform_channels: int = 0,
        waveform_length: int = 0,
        waveform_encoder: str = "none",
        waveform_embedding_dim: int = 64,
        waveform_transformer_heads: int = 4,
        waveform_transformer_layers: int = 1,
        waveform_transformer_max_tokens: int = 128,
        waveform_transformer_downsample: str = "adaptive_avg",
        classification_arch: str = "legacy",
    ):
        super().__init__()
        classification_arch = str(classification_arch).lower()
        if classification_arch not in {"legacy", "enhanced"}:
            raise ValueError("classification_arch must be 'legacy' or 'enhanced'")
        raw_detector_lids = [] if detector_lids is None else detector_lids
        detector_lids_list = sorted({int(lid) for lid in raw_detector_lids if int(lid) >= 0})
        self.config = {
            "architecture": "physics",
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "pulse_dim": pulse_dim,
            "target_dim": target_dim,
            "classification_dim": classification_dim,
            "quality_dim": quality_dim,
            "error_dim": error_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "readout_heads": readout_heads,
            "detector_lids": detector_lids_list,
            "detector_embedding_dim": int(detector_embedding_dim),
            "time_edge_encoder": bool(time_edge_encoder),
            "waveform_channels": int(waveform_channels),
            "waveform_length": int(waveform_length),
            "waveform_encoder": str(waveform_encoder),
            "waveform_embedding_dim": int(waveform_embedding_dim),
            "waveform_transformer_heads": int(waveform_transformer_heads),
            "waveform_transformer_layers": int(waveform_transformer_layers),
            "waveform_transformer_max_tokens": int(waveform_transformer_max_tokens),
            "waveform_transformer_downsample": str(waveform_transformer_downsample),
            "classification_arch": classification_arch,
        }
        self.pulse_dim = int(pulse_dim)
        self.hidden_dim = int(hidden_dim)
        self.target_dim = max(int(target_dim), 0)
        if self.target_dim not in {0, 6, 7}:
            raise ValueError("physics reconstruction target_dim must be 0, 6, or legacy 7")
        self.core_output_dim = 3 if self.target_dim == 7 else 2
        self.classification_dim = int(classification_dim)
        self.quality_dim = int(quality_dim)
        self.error_dim = int(error_dim)
        self.classification_arch = classification_arch
        self.waveform_shape_summary_enabled = int(waveform_channels) > 0 and int(waveform_length) > 0
        self.detector_encoder = DetectorIdEmbedding(detector_lids_list, detector_embedding_dim)
        self.waveform_encoder = WaveformEncoder(
            waveform_channels=waveform_channels,
            waveform_length=waveform_length,
            embedding_dim=waveform_embedding_dim,
            mode=waveform_encoder,
            dropout=dropout,
            transformer_heads=waveform_transformer_heads,
            transformer_layers=waveform_transformer_layers,
            transformer_max_tokens=waveform_transformer_max_tokens,
            transformer_downsample=waveform_transformer_downsample,
        )
        if self.pulse_dim > 0:
            self.pulse_encoder = _make_mlp(self.pulse_dim, hidden_dim, hidden_dim, dropout)
            node_input_dim = node_dim + hidden_dim * 2
        else:
            self.pulse_encoder = None
            node_input_dim = node_dim
        node_input_dim += self.detector_encoder.output_dim
        node_input_dim += self.waveform_encoder.output_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.time_edge_encoder = EdgeTimeDeltaEncoder(hidden_dim, edge_dim, dropout, enabled=time_edge_encoder)
        self.layers = nn.ModuleList(
            [GatedEdgeMessageLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = AttentiveReadout(hidden_dim, heads=readout_heads)
        pooled_dim = hidden_dim * (2 + readout_heads)
        needs_shared = (
            self.target_dim > 0
            or self.quality_dim > 0
            or self.error_dim > 0
            or (self.classification_dim > 0 and self.classification_arch != "enhanced")
        )
        self.shared = None
        if needs_shared:
            self.shared = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
            )
        self.energy_head = None
        self.core_head = None
        self.direction_head = None
        if self.target_dim > 0:
            self.energy_head = nn.Linear(hidden_dim, 1)
            self.core_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.core_output_dim),
            )
            self.direction_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.class_head = None
        if self.classification_dim > 0:
            class_input_dim = hidden_dim
            if self.classification_arch == "enhanced":
                class_input_dim = pooled_dim + hidden_dim * 2 + 3
                class_input_dim += self.waveform_encoder.output_dim * 2
                if self.waveform_shape_summary_enabled:
                    class_input_dim += WAVEFORM_SHAPE_DIM * 2
            self.class_head = _make_classification_head(
                class_input_dim,
                hidden_dim,
                self.classification_dim,
                dropout,
                enhanced=self.classification_arch == "enhanced",
            )
        self.quality_head = None
        if self.quality_dim > 0:
            self.quality_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.quality_dim),
            )
        self.error_head = None
        if self.error_dim > 0:
            self.error_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.error_dim),
            )

    def _pulse_pool(self, pulse_x: torch.Tensor, pulse_node_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self.pulse_encoder is None:
            return torch.zeros(num_nodes, 0, dtype=pulse_x.dtype, device=pulse_x.device)
        if pulse_x.numel() == 0:
            return torch.zeros(num_nodes, self.hidden_dim * 2, dtype=pulse_x.dtype, device=pulse_x.device)
        encoded = self.pulse_encoder(pulse_x)
        mean = torch.zeros(num_nodes, encoded.shape[1], dtype=encoded.dtype, device=encoded.device)
        mean.index_add_(0, pulse_node_index, encoded)
        counts = torch.zeros(num_nodes, 1, dtype=encoded.dtype, device=encoded.device)
        counts.index_add_(0, pulse_node_index, torch.ones(encoded.shape[0], 1, dtype=encoded.dtype, device=encoded.device))
        mean = mean / counts.clamp_min(1.0)
        max_pool = torch.full_like(mean, -torch.inf)
        if hasattr(max_pool, "scatter_reduce_"):
            index = pulse_node_index[:, None].expand(-1, encoded.shape[1])
            max_pool.scatter_reduce_(0, index, encoded, reduce="amax", include_self=True)
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        else:
            for node_index in torch.unique(pulse_node_index).tolist():
                mask = pulse_node_index == int(node_index)
                max_pool[int(node_index)] = torch.max(encoded[mask], dim=0).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean, max_pool], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        detector_embedding = self.detector_encoder(
            batch.get("detector_lids"),
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        waveform_embedding = self.waveform_encoder(
            batch.get("waveform_x"),
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        if self.pulse_encoder is not None:
            pulse_summary = self._pulse_pool(
                batch.get("pulse_x", x.new_zeros((0, self.pulse_dim))),
                batch.get("pulse_node_index", torch.zeros(0, dtype=torch.long, device=x.device)),
                x.shape[0],
            )
            x = torch.cat([x, pulse_summary], dim=-1)
        if detector_embedding.shape[1] > 0:
            x = torch.cat([x, detector_embedding], dim=-1)
        if waveform_embedding.shape[1] > 0:
            x = torch.cat([x, waveform_embedding], dim=-1)
        node = self.node_encoder(x)
        node_initial = node
        edge_index = batch["edge_index"]
        if edge_index.numel() == 0:
            src = None
            dst = None
            degree = None
        else:
            src = edge_index[0]
            dst = edge_index[1]
            degree = batch.get("edge_dst_degree")
            if degree is None:
                degree = torch.zeros(node.shape[0], 1, dtype=node.dtype, device=node.device)
                degree.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=node.dtype, device=node.device))
            else:
                degree = degree.to(dtype=node.dtype)
        node = self.time_edge_encoder(node, batch["edge_attr"], dst, degree)
        for layer in self.layers:
            node = layer(node, src, dst, batch["edge_attr"], degree)
        num_graphs = int(batch["num_graphs"])
        pooled = torch.cat(
            [
                _scatter_mean(node, batch["batch"], num_graphs),
                _scatter_max(node, batch["batch"], num_graphs),
                self.readout(node, batch["batch"], num_graphs),
            ],
            dim=-1,
        )
        shared = self.shared(pooled) if self.shared is not None else None
        if self.target_dim <= 0:
            reconstruction = pooled.new_zeros((num_graphs, 0))
        else:
            if shared is None or self.energy_head is None or self.core_head is None or self.direction_head is None:
                raise RuntimeError("reconstruction heads are not initialized")
            reconstruction = torch.cat(
                [
                    self.energy_head(shared),
                    self.core_head(shared),
                    self.direction_head(shared),
                ],
                dim=-1,
            )
        outputs = [reconstruction]
        if self.class_head is not None:
            if shared is None and self.classification_arch != "enhanced":
                raise RuntimeError("legacy classification head requires the shared readout")
            class_input = shared
            if self.classification_arch == "enhanced":
                initial_pooled = torch.cat(
                    [
                        _scatter_mean(node_initial, batch["batch"], num_graphs),
                        _scatter_max(node_initial, batch["batch"], num_graphs),
                    ],
                    dim=-1,
                )
                context = _graph_context(batch["batch"], src, num_graphs, dtype=node.dtype, device=node.device)
                waveform_direct = _waveform_mass_readout(
                    waveform_embedding,
                    batch.get("waveform_x"),
                    batch["batch"],
                    num_graphs,
                    summary_enabled=self.waveform_shape_summary_enabled,
                )
                class_input = torch.cat([pooled, initial_pooled, context, waveform_direct], dim=-1)
            outputs.append(self.class_head(class_input))
        if self.quality_head is not None:
            if shared is None:
                raise RuntimeError("quality head requires the shared readout")
            outputs.append(self.quality_head(shared))
        if self.error_head is not None:
            if shared is None:
                raise RuntimeError("error head requires the shared readout")
            outputs.append(self.error_head(shared))
        return torch.cat(outputs, dim=-1)


def build_model_from_config(config: dict) -> nn.Module:
    config = dict(config)
    architecture = str(config.pop("architecture", "baseline"))
    if "time_edge_encoder" not in config:
        config["time_edge_encoder"] = False
    if architecture == "baseline":
        config.pop("readout_heads", None)
        return TaleSdGNN(**config)
    if architecture == "physics":
        return PhysicsTaleSdGNN(**config)
    raise ValueError(f"unknown model architecture: {architecture}")
