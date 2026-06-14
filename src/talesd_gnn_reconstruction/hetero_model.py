from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .constants import WAVEFORM_RISE_ANCHOR_BIN, WAVEFORM_TRACE_BINS
from .hetero_data import (
    EDGE_TYPE_BY_RELATION,
    hetero_data_to_tensors,
    hetero_sample_to_tensors,
    normalize_detector_waveforms,
)
from .model import WaveformEncoder, _make_mlp, _scatter_max, _scatter_mean, _scatter_softmax


NODE_TYPE_BY_RELATION = {
    "pulse__same_detector_next__pulse": ("pulse", "pulse"),
    "pulse__same_detector_prev__pulse": ("pulse", "pulse"),
    "pulse__near_space__pulse": ("pulse", "pulse"),
    "pulse__time_causal__pulse": ("pulse", "pulse"),
    "detector__near__detector": ("detector", "detector"),
    "detector__observes__pulse": ("detector", "pulse"),
    "pulse__observed_by__detector": ("pulse", "detector"),
}


class HeteroMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dims: Mapping[str, int],
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.messages = nn.ModuleDict(
            {
                relation: _make_mlp(self.hidden_dim * 2 + int(edge_dim), self.hidden_dim, self.hidden_dim, dropout)
                for relation, edge_dim in edge_dims.items()
            }
        )
        self.detector_update = _make_mlp(self.hidden_dim * 2, self.hidden_dim, self.hidden_dim, dropout)
        self.pulse_update = _make_mlp(self.hidden_dim * 2, self.hidden_dim, self.hidden_dim, dropout)
        self.detector_norm = nn.LayerNorm(self.hidden_dim)
        self.pulse_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        node_states: dict[str, torch.Tensor],
        edge_index_by_type: Mapping[str, torch.Tensor],
        edge_features_by_type: Mapping[str, torch.Tensor],
        node_masks: Mapping[str, torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        aggregates = {node_type: torch.zeros_like(state) for node_type, state in node_states.items()}
        counts = {
            node_type: torch.zeros(state.shape[0], 1, dtype=state.dtype, device=state.device)
            for node_type, state in node_states.items()
        }
        for relation, message_mlp in self.messages.items():
            edge_index = edge_index_by_type.get(relation)
            if edge_index is None or edge_index.numel() == 0:
                continue
            edge_attr = edge_features_by_type[relation].to(
                dtype=node_states[NODE_TYPE_BY_RELATION[relation][0]].dtype,
                device=node_states[NODE_TYPE_BY_RELATION[relation][0]].device,
            )
            src_type, dst_type = NODE_TYPE_BY_RELATION[relation]
            src_index = edge_index[0].to(device=node_states[src_type].device)
            dst_index = edge_index[1].to(device=node_states[dst_type].device)
            if node_masks:
                edge_keep = torch.ones(src_index.shape[0], dtype=torch.bool, device=src_index.device)
                src_mask = node_masks.get(src_type)
                dst_mask = node_masks.get(dst_type)
                if src_mask is not None:
                    edge_keep &= src_mask.to(device=src_index.device, dtype=torch.bool).reshape(-1)[src_index]
                if dst_mask is not None:
                    edge_keep &= dst_mask.to(device=dst_index.device, dtype=torch.bool).reshape(-1)[dst_index]
                if not bool(edge_keep.any()):
                    continue
                src_index = src_index[edge_keep]
                dst_index = dst_index[edge_keep]
                edge_attr = edge_attr[edge_keep]
            message_input = torch.cat(
                [node_states[src_type][src_index], node_states[dst_type][dst_index], edge_attr],
                dim=-1,
            )
            messages = message_mlp(message_input)
            messages = messages.to(dtype=aggregates[dst_type].dtype)
            aggregates[dst_type].index_add_(0, dst_index, messages)
            counts[dst_type].index_add_(
                0,
                dst_index,
                torch.ones(dst_index.shape[0], 1, dtype=counts[dst_type].dtype, device=counts[dst_type].device),
            )
        detector_aggregate = aggregates["detector"] / counts["detector"].clamp_min(1.0)
        pulse_aggregate = aggregates["pulse"] / counts["pulse"].clamp_min(1.0)
        return {
            "detector": self.detector_norm(
                node_states["detector"]
                + self.detector_update(torch.cat([node_states["detector"], detector_aggregate], dim=-1))
            ),
            "pulse": self.pulse_norm(
                node_states["pulse"] + self.pulse_update(torch.cat([node_states["pulse"], pulse_aggregate], dim=-1))
            ),
        }


class HeteroAttentionMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dims: Mapping[str, int],
        dropout: float = 0.05,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        requested_heads = max(int(attention_heads), 1)
        self.heads = requested_heads if self.hidden_dim % requested_heads == 0 else 1
        self.head_dim = self.hidden_dim // self.heads
        self.query = nn.ModuleDict({relation: nn.Linear(self.hidden_dim, self.hidden_dim) for relation in edge_dims})
        self.key = nn.ModuleDict(
            {
                relation: nn.Linear(self.hidden_dim + int(edge_dim), self.hidden_dim)
                for relation, edge_dim in edge_dims.items()
            }
        )
        self.value = nn.ModuleDict(
            {
                relation: nn.Linear(self.hidden_dim + int(edge_dim), self.hidden_dim)
                for relation, edge_dim in edge_dims.items()
            }
        )
        self.relation_output = nn.ModuleDict(
            {relation: nn.Linear(self.hidden_dim, self.hidden_dim) for relation in edge_dims}
        )
        self.detector_update = _make_mlp(self.hidden_dim * 2, self.hidden_dim, self.hidden_dim, dropout)
        self.pulse_update = _make_mlp(self.hidden_dim * 2, self.hidden_dim, self.hidden_dim, dropout)
        self.detector_norm = nn.LayerNorm(self.hidden_dim)
        self.pulse_norm = nn.LayerNorm(self.hidden_dim)
        self.detector_ffn = _make_mlp(self.hidden_dim, self.hidden_dim * 2, self.hidden_dim, dropout)
        self.pulse_ffn = _make_mlp(self.hidden_dim, self.hidden_dim * 2, self.hidden_dim, dropout)
        self.detector_ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.pulse_ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: dict[str, torch.Tensor],
        edge_index_by_type: Mapping[str, torch.Tensor],
        edge_features_by_type: Mapping[str, torch.Tensor],
        *,
        return_attention: bool = False,
        node_masks: Mapping[str, torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, Any]]:
        aggregates = {node_type: torch.zeros_like(state) for node_type, state in node_states.items()}
        counts = {
            node_type: torch.zeros(state.shape[0], 1, dtype=state.dtype, device=state.device)
            for node_type, state in node_states.items()
        }
        attention_by_relation: dict[str, dict[str, torch.Tensor | str]] = {}
        scale = float(self.head_dim) ** -0.5
        for relation in self.query.keys():
            edge_index = edge_index_by_type.get(relation)
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_type, dst_type = NODE_TYPE_BY_RELATION[relation]
            src_state = node_states[src_type]
            dst_state = node_states[dst_type]
            src_index = edge_index[0].to(device=src_state.device)
            dst_index = edge_index[1].to(device=dst_state.device)
            edge_attr = edge_features_by_type[relation].to(dtype=src_state.dtype, device=src_state.device)
            if node_masks:
                edge_keep = torch.ones(src_index.shape[0], dtype=torch.bool, device=src_index.device)
                src_mask = node_masks.get(src_type)
                dst_mask = node_masks.get(dst_type)
                if src_mask is not None:
                    edge_keep &= src_mask.to(device=src_index.device, dtype=torch.bool).reshape(-1)[src_index]
                if dst_mask is not None:
                    edge_keep &= dst_mask.to(device=dst_index.device, dtype=torch.bool).reshape(-1)[dst_index]
                if not bool(edge_keep.any()):
                    continue
                src_index = src_index[edge_keep]
                dst_index = dst_index[edge_keep]
                edge_index = edge_index[:, edge_keep]
                edge_attr = edge_attr[edge_keep]
            src_input = torch.cat([src_state[src_index], edge_attr], dim=-1)
            query = self.query[relation](dst_state[dst_index]).view(-1, self.heads, self.head_dim)
            key = self.key[relation](src_input).view(-1, self.heads, self.head_dim)
            value = self.value[relation](src_input).view(-1, self.heads, self.head_dim)
            scores = (query * key).sum(dim=-1) * scale
            weights = _scatter_softmax(scores, dst_index, dst_state.shape[0])
            weights = weights.to(dtype=value.dtype)
            if return_attention:
                attention_by_relation[str(relation)] = {
                    "src_type": src_type,
                    "dst_type": dst_type,
                    "edge_index": edge_index.detach(),
                    "scores": scores.detach(),
                    "weights": weights.detach(),
                }
            messages = (value * weights[:, :, None]).reshape(-1, self.hidden_dim)
            messages = self.dropout(self.relation_output[relation](messages))
            messages = messages.to(dtype=aggregates[dst_type].dtype)
            aggregates[dst_type].index_add_(0, dst_index, messages)

            present = torch.zeros(dst_state.shape[0], 1, dtype=dst_state.dtype, device=dst_state.device)
            present[torch.unique(dst_index)] = 1.0
            counts[dst_type] += present

        detector_aggregate = aggregates["detector"] / counts["detector"].clamp_min(1.0)
        pulse_aggregate = aggregates["pulse"] / counts["pulse"].clamp_min(1.0)
        detector = self.detector_norm(
            node_states["detector"]
            + self.detector_update(torch.cat([node_states["detector"], detector_aggregate], dim=-1))
        )
        pulse = self.pulse_norm(
            node_states["pulse"] + self.pulse_update(torch.cat([node_states["pulse"], pulse_aggregate], dim=-1))
        )
        output = {
            "detector": self.detector_ffn_norm(detector + self.detector_ffn(detector)),
            "pulse": self.pulse_ffn_norm(pulse + self.pulse_ffn(pulse)),
        }
        if return_attention:
            return output, {"relations": attention_by_relation}
        return output


class HeteroAttentiveReadout(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = max(int(heads), 1)
        self.score = nn.Linear(self.hidden_dim, self.heads)
        self.output_dim = self.hidden_dim * (2 + self.heads)

    def forward(
        self,
        state: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weights = _scatter_softmax(self.score(state), batch, num_graphs).to(dtype=state.dtype)
        pooled = [_scatter_mean(state, batch, num_graphs), _scatter_max(state, batch, num_graphs)]
        for head in range(self.heads):
            weighted = state * weights[:, head : head + 1]
            out = torch.zeros(num_graphs, state.shape[1], dtype=state.dtype, device=state.device)
            out.index_add_(0, batch, weighted.to(dtype=out.dtype))
            pooled.append(out)
        output = torch.cat(pooled, dim=-1)
        if return_attention:
            return output, weights.detach()
        return output


def _detector_index_for_pulses(
    pulse: Mapping[str, Any],
    edge_index_by_type: Mapping[str, torch.Tensor],
    *,
    n_pulse: int,
    n_detector: int,
    device: torch.device,
) -> torch.Tensor | None:
    if n_pulse <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    detector_index = pulse.get("detector_index")
    if detector_index is not None:
        values = detector_index.to(device=device, dtype=torch.long).reshape(-1)
        if values.numel() == n_pulse and bool(((values >= 0) & (values < n_detector)).all()):
            return values

    observes = edge_index_by_type.get("detector__observes__pulse")
    if observes is None or observes.numel() == 0:
        return None
    edge_index = observes.to(device=device, dtype=torch.long)
    detector_rows = edge_index[0].reshape(-1)
    pulse_rows = edge_index[1].reshape(-1)
    valid = (detector_rows >= 0) & (detector_rows < n_detector) & (pulse_rows >= 0) & (pulse_rows < n_pulse)
    if not bool(valid.any()):
        return None
    output = torch.full((n_pulse,), -1, dtype=torch.long, device=device)
    output[pulse_rows[valid]] = detector_rows[valid]
    return output


def _pulse_waveform_windows(
    detector_waveform: torch.Tensor,
    pulse_detector_index: torch.Tensor | None,
    pulse_bounds: torch.Tensor | None,
    *,
    window_length: int,
    rise_anchor_bin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_detector, channels, waveform_length = detector_waveform.shape
    n_pulse = 0 if pulse_detector_index is None else int(pulse_detector_index.numel())
    windows = torch.zeros(
        (n_pulse, channels, int(window_length)),
        dtype=detector_waveform.dtype,
        device=detector_waveform.device,
    )
    valid = torch.zeros((n_pulse,), dtype=torch.bool, device=detector_waveform.device)
    if n_pulse == 0 or n_detector <= 0 or waveform_length <= 0 or pulse_bounds is None:
        return windows, valid

    detector_index = pulse_detector_index.to(device=detector_waveform.device, dtype=torch.long).reshape(-1)
    bounds = pulse_bounds.to(device=detector_waveform.device, dtype=torch.float32)
    if bounds.ndim != 2 or bounds.shape[0] != n_pulse or bounds.shape[1] < 4:
        return windows, valid

    detector_valid = (detector_index >= 0) & (detector_index < n_detector)
    finite_bounds = torch.isfinite(bounds[:, :4]).all(dim=1)
    valid = detector_valid & finite_bounds
    if not bool(valid.any()):
        return windows, valid

    safe_detector = detector_index.clamp(0, max(n_detector - 1, 0))
    onset = torch.minimum(bounds[:, 0], bounds[:, 2])
    start = torch.floor(onset).to(dtype=torch.long) - int(rise_anchor_bin)
    offsets = torch.arange(int(window_length), dtype=torch.long, device=detector_waveform.device)
    indices = start[:, None] + offsets[None, :]
    inside = (indices >= 0) & (indices < waveform_length) & valid[:, None]
    safe_indices = indices.clamp(0, max(waveform_length - 1, 0))
    gathered = detector_waveform[safe_detector].gather(2, safe_indices[:, None, :].expand(-1, channels, -1))
    windows = torch.where(inside[:, None, :], gathered, windows)
    return windows, valid


def _pulse_bounds_norm(
    pulse_bounds: torch.Tensor | None,
    *,
    n_pulse: int,
    waveform_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros((int(n_pulse), 4), dtype=dtype, device=device)
    valid = torch.zeros((int(n_pulse),), dtype=torch.bool, device=device)
    if pulse_bounds is None or int(n_pulse) <= 0:
        return output, valid
    bounds = pulse_bounds.to(device=device, dtype=torch.float32)
    if bounds.ndim != 2 or bounds.shape[0] != int(n_pulse) or bounds.shape[1] < 4:
        return output, valid
    finite = torch.isfinite(bounds[:, :4]).all(dim=1)
    scale = float(max(int(waveform_length) - 1, 1))
    output = torch.where(finite[:, None], (bounds[:, :4] / scale).to(dtype=dtype), output)
    return output, finite


def _relative_position_features(
    node: Mapping[str, Any],
    *,
    n_nodes: int,
    position_anchor: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    output = torch.zeros((int(n_nodes), 4), dtype=dtype, device=device)
    pos = node.get("pos")
    if pos is None or int(n_nodes) <= 0:
        return output, False
    positions = pos.to(device=device, dtype=torch.float32)
    if positions.ndim != 2 or positions.shape[0] != int(n_nodes) or positions.shape[1] < 2:
        return output, False
    batch = node.get("batch")
    if batch is None:
        batch_index = torch.zeros((int(n_nodes),), dtype=torch.long, device=device)
    else:
        batch_index = batch.to(device=device, dtype=torch.long).reshape(-1)
        if batch_index.numel() != int(n_nodes):
            batch_index = torch.zeros((int(n_nodes),), dtype=torch.long, device=device)
    if position_anchor is None:
        anchor = torch.zeros((int(n_nodes), 3), dtype=torch.float32, device=device)
    else:
        anchors = position_anchor.to(device=device, dtype=torch.float32)
        if anchors.ndim == 1:
            anchors = anchors.reshape(1, -1)
        if anchors.shape[1] < 2:
            anchor = torch.zeros((int(n_nodes), 3), dtype=torch.float32, device=device)
        else:
            safe_batch = batch_index.clamp(0, max(int(anchors.shape[0]) - 1, 0))
            anchor = torch.zeros((int(n_nodes), 3), dtype=torch.float32, device=device)
            anchor[:, : min(int(anchors.shape[1]), 3)] = anchors[safe_batch, : min(int(anchors.shape[1]), 3)]
    rel_xy = positions[:, :2] - anchor[:, :2]
    if positions.shape[1] >= 3:
        rel_z = positions[:, 2:3] - anchor[:, 2:3]
    else:
        rel_z = torch.zeros((int(n_nodes), 1), dtype=torch.float32, device=device)
    radius = torch.sqrt((rel_xy.square().sum(dim=1, keepdim=True) + rel_z.square()).clamp_min(0.0))
    return torch.cat([rel_xy, rel_z, radius], dim=1).to(dtype=dtype), True


def _masked_readout_input(
    state: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return state, batch
    keep = mask.to(device=state.device, dtype=torch.bool).reshape(-1)
    if keep.numel() != state.shape[0]:
        return state, batch
    return state[keep], batch.to(device=state.device, dtype=torch.long)[keep]


def _detector_readout_mask(detector: Mapping[str, Any], mode: str) -> torch.Tensor | None:
    mode = str(mode)
    if mode == "all":
        return None
    if mode == "signal":
        return detector.get("has_signal")
    if mode == "ising_kept":
        return detector.get("has_ising_kept")
    raise ValueError(f"unknown detector_readout_mask: {mode}")


def _pulse_readout_mask(pulse: Mapping[str, Any], mode: str) -> torch.Tensor | None:
    mode = str(mode)
    if mode == "all":
        return None
    if mode != "valid":
        raise ValueError(f"unknown pulse_readout_mask: {mode}")
    detector_index = pulse.get("detector_index")
    pulse_bounds = pulse.get("pulse_bounds")
    if detector_index is None and pulse_bounds is None:
        return None
    masks = []
    if detector_index is not None:
        masks.append(detector_index.reshape(-1).to(dtype=torch.long) >= 0)
    if pulse_bounds is not None:
        bounds = pulse_bounds
        if bounds.ndim == 2 and bounds.shape[1] >= 4:
            masks.append(torch.isfinite(bounds[:, :4]).all(dim=1))
    if not masks:
        return None
    output = masks[0].to(dtype=torch.bool)
    for mask in masks[1:]:
        output = output & mask.to(device=output.device, dtype=torch.bool)
    return output


class MinimalHeteroTaleSdGNN(nn.Module):
    def __init__(
        self,
        *,
        architecture: str = "hetero_attention",
        detector_dim: int,
        detector_context_dim: int,
        pulse_dim: int,
        edge_dims: Mapping[str, int],
        waveform_channels: int,
        waveform_length: int,
        target_dim: int = 6,
        classification_dim: int = 0,
        quality_dim: int = 0,
        error_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        waveform_encoder: str = "cnn",
        waveform_embedding_dim: int = 64,
        waveform_transformer_heads: int = 4,
        waveform_transformer_layers: int = 1,
        waveform_transformer_max_tokens: int = 128,
        waveform_transformer_downsample: str = "adaptive_avg",
        use_pulse_parent_waveform: bool = True,
        use_pulse_bounds: bool = True,
        pulse_waveform_encoder: str | None = None,
        pulse_waveform_embedding_dim: int | None = None,
        pulse_waveform_window_length: int | None = None,
        pulse_waveform_rise_anchor_bin: int | None = None,
        use_relative_positions: bool = False,
        detector_readout_mask: str = "all",
        pulse_readout_mask: str = "all",
        attention_heads: int = 4,
        readout_heads: int = 4,
    ):
        super().__init__()
        architecture = str(architecture)
        if architecture not in {"minimal_hetero", "hetero_attention"}:
            raise ValueError("architecture must be 'minimal_hetero' or 'hetero_attention'")
        if detector_readout_mask not in {"all", "signal", "ising_kept"}:
            raise ValueError("detector_readout_mask must be all, signal, or ising_kept")
        if pulse_readout_mask not in {"all", "valid"}:
            raise ValueError("pulse_readout_mask must be all or valid")
        if pulse_waveform_encoder is None:
            # Backward compatibility: checkpoints created before the explicit
            # pulse waveform mode stored only a nonzero embedding dimension.
            resolved_pulse_waveform_encoder = (
                "crop_same" if pulse_waveform_embedding_dim is not None and int(pulse_waveform_embedding_dim) > 0 else "none"
            )
        else:
            resolved_pulse_waveform_encoder = str(pulse_waveform_encoder)
        if resolved_pulse_waveform_encoder not in {"none", "bounds", "crop_cnn", "crop_same"}:
            raise ValueError("pulse_waveform_encoder must be none, bounds, crop_cnn, or crop_same")
        pulse_window_embedding_dim = (
            0
            if resolved_pulse_waveform_encoder in {"none", "bounds"}
            else max(0, 0 if pulse_waveform_embedding_dim is None else int(pulse_waveform_embedding_dim))
        )
        self.config = {
            "architecture": architecture,
            "detector_dim": int(detector_dim),
            "detector_context_dim": int(detector_context_dim),
            "pulse_dim": int(pulse_dim),
            "edge_dims": {str(key): int(value) for key, value in edge_dims.items()},
            "waveform_channels": int(waveform_channels),
            "waveform_length": int(waveform_length),
            "target_dim": int(target_dim),
            "classification_dim": int(classification_dim),
            "quality_dim": int(quality_dim),
            "error_dim": int(error_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "waveform_encoder": str(waveform_encoder),
            "waveform_embedding_dim": int(waveform_embedding_dim),
            "waveform_transformer_heads": int(waveform_transformer_heads),
            "waveform_transformer_layers": int(waveform_transformer_layers),
            "waveform_transformer_max_tokens": int(waveform_transformer_max_tokens),
            "waveform_transformer_downsample": str(waveform_transformer_downsample),
            "use_pulse_parent_waveform": bool(use_pulse_parent_waveform),
            "use_pulse_bounds": bool(use_pulse_bounds),
            "pulse_waveform_encoder": resolved_pulse_waveform_encoder,
            "pulse_waveform_embedding_dim": int(pulse_window_embedding_dim),
            "pulse_waveform_window_length": int(
                pulse_waveform_window_length
                if pulse_waveform_window_length is not None
                else min(max(int(waveform_length), 1), WAVEFORM_TRACE_BINS)
            ),
            "pulse_waveform_rise_anchor_bin": int(
                pulse_waveform_rise_anchor_bin
                if pulse_waveform_rise_anchor_bin is not None
                else min(
                    WAVEFORM_RISE_ANCHOR_BIN,
                    max(
                        int(
                            pulse_waveform_window_length
                            if pulse_waveform_window_length is not None
                            else min(max(int(waveform_length), 1), WAVEFORM_TRACE_BINS)
                        )
                        - 1,
                        0,
                    ),
                )
            ),
            "use_relative_positions": bool(use_relative_positions),
            "attention_heads": int(attention_heads),
            "readout_heads": int(readout_heads),
            "detector_readout_mask": str(detector_readout_mask),
            "pulse_readout_mask": str(pulse_readout_mask),
        }
        self.architecture = architecture
        self.hidden_dim = int(hidden_dim)
        self.target_dim = max(int(target_dim), 0)
        self.classification_dim = max(int(classification_dim), 0)
        self.quality_dim = max(int(quality_dim), 0)
        self.error_dim = max(int(error_dim), 0)
        self.use_pulse_parent_waveform = bool(use_pulse_parent_waveform)
        self.use_pulse_bounds = bool(use_pulse_bounds)
        self.pulse_waveform_encoder_kind = resolved_pulse_waveform_encoder
        self.use_relative_positions = bool(use_relative_positions)
        self.pulse_waveform_embedding_dim = int(pulse_window_embedding_dim)
        self.pulse_waveform_window_length = max(int(self.config["pulse_waveform_window_length"]), 1)
        self.pulse_waveform_rise_anchor_bin = max(int(self.config["pulse_waveform_rise_anchor_bin"]), 0)
        self.detector_readout_mask = str(detector_readout_mask)
        self.pulse_readout_mask = str(pulse_readout_mask)
        self.detector_feature_encoder = _make_mlp(detector_dim, hidden_dim, hidden_dim, dropout)
        self.detector_context_encoder = _make_mlp(detector_context_dim, hidden_dim, hidden_dim, dropout)
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
        relative_position_dim = 4 if self.use_relative_positions else 0
        detector_input_dim = hidden_dim * 2 + self.waveform_encoder.output_dim + relative_position_dim
        self.detector_node_encoder = nn.Sequential(
            nn.Linear(detector_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        pulse_waveform_mode = "none"
        if self.pulse_waveform_encoder_kind == "crop_cnn":
            pulse_waveform_mode = "cnn"
        elif self.pulse_waveform_encoder_kind == "crop_same":
            pulse_waveform_mode = str(waveform_encoder)
        self.pulse_waveform_encoder = WaveformEncoder(
            waveform_channels=waveform_channels,
            waveform_length=self.pulse_waveform_window_length,
            embedding_dim=self.pulse_waveform_embedding_dim,
            mode=pulse_waveform_mode,
            dropout=dropout,
            transformer_heads=waveform_transformer_heads,
            transformer_layers=waveform_transformer_layers,
            transformer_max_tokens=min(int(waveform_transformer_max_tokens), self.pulse_waveform_window_length),
            transformer_downsample=waveform_transformer_downsample,
        )
        pulse_extra_dim = 0
        if self.use_relative_positions:
            pulse_extra_dim += relative_position_dim
        if self.use_pulse_parent_waveform:
            pulse_extra_dim += self.waveform_encoder.output_dim
        if self.use_pulse_bounds:
            pulse_extra_dim += 4
        pulse_extra_dim += self.pulse_waveform_encoder.output_dim
        if pulse_extra_dim > 0:
            self.pulse_feature_encoder = _make_mlp(pulse_dim, hidden_dim, hidden_dim, dropout)
            pulse_input_dim = hidden_dim + pulse_extra_dim
        else:
            self.pulse_feature_encoder = None
            pulse_input_dim = pulse_dim
        self.pulse_node_encoder = nn.Sequential(
            nn.Linear(pulse_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        edge_dims_complete = {relation: int(edge_dims.get(relation, 0)) for relation in EDGE_TYPE_BY_RELATION}
        if architecture == "hetero_attention":
            self.layers = nn.ModuleList(
                [
                    HeteroAttentionMessageLayer(
                        hidden_dim,
                        edge_dims_complete,
                        dropout=dropout,
                        attention_heads=attention_heads,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.detector_readout = HeteroAttentiveReadout(hidden_dim, heads=readout_heads)
            self.pulse_readout = HeteroAttentiveReadout(hidden_dim, heads=readout_heads)
            readout_dim = self.detector_readout.output_dim + self.pulse_readout.output_dim
        else:
            self.layers = nn.ModuleList(
                [HeteroMessageLayer(hidden_dim, edge_dims_complete, dropout=dropout) for _ in range(num_layers)]
            )
            self.detector_readout = None
            self.pulse_readout = None
            readout_dim = hidden_dim * 4
        self.reconstruction_head = (
            nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.target_dim),
            )
            if self.target_dim > 0
            else None
        )
        self.classification_head = (
            nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.classification_dim),
            )
            if self.classification_dim > 0
            else None
        )
        self.quality_head = (
            nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.quality_dim),
            )
            if self.quality_dim > 0
            else None
        )
        self.error_head = (
            nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.error_dim),
            )
            if self.error_dim > 0
            else None
        )

    @classmethod
    def from_sample(
        cls,
        sample: Mapping[str, Any],
        *,
        target_dim: int = 6,
        classification_dim: int = 0,
        quality_dim: int = 0,
        error_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        waveform_encoder: str = "cnn",
        waveform_embedding_dim: int = 64,
        waveform_length: int | None = None,
        architecture: str = "hetero_attention",
        waveform_transformer_heads: int = 4,
        waveform_transformer_layers: int = 1,
        waveform_transformer_max_tokens: int = 128,
        waveform_transformer_downsample: str = "adaptive_avg",
        use_pulse_parent_waveform: bool = True,
        use_pulse_bounds: bool = True,
        pulse_waveform_encoder: str = "bounds",
        pulse_waveform_embedding_dim: int | None = None,
        pulse_waveform_window_length: int | None = None,
        pulse_waveform_rise_anchor_bin: int | None = None,
        use_relative_positions: bool = True,
        detector_readout_mask: str = "all",
        pulse_readout_mask: str = "all",
        attention_heads: int = 4,
        readout_heads: int = 4,
    ) -> "MinimalHeteroTaleSdGNN":
        waveform_shape = sample["detector_waveforms"].shape
        has_pulse_waveform_reference = "pulse_detector_index" in sample and "pulse_bounds" in sample
        pulse_waveform_encoder = str(pulse_waveform_encoder)
        requires_pulse_reference = (
            bool(use_pulse_parent_waveform)
            or bool(use_pulse_bounds)
            or pulse_waveform_encoder not in {"none"}
        )
        if requires_pulse_reference and not has_pulse_waveform_reference:
            raise ValueError(
                "pulse waveform options require pulse_detector_index and pulse_bounds; "
                "regenerate the flat training cache or set USE_PULSE_PARENT_WAVEFORM=0, "
                "USE_PULSE_BOUNDS=0, and PULSE_WAVEFORM_ENCODER=none"
            )
        resolved_pulse_waveform_embedding_dim = (
            int(waveform_embedding_dim)
            if pulse_waveform_embedding_dim is None and pulse_waveform_encoder in {"crop_cnn", "crop_same"}
            else pulse_waveform_embedding_dim
        )
        edge_dims = {
            relation: int(features.shape[1]) if features.ndim == 2 else 0
            for relation, features in sample["edge_features_by_type"].items()
        }
        return cls(
            detector_dim=int(sample["detector_features"].shape[1]),
            detector_context_dim=int(sample["detector_context_features"].shape[1]),
            pulse_dim=int(sample["pulse_features"].shape[1]),
            edge_dims=edge_dims,
            waveform_channels=int(waveform_shape[1]),
            waveform_length=int(waveform_length) if waveform_length is not None else int(waveform_shape[2]),
            target_dim=target_dim,
            classification_dim=classification_dim,
            quality_dim=quality_dim,
            error_dim=error_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            waveform_encoder=waveform_encoder,
            waveform_embedding_dim=waveform_embedding_dim,
            waveform_transformer_heads=waveform_transformer_heads,
            waveform_transformer_layers=waveform_transformer_layers,
            waveform_transformer_max_tokens=waveform_transformer_max_tokens,
            waveform_transformer_downsample=waveform_transformer_downsample,
            use_pulse_parent_waveform=use_pulse_parent_waveform,
            use_pulse_bounds=use_pulse_bounds,
            pulse_waveform_encoder=pulse_waveform_encoder,
            pulse_waveform_embedding_dim=resolved_pulse_waveform_embedding_dim,
            pulse_waveform_window_length=pulse_waveform_window_length,
            pulse_waveform_rise_anchor_bin=pulse_waveform_rise_anchor_bin,
            use_relative_positions=use_relative_positions,
            detector_readout_mask=detector_readout_mask,
            pulse_readout_mask=pulse_readout_mask,
            architecture=architecture,
            attention_heads=attention_heads,
            readout_heads=readout_heads,
        )

    @staticmethod
    def _pool(state: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        return torch.cat([_scatter_mean(state, batch, num_graphs), _scatter_max(state, batch, num_graphs)], dim=-1)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if "detector_features" in batch:
            batch = hetero_sample_to_tensors(batch, waveform_length=int(self.config["waveform_length"]))
        elif "edge_index_by_type" not in batch:
            batch = hetero_data_to_tensors(batch)
        detector = batch["detector"]
        pulse = batch["pulse"]
        detector_x = detector["x"]
        detector_context = detector["context"]
        detector_waveform = normalize_detector_waveforms(detector["waveform"], int(self.config["waveform_length"]))
        detector_waveform_valid = detector.get("waveform_valid")
        if detector_waveform_valid is None:
            detector_waveform_valid = torch.ones(detector_x.shape[0], dtype=detector_x.dtype, device=detector_x.device)
        else:
            detector_waveform_valid = detector_waveform_valid.to(dtype=detector_x.dtype, device=detector_x.device).reshape(-1)
        pulse_x = pulse["x"]
        waveform_embedding = self.waveform_encoder(
            detector_waveform,
            detector_x.shape[0],
            device=detector_x.device,
            dtype=detector_x.dtype,
            valid_mask=detector_waveform_valid,
        )

        detector_parts = [
            self.detector_feature_encoder(detector_x),
            self.detector_context_encoder(detector_context),
            waveform_embedding,
        ]
        edge_index_by_type = batch["edge_index_by_type"]
        edge_features_by_type = batch["edge_features_by_type"]
        core_anchor = batch.get("core_anchor")
        position_anchor = batch.get("position_anchor", core_anchor)
        if self.use_relative_positions:
            detector_relative_pos, detector_pos_valid = _relative_position_features(
                detector,
                n_nodes=int(detector_x.shape[0]),
                position_anchor=position_anchor,
                device=detector_x.device,
                dtype=detector_x.dtype,
            )
            if not detector_pos_valid and not getattr(self, "_warned_missing_detector_positions", False):
                print("WARNING: hetero_relative_positions detector_pos_missing=1 detector_relative_position_zero=1", flush=True)
                self._warned_missing_detector_positions = True
            detector_parts.append(detector_relative_pos)
        node_states = {
            "detector": self.detector_node_encoder(torch.cat(detector_parts, dim=-1)),
        }
        pulse_extra_parts = []
        needs_pulse_reference = (
            self.use_pulse_parent_waveform
            or self.use_pulse_bounds
            or self.pulse_waveform_encoder.output_dim > 0
        )
        pulse_detector_index = None
        if needs_pulse_reference:
            pulse_detector_index = _detector_index_for_pulses(
                pulse,
                edge_index_by_type,
                n_pulse=int(pulse_x.shape[0]),
                n_detector=int(detector_x.shape[0]),
                device=detector_x.device,
            )
        if self.use_pulse_parent_waveform and self.waveform_encoder.output_dim > 0:
            parent_waveform = torch.zeros(
                (pulse_x.shape[0], self.waveform_encoder.output_dim),
                dtype=pulse_x.dtype,
                device=pulse_x.device,
            )
            if pulse_detector_index is not None and pulse_detector_index.numel() == pulse_x.shape[0]:
                valid_parent = (pulse_detector_index >= 0) & (pulse_detector_index < detector_x.shape[0])
                if bool(valid_parent.any()):
                    safe_parent = pulse_detector_index.clamp(0, max(int(detector_x.shape[0]) - 1, 0))
                    parent_waveform = waveform_embedding[safe_parent].to(dtype=pulse_x.dtype)
                    parent_waveform = torch.where(valid_parent[:, None], parent_waveform, torch.zeros_like(parent_waveform))
            elif not getattr(self, "_warned_missing_pulse_parent_waveform", False):
                print(
                    "WARNING: hetero_pulse_parent_waveform missing_detector_index=1 parent_embedding_zero=1",
                    flush=True,
                )
                self._warned_missing_pulse_parent_waveform = True
            pulse_extra_parts.append(parent_waveform)
        if self.use_pulse_bounds:
            pulse_bounds = pulse.get("pulse_bounds")
            bounds_norm, bounds_valid = _pulse_bounds_norm(
                pulse_bounds,
                n_pulse=int(pulse_x.shape[0]),
                waveform_length=int(detector_waveform.shape[-1]),
                device=pulse_x.device,
                dtype=pulse_x.dtype,
            )
            if not bool(bounds_valid.any()) and not getattr(self, "_warned_missing_pulse_bounds", False):
                print("WARNING: hetero_pulse_bounds missing_or_invalid=1 bounds_features_zero=1", flush=True)
                self._warned_missing_pulse_bounds = True
            pulse_extra_parts.append(bounds_norm)
        if self.use_relative_positions:
            pulse_relative_pos, pulse_pos_valid = _relative_position_features(
                pulse,
                n_nodes=int(pulse_x.shape[0]),
                position_anchor=position_anchor,
                device=pulse_x.device,
                dtype=pulse_x.dtype,
            )
            if not pulse_pos_valid and not getattr(self, "_warned_missing_pulse_positions", False):
                print("WARNING: hetero_relative_positions pulse_pos_missing=1 pulse_relative_position_zero=1", flush=True)
                self._warned_missing_pulse_positions = True
            pulse_extra_parts.append(pulse_relative_pos)
        if self.pulse_waveform_encoder.output_dim > 0:
            pulse_bounds = pulse.get("pulse_bounds")
            pulse_windows, pulse_window_valid = _pulse_waveform_windows(
                detector_waveform,
                pulse_detector_index,
                pulse_bounds,
                window_length=self.pulse_waveform_window_length,
                rise_anchor_bin=self.pulse_waveform_rise_anchor_bin,
            )
            if pulse_detector_index is not None and pulse_detector_index.numel() == pulse_x.shape[0]:
                detector_valid_for_pulse = detector_waveform_valid[
                    pulse_detector_index.clamp(0, max(int(detector_x.shape[0]) - 1, 0))
                ].reshape(-1) > 0.5
                pulse_window_valid = pulse_window_valid & detector_valid_for_pulse
            if not bool(pulse_window_valid.any()) and not getattr(self, "_warned_missing_pulse_waveform_reference", False):
                print(
                    "WARNING: hetero_pulse_waveform_metadata missing_or_invalid=1 "
                    "pulse_window_embedding_zero=1",
                    flush=True,
                )
                self._warned_missing_pulse_waveform_reference = True
            pulse_window_embedding = self.pulse_waveform_encoder(
                pulse_windows,
                pulse_x.shape[0],
                device=pulse_x.device,
                dtype=pulse_x.dtype,
                valid_mask=pulse_window_valid,
            )
            pulse_extra_parts.append(pulse_window_embedding)
        if pulse_extra_parts:
            pulse_feature_embedding = self.pulse_feature_encoder(pulse_x)  # type: ignore[operator]
            node_states["pulse"] = self.pulse_node_encoder(torch.cat([pulse_feature_embedding, *pulse_extra_parts], dim=-1))
        else:
            node_states["pulse"] = self.pulse_node_encoder(pulse_x)
        detector_message_mask = _detector_readout_mask(detector, self.detector_readout_mask)
        node_message_masks: dict[str, torch.Tensor | None] | None = None
        if detector_message_mask is not None:
            node_message_masks = {"detector": detector_message_mask, "pulse": None}
        layer_attention = []
        for layer_index, layer in enumerate(self.layers):
            if return_attention and isinstance(layer, HeteroAttentionMessageLayer):
                node_states, attention = layer(
                    node_states,
                    edge_index_by_type,
                    edge_features_by_type,
                    return_attention=True,
                    node_masks=node_message_masks,
                )
                layer_attention.append({"layer": int(layer_index), **attention})
            else:
                node_states = layer(
                    node_states,
                    edge_index_by_type,
                    edge_features_by_type,
                    node_masks=node_message_masks,
                )

        num_graphs = int(batch.get("num_graphs", 1))
        readout_attention: dict[str, torch.Tensor] = {}
        detector_readout_state, detector_readout_batch = _masked_readout_input(
            node_states["detector"],
            detector["batch"],
            detector_message_mask,
        )
        pulse_readout_state, pulse_readout_batch = _masked_readout_input(
            node_states["pulse"],
            pulse["batch"],
            _pulse_readout_mask(pulse, self.pulse_readout_mask),
        )
        if self.architecture == "hetero_attention":
            if return_attention:
                detector_readout, detector_weights = self.detector_readout(
                    detector_readout_state,
                    detector_readout_batch,
                    num_graphs,
                    return_attention=True,
                )
                pulse_readout, pulse_weights = self.pulse_readout(
                    pulse_readout_state,
                    pulse_readout_batch,
                    num_graphs,
                    return_attention=True,
                )
                readout_attention = {
                    "detector": detector_weights,
                    "pulse": pulse_weights,
                }
            else:
                detector_readout = self.detector_readout(detector_readout_state, detector_readout_batch, num_graphs)
                pulse_readout = self.pulse_readout(pulse_readout_state, pulse_readout_batch, num_graphs)
            readout = torch.cat(
                [
                    detector_readout,
                    pulse_readout,
                ],
                dim=-1,
            )
        else:
            readout = torch.cat(
                [
                    self._pool(detector_readout_state, detector_readout_batch, num_graphs),
                    self._pool(pulse_readout_state, pulse_readout_batch, num_graphs),
                ],
                dim=-1,
            )
        outputs = []
        if self.reconstruction_head is not None:
            outputs.append(self.reconstruction_head(readout))
        if self.classification_head is not None:
            outputs.append(self.classification_head(readout))
        if self.quality_head is not None:
            outputs.append(self.quality_head(readout))
        if self.error_head is not None:
            outputs.append(self.error_head(readout))
        if not outputs:
            output = readout
        else:
            output = torch.cat(outputs, dim=-1)
        if return_attention:
            node_metadata: dict[str, torch.Tensor] = {}
            missing_metadata = []

            def _add_metadata(store: Mapping[str, Any], source_key: str, output_key: str) -> None:
                if source_key in store:
                    value = store[source_key]
                    node_metadata[output_key] = value.detach() if hasattr(value, "detach") else value
                else:
                    missing_metadata.append(output_key)

            _add_metadata(detector, "batch", "detector_batch")
            _add_metadata(detector, "lid", "detector_lid")
            _add_metadata(detector, "pos", "detector_pos")
            _add_metadata(pulse, "batch", "pulse_batch")
            _add_metadata(pulse, "lid", "pulse_lid")
            _add_metadata(pulse, "pos", "pulse_pos")
            _add_metadata(pulse, "detector_index", "pulse_detector_index")
            _add_metadata(pulse, "pulse_bounds", "pulse_bounds")
            if missing_metadata and not getattr(self, "_warned_missing_attention_metadata", False):
                print(
                    "hetero_attention_metadata "
                    "missing=1 data_format=fast_tensor "
                    f"fields={','.join(missing_metadata)}",
                    flush=True,
                )
                self._warned_missing_attention_metadata = True
            attention_payload = {
                "layers": layer_attention,
                "readout": readout_attention,
                "node_metadata": node_metadata,
            }
            return output, attention_payload
        return output
