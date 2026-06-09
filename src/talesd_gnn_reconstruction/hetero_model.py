from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

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
            message_input = torch.cat(
                [node_states[src_type][src_index], node_states[dst_type][dst_index], edge_attr],
                dim=-1,
            )
            messages = message_mlp(message_input)
            aggregates[dst_type].index_add_(0, dst_index, messages)
            counts[dst_type].index_add_(
                0,
                dst_index,
                torch.ones(dst_index.shape[0], 1, dtype=messages.dtype, device=messages.device),
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
            src_input = torch.cat([src_state[src_index], edge_attr], dim=-1)
            query = self.query[relation](dst_state[dst_index]).view(-1, self.heads, self.head_dim)
            key = self.key[relation](src_input).view(-1, self.heads, self.head_dim)
            value = self.value[relation](src_input).view(-1, self.heads, self.head_dim)
            scores = (query * key).sum(dim=-1) * scale
            weights = _scatter_softmax(scores, dst_index, dst_state.shape[0])
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
        weights = _scatter_softmax(self.score(state), batch, num_graphs)
        pooled = [_scatter_mean(state, batch, num_graphs), _scatter_max(state, batch, num_graphs)]
        for head in range(self.heads):
            weighted = state * weights[:, head : head + 1]
            out = torch.zeros(num_graphs, state.shape[1], dtype=state.dtype, device=state.device)
            out.index_add_(0, batch, weighted)
            pooled.append(out)
        output = torch.cat(pooled, dim=-1)
        if return_attention:
            return output, weights.detach()
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
        attention_heads: int = 4,
        readout_heads: int = 4,
    ):
        super().__init__()
        architecture = str(architecture)
        if architecture not in {"minimal_hetero", "hetero_attention"}:
            raise ValueError("architecture must be 'minimal_hetero' or 'hetero_attention'")
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
            "attention_heads": int(attention_heads),
            "readout_heads": int(readout_heads),
        }
        self.architecture = architecture
        self.hidden_dim = int(hidden_dim)
        self.target_dim = max(int(target_dim), 0)
        self.classification_dim = max(int(classification_dim), 0)
        self.quality_dim = max(int(quality_dim), 0)
        self.error_dim = max(int(error_dim), 0)
        self.detector_feature_encoder = _make_mlp(detector_dim, hidden_dim, hidden_dim, dropout)
        self.detector_context_encoder = _make_mlp(detector_context_dim, hidden_dim, hidden_dim, dropout)
        self.waveform_encoder = WaveformEncoder(
            waveform_channels=waveform_channels,
            waveform_length=waveform_length,
            embedding_dim=waveform_embedding_dim,
            mode=waveform_encoder,
            dropout=dropout,
        )
        detector_input_dim = hidden_dim * 2 + self.waveform_encoder.output_dim
        self.detector_node_encoder = nn.Sequential(
            nn.Linear(detector_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pulse_node_encoder = nn.Sequential(
            nn.Linear(pulse_dim, hidden_dim),
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
        attention_heads: int = 4,
        readout_heads: int = 4,
    ) -> "MinimalHeteroTaleSdGNN":
        waveform_shape = sample["detector_waveforms"].shape
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
        pulse_x = pulse["x"]

        detector_parts = [
            self.detector_feature_encoder(detector_x),
            self.detector_context_encoder(detector_context),
            self.waveform_encoder(
                detector_waveform,
                detector_x.shape[0],
                device=detector_x.device,
                dtype=detector_x.dtype,
            ),
        ]
        node_states = {
            "detector": self.detector_node_encoder(torch.cat(detector_parts, dim=-1)),
            "pulse": self.pulse_node_encoder(pulse_x),
        }
        edge_index_by_type = batch["edge_index_by_type"]
        edge_features_by_type = batch["edge_features_by_type"]
        layer_attention = []
        for layer_index, layer in enumerate(self.layers):
            if return_attention and isinstance(layer, HeteroAttentionMessageLayer):
                node_states, attention = layer(
                    node_states,
                    edge_index_by_type,
                    edge_features_by_type,
                    return_attention=True,
                )
                layer_attention.append({"layer": int(layer_index), **attention})
            else:
                node_states = layer(node_states, edge_index_by_type, edge_features_by_type)

        num_graphs = int(batch.get("num_graphs", 1))
        readout_attention: dict[str, torch.Tensor] = {}
        if self.architecture == "hetero_attention":
            if return_attention:
                detector_readout, detector_weights = self.detector_readout(
                    node_states["detector"],
                    detector["batch"],
                    num_graphs,
                    return_attention=True,
                )
                pulse_readout, pulse_weights = self.pulse_readout(
                    node_states["pulse"],
                    pulse["batch"],
                    num_graphs,
                    return_attention=True,
                )
                readout_attention = {
                    "detector": detector_weights,
                    "pulse": pulse_weights,
                }
            else:
                detector_readout = self.detector_readout(node_states["detector"], detector["batch"], num_graphs)
                pulse_readout = self.pulse_readout(node_states["pulse"], pulse["batch"], num_graphs)
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
                    self._pool(node_states["detector"], detector["batch"], num_graphs),
                    self._pool(node_states["pulse"], pulse["batch"], num_graphs),
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
            attention_payload = {
                "layers": layer_attention,
                "readout": readout_attention,
                "node_metadata": {
                    "detector_batch": detector["batch"].detach(),
                    "detector_lid": detector["lid"].detach(),
                    "detector_pos": detector["pos"].detach(),
                    "pulse_batch": pulse["batch"].detach(),
                    "pulse_lid": pulse["lid"].detach(),
                    "pulse_pos": pulse["pos"].detach(),
                    "pulse_detector_index": pulse["detector_index"].detach(),
                    "pulse_bounds": pulse["pulse_bounds"].detach(),
                },
            }
            return output, attention_payload
        return output
