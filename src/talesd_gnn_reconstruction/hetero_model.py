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
from .model import WaveformEncoder, _make_mlp, _scatter_max, _scatter_mean


NODE_TYPE_BY_RELATION = {
    "pulse__interacts__pulse": ("pulse", "pulse"),
    "detector__near__detector": ("detector", "detector"),
    "detector__observes__pulse": ("detector", "pulse"),
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


class MinimalHeteroTaleSdGNN(nn.Module):
    def __init__(
        self,
        *,
        detector_dim: int,
        detector_context_dim: int,
        pulse_dim: int,
        edge_dims: Mapping[str, int],
        waveform_channels: int,
        waveform_length: int,
        target_dim: int = 6,
        classification_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        waveform_encoder: str = "cnn",
        waveform_embedding_dim: int = 64,
    ):
        super().__init__()
        self.config = {
            "architecture": "minimal_hetero",
            "detector_dim": int(detector_dim),
            "detector_context_dim": int(detector_context_dim),
            "pulse_dim": int(pulse_dim),
            "edge_dims": {str(key): int(value) for key, value in edge_dims.items()},
            "waveform_channels": int(waveform_channels),
            "waveform_length": int(waveform_length),
            "target_dim": int(target_dim),
            "classification_dim": int(classification_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "waveform_encoder": str(waveform_encoder),
            "waveform_embedding_dim": int(waveform_embedding_dim),
        }
        self.hidden_dim = int(hidden_dim)
        self.target_dim = max(int(target_dim), 0)
        self.classification_dim = max(int(classification_dim), 0)
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
        self.layers = nn.ModuleList(
            [HeteroMessageLayer(hidden_dim, edge_dims_complete, dropout=dropout) for _ in range(num_layers)]
        )
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

    @classmethod
    def from_sample(
        cls,
        sample: Mapping[str, Any],
        *,
        target_dim: int = 6,
        classification_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        waveform_encoder: str = "cnn",
        waveform_embedding_dim: int = 64,
        waveform_length: int | None = None,
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
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            waveform_encoder=waveform_encoder,
            waveform_embedding_dim=waveform_embedding_dim,
        )

    @staticmethod
    def _pool(state: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        return torch.cat([_scatter_mean(state, batch, num_graphs), _scatter_max(state, batch, num_graphs)], dim=-1)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
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
        for layer in self.layers:
            node_states = layer(node_states, edge_index_by_type, edge_features_by_type)

        num_graphs = int(batch.get("num_graphs", 1))
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
        if not outputs:
            return readout
        return torch.cat(outputs, dim=-1)
