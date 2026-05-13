from __future__ import annotations

import torch
from torch import nn


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.05):
        super().__init__()
        self.message = _make_mlp(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim, dropout)
        self.update = _make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregate = torch.zeros_like(node)
        else:
            src = edge_index[0]
            dst = edge_index[1]
            message_input = torch.cat([node[src], node[dst], edge_attr], dim=-1)
            messages = self.message(message_input)
            aggregate = torch.zeros_like(node)
            aggregate.index_add_(0, dst, messages)
            degree = torch.zeros(node.shape[0], 1, dtype=node.dtype, device=node.device)
            degree.index_add_(0, dst, torch.ones(messages.shape[0], 1, dtype=node.dtype, device=node.device))
            aggregate = aggregate / degree.clamp_min(1.0)
        return self.norm(node + self.update(torch.cat([node, aggregate], dim=-1)))


class TaleSdGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        pulse_dim: int = 0,
        target_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.config = {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "pulse_dim": pulse_dim,
            "target_dim": target_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.pulse_dim = int(pulse_dim)
        self.hidden_dim = int(hidden_dim)
        if self.pulse_dim > 0:
            self.pulse_encoder = _make_mlp(self.pulse_dim, hidden_dim, hidden_dim, dropout)
            node_input_dim = node_dim + hidden_dim * 2
        else:
            self.pulse_encoder = None
            node_input_dim = node_dim
        self.node_encoder = nn.Sequential(nn.Linear(node_input_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(
            [EdgeMessageLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, target_dim),
        )

    def _pool(self, node: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        mean = torch.zeros(num_graphs, node.shape[1], dtype=node.dtype, device=node.device)
        mean.index_add_(0, batch, node)
        counts = torch.zeros(num_graphs, 1, dtype=node.dtype, device=node.device)
        counts.index_add_(0, batch, torch.ones(node.shape[0], 1, dtype=node.dtype, device=node.device))
        mean = mean / counts.clamp_min(1.0)

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
        if self.pulse_encoder is not None:
            pulse_summary = self._pulse_pool(
                batch.get("pulse_x", x.new_zeros((0, self.pulse_dim))),
                batch.get("pulse_node_index", torch.zeros(0, dtype=torch.long, device=x.device)),
                x.shape[0],
            )
            x = torch.cat([x, pulse_summary], dim=-1)

        node = self.node_encoder(x)
        for layer in self.layers:
            node = layer(node, batch["edge_index"], batch["edge_attr"])
        pooled = self._pool(node, batch["batch"], int(batch["num_graphs"]))
        return self.head(pooled)
