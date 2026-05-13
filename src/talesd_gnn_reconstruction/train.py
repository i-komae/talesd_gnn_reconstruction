from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .dataset import H5GraphDataset, StandardScaler, collate_graphs, fit_scalers
from .metrics import reconstruction_metrics

if TYPE_CHECKING:
    from .model import TaleSdGNN


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _batches(indices: list[int], batch_size: int, shuffle: bool) -> list[list[int]]:
    indices = list(indices)
    if shuffle:
        random.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def split_indices(
    n_items: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 12345,
) -> dict[str, list[int]]:
    if n_items <= 0:
        raise ValueError("no graphs available")
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if n_items < 3:
        return {"train": indices, "val": indices, "test": indices}

    n_test = max(1, int(round(n_items * test_fraction)))
    n_test = min(n_test, n_items - 2)
    remaining = n_items - n_test
    n_val = max(1, int(round(n_items * val_fraction)))
    n_val = min(n_val, remaining - 1)

    test_indices = indices[:n_test]
    val_indices = indices[n_test : n_test + n_val]
    train_indices = indices[n_test + n_val :]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def _predict_numpy(
    model: "TaleSdGNN",
    dataset: H5GraphDataset,
    indices: list[int],
    scalers: dict[str, StandardScaler],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    pred_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size, shuffle=False):
            samples = [dataset[i] for i in batch_indices]
            batch = collate_graphs(samples, scalers=scalers, device=device, require_target=True)
            pred_scaled = model(batch).detach().cpu().numpy()
            target_scaled = batch["y"].detach().cpu().numpy()
            pred_rows.append(scalers["target"].inverse_transform(pred_scaled))
            target_rows.append(scalers["target"].inverse_transform(target_scaled))
    return np.concatenate(pred_rows, axis=0), np.concatenate(target_rows, axis=0)


def train_model(
    graphs_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1.0e-3,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.05,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 12345,
    device: str = "auto",
    sample_cache_size: int = 1024,
) -> dict[str, Any]:
    import torch
    from torch import nn

    from .model import TaleSdGNN

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(device)
    dataset = H5GraphDataset(graphs_path, require_target=True, cache_size=sample_cache_size)
    if len(dataset) < 2:
        raise ValueError("training needs at least two graphs with MC targets")

    split = split_indices(
        len(dataset),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]

    scalers = fit_scalers(dataset, train_indices)
    first = dataset[train_indices[0]]
    model = TaleSdGNN(
        node_dim=first["node_features"].shape[1],
        edge_dim=first["edge_features"].shape[1],
        pulse_dim=max(first["pulse_features"].shape[1] - 1, 0),
        target_dim=first["target"].shape[0],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_indices in _batches(train_indices, batch_size, shuffle=True):
            samples = [dataset[i] for i in batch_indices]
            batch = collate_graphs(samples, scalers=scalers, device=device, require_target=True)
            pred = model(batch)
            loss = loss_fn(pred, batch["y"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_indices in _batches(val_indices, batch_size, shuffle=False):
                samples = [dataset[i] for i in batch_indices]
                batch = collate_graphs(samples, scalers=scalers, device=device, require_target=True)
                val_losses.append(float(loss_fn(model(batch), batch["y"]).detach().cpu()))

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        history.append(epoch_row)
        if epoch_row["val_loss"] < best_val:
            best_val = epoch_row["val_loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} train_loss={epoch_row['train_loss']:.6f} "
                f"val_loss={epoch_row['val_loss']:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_val, target_val = _predict_numpy(model, dataset, val_indices, scalers, batch_size, device)
    val_metrics = reconstruction_metrics(pred_val, target_val)
    pred_test, target_test = _predict_numpy(model, dataset, test_indices, scalers, batch_size, device)
    test_metrics = reconstruction_metrics(pred_test, target_test)
    print("validation metrics:", json.dumps(val_metrics, sort_keys=True))
    print("test metrics:", json.dumps(test_metrics, sort_keys=True))

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "scalers": {name: scaler.to_dict() for name, scaler in scalers.items()},
        "history": history,
        "metrics": {"validation": val_metrics, "test": test_metrics},
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "split": {
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "n_test": len(test_indices),
        },
    }
    torch.save(checkpoint, output)

    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "history": history,
                "metrics": {"validation": val_metrics, "test": test_metrics},
                "split": checkpoint["split"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    dataset.close()
    return {
        "checkpoint": str(output),
        "metrics_path": str(metrics_path),
        "metrics": {"validation": val_metrics, "test": test_metrics},
    }
