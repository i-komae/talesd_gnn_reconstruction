from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import H5GraphDataset, StandardScaler, collate_graphs
from .metrics import angular_error_deg, direction_to_angles, normalize_directions
from .model import TaleSdGNN
from .train import resolve_device


def _load_checkpoint(path: str | Path, device: str) -> dict[str, Any]:
    import torch

    return torch.load(Path(path).expanduser(), map_location=device)


def predict_graphs(
    graphs_path: str | Path | Sequence[str | Path],
    checkpoint_path: str | Path,
    output_csv: str | Path,
    batch_size: int = 64,
    device: str = "auto",
    include_truth: bool = True,
) -> str:
    import torch

    device = resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    scalers = {name: StandardScaler.from_dict(data) for name, data in checkpoint["scalers"].items()}
    model = TaleSdGNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = H5GraphDataset(graphs_path, require_target=False)
    output = Path(output_csv).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_id",
        "source_path",
        "source_index",
        "n_nodes",
        "n_edges",
        "log10_energy_eV",
        "energy_eV",
        "core_x_km",
        "core_y_km",
        "core_z_km",
        "zenith_deg",
        "azimuth_deg",
    ]
    truth_fields = [
        "true_log10_energy_eV",
        "true_core_x_km",
        "true_core_y_km",
        "true_core_z_km",
        "true_zenith_deg",
        "true_azimuth_deg",
        "delta_log10_energy",
        "core_error_km",
        "angular_error_deg",
    ]
    if include_truth:
        fieldnames.extend(truth_fields)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                batch = collate_graphs(samples, scalers=scalers, device=device, require_target=False)
                pred_scaled = model(batch).detach().cpu().numpy()
                pred = scalers["target"].inverse_transform(pred_scaled)
                direction = normalize_directions(pred)
                zenith, azimuth = direction_to_angles(direction)
                pred[:, 4:7] = direction

                for row_idx, sample in enumerate(samples):
                    attrs = sample["attrs"]
                    row = {
                        "event_id": sample["event_id"],
                        "source_path": attrs.get("source_path", ""),
                        "source_index": int(attrs.get("source_index", -1)),
                        "n_nodes": int(attrs.get("n_nodes", sample["node_features"].shape[0])),
                        "n_edges": int(attrs.get("n_edges", sample["edge_features"].shape[0])),
                        "log10_energy_eV": float(pred[row_idx, 0]),
                        "energy_eV": float(10.0 ** pred[row_idx, 0]),
                        "core_x_km": float(pred[row_idx, 1]),
                        "core_y_km": float(pred[row_idx, 2]),
                        "core_z_km": float(pred[row_idx, 3]),
                        "zenith_deg": float(zenith[row_idx]),
                        "azimuth_deg": float(azimuth[row_idx]),
                    }
                    target = sample["target"]
                    if include_truth and target is not None:
                        target_dir = normalize_directions(target[None, :])
                        true_zenith, true_azimuth = direction_to_angles(target_dir)
                        row.update(
                            {
                                "true_log10_energy_eV": float(target[0]),
                                "true_core_x_km": float(target[1]),
                                "true_core_y_km": float(target[2]),
                                "true_core_z_km": float(target[3]),
                                "true_zenith_deg": float(true_zenith[0]),
                                "true_azimuth_deg": float(true_azimuth[0]),
                                "delta_log10_energy": float(pred[row_idx, 0] - target[0]),
                                "core_error_km": float(np.linalg.norm(pred[row_idx, 1:4] - target[1:4])),
                                "angular_error_deg": float(angular_error_deg(pred[row_idx : row_idx + 1], target[None, :])[0]),
                            }
                        )
                    writer.writerow(row)

    dataset.close()
    return str(output)
