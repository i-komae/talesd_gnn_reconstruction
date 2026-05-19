from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import H5GraphDataset, StandardScaler, collate_graphs
from .metrics import angular_error_deg, direction_to_angles, normalize_directions
from .model import build_model_from_config
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
    model_config = dict(checkpoint["model_config"])
    model = build_model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_dim = int(model_config.get("target_dim", 7))
    classification_dim = int(model_config.get("classification_dim", 0))
    load_detector_lids = int(model_config.get("detector_embedding_dim", 0)) > 0

    dataset = H5GraphDataset(
        graphs_path,
        require_target=False,
        load_particle_label=classification_dim > 0,
        load_detector_lids=load_detector_lids,
    )
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
    if classification_dim > 0:
        fieldnames.extend(["p_iron", "p_proton", "pred_parttype"])
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
    if classification_dim > 0:
        truth_fields.extend(["true_parttype", "true_particle_label", "mass_correct"])
    if include_truth:
        fieldnames.extend(truth_fields)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                batch = collate_graphs(samples, scalers=scalers, device=device, require_target=False)
                pred_all = model(batch).detach().cpu().numpy()
                pred_scaled = pred_all[:, :target_dim]
                pred = scalers["target"].inverse_transform(pred_scaled)
                direction = normalize_directions(pred)
                zenith, azimuth = direction_to_angles(direction)
                pred[:, 4:7] = direction
                p_iron = None
                pred_is_iron = None
                if classification_dim > 0:
                    logits = pred_all[:, target_dim]
                    p_iron = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
                    pred_is_iron = p_iron >= 0.5

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
                    if classification_dim > 0 and p_iron is not None and pred_is_iron is not None:
                        row.update(
                            {
                                "p_iron": float(p_iron[row_idx]),
                                "p_proton": float(1.0 - p_iron[row_idx]),
                                "pred_parttype": 5626 if bool(pred_is_iron[row_idx]) else 14,
                            }
                        )
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
                    if include_truth and classification_dim > 0 and "particle_label" in sample:
                        label = sample.get("particle_label")
                        if label is not None and np.isfinite(float(label)):
                            true_is_iron = float(label) >= 0.5
                            row.update(
                                {
                                    "true_parttype": 5626 if true_is_iron else 14,
                                    "true_particle_label": float(label),
                                    "mass_correct": int(bool(pred_is_iron[row_idx]) == true_is_iron)
                                    if pred_is_iron is not None
                                    else "",
                                }
                            )
                    writer.writerow(row)

    dataset.close()
    return str(output)
