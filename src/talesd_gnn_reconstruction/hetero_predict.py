from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .core_coordinates import (
    core_anchor_from_sample,
    core_anchor_weight_column,
    filter_feature_matrix,
    filtered_columns,
    inverse_transform_core_target,
    normalize_coordinate_feature_mode,
    normalize_core_target_mode,
    parse_columns_json,
)
from .hetero_data import sample_to_hetero_data
from .hetero_graph_io import graph_event_to_sample
from .hetero_model import MinimalHeteroTaleSdGNN
from .metrics import direction_columns_for_dim, direction_to_angles, normalize_directions
from .train import resolve_device


def _load_checkpoint(path: str | Path, device: str) -> dict[str, Any]:
    import torch

    return torch.load(Path(path).expanduser(), map_location=device, weights_only=False)


def _build_hetero_model(config: Mapping[str, Any]) -> MinimalHeteroTaleSdGNN:
    model_config = dict(config)
    architecture = str(model_config.pop("architecture", ""))
    if architecture not in {"minimal_hetero", "hetero_attention"}:
        raise ValueError(f"checkpoint architecture {architecture!r} is not supported for hetero DST reconstruction")
    return MinimalHeteroTaleSdGNN(architecture=architecture, **model_config)


def _inverse_target(pred_scaled: np.ndarray, scalers: Mapping[str, Any]) -> np.ndarray:
    scaler = scalers.get("target")
    if scaler is None:
        return pred_scaled
    mean = np.asarray(scaler["mean"] if isinstance(scaler, Mapping) else scaler.mean, dtype=np.float32)
    std = np.asarray(scaler["std"] if isinstance(scaler, Mapping) else scaler.std, dtype=np.float32)
    if pred_scaled.shape[1] != mean.shape[0]:
        raise ValueError(f"target scaler dimension mismatch: pred_dim={pred_scaled.shape[1]} scaler_dim={mean.shape[0]}")
    return pred_scaled * std[None, :] + mean[None, :]


def _batched(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _columns_from_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("raw_columns_json", "columns_json"):
        value = runtime.get(key)
        if value:
            columns = parse_columns_json(value)
            if columns:
                return columns
    try:
        import dstio.tale.graph as tale_graph

        return dict(tale_graph.graph_columns())
    except Exception:
        columns: dict[str, Any] = {}
        detector_columns = runtime.get("detector_feature_columns")
        pulse_columns = runtime.get("pulse_feature_columns")
        if detector_columns:
            columns["detector_features"] = list(detector_columns)
        if pulse_columns:
            columns["pulse_features"] = list(pulse_columns)
        return columns


def _prepare_reconstruct_sample(
    graph: Any,
    *,
    columns: Mapping[str, Any],
    core_anchor_mode: str,
    coordinate_feature_mode: str,
) -> dict[str, Any]:
    sample = graph_event_to_sample(graph)
    core_anchor = core_anchor_from_sample(sample, columns=columns, core_anchor_mode=core_anchor_mode)
    sample["core_anchor"] = np.asarray(core_anchor, dtype=np.float32).reshape(-1)[:2]
    detector_columns = list(columns.get("detector_features", []))
    pulse_columns = list(columns.get("pulse_features", []))
    sample["detector_features"] = filter_feature_matrix(
        np.asarray(sample["detector_features"], dtype=np.float32),
        detector_columns,
        coordinate_feature_mode,
    )
    sample["pulse_features"] = filter_feature_matrix(
        np.asarray(sample["pulse_features"], dtype=np.float32),
        pulse_columns,
        coordinate_feature_mode,
    )
    sample["detector_feature_columns"] = filtered_columns(detector_columns, coordinate_feature_mode)
    sample["pulse_feature_columns"] = filtered_columns(pulse_columns, coordinate_feature_mode)
    return sample


def _validate_reconstruct_sample_dims(
    sample: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    coordinate_feature_mode: str,
) -> None:
    expected_detector = int(model_config.get("detector_dim", sample["detector_features"].shape[1]))
    actual_detector = int(sample["detector_features"].shape[1])
    if actual_detector != expected_detector:
        raise RuntimeError(
            "detector feature dimension mismatch in reconstruct_dst: "
            f"checkpoint detector_dim={expected_detector}, sample detector_dim={actual_detector}, "
            f"coordinate_feature_mode={coordinate_feature_mode}"
        )
    expected_pulse = int(model_config.get("pulse_dim", sample["pulse_features"].shape[1]))
    actual_pulse = int(sample["pulse_features"].shape[1])
    if actual_pulse != expected_pulse:
        raise RuntimeError(
            "pulse feature dimension mismatch in reconstruct_dst: "
            f"checkpoint pulse_dim={expected_pulse}, sample pulse_dim={actual_pulse}, "
            f"coordinate_feature_mode={coordinate_feature_mode}"
        )


def _prediction_rows(
    samples: Sequence[Mapping[str, Any]],
    pred_all: np.ndarray,
    *,
    target_dim: int,
    classification_dim: int,
    quality_dim: int,
    error_dim: int,
    error_energy_scale: float,
    error_angular_scale_deg: float,
    error_core_scale_km: float,
    core_anchor: np.ndarray | None = None,
    delta_core_xy: np.ndarray | None = None,
    core_target_mode: str = "absolute",
    core_anchor_mode: str = "absolute",
) -> list[dict[str, Any]]:
    pred = None
    zenith = None
    azimuth = None
    if target_dim >= 6:
        pred = pred_all[:, :target_dim].copy()
        direction = normalize_directions(pred)
        zenith, azimuth = direction_to_angles(direction)
        pred[:, direction_columns_for_dim(target_dim)] = direction
    p_iron = None
    if classification_dim > 0:
        logits = pred_all[:, target_dim]
        p_iron = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    offset = target_dim + max(int(classification_dim), 0)
    quality = None
    if quality_dim > 0:
        quality_logits = pred_all[:, offset]
        offset += quality_dim
        quality = 1.0 / (1.0 + np.exp(-np.clip(quality_logits, -80.0, 80.0)))
    predicted_errors = None
    if error_dim > 0:
        raw = pred_all[:, offset : offset + error_dim]
        raw = raw[:, :3]
        softplus = np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0)
        predicted_errors = softplus * np.asarray(
            [error_energy_scale, error_angular_scale_deg, error_core_scale_km],
            dtype=np.float64,
        )

    rows = []
    for index, sample in enumerate(samples):
        metadata = dict(sample.get("metadata", {}))
        row: dict[str, Any] = {
            "event_id": sample.get("event_id", metadata.get("event_id", "")),
            "source_path": metadata.get("source_path", ""),
            "source_index": metadata.get("source_index", -1),
            "date": metadata.get("date", ""),
            "time": metadata.get("time", ""),
            "usec": metadata.get("usec", ""),
            "n_detector_nodes": int(sample["detector_features"].shape[0]),
            "n_pulse_nodes": int(sample["pulse_features"].shape[0]),
            "node_policy": metadata.get("node_policy", ""),
            "cleaning_mode": metadata.get("cleaning_mode", ""),
            "has_reference_core": metadata.get("has_reference_core", ""),
            "core_relative_features_valid": metadata.get("core_relative_features_valid", ""),
        }
        if pred is not None and zenith is not None and azimuth is not None:
            if core_anchor is not None and core_anchor.shape[0] > index:
                anchor_x = float(core_anchor[index, 0])
                anchor_y = float(core_anchor[index, 1])
            else:
                anchor_x = 0.0
                anchor_y = 0.0
            if delta_core_xy is not None and delta_core_xy.shape[0] > index:
                delta_x = float(delta_core_xy[index, 0])
                delta_y = float(delta_core_xy[index, 1])
            else:
                delta_x = 0.0
                delta_y = 0.0
            row.update(
                {
                    "log10_energy_eV": float(pred[index, 0]),
                    "energy_eV": float(10.0 ** pred[index, 0]),
                    "core_x_km": float(pred[index, 1]),
                    "core_y_km": float(pred[index, 2]),
                    "pred_delta_core_x_km": delta_x,
                    "pred_delta_core_y_km": delta_y,
                    "core_anchor_x_km": anchor_x,
                    "core_anchor_y_km": anchor_y,
                    "core_target_mode": core_target_mode,
                    "core_anchor_mode": core_anchor_mode,
                    "zenith_deg": float(zenith[index]),
                    "azimuth_deg": float(azimuth[index]),
                }
            )
        if p_iron is not None:
            row.update(
                {
                    "p_iron": float(p_iron[index]),
                    "p_proton": float(1.0 - p_iron[index]),
                    "pred_parttype": 5626 if p_iron[index] >= 0.5 else 14,
                }
            )
        if quality is not None:
            row["quality"] = float(quality[index])
        if predicted_errors is not None:
            row.update(
                {
                    "pred_energy_abs_relative_error": float(predicted_errors[index, 0]),
                    "pred_opening_angle_deg": float(predicted_errors[index, 1]),
                    "pred_core_error_km": float(predicted_errors[index, 2]),
                }
            )
        rows.append(row)
    return rows


def reconstruct_dst(
    inputs: str | Path | Sequence[str | Path],
    checkpoint_path: str | Path,
    output_csv: str | Path,
    *,
    kind: str = "auto",
    const_dst: str | Path | None = None,
    mc_calib_dir: str | Path | None = None,
    batch_size: int = 128,
    max_events: int | None = None,
    device: str = "auto",
    cleaning: str = "ising",
    node_policy: str = "all_candidates_with_ising",
    require_reference_core: bool = True,
    skip_errors: bool = False,
    skip_missing_mc_calibration: bool = False,
    open_retries: int = 1,
    open_retry_delay: float = 0.0,
) -> dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    import dstio.tale.graph as tale_graph

    device = resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    scalers = dict(checkpoint.get("hetero_scalers", {}))
    model_config = dict(checkpoint["model_config"])
    model = _build_hetero_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_dim = int(model_config.get("target_dim", 6))
    classification_dim = int(model_config.get("classification_dim", 0))
    quality_dim = int(model_config.get("quality_dim", 0))
    error_dim = int(model_config.get("error_dim", 0))
    waveform_length = int(model_config["waveform_length"])
    runtime = dict(checkpoint.get("runtime", {}))
    core_target_mode = normalize_core_target_mode(runtime.get("core_target_mode", "absolute"))
    core_anchor_mode = normalize_core_target_mode(runtime.get("core_anchor_mode", core_target_mode))
    coordinate_feature_mode = normalize_coordinate_feature_mode(
        runtime.get("coordinate_feature_mode", "absolute_and_relative")
    )
    columns = _columns_from_runtime(runtime)
    error_energy_scale = float(runtime.get("error_energy_scale", runtime.get("quality_energy_scale", 0.10)))
    error_angular_scale_deg = float(runtime.get("error_angular_scale_deg", runtime.get("quality_angular_scale_deg", 1.0)))
    error_core_scale_km = float(runtime.get("error_core_scale_km", runtime.get("quality_core_scale_km", 0.05)))
    print(
        "hetero_reconstruct_config "
        f"core_target_mode={core_target_mode} "
        f"core_anchor_mode={core_anchor_mode} "
        f"coordinate_feature_mode={coordinate_feature_mode}",
        flush=True,
    )
    if core_anchor_mode == "signal_bary_relative":
        weight_column = core_anchor_weight_column(list(columns.get("pulse_features", [])))
        print(
            "hetero_core_anchor "
            f"mode={core_anchor_mode} "
            f"weight_column={weight_column} "
            f"warning={'no_rho_column' if weight_column == 'uniform' else 'none'}",
            flush=True,
        )
        print(
            'hetero_core_anchor_source source=recomputed_from_pulse_positions '
            'note="future exports should store core_anchor explicitly"',
            flush=True,
        )
    elif core_anchor_mode == "absolute":
        print("hetero_core_anchor mode=absolute weight_column=none warning=none", flush=True)
        print("hetero_core_anchor_source source=zero_anchor", flush=True)
    elif core_anchor_mode == "fit_core_relative":
        print("hetero_core_anchor mode=fit_core_relative weight_column=none warning=none", flush=True)
        print("hetero_core_anchor_source source=metadata_reference_core", flush=True)

    input_list = [Path(inputs).expanduser()] if isinstance(inputs, (str, Path)) else [Path(item).expanduser() for item in inputs]
    graphs = tale_graph.iter_graphs(
        input_list,
        kind=kind,
        cleaning=cleaning,
        node_policy=node_policy,
        const_dst=Path(const_dst).expanduser() if const_dst is not None else None,
        mc_calib_dir=Path(mc_calib_dir).expanduser() if mc_calib_dir is not None else None,
        max_events=max_events,
        require_reference_core=bool(require_reference_core),
        skip_errors=bool(skip_errors),
        skip_missing_mc_calibration=bool(skip_missing_mc_calibration),
        open_retries=open_retries,
        open_retry_delay=open_retry_delay,
    )

    fieldnames = [
        "event_id",
        "source_path",
        "source_index",
        "date",
        "time",
        "usec",
        "n_detector_nodes",
        "n_pulse_nodes",
        "node_policy",
        "cleaning_mode",
        "has_reference_core",
        "core_relative_features_valid",
    ]
    if target_dim >= 6:
        fieldnames.extend(
            [
                "log10_energy_eV",
                "energy_eV",
                "core_x_km",
                "core_y_km",
                "pred_delta_core_x_km",
                "pred_delta_core_y_km",
                "core_anchor_x_km",
                "core_anchor_y_km",
                "core_target_mode",
                "core_anchor_mode",
                "zenith_deg",
                "azimuth_deg",
            ]
        )
    if classification_dim > 0:
        fieldnames.extend(["p_iron", "p_proton", "pred_parttype"])
    if quality_dim > 0:
        fieldnames.append("quality")
    if error_dim > 0:
        fieldnames.extend(["pred_energy_abs_relative_error", "pred_opening_angle_deg", "pred_core_error_km"])

    output = Path(output_csv).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for graph_batch in _batched(graphs, max(int(batch_size), 1)):
                samples = []
                for graph in graph_batch:
                    try:
                        sample = _prepare_reconstruct_sample(
                            graph,
                            columns=columns,
                            core_anchor_mode=core_anchor_mode,
                            coordinate_feature_mode=coordinate_feature_mode,
                        )
                        _validate_reconstruct_sample_dims(
                            sample,
                            model_config=model_config,
                            coordinate_feature_mode=coordinate_feature_mode,
                        )
                    except Exception as exc:
                        if not skip_errors:
                            raise
                        event_id = getattr(graph, "event_id", "")
                        print(f"hetero_reconstruct_skip event_id={event_id} reason={exc}", flush=True)
                        continue
                    samples.append(sample)
                if not samples:
                    continue
                data_list = [
                    sample_to_hetero_data(
                        sample,
                        scalers=scalers,
                        waveform_length=waveform_length,
                    )
                    for sample in samples
                ]
                loader = DataLoader(data_list, batch_size=len(data_list))
                batch = next(iter(loader)).to(device)
                pred_all = model(batch).detach().cpu().numpy()
                core_anchor_values = None
                delta_core_xy = None
                if target_dim >= 6:
                    pred_target = _inverse_target(pred_all[:, :target_dim], scalers)
                    core_anchor_values = np.stack(
                        [np.asarray(sample["core_anchor"], dtype=np.float32).reshape(-1)[:2] for sample in samples],
                        axis=0,
                    ).astype(np.float32)
                    if core_target_mode != "absolute":
                        delta_core_xy = pred_target[:, 1:3].copy()
                    pred_absolute = inverse_transform_core_target(pred_target, core_anchor_values, core_target_mode)
                    pred_all = pred_all.copy()
                    pred_all[:, :target_dim] = pred_absolute
                for row in _prediction_rows(
                    samples,
                    pred_all,
                    target_dim=target_dim,
                    classification_dim=classification_dim,
                    quality_dim=quality_dim,
                    error_dim=error_dim,
                    error_energy_scale=error_energy_scale,
                    error_angular_scale_deg=error_angular_scale_deg,
                    error_core_scale_km=error_core_scale_km,
                    core_anchor=core_anchor_values,
                    delta_core_xy=delta_core_xy,
                    core_target_mode=core_target_mode,
                    core_anchor_mode=core_anchor_mode,
                ):
                    writer.writerow(row)
                    written += 1
    return {
        "output": str(output),
        "events_written": written,
        "inputs": [str(path) for path in input_list],
        "require_reference_core": bool(require_reference_core),
        "node_policy": node_policy,
        "cleaning": cleaning,
    }
