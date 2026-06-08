from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import StandardScaler
from .feature_analysis import expand_graph_paths
from .hetero_data import hetero_sample_to_tensors
from .hetero_feature_analysis import _scalers_from_checkpoint, _selected_hetero_checkpoint_indices
from .hetero_graph_io import H5HeteroGraphDataset
from .hetero_model import MinimalHeteroTaleSdGNN
from .progress import progress as _progress
from .progress import write as _progress_write
from .train import _split_model_output, resolve_device


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


def _add_array(
    arrays: dict[str, np.ndarray],
    records: dict[str, dict[str, Any]],
    key: str,
    value: Any,
) -> str:
    array = _to_numpy(value)
    arrays[key] = array
    records[key] = {
        "key": key,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
    }
    return key


def _build_hetero_model_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[MinimalHeteroTaleSdGNN, dict[str, Any]]:
    model_config = dict(checkpoint["model_config"])
    architecture = str(model_config.pop("architecture", ""))
    if architecture != "hetero_attention":
        raise ValueError(
            "attention map export requires a hetero_attention checkpoint; "
            f"got architecture={architecture!r}"
        )
    return MinimalHeteroTaleSdGNN(architecture=architecture, **model_config), model_config


def _event_metadata(dataset: H5HeteroGraphDataset, sample: dict[str, Any], index: int) -> dict[str, Any]:
    attrs = dict(sample.get("attrs", {}))
    metadata = dict(sample.get("metadata", {}))
    event_id = metadata.get("event_id", attrs.get("event_id", ""))
    if not event_id:
        event_id = f"graph_{int(index):08d}"
    return {
        "graph_index": int(index),
        "event_id": str(event_id),
        "source_path": str(metadata.get("source_path", attrs.get("source_path", dataset.source_path(int(index))))),
        "source_index": int(metadata.get("source_index", attrs.get("source_index", -1))),
        "date": metadata.get("date", attrs.get("date")),
        "time": metadata.get("time", attrs.get("time")),
        "usec": metadata.get("usec", attrs.get("usec")),
        "particle_label": None if sample.get("particle_label") is None else float(sample["particle_label"]),
    }


def _selected_indices(
    checkpoint: dict[str, Any],
    *,
    split: str,
    max_graphs: int,
    seed: int,
    indices: Sequence[int] | None,
) -> list[int]:
    if indices is not None:
        selected = [int(value) for value in indices]
    else:
        selected = _selected_hetero_checkpoint_indices(checkpoint, split)
    if max_graphs > 0 and len(selected) > max_graphs:
        rng = random.Random(seed)
        selected = sorted(rng.sample(selected, max_graphs))
    return selected


def _prediction_payload(
    pred_all: Any,
    *,
    target_dim: int,
    classification_dim: int,
    quality_dim: int,
    error_dim: int,
    scalers: dict[str, StandardScaler],
) -> dict[str, Any]:
    pred_scaled, mass_logit, quality_logit, error_raw = _split_model_output(
        pred_all,
        target_dim,
        classification_dim > 0,
        quality_prediction=quality_dim > 0,
        error_prediction=error_dim > 0,
    )
    payload: dict[str, Any] = {}
    if target_dim > 0:
        payload["reconstruction"] = scalers["target"].inverse_transform(_to_numpy(pred_scaled)).reshape(-1).tolist()
    if mass_logit is not None:
        payload["mass_logit"] = _to_numpy(mass_logit).reshape(-1).tolist()
    if quality_logit is not None:
        payload["quality_logit"] = _to_numpy(quality_logit).reshape(-1).tolist()
    if error_raw is not None:
        payload["error_raw"] = _to_numpy(error_raw).reshape(-1).tolist()
    return payload


def save_hetero_attention_maps(
    graphs_path: str | Path | Sequence[str | Path],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    max_graphs: int = 16,
    indices: Sequence[int] | None = None,
    device: str = "auto",
    seed: int = 12345,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Save relation/readout attention weights for selected hetero events."""

    import torch

    resolved_device = resolve_device(device)
    checkpoint_file = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_file, map_location=resolved_device, weights_only=False)
    model, model_config = _build_hetero_model_from_checkpoint(checkpoint)
    model = model.to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scalers = _scalers_from_checkpoint(checkpoint)
    target_dim = int(model_config.get("target_dim", 6))
    classification_dim = int(model_config.get("classification_dim", 0))
    quality_dim = int(model_config.get("quality_dim", 0))
    error_dim = int(model_config.get("error_dim", 0))
    waveform_length = int(model_config["waveform_length"])
    selected = _selected_indices(
        checkpoint,
        split=split,
        max_graphs=int(max_graphs),
        seed=int(seed),
        indices=indices,
    )
    paths = expand_graph_paths(graphs_path)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    array_records: dict[str, dict[str, Any]] = {}
    event_records: list[dict[str, Any]] = []

    if show_progress:
        _progress_write(f"stage=start hetero_attention_maps graphs={len(paths)} selected={len(selected)}")
    dataset = H5HeteroGraphDataset(
        paths,
        require_target=target_dim > 0,
        require_particle_label=classification_dim > 0,
        load_attrs=True,
    )
    try:
        with torch.no_grad():
            for event_number, graph_index in enumerate(
                _progress(selected, desc="hetero attention maps", total=len(selected), enabled=show_progress)
            ):
                sample = dataset[int(graph_index)]
                tensors = hetero_sample_to_tensors(
                    sample,
                    device=resolved_device,
                    scalers=scalers,
                    waveform_length=waveform_length,
                )
                pred_all, attention = model(tensors, return_attention=True)
                prefix = f"event_{event_number:04d}"
                event = _event_metadata(dataset, sample, int(graph_index))
                event["target"] = None if sample.get("target") is None else np.asarray(sample["target"]).reshape(-1).tolist()
                event["prediction"] = _prediction_payload(
                    pred_all,
                    target_dim=target_dim,
                    classification_dim=classification_dim,
                    quality_dim=quality_dim,
                    error_dim=error_dim,
                    scalers=scalers,
                )
                event_arrays: dict[str, str] = {}
                event_arrays["detector_lids"] = _add_array(
                    arrays, array_records, f"{prefix}_detector_lids", sample["detector_lids"]
                )
                event_arrays["pulse_lids"] = _add_array(arrays, array_records, f"{prefix}_pulse_lids", sample["pulse_lids"])
                event_arrays["detector_positions_km"] = _add_array(
                    arrays, array_records, f"{prefix}_detector_positions_km", sample["detector_positions_km"]
                )
                event_arrays["pulse_positions_km"] = _add_array(
                    arrays, array_records, f"{prefix}_pulse_positions_km", sample["pulse_positions_km"]
                )
                event_arrays["pulse_detector_index"] = _add_array(
                    arrays, array_records, f"{prefix}_pulse_detector_index", sample["pulse_detector_index"]
                )
                event_arrays["pulse_bounds"] = _add_array(
                    arrays, array_records, f"{prefix}_pulse_bounds", sample["pulse_bounds"]
                )
                readout = dict(attention.get("readout", {}))
                if "detector" in readout:
                    event_arrays["readout_detector_weights"] = _add_array(
                        arrays,
                        array_records,
                        f"{prefix}_readout_detector_weights",
                        readout["detector"],
                    )
                if "pulse" in readout:
                    event_arrays["readout_pulse_weights"] = _add_array(
                        arrays,
                        array_records,
                        f"{prefix}_readout_pulse_weights",
                        readout["pulse"],
                    )
                relation_records: list[dict[str, Any]] = []
                for layer in attention.get("layers", []):
                    layer_index = int(layer["layer"])
                    for relation, payload in dict(layer.get("relations", {})).items():
                        safe_relation = _safe_key(relation)
                        base = f"{prefix}_layer_{layer_index:02d}_{safe_relation}"
                        weights = _to_numpy(payload["weights"])
                        relation_record = {
                            "layer": layer_index,
                            "relation": str(relation),
                            "src_type": str(payload.get("src_type", "")),
                            "dst_type": str(payload.get("dst_type", "")),
                            "edge_index": _add_array(arrays, array_records, f"{base}_edge_index", payload["edge_index"]),
                            "attention_scores": _add_array(
                                arrays,
                                array_records,
                                f"{base}_attention_scores",
                                payload["scores"],
                            ),
                            "attention_weights": _add_array(
                                arrays,
                                array_records,
                                f"{base}_attention_weights",
                                weights,
                            ),
                            "attention_weight_mean": _add_array(
                                arrays,
                                array_records,
                                f"{base}_attention_weight_mean",
                                weights.mean(axis=1) if weights.ndim == 2 else weights,
                            ),
                        }
                        relation_records.append(relation_record)
                event["arrays"] = event_arrays
                event["relations"] = relation_records
                event_records.append(event)
    finally:
        dataset.close()

    npz_path = output / "attention_maps.npz"
    json_path = output / "attention_maps.json"
    np.savez_compressed(npz_path, **arrays)
    result = {
        "format": "hetero_attention_maps_v1",
        "checkpoint": str(checkpoint_file),
        "graphs": paths,
        "split": split,
        "n_graphs": len(event_records),
        "selected_indices": [int(value) for value in selected],
        "attention_note": "Attention weights are diagnostic values; use ablation or perturbation tests for feature importance.",
        "model_config": checkpoint.get("model_config", {}),
        "array_file": str(npz_path),
        "arrays": array_records,
        "events": event_records,
        "summary_json": str(json_path),
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    if show_progress:
        _progress_write(f"stage=done hetero_attention_maps summary_json={json_path} array_file={npz_path}")
    return result
