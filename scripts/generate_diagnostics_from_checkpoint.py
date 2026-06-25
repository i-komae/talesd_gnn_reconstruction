#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths
from talesd_gnn_reconstruction.dataset import H5GraphDataset, StandardScaler
from talesd_gnn_reconstruction.diagnostics import save_training_diagnostics
from talesd_gnn_reconstruction.metrics import (
    balanced_accuracy_threshold,
    binary_classification_metrics,
    energy_particle_bias_metrics,
    reconstruction_metrics,
)
from talesd_gnn_reconstruction.model import build_model_from_config
from talesd_gnn_reconstruction.train import (
    _make_graph_loader,
    _predict_numpy,
    _resolve_collate_backend,
    resolve_device,
    source_group_key,
)


_HETERO_ARCHITECTURES = {"minimal_hetero", "hetero_attention"}


class _SourcePathLookup:
    def __init__(self, paths: list[Path], *, max_open_files: int = 4):
        self.paths = list(paths)
        self.max_open_files = max(int(max_open_files), 0)
        self._cumulative_lengths: list[int] = []
        self._source_path_arrays: list[np.ndarray | None] = []
        self._key_lists: list[list[str] | None] = []
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        total = 0
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                events = handle["events"]
                key_list_all = sorted(events.keys())
                n_events = len(key_list_all)
                dense_numeric_keys = n_events > 0 and all(
                    key == f"{index:08d}" for index, key in enumerate(key_list_all)
                )
                key_list = None if dense_numeric_keys else key_list_all
                metadata = handle.get("metadata")
                source_paths = None
                if metadata is not None and "source_path" in metadata and len(metadata["source_path"]) >= n_events:
                    source_paths = np.asarray(metadata["source_path"][:n_events])
                self._source_path_arrays.append(source_paths)
                self._key_lists.append(key_list)
                total += n_events
                self._cumulative_lengths.append(total)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _handle(self, path_index: int) -> h5py.File:
        if path_index in self._handles:
            handle = self._handles.pop(path_index)
            self._handles[path_index] = handle
            return handle
        handle = h5py.File(self.paths[path_index], "r")
        self._handles[path_index] = handle
        if self.max_open_files > 0:
            while len(self._handles) > self.max_open_files:
                _old_path_index, old_handle = self._handles.popitem(last=False)
                old_handle.close()
        return handle

    def _locate(self, index: int) -> tuple[int, int, str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path_index = bisect_right(self._cumulative_lengths, index)
        previous = self._cumulative_lengths[path_index - 1] if path_index > 0 else 0
        local_index = index - previous
        key_list = self._key_lists[path_index]
        key = f"{local_index:08d}" if key_list is None else key_list[local_index]
        return path_index, local_index, key

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def source_path(self, index: int) -> str:
        path_index, local_index, key = self._locate(index)
        source_paths = self._source_path_arrays[path_index]
        if source_paths is not None:
            value = source_paths[local_index]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
        value = self._handle(path_index)["events"][key].attrs.get("source_path", "")
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)


def _auto_workers(n_graphs: int, requested: int) -> int:
    if requested >= 0:
        return max(int(requested), 0)
    return 0 if n_graphs < 1024 else min(4, max((os.cpu_count() or 2) // 2, 1))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _update_metrics_json(checkpoint_path: Path, diagnostics: dict[str, Any], metrics: dict[str, Any], elapsed: float) -> None:
    metrics_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".metrics.json")
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text())
    else:
        payload = {}
    payload["diagnostics"] = diagnostics
    payload["metrics"] = metrics
    runtime = payload.setdefault("runtime", {})
    runtime["diagnostics_regenerated_seconds"] = round(float(elapsed), 3)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _default_prediction_cache_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".diagnostics") / "prediction_cache.npz"


def _default_diagnostics_dir(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".diagnostics")


def _checkpoint_training_task(ckpt: dict[str, Any]) -> str:
    task = str(dict(ckpt.get("runtime", {})).get("training_task", "reconstruction")).lower()
    return task if task in {"reconstruction", "mass"} else "reconstruction"


def _index_list(values: Any, *, name: str) -> list[int]:
    if values is None:
        raise ValueError(f"checkpoint has no {name}")
    return [int(value) for value in np.asarray(values).reshape(-1)]


def _source_key_for_index(dataset: Any, index: int, *, mode: str) -> str:
    source_path = dataset.source_path(index)
    if not source_path:
        return f"unknown:{index}"
    if mode == "raw":
        return str(source_path)
    if mode == "dat":
        return source_group_key(str(source_path))
    raise ValueError("source group mode must be 'dat' or 'raw'")


def _source_keys_for_indices(
    dataset: Any,
    indices: list[int],
    *,
    mode: str,
    label: str,
    progress_interval_sec: float = 30.0,
) -> dict[int, str]:
    started = time.monotonic()
    last = started
    result: dict[int, str] = {}
    total = len(indices)
    for offset, index in enumerate(indices, start=1):
        result[index] = _source_key_for_index(dataset, int(index), mode=mode)
        now = time.monotonic()
        if now - last >= progress_interval_sec:
            rate = offset / max(now - started, 1e-9)
            print(
                f"source_leakage_scan split={label} done={offset}/{total} "
                f"elapsed={now - started:.1f}s rate={rate:.1f}/s",
                flush=True,
            )
            last = now
    return result


def _test_seen_unseen_split(
    dataset: Any,
    *,
    train_indices: list[int],
    test_indices: list[int],
    source_group_mode: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    train_key_by_index = _source_keys_for_indices(
        dataset,
        train_indices,
        mode=source_group_mode,
        label="train",
    )
    test_key_by_index = _source_keys_for_indices(
        dataset,
        test_indices,
        mode=source_group_mode,
        label="test",
    )
    train_keys = set(train_key_by_index.values())
    test_keys = set(test_key_by_index.values())
    seen_indices = [index for index in test_indices if test_key_by_index[index] in train_keys]
    unseen_indices = [index for index in test_indices if test_key_by_index[index] not in train_keys]
    seen_keys = {test_key_by_index[index] for index in seen_indices}
    unseen_keys = {test_key_by_index[index] for index in unseen_indices}
    leaked_source_keys = test_keys & train_keys
    summary = {
        "source_group_mode": source_group_mode,
        "train_graphs": len(train_indices),
        "test_graphs": len(test_indices),
        "train_sources": len(train_keys),
        "test_sources": len(test_keys),
        "test_seen_train_source_graphs": len(seen_indices),
        "test_unseen_train_source_graphs": len(unseen_indices),
        "test_seen_train_source_fraction": len(seen_indices) / max(len(test_indices), 1),
        "test_unseen_train_source_fraction": len(unseen_indices) / max(len(test_indices), 1),
        "test_sources_seen_in_train": len(seen_keys),
        "test_sources_unseen_in_train": len(unseen_keys),
        "test_source_leakage_fraction": len(leaked_source_keys) / max(len(test_keys), 1),
        "example_seen_sources": sorted(seen_keys)[:20],
        "example_unseen_sources": sorted(unseen_keys)[:20],
    }
    return seen_indices, unseen_indices, summary


def _rows_for_indices(reference_indices: list[int], subset_indices: list[int]) -> np.ndarray:
    subset = set(int(index) for index in subset_indices)
    return np.asarray([int(index) in subset for index in sorted(reference_indices)], dtype=bool)


def _slice_optional(values: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    return values[mask]


def _prediction_metric_bundle(
    *,
    training_task: str,
    pred: np.ndarray,
    target: np.ndarray,
    mass_logit: np.ndarray | None,
    mass_label: np.ndarray | None,
    quality: np.ndarray | None,
    predicted_error: np.ndarray | None,
    mass_threshold: float,
    tuned_mass_threshold: float,
    energy_bin_width: float,
    min_bin_count: int,
) -> dict[str, Any]:
    del quality, predicted_error
    if pred.shape[0] == 0:
        return {
            "n_graphs": 0,
            "reconstruction": None,
            "mass": None,
            "mass_tuned": None,
        }
    payload: dict[str, Any] = {
        "n_graphs": int(pred.shape[0]),
        "reconstruction": _reconstruction_metrics_for_task(
            training_task,
            pred,
            target,
            mass_label,
            energy_bin_width=energy_bin_width,
            min_bin_count=min_bin_count,
        ),
        "mass": None,
        "mass_tuned": None,
    }
    if mass_logit is not None and mass_label is not None and len(mass_logit) > 0:
        payload["mass"] = binary_classification_metrics(mass_logit, mass_label, threshold=mass_threshold)
        payload["mass_tuned"] = binary_classification_metrics(
            mass_logit,
            mass_label,
            threshold=tuned_mass_threshold,
        )
    return payload


def _source_overlap_metrics_from_test_predictions(
    *,
    test_indices: list[int],
    seen_indices: list[int],
    unseen_indices: list[int],
    training_task: str,
    pred_test: np.ndarray,
    target_test: np.ndarray,
    mass_logit_test: np.ndarray | None,
    mass_label_test: np.ndarray | None,
    quality_test: np.ndarray | None,
    predicted_error_test: np.ndarray | None,
    mass_threshold: float,
    tuned_mass_threshold: float,
    energy_bin_width: float,
    min_bin_count: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, indices in [
        ("test_seen_train_source", seen_indices),
        ("test_unseen_train_source", unseen_indices),
    ]:
        mask = _rows_for_indices(test_indices, indices)
        output[name] = _prediction_metric_bundle(
            training_task=training_task,
            pred=pred_test[mask],
            target=target_test[mask],
            mass_logit=_slice_optional(mass_logit_test, mask),
            mass_label=_slice_optional(mass_label_test, mask),
            quality=_slice_optional(quality_test, mask),
            predicted_error=_slice_optional(predicted_error_test, mask),
            mass_threshold=mass_threshold,
            tuned_mass_threshold=tuned_mass_threshold,
            energy_bin_width=energy_bin_width,
            min_bin_count=min_bin_count,
        )
    return output


def _write_source_overlap_outputs(
    *,
    diagnostics_dir: Path,
    summary: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / "source_leakage_summary.json"
    metrics_path = diagnostics_dir / "source_leakage_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"source leakage summary: {summary_path}", flush=True)
    print(f"source leakage metrics: {metrics_path}", flush=True)


def _is_hetero_checkpoint(ckpt: dict[str, Any]) -> bool:
    model_config = dict(ckpt.get("model_config", {}))
    architecture = str(model_config.get("architecture", "")).strip()
    return "hetero_scalers" in ckpt or architecture in _HETERO_ARCHITECTURES


def _reconstruction_metrics_for_task(
    training_task: str,
    pred: np.ndarray,
    target: np.ndarray,
    particle_labels: np.ndarray | None = None,
    *,
    energy_bin_width: float = 0.1,
    min_bin_count: int = 8,
) -> dict[str, Any] | None:
    if training_task == "mass":
        return None
    metrics: dict[str, Any] = dict(reconstruction_metrics(pred, target))
    if particle_labels is not None:
        metrics.update(
            energy_particle_bias_metrics(
                pred,
                target,
                particle_labels,
                bin_width=energy_bin_width,
                min_bin_count=min_bin_count,
            )
        )
    return metrics


def _print_diagnostics_paths(diagnostics: dict[str, Any]) -> None:
    print("diagnostics directory:", diagnostics["directory"], flush=True)
    print("learning curve:", diagnostics["learning_curve_pdf"], flush=True)
    if diagnostics.get("loss_component_curves_pdf"):
        print("loss component curves:", diagnostics["loss_component_curves_pdf"], flush=True)
    if diagnostics.get("mass_metric_curves_pdf"):
        print("mass metric curves:", diagnostics["mass_metric_curves_pdf"], flush=True)
    for key, label in [
        ("validation", "validation diagnostics"),
        ("test", "test diagnostics"),
        ("validation_species", "validation species diagnostics"),
        ("test_species", "test species diagnostics"),
        ("validation_mass", "validation mass diagnostics"),
        ("test_mass", "test mass diagnostics"),
    ]:
        if key in diagnostics:
            print(f"{label}: {diagnostics[key]['directory']}", flush=True)
    print("diagnostics summary:", diagnostics["summary_json"], flush=True)


def _none_to_cache_array(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float32)
    return np.asarray(values)


def _cache_array_to_none(values: np.ndarray) -> np.ndarray | None:
    return None if values.size == 0 else values


def _load_prediction_cache(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["pred_val"]),
            np.asarray(data["target_val"]),
            _cache_array_to_none(np.asarray(data["mass_logit_val"])),
            _cache_array_to_none(np.asarray(data["mass_label_val"])),
            _cache_array_to_none(np.asarray(data["quality_val"])) if "quality_val" in data else None,
            _cache_array_to_none(np.asarray(data["predicted_error_val"])) if "predicted_error_val" in data else None,
            np.asarray(data["pred_test"]),
            np.asarray(data["target_test"]),
            _cache_array_to_none(np.asarray(data["mass_logit_test"])),
            _cache_array_to_none(np.asarray(data["mass_label_test"])),
            _cache_array_to_none(np.asarray(data["quality_test"])) if "quality_test" in data else None,
            _cache_array_to_none(np.asarray(data["predicted_error_test"])) if "predicted_error_test" in data else None,
        )


def _save_prediction_cache(
    path: Path,
    *,
    pred_val: np.ndarray,
    target_val: np.ndarray,
    mass_logit_val: np.ndarray | None,
    mass_label_val: np.ndarray | None,
    quality_val: np.ndarray | None,
    predicted_error_val: np.ndarray | None,
    pred_test: np.ndarray,
    target_test: np.ndarray,
    mass_logit_test: np.ndarray | None,
    mass_label_test: np.ndarray | None,
    quality_test: np.ndarray | None,
    predicted_error_test: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_val=np.asarray(pred_val),
        target_val=np.asarray(target_val),
        mass_logit_val=_none_to_cache_array(mass_logit_val),
        mass_label_val=_none_to_cache_array(mass_label_val),
        quality_val=_none_to_cache_array(quality_val),
        predicted_error_val=_none_to_cache_array(predicted_error_val),
        pred_test=np.asarray(pred_test),
        target_test=np.asarray(target_test),
        mass_logit_test=_none_to_cache_array(mass_logit_test),
        mass_label_test=_none_to_cache_array(mass_label_test),
        quality_test=_none_to_cache_array(quality_test),
        predicted_error_test=_none_to_cache_array(predicted_error_test),
    )


def _run_hetero_reprediction(
    *,
    ckpt: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    cache_path: Path,
    graph_paths: list[Path],
    args: argparse.Namespace,
    started: float,
    training_task: str,
    save_reconstruction: bool,
) -> None:
    import torch

    from talesd_gnn_reconstruction.core_coordinates import (
        normalize_coordinate_feature_mode,
        normalize_core_target_mode,
    )
    from talesd_gnn_reconstruction.hetero_data import EDGE_TYPE_BY_RELATION
    from talesd_gnn_reconstruction.hetero_graph_io import H5PyGHeteroGraphDataset
    from talesd_gnn_reconstruction.hetero_model import MinimalHeteroTaleSdGNN
    from talesd_gnn_reconstruction.hetero_training import (
        H5TensorHeteroGraphDataset,
        _make_hetero_loader,
        _milestone_metric_summary,
        _predict_hetero_numpy,
        _scalers_from_dict,
    )

    model_config = dict(ckpt["model_config"])
    architecture = str(model_config.pop("architecture", "")).strip()
    if architecture not in _HETERO_ARCHITECTURES:
        raise ValueError(f"hetero checkpoint has unsupported architecture {architecture!r}")

    runtime = dict(ckpt.get("runtime", {}))
    core_target_mode = normalize_core_target_mode(runtime.get("core_target_mode", "absolute"))
    coordinate_feature_mode = normalize_coordinate_feature_mode(
        runtime.get("coordinate_feature_mode", "absolute_and_relative")
    )
    target_dim = int(model_config.get("target_dim", 6))
    mass_classification = int(model_config.get("classification_dim", 0)) > 0
    quality_prediction = int(model_config.get("quality_dim", 0)) > 0
    error_prediction = int(model_config.get("error_dim", 0)) > 0
    waveform_length = int(model_config.get("waveform_length", runtime.get("waveform_length", 0)) or 0)
    if waveform_length <= 0:
        raise ValueError("hetero checkpoint model_config has no valid waveform_length")

    data_format = str(args.hetero_data_format).strip().lower()
    if data_format not in {"fast_tensor", "pyg"}:
        raise ValueError("--hetero-data-format must be fast_tensor or pyg")

    device = resolve_device(args.device)
    scalers = _scalers_from_dict(dict(ckpt["hetero_scalers"]))
    dataset_class: Any = H5TensorHeteroGraphDataset if data_format == "fast_tensor" else H5PyGHeteroGraphDataset
    dataset = dataset_class(
        graph_paths,
        require_target=True,
        require_particle_label=mass_classification,
        core_target_mode=core_target_mode,
        coordinate_feature_mode=coordinate_feature_mode,
        scalers=scalers,
        waveform_length=waveform_length,
    )
    try:
        split_info = dict(ckpt.get("split", {}))
        if "val_indices" not in split_info or "test_indices" not in split_info:
            raise ValueError("hetero checkpoint has no split.val_indices/test_indices")
        val_indices = [int(value) for value in np.asarray(split_info["val_indices"]).reshape(-1)]
        test_indices = [int(value) for value in np.asarray(split_info["test_indices"]).reshape(-1)]
        num_workers = _auto_workers(len(dataset), int(args.num_workers))
        pin_memory = device.startswith("cuda")
        torch.set_num_threads(max(min(os.cpu_count() or 1, 8), 1))
        model = MinimalHeteroTaleSdGNN(architecture=architecture, **model_config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        enabled_relations = set(runtime.get("enabled_relations") or EDGE_TYPE_BY_RELATION)
        error_angular_scale_deg = float(
            runtime.get("error_angular_scale_deg", runtime.get("quality_angular_scale_deg", 1.0))
        )
        error_core_scale_km = float(runtime.get("error_core_scale_km", runtime.get("quality_core_scale_km", 0.05)))
        error_energy_scale = float(runtime.get("error_energy_scale", runtime.get("quality_energy_scale", 0.10)))

        print(
            "hetero_diagnostics "
            f"graphs={len(dataset)} val={len(val_indices)} test={len(test_indices)} "
            f"device={device} workers={num_workers} batch_size={args.batch_size} "
            f"data_format={data_format} architecture={architecture} "
            f"core_target_mode={core_target_mode} coordinate_feature_mode={coordinate_feature_mode}",
            flush=True,
        )

        def _predict_split(split_name: str, indices: list[int]):
            loader = _make_hetero_loader(
                dataset,
                indices,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=False,
                split_name=f"{split_name}_diagnostics",
                timeout_sec=120.0,
                data_format=data_format,
            )
            return _predict_hetero_numpy(
                model,
                loader,
                scalers,
                device,
                target_dim=target_dim,
                mass_classification=mass_classification,
                quality_prediction=quality_prediction,
                error_prediction=error_prediction,
                error_angular_scale_deg=error_angular_scale_deg,
                error_core_scale_km=error_core_scale_km,
                error_energy_scale=error_energy_scale,
                desc=f"{split_name} hetero predict",
                show_progress=not args.no_progress,
                enabled_relations=enabled_relations,
                max_neighbors={},
                non_blocking=pin_memory,
                progress_interval_sec=float(os.environ.get("TALESD_GNN_PREDICT_PROGRESS_INTERVAL_SEC", "60")),
                split_name=f"{split_name}_diagnostics",
                core_target_mode=core_target_mode,
            )

        pred_val, target_val, mass_logit_val, mass_label_val, quality_val, predicted_error_val = _predict_split(
            "validation",
            val_indices,
        )
        pred_test, target_test, mass_logit_test, mass_label_test, quality_test, predicted_error_test = _predict_split(
            "test",
            test_indices,
        )

        val_metrics = _milestone_metric_summary(
            pred_val,
            target_val,
            mass_logit_val,
            mass_label_val,
            quality_val,
            predicted_error_val,
            energy_particle_bias_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
            energy_particle_bias_min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
        )
        test_metrics = _milestone_metric_summary(
            pred_test,
            target_test,
            mass_logit_test,
            mass_label_test,
            quality_test,
            predicted_error_test,
            energy_particle_bias_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
            energy_particle_bias_min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
        )
        print("validation metrics:", json.dumps(val_metrics, sort_keys=True), flush=True)
        print("test metrics:", json.dumps(test_metrics, sort_keys=True), flush=True)

        if not args.no_prediction_cache:
            _save_prediction_cache(
                cache_path,
                pred_val=pred_val,
                target_val=target_val,
                mass_logit_val=mass_logit_val,
                mass_label_val=mass_label_val,
                quality_val=quality_val,
                predicted_error_val=predicted_error_val,
                pred_test=pred_test,
                target_test=target_test,
                mass_logit_test=mass_logit_test,
                mass_label_test=mass_label_test,
                quality_test=quality_test,
                predicted_error_test=predicted_error_test,
            )
            print(f"prediction cache: {cache_path}", flush=True)

        diagnostics = save_training_diagnostics(
            output_path,
            history=ckpt["history"],
            validation=(pred_val, target_val),
            test=(pred_test, target_test),
            validation_mass=(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else None,
            test_mass=(mass_logit_test, mass_label_test)
            if mass_logit_test is not None and mass_label_test is not None
            else None,
            validation_particle_labels=mass_label_val,
            test_particle_labels=mass_label_test,
            validation_quality=quality_val,
            test_quality=quality_test,
            validation_predicted_errors=predicted_error_val,
            test_predicted_errors=predicted_error_test,
            energy_bin_width=args.diagnostic_energy_bin_width,
            min_bin_count=args.diagnostic_min_bin_count,
            save_reconstruction=save_reconstruction,
        )
        elapsed = time.perf_counter() - started
        metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_mass": None,
            "test_mass": None,
            "validation_mass_tuned": None,
            "test_mass_tuned": None,
        }
        if output_path == checkpoint_path:
            _update_metrics_json(checkpoint_path, diagnostics, metrics, elapsed)
        _print_diagnostics_paths(diagnostics)
        print(f"elapsed_seconds={elapsed:.3f}", flush=True)
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation/test diagnostic plots from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="trainで保存したcheckpoint .pt")
    parser.add_argument("--graphs", nargs="*", default=[], help="HDF5 graph shard、またはprefix。prediction cacheがある場合は不要")
    parser.add_argument("-o", "--output", default=None, help="診断図のprefix。省略時はcheckpoint名を使う")
    parser.add_argument("--prediction-cache", default=None, help="validation/test prediction cache .npz。省略時はdiagnostics/prediction_cache.npz")
    parser.add_argument("--refresh-prediction-cache", action="store_true", help="既存prediction cacheを無視して再推論する")
    parser.add_argument("--no-prediction-cache", action="store_true", help="prediction cacheの読み書きを無効化する")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--hetero-data-format",
        choices=["fast_tensor", "pyg"],
        default="fast_tensor",
        help="hetero checkpoint再推論用のdataset形式",
    )
    parser.add_argument("--collate-backend", choices=["auto", "cpp", "python"], default="auto")
    parser.add_argument("--collate-threads", type=int, default=1)
    parser.add_argument("--diagnostic-energy-bin-width", type=float, default=0.1)
    parser.add_argument("--diagnostic-min-bin-count", type=int, default=20)
    parser.add_argument(
        "--split-test-by-train-source-group",
        action="store_true",
        help="testをtrainと同じsource groupを持つeventと、持たないeventに分けて追加評価する",
    )
    parser.add_argument(
        "--source-group-mode",
        choices=["dat", "raw"],
        default="dat",
        help="leakage判定に使うsource group。datはDAT??????_gea_trg_XXXをDAT??????へ畳む",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser() if args.output else checkpoint_path
    cache_path = Path(args.prediction_cache).expanduser() if args.prediction_cache else _default_prediction_cache_path(output_path)
    ckpt = _load_checkpoint(checkpoint_path)
    target_dim = int(ckpt["model_config"].get("target_dim", 7))
    mass_classification = int(ckpt["model_config"].get("classification_dim", 0)) > 0
    quality_prediction = int(ckpt["model_config"].get("quality_dim", 0)) > 0
    error_prediction = int(ckpt["model_config"].get("error_dim", 0)) > 0
    runtime = dict(ckpt.get("runtime", {}))
    training_task = _checkpoint_training_task(ckpt)
    save_reconstruction = training_task != "mass"
    error_angular_scale_deg = float(runtime.get("error_angular_scale_deg", runtime.get("quality_angular_scale_deg", 1.0)))
    error_core_scale_km = float(runtime.get("error_core_scale_km", runtime.get("quality_core_scale_km", 0.05)))
    error_energy_scale = float(runtime.get("error_energy_scale", runtime.get("quality_energy_scale", 0.10)))
    load_detector_lids = int(ckpt["model_config"].get("detector_embedding_dim", 0)) > 0
    if not args.no_prediction_cache and cache_path.exists() and not args.refresh_prediction_cache:
        print(f"using prediction cache: {cache_path}", flush=True)
        (
            pred_val,
            target_val,
            mass_logit_val,
            mass_label_val,
            quality_val,
            predicted_error_val,
            pred_test,
            target_test,
            mass_logit_test,
            mass_label_test,
            quality_test,
            predicted_error_test,
        ) = _load_prediction_cache(cache_path)
        val_metrics = _reconstruction_metrics_for_task(
            training_task,
            pred_val,
            target_val,
            mass_label_val,
            energy_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
            min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
        )
        mass_threshold = 0.5
        tuned_mass_threshold = (
            balanced_accuracy_threshold(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else 0.5
        )
        val_mass_metrics = (
            binary_classification_metrics(mass_logit_val, mass_label_val, threshold=mass_threshold)
            if mass_logit_val is not None and mass_label_val is not None
            else None
        )
        val_mass_tuned_metrics = (
            binary_classification_metrics(mass_logit_val, mass_label_val, threshold=tuned_mass_threshold)
            if mass_logit_val is not None and mass_label_val is not None
            else None
        )
        test_metrics = _reconstruction_metrics_for_task(
            training_task,
            pred_test,
            target_test,
            mass_label_test,
            energy_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
            min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
        )
        test_mass_metrics = (
            binary_classification_metrics(mass_logit_test, mass_label_test, threshold=mass_threshold)
            if mass_logit_test is not None and mass_label_test is not None
            else None
        )
        test_mass_tuned_metrics = (
            binary_classification_metrics(mass_logit_test, mass_label_test, threshold=tuned_mass_threshold)
            if mass_logit_test is not None and mass_label_test is not None
            else None
        )
        source_overlap_metrics = None
        if args.split_test_by_train_source_group:
            graph_paths = _expand_h5_graph_paths(args.graphs)
            if not graph_paths:
                raise SystemExit("--split-test-by-train-source-group requires --graphs even when using prediction cache")
            lookup = _SourcePathLookup(graph_paths)
            try:
                train_indices = _index_list(ckpt.get("train_indices"), name="train_indices")
                test_indices = _index_list(ckpt.get("test_indices"), name="test_indices")
                seen_indices, unseen_indices, source_overlap_summary = _test_seen_unseen_split(
                    lookup,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    source_group_mode=args.source_group_mode,
                )
            finally:
                lookup.close()
            source_overlap_metrics = _source_overlap_metrics_from_test_predictions(
                test_indices=test_indices,
                seen_indices=seen_indices,
                unseen_indices=unseen_indices,
                training_task=training_task,
                pred_test=pred_test,
                target_test=target_test,
                mass_logit_test=mass_logit_test,
                mass_label_test=mass_label_test,
                quality_test=quality_test,
                predicted_error_test=predicted_error_test,
                mass_threshold=mass_threshold,
                tuned_mass_threshold=tuned_mass_threshold,
                energy_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
                min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
            )
            _write_source_overlap_outputs(
                diagnostics_dir=_default_diagnostics_dir(output_path),
                summary=source_overlap_summary,
                metrics=source_overlap_metrics,
            )
            print("source leakage metrics:", json.dumps(source_overlap_metrics, sort_keys=True), flush=True)
        diagnostics = save_training_diagnostics(
            output_path,
            history=ckpt["history"],
            validation=(pred_val, target_val),
            test=(pred_test, target_test),
            validation_mass=(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else None,
            test_mass=(mass_logit_test, mass_label_test)
            if mass_logit_test is not None and mass_label_test is not None
            else None,
            validation_particle_labels=mass_label_val,
            test_particle_labels=mass_label_test,
            validation_quality=quality_val,
            test_quality=quality_test,
            validation_predicted_errors=predicted_error_val,
            test_predicted_errors=predicted_error_test,
            energy_bin_width=args.diagnostic_energy_bin_width,
            min_bin_count=args.diagnostic_min_bin_count,
            save_reconstruction=save_reconstruction,
        )
        elapsed = time.perf_counter() - started
        metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_mass": val_mass_metrics,
            "test_mass": test_mass_metrics,
            "validation_mass_tuned": val_mass_tuned_metrics,
            "test_mass_tuned": test_mass_tuned_metrics,
        }
        if source_overlap_metrics is not None:
            metrics["source_overlap_test"] = source_overlap_metrics
        if output_path == checkpoint_path:
            _update_metrics_json(checkpoint_path, diagnostics, metrics, elapsed)
        _print_diagnostics_paths(diagnostics)
        print(f"elapsed_seconds={elapsed:.3f}", flush=True)
        return

    graph_paths = _expand_h5_graph_paths(args.graphs)
    if not graph_paths:
        raise SystemExit(
            "no prediction cache was available, so --graphs is required for re-evaluation. "
            f"Expected cache: {cache_path}"
        )

    if _is_hetero_checkpoint(ckpt):
        if args.split_test_by_train_source_group:
            raise SystemExit("--split-test-by-train-source-group currently supports homogeneous checkpoints")
        _run_hetero_reprediction(
            ckpt=ckpt,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            cache_path=cache_path,
            graph_paths=graph_paths,
            args=args,
            started=started,
            training_task=training_task,
            save_reconstruction=save_reconstruction,
        )
        return

    dataset = H5GraphDataset(
        graph_paths,
        require_target=True,
        cache_size=0,
        load_node_positions=False,
        load_attrs=False,
        load_particle_label=True,
        load_detector_lids=load_detector_lids,
    )
    try:
        train_indices = _index_list(ckpt.get("train_indices"), name="train_indices")
        val_indices = _index_list(ckpt.get("val_indices"), name="val_indices")
        test_indices = _index_list(ckpt.get("test_indices"), name="test_indices")
        scalers = {name: StandardScaler.from_dict(data) for name, data in ckpt["scalers"].items()}
        device = resolve_device(args.device)
        num_workers = _auto_workers(len(dataset), int(args.num_workers))
        collate_backend = _resolve_collate_backend(args.collate_backend, n_graphs=len(dataset), num_workers=num_workers)
        collate_threads = max(int(args.collate_threads), 0)
        pin_memory = device.startswith("cuda")

        import torch

        model = build_model_from_config(ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        torch.set_num_threads(max(min(os.cpu_count() or 1, 8), 1))
        print(
            f"graphs={len(dataset)} val={len(val_indices)} test={len(test_indices)} "
            f"device={device} workers={num_workers} batch_size={args.batch_size} "
            f"collate_backend={collate_backend} collate_threads={collate_threads or 'auto'}",
            flush=True,
        )

        val_loader = _make_graph_loader(
            dataset,
            val_indices,
            scalers=scalers,
            batch_size=args.batch_size,
            shuffle=False,
            require_target=True,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=12345,
            pin_memory=pin_memory,
            persistent_workers=False,
            collate_backend=collate_backend,
            collate_threads=collate_threads,
        )
        pred_val, target_val, mass_logit_val, mass_label_val, quality_val, predicted_error_val = _predict_numpy(
            model,
            val_loader,
            scalers,
            device,
            non_blocking=pin_memory,
            desc="validation predict",
            show_progress=not args.no_progress,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            target_dim=target_dim,
            mass_logit_offset=0.0,
            error_angular_scale_deg=error_angular_scale_deg,
            error_core_scale_km=error_core_scale_km,
            error_energy_scale=error_energy_scale,
        )
        val_metrics = _reconstruction_metrics_for_task(
            training_task,
            pred_val,
            target_val,
            mass_label_val,
            energy_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
            min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
        )
        mass_threshold = 0.5
        tuned_mass_threshold = (
            balanced_accuracy_threshold(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else 0.5
        )
        val_mass_metrics = (
            binary_classification_metrics(mass_logit_val, mass_label_val, threshold=mass_threshold)
            if mass_logit_val is not None and mass_label_val is not None
            else None
        )
        val_mass_tuned_metrics = (
            binary_classification_metrics(mass_logit_val, mass_label_val, threshold=tuned_mass_threshold)
            if mass_logit_val is not None and mass_label_val is not None
            else None
        )
        if val_metrics is not None:
            print("validation metrics:", json.dumps(val_metrics, sort_keys=True), flush=True)
        if val_mass_metrics is not None:
            print("validation mass metrics:", json.dumps(val_mass_metrics, sort_keys=True), flush=True)
        if val_mass_tuned_metrics is not None:
            print("validation mass tuned metrics:", json.dumps(val_mass_tuned_metrics, sort_keys=True), flush=True)

        test_loader = _make_graph_loader(
            dataset,
            test_indices,
            scalers=scalers,
            batch_size=args.batch_size,
            shuffle=False,
            require_target=True,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=12345,
            pin_memory=pin_memory,
            persistent_workers=False,
            collate_backend=collate_backend,
            collate_threads=collate_threads,
        )
        pred_test, target_test, mass_logit_test, mass_label_test, quality_test, predicted_error_test = _predict_numpy(
            model,
            test_loader,
            scalers,
            device,
            non_blocking=pin_memory,
            desc="test predict",
            show_progress=not args.no_progress,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            target_dim=target_dim,
            mass_logit_offset=0.0,
            error_angular_scale_deg=error_angular_scale_deg,
            error_core_scale_km=error_core_scale_km,
            error_energy_scale=error_energy_scale,
        )
        test_metrics = _reconstruction_metrics_for_task(
            training_task,
            pred_test,
            target_test,
            mass_label_test,
            energy_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
            min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
        )
        test_mass_metrics = (
            binary_classification_metrics(mass_logit_test, mass_label_test, threshold=mass_threshold)
            if mass_logit_test is not None and mass_label_test is not None
            else None
        )
        test_mass_tuned_metrics = (
            binary_classification_metrics(mass_logit_test, mass_label_test, threshold=tuned_mass_threshold)
            if mass_logit_test is not None and mass_label_test is not None
            else None
        )
        if test_metrics is not None:
            print("test metrics:", json.dumps(test_metrics, sort_keys=True), flush=True)
        if test_mass_metrics is not None:
            print("test mass metrics:", json.dumps(test_mass_metrics, sort_keys=True), flush=True)
        if test_mass_tuned_metrics is not None:
            print("test mass tuned metrics:", json.dumps(test_mass_tuned_metrics, sort_keys=True), flush=True)
        source_overlap_metrics = None
        if args.split_test_by_train_source_group:
            seen_indices, unseen_indices, source_overlap_summary = _test_seen_unseen_split(
                dataset,
                train_indices=train_indices,
                test_indices=test_indices,
                source_group_mode=args.source_group_mode,
            )
            source_overlap_metrics = _source_overlap_metrics_from_test_predictions(
                test_indices=test_indices,
                seen_indices=seen_indices,
                unseen_indices=unseen_indices,
                training_task=training_task,
                pred_test=pred_test,
                target_test=target_test,
                mass_logit_test=mass_logit_test,
                mass_label_test=mass_label_test,
                quality_test=quality_test,
                predicted_error_test=predicted_error_test,
                mass_threshold=mass_threshold,
                tuned_mass_threshold=tuned_mass_threshold,
                energy_bin_width=float(runtime.get("energy_bias_bin_width", args.diagnostic_energy_bin_width)),
                min_bin_count=int(runtime.get("energy_bias_min_bin_count", args.diagnostic_min_bin_count)),
            )
            _write_source_overlap_outputs(
                diagnostics_dir=_default_diagnostics_dir(output_path),
                summary=source_overlap_summary,
                metrics=source_overlap_metrics,
            )
            print("source leakage metrics:", json.dumps(source_overlap_metrics, sort_keys=True), flush=True)
        if not args.no_prediction_cache:
            _save_prediction_cache(
                cache_path,
                pred_val=pred_val,
                target_val=target_val,
                mass_logit_val=mass_logit_val,
                mass_label_val=mass_label_val,
                quality_val=quality_val,
                predicted_error_val=predicted_error_val,
                pred_test=pred_test,
                target_test=target_test,
                mass_logit_test=mass_logit_test,
                mass_label_test=mass_label_test,
                quality_test=quality_test,
                predicted_error_test=predicted_error_test,
            )
            print(f"prediction cache: {cache_path}", flush=True)

        diagnostics = save_training_diagnostics(
            output_path,
            history=ckpt["history"],
            validation=(pred_val, target_val),
            test=(pred_test, target_test),
            validation_mass=(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else None,
            test_mass=(mass_logit_test, mass_label_test)
            if mass_logit_test is not None and mass_label_test is not None
            else None,
            validation_particle_labels=mass_label_val,
            test_particle_labels=mass_label_test,
            validation_quality=quality_val,
            test_quality=quality_test,
            validation_predicted_errors=predicted_error_val,
            test_predicted_errors=predicted_error_test,
            energy_bin_width=args.diagnostic_energy_bin_width,
            min_bin_count=args.diagnostic_min_bin_count,
            save_reconstruction=save_reconstruction,
        )
        elapsed = time.perf_counter() - started
        metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_mass": val_mass_metrics,
            "test_mass": test_mass_metrics,
            "validation_mass_tuned": val_mass_tuned_metrics,
            "test_mass_tuned": test_mass_tuned_metrics,
        }
        if source_overlap_metrics is not None:
            metrics["source_overlap_test"] = source_overlap_metrics
        if output_path == checkpoint_path:
            _update_metrics_json(checkpoint_path, diagnostics, metrics, elapsed)
        _print_diagnostics_paths(diagnostics)
        print(f"elapsed_seconds={elapsed:.3f}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
