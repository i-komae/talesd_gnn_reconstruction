#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

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
)


_HETERO_ARCHITECTURES = {"minimal_hetero", "hetero_attention"}


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


def _checkpoint_training_task(ckpt: dict[str, Any]) -> str:
    task = str(dict(ckpt.get("runtime", {})).get("training_task", "reconstruction")).lower()
    return task if task in {"reconstruction", "mass"} else "reconstruction"


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
        val_indices = list(ckpt["val_indices"])
        test_indices = list(ckpt["test_indices"])
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
        if output_path == checkpoint_path:
            _update_metrics_json(checkpoint_path, diagnostics, metrics, elapsed)
        _print_diagnostics_paths(diagnostics)
        print(f"elapsed_seconds={elapsed:.3f}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
