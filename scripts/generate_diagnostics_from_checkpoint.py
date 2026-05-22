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
from talesd_gnn_reconstruction.metrics import binary_classification_metrics, reconstruction_metrics
from talesd_gnn_reconstruction.model import build_model_from_config
from talesd_gnn_reconstruction.train import (
    _make_graph_loader,
    _predict_numpy,
    _resolve_collate_backend,
    resolve_device,
)


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
    np.ndarray,
    np.ndarray,
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
            np.asarray(data["pred_test"]),
            np.asarray(data["target_test"]),
            _cache_array_to_none(np.asarray(data["mass_logit_test"])),
            _cache_array_to_none(np.asarray(data["mass_label_test"])),
            _cache_array_to_none(np.asarray(data["quality_test"])) if "quality_test" in data else None,
        )


def _save_prediction_cache(
    path: Path,
    *,
    pred_val: np.ndarray,
    target_val: np.ndarray,
    mass_logit_val: np.ndarray | None,
    mass_label_val: np.ndarray | None,
    quality_val: np.ndarray | None,
    pred_test: np.ndarray,
    target_test: np.ndarray,
    mass_logit_test: np.ndarray | None,
    mass_label_test: np.ndarray | None,
    quality_test: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_val=np.asarray(pred_val),
        target_val=np.asarray(target_val),
        mass_logit_val=_none_to_cache_array(mass_logit_val),
        mass_label_val=_none_to_cache_array(mass_label_val),
        quality_val=_none_to_cache_array(quality_val),
        pred_test=np.asarray(pred_test),
        target_test=np.asarray(target_test),
        mass_logit_test=_none_to_cache_array(mass_logit_test),
        mass_label_test=_none_to_cache_array(mass_label_test),
        quality_test=_none_to_cache_array(quality_test),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation/test diagnostic plots from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="trainで保存したcheckpoint .pt")
    parser.add_argument("--graphs", nargs="+", required=True, help="HDF5 graph shard、またはprefix")
    parser.add_argument("-o", "--output", default=None, help="診断図のprefix。省略時はcheckpoint名を使う")
    parser.add_argument("--prediction-cache", default=None, help="validation/test prediction cache .npz。省略時はdiagnostics/prediction_cache.npz")
    parser.add_argument("--refresh-prediction-cache", action="store_true", help="既存prediction cacheを無視して再推論する")
    parser.add_argument("--no-prediction-cache", action="store_true", help="prediction cacheの読み書きを無効化する")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
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
    graph_paths = _expand_h5_graph_paths(args.graphs)
    if not graph_paths:
        raise SystemExit("no graph files matched --graphs")

    ckpt = _load_checkpoint(checkpoint_path)
    target_dim = int(ckpt["model_config"].get("target_dim", 7))
    mass_classification = int(ckpt["model_config"].get("classification_dim", 0)) > 0
    quality_prediction = int(ckpt["model_config"].get("quality_dim", 0)) > 0
    load_detector_lids = int(ckpt["model_config"].get("detector_embedding_dim", 0)) > 0
    if not args.no_prediction_cache and cache_path.exists() and not args.refresh_prediction_cache:
        print(f"using prediction cache: {cache_path}", flush=True)
        (
            pred_val,
            target_val,
            mass_logit_val,
            mass_label_val,
            quality_val,
            pred_test,
            target_test,
            mass_logit_test,
            mass_label_test,
            quality_test,
        ) = _load_prediction_cache(cache_path)
        val_metrics = reconstruction_metrics(pred_val, target_val)
        val_mass_metrics = (
            binary_classification_metrics(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else None
        )
        test_metrics = reconstruction_metrics(pred_test, target_test)
        test_mass_metrics = (
            binary_classification_metrics(mass_logit_test, mass_label_test)
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
            energy_bin_width=args.diagnostic_energy_bin_width,
            min_bin_count=args.diagnostic_min_bin_count,
        )
        elapsed = time.perf_counter() - started
        metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_mass": val_mass_metrics,
            "test_mass": test_mass_metrics,
        }
        if output_path == checkpoint_path:
            _update_metrics_json(checkpoint_path, diagnostics, metrics, elapsed)
        print("diagnostics directory:", diagnostics["directory"], flush=True)
        print("learning curve:", diagnostics["learning_curve_pdf"], flush=True)
        print("validation diagnostics:", diagnostics["validation"]["directory"], flush=True)
        print("test diagnostics:", diagnostics["test"]["directory"], flush=True)
        if "validation_species" in diagnostics:
            print("validation species diagnostics:", diagnostics["validation_species"]["directory"], flush=True)
        if "test_species" in diagnostics:
            print("test species diagnostics:", diagnostics["test_species"]["directory"], flush=True)
        print("diagnostics summary:", diagnostics["summary_json"], flush=True)
        print(f"elapsed_seconds={elapsed:.3f}", flush=True)
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
        pred_val, target_val, mass_logit_val, mass_label_val, quality_val = _predict_numpy(
            model,
            val_loader,
            scalers,
            device,
            non_blocking=pin_memory,
            desc="validation predict",
            show_progress=not args.no_progress,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            target_dim=target_dim,
        )
        val_metrics = reconstruction_metrics(pred_val, target_val)
        val_mass_metrics = (
            binary_classification_metrics(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else None
        )
        print("validation metrics:", json.dumps(val_metrics, sort_keys=True), flush=True)
        if val_mass_metrics is not None:
            print("validation mass metrics:", json.dumps(val_mass_metrics, sort_keys=True), flush=True)

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
        pred_test, target_test, mass_logit_test, mass_label_test, quality_test = _predict_numpy(
            model,
            test_loader,
            scalers,
            device,
            non_blocking=pin_memory,
            desc="test predict",
            show_progress=not args.no_progress,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            target_dim=target_dim,
        )
        test_metrics = reconstruction_metrics(pred_test, target_test)
        test_mass_metrics = (
            binary_classification_metrics(mass_logit_test, mass_label_test)
            if mass_logit_test is not None and mass_label_test is not None
            else None
        )
        print("test metrics:", json.dumps(test_metrics, sort_keys=True), flush=True)
        if test_mass_metrics is not None:
            print("test mass metrics:", json.dumps(test_mass_metrics, sort_keys=True), flush=True)
        if not args.no_prediction_cache:
            _save_prediction_cache(
                cache_path,
                pred_val=pred_val,
                target_val=target_val,
                mass_logit_val=mass_logit_val,
                mass_label_val=mass_label_val,
                quality_val=quality_val,
                pred_test=pred_test,
                target_test=target_test,
                mass_logit_test=mass_logit_test,
                mass_label_test=mass_label_test,
                quality_test=quality_test,
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
            energy_bin_width=args.diagnostic_energy_bin_width,
            min_bin_count=args.diagnostic_min_bin_count,
        )
        elapsed = time.perf_counter() - started
        metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_mass": val_mass_metrics,
            "test_mass": test_mass_metrics,
        }
        if output_path == checkpoint_path:
            _update_metrics_json(checkpoint_path, diagnostics, metrics, elapsed)
        print("diagnostics directory:", diagnostics["directory"], flush=True)
        print("learning curve:", diagnostics["learning_curve_pdf"], flush=True)
        print("validation diagnostics:", diagnostics["validation"]["directory"], flush=True)
        print("test diagnostics:", diagnostics["test"]["directory"], flush=True)
        if "validation_species" in diagnostics:
            print("validation species diagnostics:", diagnostics["validation_species"]["directory"], flush=True)
        if "test_species" in diagnostics:
            print("test species diagnostics:", diagnostics["test_species"]["directory"], flush=True)
        if "validation_mass" in diagnostics:
            print("validation mass diagnostics:", diagnostics["validation_mass"]["directory"], flush=True)
        if "test_mass" in diagnostics:
            print("test mass diagnostics:", diagnostics["test_mass"]["directory"], flush=True)
        print("diagnostics summary:", diagnostics["summary_json"], flush=True)
        print(f"elapsed_seconds={elapsed:.3f}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
