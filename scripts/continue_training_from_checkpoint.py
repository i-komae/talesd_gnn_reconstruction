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
from talesd_gnn_reconstruction.metrics import reconstruction_metrics
from talesd_gnn_reconstruction.model import build_model_from_config
from talesd_gnn_reconstruction.progress import write as _progress_write
from talesd_gnn_reconstruction.train import (
    _batch_to_device,
    _make_graph_loader,
    _predict_numpy,
    _progress,
    _resolve_collate_backend,
    _split_model_output,
    resolve_device,
)


def _auto_workers(n_graphs: int, requested: int) -> int:
    if requested >= 0:
        return max(int(requested), 0)
    return 0 if n_graphs < 1024 else min(4, max((os.cpu_count() or 2) // 2, 1))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _base_epoch(history: list[dict[str, Any]]) -> int:
    if not history:
        return 0
    return max(int(row.get("epoch", 0)) for row in history)


def _parse_target_weights(raw: str, target_dim: int) -> np.ndarray:
    if str(raw).strip().lower() in {"", "auto", "none"}:
        return np.ones(int(target_dim), dtype=np.float32)
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != target_dim:
        raise SystemExit(f"--target-weights needs {target_dim} comma-separated values")
    weights = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise SystemExit("--target-weights must be finite non-negative values")
    mean = float(np.mean(weights))
    if mean <= 0.0:
        raise SystemExit("--target-weights must contain at least one positive value")
    return weights / mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue reconstruction training from an existing checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="既存checkpoint .pt")
    parser.add_argument("--graphs", nargs="+", required=True, help="HDF5 graph shard、またはprefix")
    parser.add_argument("-o", "--output", required=True, help="継続学習後のcheckpoint .pt")
    parser.add_argument("--additional-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument(
        "--target-weights",
        default="auto",
        help="scaled target MSE weights. Current schema is logE,core_x,core_y,dir_x,dir_y,dir_z; legacy checkpoints may include core_z. Normalized to mean 1.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true", help="DataLoader workersをepoch間で保持する")
    parser.add_argument("--collate-backend", choices=["auto", "cpp", "python"], default="cpp")
    parser.add_argument("--collate-threads", type=int, default=0)
    parser.add_argument("--diagnostic-energy-bin-width", type=float, default=0.1)
    parser.add_argument("--diagnostic-min-bin-count", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_paths = _expand_h5_graph_paths(args.graphs)
    if not graph_paths:
        raise SystemExit("no graph files matched --graphs")

    ckpt = _load_checkpoint(checkpoint_path)
    model_config = dict(ckpt["model_config"])
    if int(model_config.get("classification_dim", 0)) != 0:
        raise SystemExit("this continuation script is for mass-free reconstruction checkpoints only")

    dataset = H5GraphDataset(
        graph_paths,
        require_target=True,
        cache_size=0,
        load_node_positions=False,
        load_attrs=False,
        load_particle_label=True,
    )
    try:
        import torch
        from torch import nn

        torch.set_num_threads(max(min(os.cpu_count() or 1, 8), 1))
        train_indices = list(ckpt["train_indices"])
        val_indices = list(ckpt["val_indices"])
        test_indices = list(ckpt["test_indices"])
        scalers = {name: StandardScaler.from_dict(data) for name, data in ckpt["scalers"].items()}
        device = resolve_device(args.device)
        num_workers = _auto_workers(len(dataset), int(args.num_workers))
        collate_backend = _resolve_collate_backend(args.collate_backend, n_graphs=len(dataset), num_workers=num_workers)
        collate_threads = max(int(args.collate_threads), 0)
        pin_memory = device.startswith("cuda")
        target_dim = int(model_config.get("target_dim", 7))
        quality_prediction = int(model_config.get("quality_dim", 0)) > 0
        error_prediction = int(model_config.get("error_dim", 0)) > 0
        runtime = dict(ckpt.get("runtime", {}))
        error_angular_scale_deg = float(runtime.get("error_angular_scale_deg", runtime.get("quality_angular_scale_deg", 1.0)))
        error_core_scale_km = float(runtime.get("error_core_scale_km", runtime.get("quality_core_scale_km", 0.05)))
        error_energy_scale = float(runtime.get("error_energy_scale", runtime.get("quality_energy_scale", 0.10)))
        target_weights_np = _parse_target_weights(args.target_weights, target_dim)
        target_weights = torch.as_tensor(target_weights_np, dtype=torch.float32, device=device)

        model = build_model_from_config(model_config).to(device)
        model.load_state_dict(ckpt["model_state"])
        if num_workers > 0:
            dataset.close()

        train_loader = _make_graph_loader(
            dataset,
            train_indices,
            scalers=scalers,
            batch_size=args.batch_size,
            shuffle=True,
            require_target=True,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=12345 + _base_epoch(list(ckpt.get("history", []))),
            pin_memory=pin_memory,
            persistent_workers=bool(args.persistent_workers),
            collate_backend=collate_backend,
            collate_threads=collate_threads,
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
            persistent_workers=bool(args.persistent_workers),
            collate_backend=collate_backend,
            collate_threads=collate_threads,
        )
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

        print(
            f"graphs={len(dataset)} train={len(train_indices)} val={len(val_indices)} test={len(test_indices)} "
            f"device={device} workers={num_workers} batch_size={args.batch_size} "
            f"prefetch_factor={args.prefetch_factor} persistent_workers={int(bool(args.persistent_workers))} "
            f"collate_backend={collate_backend} collate_threads={collate_threads or 'auto'} "
            f"additional_epochs={args.additional_epochs} lr={args.lr} "
            f"target_weights={target_weights_np.tolist()}",
            flush=True,
        )

        history = [dict(row) for row in ckpt.get("history", [])]
        start_epoch = _base_epoch(history)
        if np.allclose(target_weights_np, np.ones_like(target_weights_np)):
            best_val = min((float(row["val_loss"]) for row in history if "val_loss" in row), default=float("inf"))
        else:
            best_val = float("inf")
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
        loss_fn = nn.MSELoss()

        stage_seconds: dict[str, float] = {}
        stage_started = time.perf_counter()
        epoch_iter = _progress(
            range(1, int(args.additional_epochs) + 1),
            desc="continue epochs",
            total=int(args.additional_epochs),
            enabled=not args.no_progress,
            position=0,
        )
        for local_epoch in epoch_iter:
            epoch = start_epoch + local_epoch
            model.train()
            train_losses: list[float] = []
            train_desc = f"epoch {epoch} train"
            for batch_cpu in _progress(
                train_loader,
                desc=train_desc,
                total=len(train_loader),
                enabled=not args.no_progress,
                leave=False,
                position=1,
            ):
                batch = _batch_to_device(batch_cpu, device, non_blocking=pin_memory)
                pred_all = model(batch)
                pred, _mass_logit, _quality_logit, _error_raw = _split_model_output(
                    pred_all,
                    target_dim,
                    mass_classification=False,
                    quality_prediction=quality_prediction,
                    error_prediction=error_prediction,
                )
                loss = torch.mean((pred - batch["y"]) ** 2 * target_weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                val_desc = f"epoch {epoch} val"
                for batch_cpu in _progress(
                    val_loader,
                    desc=val_desc,
                    total=len(val_loader),
                    enabled=not args.no_progress,
                    leave=False,
                    position=1,
                ):
                    batch = _batch_to_device(batch_cpu, device, non_blocking=pin_memory)
                    pred_all = model(batch)
                    pred, _mass_logit, _quality_logit, _error_raw = _split_model_output(
                        pred_all,
                        target_dim,
                        mass_classification=False,
                        quality_prediction=quality_prediction,
                        error_prediction=error_prediction,
                    )
                    loss = torch.mean((pred - batch["y"]) ** 2 * target_weights)
                    val_losses.append(float(loss.detach().cpu()))

            epoch_row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(np.mean(val_losses)),
                "train_reconstruction_loss": float(np.mean(train_losses)),
                "val_reconstruction_loss": float(np.mean(val_losses)),
            }
            history.append(epoch_row)
            if epoch_row["val_loss"] < best_val:
                best_val = epoch_row["val_loss"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            _progress_write(
                f"epoch={epoch:04d} train_loss={epoch_row['train_loss']:.6f} "
                f"val_loss={epoch_row['val_loss']:.6f}",
            )
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(train_loss=f"{epoch_row['train_loss']:.4g}", val_loss=f"{epoch_row['val_loss']:.4g}")
        stage_seconds["epochs"] = time.perf_counter() - stage_started

        model.load_state_dict(best_state)

        stage_started = time.perf_counter()
        pred_val, target_val, _mass_logit_val, mass_label_val, _quality_val, predicted_error_val = _predict_numpy(
            model,
            val_loader,
            scalers,
            device,
            non_blocking=pin_memory,
            desc="validation predict",
            show_progress=not args.no_progress,
            mass_classification=False,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            target_dim=target_dim,
            error_angular_scale_deg=error_angular_scale_deg,
            error_core_scale_km=error_core_scale_km,
            error_energy_scale=error_energy_scale,
        )
        val_metrics = reconstruction_metrics(pred_val, target_val)
        stage_seconds["validation_predict"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        pred_test, target_test, _mass_logit_test, mass_label_test, _quality_test, predicted_error_test = _predict_numpy(
            model,
            test_loader,
            scalers,
            device,
            non_blocking=pin_memory,
            desc="test predict",
            show_progress=not args.no_progress,
            mass_classification=False,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            target_dim=target_dim,
            error_angular_scale_deg=error_angular_scale_deg,
            error_core_scale_km=error_core_scale_km,
            error_energy_scale=error_energy_scale,
        )
        test_metrics = reconstruction_metrics(pred_test, target_test)
        stage_seconds["test_predict"] = time.perf_counter() - stage_started
        print("validation metrics:", json.dumps(val_metrics, sort_keys=True), flush=True)
        print("test metrics:", json.dumps(test_metrics, sort_keys=True), flush=True)

        stage_started = time.perf_counter()
        diagnostics = save_training_diagnostics(
            output_path,
            history=history,
            validation=(pred_val, target_val),
            test=(pred_test, target_test),
            validation_particle_labels=mass_label_val,
            test_particle_labels=mass_label_test,
            validation_predicted_errors=predicted_error_val,
            test_predicted_errors=predicted_error_test,
            energy_bin_width=args.diagnostic_energy_bin_width,
            min_bin_count=args.diagnostic_min_bin_count,
        )
        stage_seconds["diagnostics"] = time.perf_counter() - stage_started
        stage_seconds["total"] = time.perf_counter() - started

        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "scalers": {name: scaler.to_dict() for name, scaler in scalers.items()},
            "history": history,
            "metrics": {"validation": val_metrics, "test": test_metrics},
            "diagnostics": diagnostics,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "split": dict(ckpt.get("split", {})),
            "runtime": {
                "continued_from": str(checkpoint_path),
                "device": device,
                "num_workers": num_workers,
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": bool(args.persistent_workers),
                "collate_backend": collate_backend,
                "collate_threads": collate_threads,
                "learning_rate": float(args.lr),
                "additional_epochs": int(args.additional_epochs),
                "target_weights": target_weights_np.tolist(),
                "quality_prediction": quality_prediction,
                "error_prediction": error_prediction,
                "error_angular_scale_deg": error_angular_scale_deg,
                "error_core_scale_km": error_core_scale_km,
                "error_energy_scale": error_energy_scale,
                "stage_seconds": {name: round(value, 3) for name, value in stage_seconds.items()},
            },
        }
        torch.save(checkpoint, output_path)
        metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
        metrics_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "metrics": {"validation": val_metrics, "test": test_metrics},
                    "split": checkpoint["split"],
                    "runtime": checkpoint["runtime"],
                    "diagnostics": diagnostics,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print("checkpoint:", output_path, flush=True)
        print("metrics:", metrics_path, flush=True)
        print("diagnostics directory:", diagnostics["directory"], flush=True)
        print("stage_seconds:", json.dumps(checkpoint["runtime"]["stage_seconds"], sort_keys=True), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
