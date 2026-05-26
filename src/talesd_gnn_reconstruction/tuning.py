from __future__ import annotations

import copy
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cli import _expand_h5_graph_paths
from .train import train_model


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return data


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError("empty override key")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted_key!r}: {part!r} is not an object")
        current = child
    current[parts[-1]] = value


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any] | Sequence[str] | None) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    if not overrides:
        return updated
    if isinstance(overrides, Mapping):
        items = overrides.items()
    else:
        parsed_items: list[tuple[str, Any]] = []
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must be KEY=VALUE: {item!r}")
            key, raw_value = item.split("=", 1)
            parsed_items.append((key, _parse_value(raw_value)))
        items = parsed_items
    for key, value in items:
        _set_dotted(updated, str(key), value)
    return updated


def _as_graph_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("config['graphs'] must be a string or a list of strings")


def _default_config_name(config: Mapping[str, Any], train_config: Mapping[str, Any]) -> str:
    task = str(train_config.get("training_task", "reconstruction"))
    arch = str(train_config.get("model_architecture", "physics"))
    waveform = str(train_config.get("waveform_encoder", "cnn-gru"))
    hidden = int(train_config.get("hidden_dim", 128))
    layers = int(train_config.get("num_layers", train_config.get("layers", 4)))
    epochs = int(train_config.get("epochs", 8))
    if task == "mass":
        loss = str(train_config.get("mass_loss_mode", "bce"))
        return f"small_mass_{arch}_wf{waveform}_h{hidden}_l{layers}_{loss}_{epochs}epoch"
    loss = str(train_config.get("loss_mode", "physics"))
    suffix = "quality" if bool(train_config.get("quality_prediction", False)) else "reco"
    return f"small_{suffix}_{arch}_wf{waveform}_h{hidden}_l{layers}_{loss}_{epochs}epoch"


def resolve_training_config(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    if "graphs" not in config:
        raise ValueError("config must define 'graphs'")

    train_config = dict(config.get("train", {}))
    if "layers" in train_config and "num_layers" not in train_config:
        train_config["num_layers"] = train_config.pop("layers")
    if "lr" in train_config and "learning_rate" not in train_config:
        train_config["learning_rate"] = train_config.pop("lr")

    graph_paths = _expand_h5_graph_paths(_as_graph_list(config["graphs"]))
    if not graph_paths:
        raise ValueError("no graph HDF5 files matched config['graphs']")

    output_path = config.get("output_path")
    if output_path:
        checkpoint_path = Path(str(output_path)).expanduser()
        run_dir = checkpoint_path.parents[1] if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
        config_name = checkpoint_path.stem
    else:
        output_root = Path(str(config.get("output_root", "outputs/talesd_gnn_reconstruction/small_tuning"))).expanduser()
        run_name = str(config.get("run_name", f"small_tuning_{time.strftime('%Y%m%d_%H%M%S')}"))
        config_name = str(config.get("config_name", _default_config_name(config, train_config)))
        run_dir = output_root / "runs" / run_name
        checkpoint_path = run_dir / "checkpoints" / f"{config_name}.pt"

    signature = inspect.signature(train_model)
    allowed = set(signature.parameters)
    passthrough = {
        key: value
        for key, value in train_config.items()
        if key in allowed and key not in {"graphs_path", "output_path"}
    }
    ignored = sorted(key for key in train_config if key not in allowed and key not in {"h5_max_open_files"})
    h5_max_open_files = train_config.get("h5_max_open_files", config.get("h5_max_open_files"))

    resolved = {
        "config_path": str(Path(config_path).expanduser()) if config_path is not None else None,
        "graphs": graph_paths,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "config_name": config_name,
        "train_kwargs": passthrough,
        "ignored_train_keys": ignored,
        "h5_max_open_files": h5_max_open_files,
    }
    return resolved


def run_training_from_config(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    updated = apply_overrides(config, overrides)
    resolved = resolve_training_config(updated, config_path=config_path)
    if dry_run:
        return resolved

    run_dir = Path(resolved["run_dir"])
    checkpoint_path = Path(resolved["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)

    h5_max_open_files = resolved.get("h5_max_open_files")
    if h5_max_open_files is not None:
        os.environ["TALESD_GNN_H5_MAX_OPEN_FILES"] = str(h5_max_open_files)

    resolved_path = run_dir / "config" / "small_tuning_resolved.json"
    resolved_payload = {
        "input_config": updated,
        "resolved": resolved,
    }
    resolved_path.write_text(json.dumps(resolved_payload, indent=2, sort_keys=True))
    resolved["resolved_config_path"] = str(resolved_path)

    result = train_model(
        graphs_path=resolved["graphs"],
        output_path=checkpoint_path,
        **resolved["train_kwargs"],
    )
    resolved["result"] = result
    return resolved
