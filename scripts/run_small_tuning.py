#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from talesd_gnn_reconstruction.tuning import load_config, run_training_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small HDF5 graph training job from a JSON config. "
        "The same config can be edited from Jupyter."
    )
    parser.add_argument("--config", required=True, help="small tuning JSON config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a JSON field with dotted syntax, e.g. train.epochs=4 or train.device=\"cuda\"",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and parameters without training")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    config = load_config(config_path)
    resolved = run_training_from_config(
        config,
        config_path=config_path,
        overrides=args.overrides,
        dry_run=args.dry_run,
    )
    printable = dict(resolved)
    if "result" in printable:
        result = dict(printable["result"])
        printable["result"] = {
            "checkpoint": result.get("checkpoint"),
            "metrics_path": result.get("metrics_path"),
            "diagnostics": result.get("diagnostics"),
        }
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
