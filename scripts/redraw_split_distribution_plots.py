#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talesd_gnn_reconstruction.progress import write as progress_write
from scripts.summarize_split_distributions import redraw_split_distribution_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw split distribution PDFs from split_distribution_plot_data.json without reading HDF5."
    )
    parser.add_argument("plot_data_json", help="split_distribution_plot_data.json")
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="output directory for PDFs; defaults to the JSON parent directory",
    )
    args = parser.parse_args()

    plot_data_json = Path(args.plot_data_json).expanduser()
    output_dir = Path(args.plot_dir).expanduser() if args.plot_dir else plot_data_json.parent
    progress_write(f"stage=start redraw_split_distribution_plots plot_data={plot_data_json}")
    result = redraw_split_distribution_plots(plot_data_json, output_dir=output_dir)
    for path in result["plot_files"]:
        print(path)
    progress_write(
        "stage=done redraw_split_distribution_plots "
        f"plot_files={len(result['plot_files'])} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
