from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .constants import DEFAULT_EDGE_K, DEFAULT_EDGE_RADIUS_KM, DEFAULT_MIN_NODES
from .layout import default_const_dst_path, load_tale_const_positions


def _progress(iterable, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def _chunked(iterable: Iterable[Any], chunk_size: int) -> Iterator[list[Any]]:
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _build_graph_chunk(payload: tuple[list[Any], dict[int, Any] | None, int, float, int]) -> list[Any]:
    from .event_graph import build_graph_event

    records, detector_positions, min_nodes, edge_radius_km, edge_k = payload
    return [
        build_graph_event(
            record,
            detector_positions=detector_positions,
            min_nodes=min_nodes,
            edge_radius_km=edge_radius_km,
            edge_k=edge_k,
        )
        for record in records
    ]


def _iter_graphs(
    records: Iterable[Any],
    args: argparse.Namespace,
    detector_positions: dict[int, Any] | None,
) -> Iterator[Any]:
    from .event_graph import build_graph_event

    workers = max(int(args.workers), 1)
    if workers == 1:
        for record in _progress(records, desc="export graphs"):
            yield build_graph_event(
                record,
                detector_positions=detector_positions,
                min_nodes=args.min_nodes,
                edge_radius_km=args.edge_radius_km,
                edge_k=args.edge_k,
            )
        return

    max_pending = max(workers * 2, 2)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending: deque[Any] = deque()
            for chunk in _progress(_chunked(records, max(int(args.chunk_size), 1)), desc="export chunks"):
                pending.append(
                    pool.submit(
                        _build_graph_chunk,
                        (chunk, detector_positions, args.min_nodes, args.edge_radius_km, args.edge_k),
                    )
                )
                while len(pending) >= max_pending:
                    for graph in pending.popleft().result():
                        yield graph
            while pending:
                for graph in pending.popleft().result():
                    yield graph
    except (OSError, PermissionError) as exc:
        print(f"warning: worker export failed ({exc}); falling back to single-process export")
        for record in _progress(records, desc="export graphs"):
            yield build_graph_event(
                record,
                detector_positions=detector_positions,
                min_nodes=args.min_nodes,
                edge_radius_km=args.edge_radius_km,
                edge_k=args.edge_k,
            )


def _shard_path(output: str | Path, shard_index: int) -> Path:
    base = Path(output).expanduser()
    suffix = base.suffix if base.suffix else ".h5"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}_{shard_index:04d}{suffix}")


def _cmd_export(args: argparse.Namespace) -> None:
    from .dst_reader import iter_dst_banks
    from .graph_io import create_graph_file, write_graph

    const_dst = Path(args.const_dst).expanduser() if args.const_dst else default_const_dst_path()
    detector_positions = None
    if args.kind == "mc":
        detector_positions = load_tale_const_positions(const_dst)
    elif args.kind == "auto" and const_dst is not None:
        detector_positions = load_tale_const_positions(const_dst)
    config = {
        "input": [str(Path(path).expanduser()) for path in args.input],
        "kind": args.kind,
        "const_dst": str(const_dst) if const_dst is not None else None,
        "max_events": args.max_events,
        "min_nodes": args.min_nodes,
        "edge_radius_km": args.edge_radius_km,
        "edge_k": args.edge_k,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "shard_size": args.shard_size,
    }
    written_total = 0
    written_in_file = 0
    skipped = 0
    shard_index = 0
    shard_size = max(int(args.shard_size), 0)
    written_paths: list[Path] = []
    handle = None
    if shard_size == 0:
        output_path = Path(args.output).expanduser()
        handle = create_graph_file(output_path, config=config)
        written_paths.append(output_path)

    records = iter_dst_banks(
        args.input,
        detector_positions=detector_positions,
        kind=args.kind,
        max_events=args.max_events,
        require_trigger_mode0=not args.keep_non_mode0,
    )
    try:
        for graph in _iter_graphs(records, args, detector_positions):
            if graph is None:
                skipped += 1
                continue
            if handle is None or (shard_size > 0 and written_in_file >= shard_size):
                if handle is not None:
                    handle.close()
                    shard_index += 1
                output_path = _shard_path(args.output, shard_index) if shard_size > 0 else Path(args.output).expanduser()
                shard_config = dict(config)
                shard_config["shard_index"] = shard_index if shard_size > 0 else None
                handle = create_graph_file(output_path, config=shard_config)
                written_paths.append(output_path)
                written_in_file = 0
            write_graph(handle, written_in_file, graph)
            written_total += 1
            written_in_file += 1
    finally:
        if handle is not None:
            handle.close()

    targets = ", ".join(str(path) for path in written_paths) if written_paths else str(args.output)
    print(f"wrote {written_total} graphs to {targets} (skipped {skipped} events)")


def _cmd_train(args: argparse.Namespace) -> None:
    from .train import train_model

    result = train_model(
        graphs_path=args.graphs,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        device=args.device,
        sample_cache_size=args.sample_cache_size,
    )
    print(f"checkpoint: {result['checkpoint']}")
    print(f"metrics: {result['metrics_path']}")


def _cmd_predict(args: argparse.Namespace) -> None:
    from .predict import predict_graphs

    output = predict_graphs(
        graphs_path=args.graphs,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        batch_size=args.batch_size,
        device=args.device,
        include_truth=not args.no_truth,
    )
    print(f"wrote predictions to {output}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    from .visualize import visualize_graphs

    outputs = visualize_graphs(
        graphs=args.graphs,
        output=args.output,
        index=args.index,
        event_id=args.event_id,
        count=args.count,
        show_edges=not args.no_edges,
        annotate_lids=args.annotate_lids,
        max_edges=args.max_edges,
        dpi=args.dpi,
        const_dst=args.const_dst,
    )
    for output in outputs:
        print(f"wrote graph visualization to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talesd-gnn",
        description="TALE-SD GNN reconstruction: DST export, training, and prediction",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="DSTをGNN用HDF5グラフへ変換")
    export.add_argument("input", nargs="+", help="入力DSTファイル。複数指定可")
    export.add_argument("-o", "--output", required=True, help="出力HDF5グラフファイル")
    export.add_argument("--kind", choices=["auto", "data", "mc"], default="auto", help="入力DSTの種類")
    export.add_argument("--const-dst", default=None, help="TALE-SD calibration DST talesdconst_pass2.dst。MC入力で必須")
    export.add_argument("--max-events", type=int, default=None, help="読み込む最大イベント数")
    export.add_argument("--min-nodes", type=int, default=DEFAULT_MIN_NODES, help="採用する最小ヒットSD数")
    export.add_argument("--edge-radius-km", type=float, default=DEFAULT_EDGE_RADIUS_KM, help="グラフ辺を張る半径[km]")
    export.add_argument("--edge-k", type=int, default=DEFAULT_EDGE_K, help="各ノードの最低k近傍数")
    export.add_argument("--workers", type=int, default=1, help="グラフ構築に使うworker数。DST読み込みとHDF5書き込みは親processで行う")
    export.add_argument("--chunk-size", type=int, default=128, help="workerへ渡すイベントchunkサイズ")
    export.add_argument("--shard-size", type=int, default=0, help="NグラフごとにHDF5を分割する。0なら分割しない")
    export.add_argument("--keep-non-mode0", action="store_true", help="trgMode != 0 も残す")
    export.set_defaults(func=_cmd_export)

    train = sub.add_parser("train", help="MC truth付きグラフでGNNを学習")
    train.add_argument("--graphs", nargs="+", required=True, help="exportで作成したMC HDF5グラフ。shardを複数指定可")
    train.add_argument("-o", "--output", required=True, help="出力checkpoint .pt")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1.0e-3)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--test-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--device", default="auto", help="auto, cpu, mps, cuda など")
    train.add_argument("--sample-cache-size", type=int, default=1024, help="学習中にLRU cacheするグラフ数。0で無効")
    train.set_defaults(func=_cmd_train)

    predict = sub.add_parser("predict", help="学習済みGNNで再構成結果CSVを作成")
    predict.add_argument("--graphs", nargs="+", required=True, help="exportで作成したHDF5グラフ。shardを複数指定可")
    predict.add_argument("--checkpoint", required=True, help="trainで作成したcheckpoint .pt")
    predict.add_argument("-o", "--output", required=True, help="出力CSV")
    predict.add_argument("--batch-size", type=int, default=64)
    predict.add_argument("--device", default="auto", help="auto, cpu, mps, cuda など")
    predict.add_argument("--no-truth", action="store_true", help="truth列を出力しない")
    predict.set_defaults(func=_cmd_predict)

    visualize = sub.add_parser("visualize", help="HDF5グラフをPDFとして描画")
    visualize.add_argument("--graphs", nargs="+", required=True, help="exportで作成したHDF5グラフ。shardを複数指定可")
    visualize.add_argument("-o", "--output", required=True, help="出力PDF。複数描画時または拡張子なしなら出力ディレクトリ")
    visualize.add_argument("--index", type=int, default=0, help="描画するグラフindex")
    visualize.add_argument("--event-id", default=None, help="event_idで選ぶ。指定時は--indexより優先")
    visualize.add_argument("--const-dst", default=None, help="背景SD配置に使うTALE-SD calibration DST")
    visualize.add_argument("--count", type=int, default=1, help="連続して描画するイベント数")
    visualize.add_argument("--no-edges", action="store_true", help="GNN edgeを描画しない")
    visualize.add_argument("--annotate-lids", action="store_true", help="各ノードにSD lidを表示")
    visualize.add_argument("--max-edges", type=int, default=2000, help="描画する最大edge数")
    visualize.add_argument("--dpi", type=int, default=160, help="出力PDF内のraster要素DPI")
    visualize.set_defaults(func=_cmd_visualize)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
