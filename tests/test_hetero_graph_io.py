from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

import dstio.tale.graph as tale_graph
from talesd_gnn_reconstruction.hetero_data import (
    TorchGeometricUnavailableError,
    hetero_sample_to_tensors,
    sample_to_hetero_data,
)
from talesd_gnn_reconstruction.hetero_attention_analysis import save_hetero_attention_maps
from talesd_gnn_reconstruction.hetero_feature_analysis import (
    save_hetero_feature_group_importance,
    save_hetero_input_distributions,
)
from talesd_gnn_reconstruction.hetero_graph_io import (
    EDGE_RELATIONS,
    FLAT_FORMAT_NAME,
    FORMAT_NAME,
    GRAPH_DEFINITION,
    H5FlatHeteroGraphDataset,
    H5HeteroGraphDataset,
    H5PyGHeteroGraphDataset,
    WAVEFORM_SCHEMA,
    convert_hetero_to_flat_cache,
    create_hetero_graph_file,
    hetero_dataset_class_for_paths,
    hetero_graph_count,
    write_hetero_graph,
)
from talesd_gnn_reconstruction.hetero_model import MinimalHeteroTaleSdGNN
from talesd_gnn_reconstruction.hetero_predict import reconstruct_dst
from talesd_gnn_reconstruction.hetero_training import (
    H5TensorHeteroGraphDataset,
    _collate_tensor_hetero_graphs,
    _estimate_graph_bytes,
    _filter_batch_relations,
    _resolve_loader_settings,
    _split_dataset,
    train_hetero_model,
)
from talesd_gnn_reconstruction.cli import (
    _cmd_export_hetero,
    _cmd_reshard_hetero,
    _expand_h5_graph_paths,
    _export_light_hetero_worker,
    _ordered_selected_entries,
    _select_light_hetero_source_groups,
    _selected_entries_from_path_indices,
)
import talesd_gnn_reconstruction.hetero_training as hetero_training
from scripts.summarize_split_distributions import _energy_bin, redraw_split_distribution_plots, summarize


DATA_SAMPLE = Path("/Users/ikomae/TALE/dstio/test/data/tale_data_talesdcalibev_single_event.dst")
MC_SAMPLE = Path("/Users/ikomae/TALE/dstio/test/data/tale_mc_DAT000327_gea_sel_10_events.dst.gz")
CONST_DST = Path("/Users/ikomae/TALE/TASoft/development/data/SD/talesdconst_pass2.dst")
MC_CALIB_DIR = Path("/Users/ikomae/TALE/TASoft/development/data/SD")


GRAPH_COLUMNS = tale_graph.graph_columns()
DETECTOR_FEATURE_DIM = len(GRAPH_COLUMNS["detector_features"])
DETECTOR_CONTEXT_DIM = len(GRAPH_COLUMNS["detector_context_features"])
PULSE_FEATURE_DIM = len(GRAPH_COLUMNS["pulse_features"])
EDGE_FEATURE_DIMS = {
    relation: len(GRAPH_COLUMNS["edge_features_by_type"].get(relation, []))
    for relation in EDGE_RELATIONS
}
DETECTOR_FEATURE_INDEX = {name: index for index, name in enumerate(GRAPH_COLUMNS["detector_features"])}
DETECTOR_ISING_FEATURE_COLUMNS = (
    "detector_has_ising_kept_pulse",
    "detector_ising_kept_pulse_count",
    "detector_ising_removed_pulse_count",
)


def _edge_features(rng: np.random.Generator, relation: str, count: int) -> np.ndarray:
    return rng.normal(size=(count, EDGE_FEATURE_DIMS[relation])).astype(np.float32)


def _synthetic_graph(index: int) -> SimpleNamespace:
    rng = np.random.default_rng(1000 + index)
    detector_count = 3
    pulse_count = 4
    pulse_detector_index = np.asarray([0, 1, 1, 2], dtype=np.int64)
    pulse_lids = np.asarray([101, 102, 102, 103], dtype=np.int64)
    label = float(index % 2)
    target = np.asarray(
        [
            17.5 + 0.05 * index,
            0.01 * index,
            -0.02 * index,
            0.0,
            0.1,
            0.995,
        ],
        dtype=np.float32,
    )
    detector_features = rng.normal(size=(detector_count, DETECTOR_FEATURE_DIM)).astype(np.float32)
    for name in (
        "detector_has_signal",
        "detector_arrival_time_valid",
        "detector_live_status",
        "detector_waveform_valid",
    ):
        if name in DETECTOR_FEATURE_INDEX:
            detector_features[:, DETECTOR_FEATURE_INDEX[name]] = 1.0
    if all(name in DETECTOR_FEATURE_INDEX for name in DETECTOR_ISING_FEATURE_COLUMNS):
        detector_features[:, DETECTOR_FEATURE_INDEX["detector_has_ising_kept_pulse"]] = np.asarray(
            [1.0, 1.0, 0.0],
            dtype=np.float32,
        )
        detector_features[:, DETECTOR_FEATURE_INDEX["detector_ising_kept_pulse_count"]] = np.asarray(
            [1.0, 1.0, 0.0],
            dtype=np.float32,
        )
        detector_features[:, DETECTOR_FEATURE_INDEX["detector_ising_removed_pulse_count"]] = np.asarray(
            [0.0, 1.0, 1.0],
            dtype=np.float32,
        )
    return SimpleNamespace(
        event_id=f"synthetic_{index:04d}",
        detector_features=detector_features,
        detector_context_features=rng.normal(size=(detector_count, DETECTOR_CONTEXT_DIM)).astype(np.float32),
        detector_positions_km=rng.normal(size=(detector_count, 3)).astype(np.float32),
        detector_lids=np.asarray([101, 102, 103], dtype=np.int64),
        detector_waveforms=rng.normal(size=(detector_count, 2, 16)).astype(np.float32),
        pulse_features=rng.normal(size=(pulse_count, PULSE_FEATURE_DIM)).astype(np.float32),
        pulse_positions_km=rng.normal(size=(pulse_count, 3)).astype(np.float32),
        pulse_lids=pulse_lids,
        pulse_detector_index=pulse_detector_index,
        pulse_bounds=np.asarray(
            [[1, 3, 5, 8], [2, 4, 6, 9], [3, 5, 7, 10], [4, 6, 8, 11]],
            dtype=np.float32,
        ),
        edge_index_by_type={
            "pulse__same_detector_next__pulse": np.asarray([[1], [2]], dtype=np.int64),
            "pulse__same_detector_prev__pulse": np.asarray([[2], [1]], dtype=np.int64),
            "pulse__near_space__pulse": np.asarray([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
            "pulse__time_causal__pulse": np.asarray([[0, 2], [2, 0]], dtype=np.int64),
            "detector__near__detector": np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
            "detector__observes__pulse": np.asarray(
                [pulse_detector_index, np.arange(pulse_count, dtype=np.int64)],
                dtype=np.int64,
            ),
            "pulse__observed_by__detector": np.asarray(
                [np.arange(pulse_count, dtype=np.int64), pulse_detector_index],
                dtype=np.int64,
            ),
        },
        edge_features_by_type={
            "pulse__same_detector_next__pulse": _edge_features(rng, "pulse__same_detector_next__pulse", 1),
            "pulse__same_detector_prev__pulse": _edge_features(rng, "pulse__same_detector_prev__pulse", 1),
            "pulse__near_space__pulse": _edge_features(rng, "pulse__near_space__pulse", 4),
            "pulse__time_causal__pulse": _edge_features(rng, "pulse__time_causal__pulse", 2),
            "detector__near__detector": _edge_features(rng, "detector__near__detector", 3),
            "detector__observes__pulse": _edge_features(rng, "detector__observes__pulse", pulse_count),
            "pulse__observed_by__detector": _edge_features(rng, "pulse__observed_by__detector", pulse_count),
        },
        target=target,
        particle_label=label,
        metadata={
            "graph_definition": GRAPH_DEFINITION,
            "event_id": f"synthetic_{index:04d}",
            "source_path": f"/synthetic/source_{index // 2:03d}.dst.gz",
            "source_index": index,
            "parttype": 5626 if label >= 0.5 else 14,
            "date": 260606,
            "time": 120000 + index,
            "usec": index,
            "node_policy": "all_candidates_with_ising",
            "cleaning_mode": "ising",
            "has_reference_core": True,
            "core_relative_features_valid": True,
        },
    )


class SyntheticHeteroGraphIoTest(unittest.TestCase):
    def test_hetero_transformer_training_defaults_are_v100_safe(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            graph_dir = tmp / "graphs"
            graph_dir.mkdir()
            env = os.environ.copy()
            env.pop("REPO", None)
            env.update(
                {
                    "GRAPH_INPUT": str(graph_dir),
                    "OUTPUT_ROOT": str(tmp / "output"),
                    "RUN_ID": "dry",
                    "PARTITION": "v100-al9_long",
                    "WAVEFORM_ENCODER": "transformer",
                    "DRY_RUN": "1",
                }
            )
            submit_result = subprocess.run(
                ["bash", "scripts/submit_server_hetero_reco_mass_quality_training.sh"],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("batch_size=32", submit_result.stdout)
            self.assertIn("gradient_accumulation_steps=4", submit_result.stdout)
            self.assertIn("pin_memory=0", submit_result.stdout)
            self.assertIn("prefetch_factor=1", submit_result.stdout)
            self.assertIn("persistent_workers=1", submit_result.stdout)
            self.assertIn("train_workers=4", submit_result.stdout)
            self.assertIn("prepare_fast_cache=1", submit_result.stdout)
            self.assertIn("final_eval_data_format=fast_tensor", submit_result.stdout)
            self.assertIn("attention_maps=0", submit_result.stdout)
            self.assertIn("feature_importance=0", submit_result.stdout)
            self.assertIn("waveform_transformer_max_tokens=128", submit_result.stdout)
            self.assertIn("hetero_training_data_format=fast_tensor", submit_result.stdout)
            sbatch_path = (
                tmp
                / "output"
                / "runs"
                / "server_hetero_reco_mass_quality_v100_128epoch_dry"
                / "slurm"
                / "server_hetero_reco_mass_quality_v100_128epoch_dry.sbatch"
            )
            self.assertIn(f"runtime_source={repo}", sbatch_path.read_text())

            direct_env = env.copy()
            direct_env["OUTPUT_ROOT"] = str(tmp / "runner_output")
            direct_result = subprocess.run(
                ["bash", "scripts/train_hetero_existing_graphs.sh"],
                cwd=repo,
                env=direct_env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("--batch-size 32", direct_result.stdout)
            self.assertIn("--gradient-accumulation-steps 4", direct_result.stdout)
            self.assertIn("--no-pin-memory", direct_result.stdout)
            self.assertIn("--prefetch-factor 1", direct_result.stdout)
            self.assertIn("--waveform-transformer-max-tokens 128", direct_result.stdout)
            self.assertIn("--training-data-format fast_tensor", direct_result.stdout)
            self.assertIn("--final-eval-data-format fast_tensor", direct_result.stdout)
            self.assertIn("--scaler-cache", direct_result.stdout)
            self.assertIn("--reuse-scaler-cache", direct_result.stdout)

    def test_dstio_v3_columns_are_used(self) -> None:
        self.assertEqual(GRAPH_DEFINITION, tale_graph.GRAPH_DEFINITION)
        self.assertEqual(GRAPH_DEFINITION, "tale_sd_hetero_ising_pulse_detector_graph_v3")
        self.assertEqual(list(GRAPH_COLUMNS["detector_features"]), list(tale_graph.graph_columns()["detector_features"]))
        for name in DETECTOR_ISING_FEATURE_COLUMNS:
            self.assertIn(name, DETECTOR_FEATURE_INDEX)

    def test_flat_cache_roundtrip_and_auto_dataset_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            flat_path = Path(tmpdir) / "synthetic_hetero.flat.h5"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(3):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            result = convert_hetero_to_flat_cache(graph_path, flat_path, compression="none")

            self.assertEqual(result["format"], FLAT_FORMAT_NAME)
            self.assertEqual(result["graphs"], 3)
            with h5py.File(flat_path, "r") as handle:
                self.assertIn("detector_features_all", handle)
                self.assertIn("detector_context_features_all", handle)
                self.assertIn("detector_waveforms_all", handle)
                self.assertIn("pulse_features_all", handle)
                self.assertIn("target_all", handle)
                self.assertIn("particle_label_all", handle)
                self.assertIn("detector_offsets", handle)
                self.assertIn("pulse_offsets", handle)
                self.assertIn("edge_offsets_by_relation", handle)
                self.assertIn("pulse__near_space__pulse", handle["edge_index_all_by_relation"])
                self.assertIsNone(handle["detector_features_all"].compression)
            self.assertIs(hetero_dataset_class_for_paths(graph_path), H5HeteroGraphDataset)
            self.assertIs(hetero_dataset_class_for_paths(flat_path), H5FlatHeteroGraphDataset)
            original = H5HeteroGraphDataset(graph_path, require_target=True, require_particle_label=True)
            flat = H5FlatHeteroGraphDataset(flat_path, require_target=True, require_particle_label=True)
            try:
                self.assertEqual(len(flat), len(original))
                self.assertEqual(flat.source_path(1), original.source_path(1))
                np.testing.assert_allclose(flat.target(1), original.target(1))
                sample = flat[2]
                self.assertEqual(sample["detector_features"].shape[1], DETECTOR_FEATURE_DIM)
                self.assertEqual(sample["pulse_features"].shape[1], PULSE_FEATURE_DIM)
                self.assertEqual(sample["metadata"]["date"], 260606)
            finally:
                original.close()
                flat.close()

            cli_flat_path = Path(tmpdir) / "synthetic_hetero.cli.flat.h5"
            cli_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "talesd_gnn_reconstruction.cli",
                    "convert-hetero-to-flat-cache",
                    "--input",
                    str(graph_path),
                    "--output",
                    str(cli_flat_path),
                    "--compression",
                    "none",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("hetero_graph_io format=flat_hdf5", cli_result.stdout)
            self.assertIn("hetero_flat_cache", cli_result.stdout)
            self.assertEqual(hetero_graph_count(cli_flat_path), 3)

    def test_flat_split_distribution_summary_uses_flat_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            flat_path = Path(tmpdir) / "synthetic_hetero.flat.h5"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(1030):
                    graph = _synthetic_graph(index)
                    particle = "proton" if index % 2 == 0 else "iron"
                    graph.particle_label = 0.0 if particle == "proton" else 1.0
                    graph.metadata["source_path"] = f"/synthetic/{particle}/DAT{index:06d}16_gea_trg_000.dst.gz"
                    graph.metadata["time"] = 120000 + (index % 60)
                    graph.metadata["parttype"] = 14 if particle == "proton" else 5626
                    write_hetero_graph(handle, index, graph)
            convert_hetero_to_flat_cache(graph_path, flat_path, compression="none")
            dataset = H5FlatHeteroGraphDataset(flat_path, require_target=True, require_particle_label=True)
            try:
                payload = summarize(
                    dataset,
                    val_fraction=0.20,
                    test_fraction=0.20,
                    source_val_fraction=0.20,
                    source_test_fraction=0.20,
                    seed=123,
                    energy_bin_width=0.1,
                    split_workers=0,
                    show_progress=False,
                    plot_dir=None,
                )
            finally:
                dataset.close()

        self.assertEqual(payload["config"]["graph_format"], "hetero_flat")
        total_events = sum(split["events"] for split in payload["totals"].values())
        self.assertEqual(total_events, 1030)
        for split in payload["totals"].values():
            self.assertEqual(split["sources"], split["independent_showers"])
            self.assertGreater(split["detector_nodes"]["n"], 0)
            self.assertGreater(split["pulse_nodes"]["n"], 0)
            self.assertGreater(split["event_time_hour"]["n"], 0)

    def test_fast_tensor_batch_collates_and_relation_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(2):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            dataset = H5TensorHeteroGraphDataset(graph_path, require_target=True, require_particle_label=True)
            try:
                with mock.patch.object(H5HeteroGraphDataset, "__getitem__", side_effect=AssertionError):
                    fast_sample = dataset[0]
                self.assertNotIn("metadata", fast_sample)
                self.assertNotIn("pos", fast_sample["detector"])
                self.assertNotIn("lid", fast_sample["detector"])
                self.assertNotIn("pulse_bounds", fast_sample["pulse"])
                batch = _collate_tensor_hetero_graphs([dataset[0], dataset[1]])
                self.assertEqual(batch["num_graphs"], 2)
                self.assertEqual(batch["detector"]["x"].shape[0], 6)
                self.assertEqual(batch["pulse"]["x"].shape[0], 8)
                self.assertEqual(batch["target"].shape, (2, 6))
                self.assertGreater(batch["edge_index_by_type"]["pulse__near_space__pulse"].shape[1], 0)

                filtered = _filter_batch_relations(
                    batch,
                    enabled_relations=set(EDGE_RELATIONS) - {"pulse__near_space__pulse"},
                    max_neighbors={},
                )
                self.assertEqual(filtered["edge_index_by_type"]["pulse__near_space__pulse"].shape, (2, 0))
                self.assertEqual(
                    filtered["edge_features_by_type"]["pulse__near_space__pulse"].shape[1],
                    EDGE_FEATURE_DIMS["pulse__near_space__pulse"],
                )
                model = MinimalHeteroTaleSdGNN.from_sample(
                    _synthetic_graph(0).__dict__,
                    target_dim=6,
                    classification_dim=1,
                    hidden_dim=16,
                    num_layers=1,
                    dropout=0.0,
                    waveform_embedding_dim=8,
                )
                model.eval()
                with torch.no_grad():
                    output = model(filtered)
                    attention_output, attention = model(filtered, return_attention=True)
                self.assertEqual(output.shape, (2, 7))
                self.assertEqual(attention_output.shape, (2, 7))
                self.assertIn("node_metadata", attention)
                self.assertIn("detector_batch", attention["node_metadata"])
                self.assertNotIn("detector_lid", attention["node_metadata"])
            finally:
                dataset.close()

    def test_training_scaler_sample_does_not_load_waveforms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "synthetic_hetero.h5"
            with create_hetero_graph_file(output) as handle:
                write_hetero_graph(handle, 0, _synthetic_graph(0))

            dataset = H5HeteroGraphDataset(output, require_target=True, require_particle_label=True)
            try:
                scaler_sample = dataset.scaler_sample(0)
                self.assertNotIn("detector_waveforms", scaler_sample)
                self.assertNotIn("detector_positions_km", scaler_sample)
                self.assertNotIn("edge_index_by_type", scaler_sample)
                self.assertEqual(dataset.detector_waveform_shape(0), (3, 2, 16))
                self.assertGreaterEqual(dataset.graph_nbytes(0), 3 * 2 * 16 * np.dtype(np.float32).itemsize)
                self.assertEqual(scaler_sample["detector_features"].shape, (3, DETECTOR_FEATURE_DIM))
                self.assertEqual(DETECTOR_FEATURE_DIM, 17)
                np.testing.assert_array_equal(
                    scaler_sample["detector_features"][:, DETECTOR_FEATURE_INDEX["detector_has_ising_kept_pulse"]],
                    np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    scaler_sample["detector_features"][:, DETECTOR_FEATURE_INDEX["detector_ising_kept_pulse_count"]],
                    np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    scaler_sample["detector_features"][:, DETECTOR_FEATURE_INDEX["detector_ising_removed_pulse_count"]],
                    np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
                )
                self.assertEqual(scaler_sample["pulse_features"].shape, (4, PULSE_FEATURE_DIM))
                self.assertEqual(set(scaler_sample["edge_features_by_type"]), set(EDGE_RELATIONS))
            finally:
                dataset.close()

    def test_flat_cache_scaler_sample_does_not_load_waveforms_or_call_getitem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            grouped = Path(tmpdir) / "synthetic_hetero.h5"
            flat = Path(tmpdir) / "synthetic_hetero.flat.h5"
            with create_hetero_graph_file(grouped) as handle:
                for index in range(2):
                    write_hetero_graph(handle, index, _synthetic_graph(index))
            summary = convert_hetero_to_flat_cache(grouped, flat, compression="none", verify_samples=2)
            self.assertEqual(summary["verified_samples"], 2)

            dataset = H5FlatHeteroGraphDataset(flat, require_target=True, require_particle_label=True)
            try:
                with mock.patch.object(H5FlatHeteroGraphDataset, "__getitem__", side_effect=AssertionError):
                    scaler_sample = dataset.scaler_sample(0)
                    graph_nbytes = dataset.graph_nbytes(0)
                self.assertNotIn("detector_waveforms", scaler_sample)
                self.assertNotIn("detector_positions_km", scaler_sample)
                self.assertNotIn("detector_lids", scaler_sample)
                self.assertNotIn("pulse_bounds", scaler_sample)
                self.assertGreater(graph_nbytes, 0)
                self.assertEqual(scaler_sample["detector_features"].shape, (3, DETECTOR_FEATURE_DIM))
                self.assertEqual(set(scaler_sample["edge_features_by_type"]), set(EDGE_RELATIONS))
            finally:
                dataset.close()

    def test_training_loader_worker_count_is_memory_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "synthetic_hetero.h5"
            with create_hetero_graph_file(output) as handle:
                for index in range(4):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            dataset = H5HeteroGraphDataset(output, require_target=True, require_particle_label=True)
            try:
                summary = _estimate_graph_bytes(dataset, [0, 1, 2, 3], max_samples=4)
            finally:
                dataset.close()

        high_budget = _resolve_loader_settings(
            requested_workers=2,
            batch_size=2,
            prefetch_factor=2,
            pin_memory=True,
            loader_memory_budget_gib=1.0,
            graph_byte_summary=summary,
        )
        low_budget = _resolve_loader_settings(
            requested_workers=2,
            batch_size=2,
            prefetch_factor=2,
            pin_memory=True,
            loader_memory_budget_gib=1.0e-9,
            graph_byte_summary=summary,
        )

        self.assertEqual(high_budget["resolved_workers"], 2)
        self.assertEqual(low_budget["resolved_workers"], 0)
        self.assertGreater(high_budget["estimated_loader_bytes"], low_budget["estimated_loader_bytes"])

    def test_h5_path_expansion_finds_nested_dstio_worker_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top = root / "top.h5"
            nested = root / "worker_0000" / "graphs_0000.h5"
            nested.parent.mkdir(parents=True)
            top.touch()
            nested.touch()

            expanded = [Path(path) for path in _expand_h5_graph_paths([str(root)])]

        self.assertEqual(expanded, [top, nested])

    def test_source_stratified_split_passes_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "synthetic_hetero.h5"
            with create_hetero_graph_file(output) as handle:
                for index in range(4):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            dataset = H5HeteroGraphDataset(output, require_target=True, require_particle_label=True)
            try:
                with mock.patch.object(
                    hetero_training,
                    "split_indices_by_stratified_source_path",
                    return_value={"train": [0, 1], "val": [2], "test": [3]},
                ) as patched:
                    split = _split_dataset(
                        dataset,
                        split_mode="source-stratified",
                        val_fraction=0.25,
                        test_fraction=0.25,
                        seed=123,
                        source_val_fraction=0.25,
                        source_test_fraction=0.25,
                        show_progress=False,
                        split_workers=3,
                    )
            finally:
                dataset.close()

        self.assertEqual(split["train"], [0, 1])
        self.assertEqual(patched.call_args.kwargs["workers"], 3)

    def test_export_hetero_balanced_delegates_refill_to_dstio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "balanced.h5"
            summary = Path(tmpdir) / "summary.json"
            args = SimpleNamespace(
                input=["/mc/proton/sel/DAT000016_gea_trg_000.dst.gz"],
                input_list=[],
                input_dir=[],
                output=str(output),
                kind="mc",
                const_dst="/calib/talesdconst_pass2.dst",
                mc_calib_dir="/calib",
                max_events=None,
                min_event_date=191002,
                energy_sample_per_bin=2,
                energy_sample_stratify_particle=True,
                energy_bin_width=0.1,
                refill_attempts=2,
                refill_safety_factor=1.25,
                refill_min_efficiency=0.01,
                seed=12345,
                workers=4,
                scan_workers=3,
                selection_workers=1,
                worker_max_files=0,
                balance_cell_preselect=8,
                balance_zenith_bin_width_deg=10.0,
                balance_azimuth_bin_width_deg=30.0,
                balance_core_bin_width_km=0.5,
                balance_time_bin_width_sec=3600,
                selection_summary=str(summary),
                dry_run_selection=False,
                output_order="interleaved",
                output_locality_run_size=32,
                write_block_size=2048,
                h5_backend="auto",
                h5_progress_interval_sec=30.0,
                source_scan_progress_interval_events=1000,
                cleaning="ising",
                node_policy="all_candidates_with_ising",
                require_reference_core=True,
                shard_size=100000,
                open_retries=3,
                open_retry_delay=1.0,
                keep_non_mode0=False,
                skip_errors=True,
                skip_missing_mc_calibration=True,
            )
            fake_result = {
                "complete": True,
                "output_paths": [str(Path(tmpdir) / "worker_0000" / "graphs_0000.h5")],
                "splits": {"all": {"written_counts_by_stratum": {"16|proton": 2, "16|iron": 2}}},
            }

            with mock.patch.object(tale_graph, "write_balanced_graph_h5", return_value=fake_result) as patched:
                _cmd_export_hetero(args)

            self.assertTrue(summary.exists())
            self.assertEqual(patched.call_args.args[1], output.parent)
            self.assertEqual(patched.call_args.kwargs["workers"], 4)
            self.assertEqual(patched.call_args.kwargs["scan_workers"], 3)
            self.assertEqual(patched.call_args.kwargs["selection_workers"], 1)
            self.assertEqual(patched.call_args.kwargs["h5_backend"], "auto")
            self.assertEqual(patched.call_args.kwargs["progress_interval_events"], 1000)

    def test_light_hetero_selection_uses_filename_source_groups_only(self) -> None:
        inputs = [
            "/mc/proton/sel/DAT000116_gea_trg_000.dst.gz",
            "/mc/proton/sel/DAT000116_gea_trg_001.dst.gz",
            "/mc/proton/sel/DAT000216_gea_trg_000.dst.gz",
            "/mc/proton/sel/DAT000316_gea_trg_000.dst.gz",
            "/mc/iron/sel/DAT100116_gea_trg_000.dst.gz",
            "/mc/iron/sel/DAT100216_gea_trg_000.dst.gz",
            "/mc/proton/sel/DAT000117_gea_trg_000.dst.gz",
            "/mc/iron/sel/DAT100117_gea_trg_000.dst.gz",
            "/mc/iron/sel/DAT100217_gea_trg_000.dst.gz",
        ]

        selected, summary = _select_light_hetero_source_groups(inputs, seed=7)

        self.assertEqual(
            summary["source_groups_by_stratum"],
            {"iron:16": 2, "iron:17": 2, "proton:16": 3, "proton:17": 1},
        )
        self.assertEqual(summary["selected_source_groups_per_stratum"], 1)
        self.assertEqual(summary["selected_source_groups"], 4)
        self.assertTrue(summary["does_not_prescan_events"])
        self.assertEqual({group.stratum for group in selected}, {"proton:16", "proton:17", "iron:16", "iron:17"})
        proton16 = [group for group in selected if group.stratum == "proton:16"]
        self.assertEqual(len(proton16), 1)
        if proton16[0].dat_tag == "DAT000116":
            self.assertEqual(
                [Path(path).name for path in proton16[0].paths],
                ["DAT000116_gea_trg_000.dst.gz", "DAT000116_gea_trg_001.dst.gz"],
            )

    def test_light_hetero_worker_refills_short_source_groups(self) -> None:
        class FakeHandle:
            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        path_counts = {
            "short.dst.gz": 5,
            "full.dst.gz": 20,
        }
        written: list[tuple[int, str]] = []

        def fake_iter_graphs(paths: list[str], **_: object) -> list[str]:
            path = Path(paths[0]).name
            return [path] * path_counts[path]

        def fake_write(_handle: FakeHandle, index: int, graph: str) -> None:
            written.append((index, graph))

        payload = {
            "worker_index": 0,
            "strata": [
                {
                    "stratum": "proton:16",
                    "target_graphs": 20,
                    "groups": [
                        {
                            "source_group": "/mc/proton/DAT000116",
                            "dat_tag": "DAT000116",
                            "energy_bin_code": "16",
                            "particle": "proton",
                            "stratum": "proton:16",
                            "paths": ["/mc/proton/short.dst.gz"],
                        },
                        {
                            "source_group": "/mc/proton/DAT000216",
                            "dat_tag": "DAT000216",
                            "energy_bin_code": "16",
                            "particle": "proton",
                            "stratum": "proton:16",
                            "paths": ["/mc/proton/full.dst.gz"],
                        },
                    ],
                }
            ],
            "output_base": tempfile.mkdtemp(),
            "graphs_per_source_group": 10,
            "source_group_overdraw_factor": 10.0,
            "seed": 12345,
            "cleaning": "ising",
            "node_policy": "all_candidates_with_ising",
            "const_dst": None,
            "mc_calib_dir": None,
            "require_trigger_mode0": True,
            "require_reference_core": True,
            "skip_errors": True,
            "skip_missing_mc_calibration": True,
            "min_event_date": None,
            "open_retries": 0,
            "open_retry_delay": 0.0,
            "shard_size": 1000,
            "progress_interval_sec": 0.0,
            "config": {},
        }
        with mock.patch.object(tale_graph, "create_graph_h5_file", return_value=FakeHandle()):
            with mock.patch.object(tale_graph, "iter_graphs", side_effect=fake_iter_graphs):
                with mock.patch.object(tale_graph, "write_graph_h5_event", side_effect=fake_write):
                    result = _export_light_hetero_worker(payload)

        self.assertEqual(result["graphs_written"], 20)
        self.assertEqual([row["graphs_written"] for row in result["source_groups"]], [5, 15])
        self.assertEqual([row["primary_graphs_written"] for row in result["source_groups"]], [5, 10])
        self.assertEqual([row["refill_graphs_written"] for row in result["source_groups"]], [0, 5])
        self.assertEqual([row["complete"] for row in result["source_groups"]], [False, True])
        self.assertEqual(result["strata"][0]["target_met"], True)
        self.assertEqual(result["strata"][0]["attempted_source_groups"], 2)
        self.assertEqual([graph for _index, graph in written].count("short.dst.gz"), 5)
        self.assertEqual([graph for _index, graph in written].count("full.dst.gz"), 15)

    def test_light_hetero_worker_spreads_refill_over_surplus_groups(self) -> None:
        class FakeHandle:
            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        path_counts = {
            "short.dst.gz": 5,
            "full_a.dst.gz": 20,
            "full_b.dst.gz": 20,
        }
        written: list[tuple[int, str]] = []

        def fake_iter_graphs(paths: list[str], **_: object) -> list[str]:
            path = Path(paths[0]).name
            return [path] * path_counts[path]

        def fake_write(_handle: FakeHandle, index: int, graph: str) -> None:
            written.append((index, graph))

        payload = {
            "worker_index": 0,
            "strata": [
                {
                    "stratum": "proton:16",
                    "target_graphs": 30,
                    "groups": [
                        {
                            "source_group": "/mc/proton/DAT000116",
                            "dat_tag": "DAT000116",
                            "energy_bin_code": "16",
                            "particle": "proton",
                            "stratum": "proton:16",
                            "paths": ["/mc/proton/short.dst.gz"],
                        },
                        {
                            "source_group": "/mc/proton/DAT000216",
                            "dat_tag": "DAT000216",
                            "energy_bin_code": "16",
                            "particle": "proton",
                            "stratum": "proton:16",
                            "paths": ["/mc/proton/full_a.dst.gz"],
                        },
                        {
                            "source_group": "/mc/proton/DAT000316",
                            "dat_tag": "DAT000316",
                            "energy_bin_code": "16",
                            "particle": "proton",
                            "stratum": "proton:16",
                            "paths": ["/mc/proton/full_b.dst.gz"],
                        },
                    ],
                }
            ],
            "output_base": tempfile.mkdtemp(),
            "graphs_per_source_group": 10,
            "source_group_overdraw_factor": 10.0,
            "seed": 12345,
            "cleaning": "ising",
            "node_policy": "all_candidates_with_ising",
            "const_dst": None,
            "mc_calib_dir": None,
            "require_trigger_mode0": True,
            "require_reference_core": True,
            "skip_errors": True,
            "skip_missing_mc_calibration": True,
            "min_event_date": None,
            "open_retries": 0,
            "open_retry_delay": 0.0,
            "shard_size": 1000,
            "progress_interval_sec": 0.0,
            "config": {},
        }
        with mock.patch.object(tale_graph, "create_graph_h5_file", return_value=FakeHandle()):
            with mock.patch.object(tale_graph, "iter_graphs", side_effect=fake_iter_graphs):
                with mock.patch.object(tale_graph, "write_graph_h5_event", side_effect=fake_write):
                    result = _export_light_hetero_worker(payload)

        written_by_path = {path: [graph for _index, graph in written].count(path) for path in path_counts}
        self.assertEqual(result["graphs_written"], 30)
        self.assertEqual(written_by_path["short.dst.gz"], 5)
        self.assertGreater(written_by_path["full_a.dst.gz"], 10)
        self.assertGreater(written_by_path["full_b.dst.gz"], 10)
        self.assertLessEqual(written_by_path["full_a.dst.gz"], 13)
        self.assertLessEqual(written_by_path["full_b.dst.gz"], 13)
        self.assertEqual(result["strata"][0]["target_met"], True)

    def test_light_hetero_worker_discards_overdrawn_graphs_by_seeded_score(self) -> None:
        class FakeHandle:
            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        written: list[str] = []
        graphs = [SimpleNamespace(event_id=f"event_{index:02d}") for index in range(20)]

        def fake_iter_graphs(paths: list[str], **_: object) -> list[SimpleNamespace]:
            self.assertEqual(Path(paths[0]).name, "full.dst.gz")
            return graphs

        def fake_write(_handle: FakeHandle, _index: int, graph: SimpleNamespace) -> None:
            written.append(str(graph.event_id))

        payload = {
            "worker_index": 0,
            "strata": [
                {
                    "stratum": "proton:16",
                    "target_graphs": 10,
                    "groups": [
                        {
                            "source_group": "/mc/proton/DAT000116",
                            "dat_tag": "DAT000116",
                            "energy_bin_code": "16",
                            "particle": "proton",
                            "stratum": "proton:16",
                            "paths": ["/mc/proton/full.dst.gz"],
                        },
                    ],
                }
            ],
            "output_base": tempfile.mkdtemp(),
            "graphs_per_source_group": 10,
            "source_group_overdraw_factor": 2.0,
            "seed": 12345,
            "cleaning": "ising",
            "node_policy": "all_candidates_with_ising",
            "const_dst": None,
            "mc_calib_dir": None,
            "require_trigger_mode0": True,
            "require_reference_core": True,
            "skip_errors": True,
            "skip_missing_mc_calibration": True,
            "min_event_date": None,
            "open_retries": 0,
            "open_retry_delay": 0.0,
            "shard_size": 1000,
            "progress_interval_sec": 0.0,
            "config": {},
        }
        with mock.patch.object(tale_graph, "create_graph_h5_file", return_value=FakeHandle()):
            with mock.patch.object(tale_graph, "iter_graphs", side_effect=fake_iter_graphs):
                with mock.patch.object(tale_graph, "write_graph_h5_event", side_effect=fake_write):
                    result = _export_light_hetero_worker(payload)

        self.assertEqual(result["graphs_written"], 10)
        self.assertEqual(result["source_groups"][0]["graphs_found"], 20)
        self.assertEqual(result["source_groups"][0]["discarded_graphs"], 10)
        self.assertTrue(result["source_groups"][0]["overdraw_cap_reached"])
        self.assertNotEqual(written, [f"event_{index:02d}" for index in range(10)])

    def test_split_distribution_energy_bins_are_centered(self) -> None:
        self.assertEqual(_energy_bin(17.99, 0.1), "17.95-18.05")
        self.assertEqual(_energy_bin(18.90, 0.1), "18.85-18.95")

    def test_export_hetero_unbalanced_uses_dstio_h5_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "graphs.h5"
            args = SimpleNamespace(
                input=["/data/event.dst.gz"],
                input_list=[],
                input_dir=[],
                output=str(output),
                kind="data",
                const_dst=None,
                mc_calib_dir=None,
                max_events=1,
                min_event_date=None,
                energy_sample_per_bin=None,
                energy_sample_stratify_particle=False,
                energy_bin_width=0.1,
                refill_attempts=0,
                refill_safety_factor=1.25,
                refill_min_efficiency=0.01,
                seed=12345,
                workers=1,
                scan_workers=None,
                selection_workers=1,
                worker_max_files=0,
                balance_cell_preselect=8,
                balance_zenith_bin_width_deg=10.0,
                balance_azimuth_bin_width_deg=30.0,
                balance_core_bin_width_km=0.5,
                balance_time_bin_width_sec=3600,
                selection_summary=None,
                dry_run_selection=False,
                output_order="interleaved",
                output_locality_run_size=32,
                write_block_size=2048,
                h5_backend="python",
                h5_progress_interval_sec=10.0,
                source_scan_progress_interval_events=0,
                cleaning="ising",
                node_policy="all_candidates_with_ising",
                require_reference_core=False,
                shard_size=0,
                open_retries=1,
                open_retry_delay=0.0,
                keep_non_mode0=False,
                skip_errors=False,
                skip_missing_mc_calibration=False,
            )
            fake_result = {"graphs_written": 1, "output_paths": [str(output)]}

            with mock.patch.object(tale_graph, "write_graph_h5", return_value=fake_result) as patched:
                _cmd_export_hetero(args)

            self.assertEqual(patched.call_args.args[1], output)
            self.assertEqual(patched.call_args.kwargs["h5_backend"], "python")
            self.assertEqual(patched.call_args.kwargs["write_block_size"], 2048)


@unittest.skipUnless(DATA_SAMPLE.exists(), "dstio TALE data sample is not available")
class HeteroGraphIoTest(unittest.TestCase):
    def test_write_and_read_dstio_hetero_graph(self) -> None:
        graph = next(tale_graph.iter_graphs(DATA_SAMPLE, kind="data", max_events=1))
        self.assertEqual(graph.metadata["graph_definition"], GRAPH_DEFINITION)
        self.assertEqual(graph.detector_waveforms.ndim, 3)
        self.assertEqual(graph.pulse_features.shape[0], graph.pulse_detector_index.shape[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hetero_graph.h5"
            with create_hetero_graph_file(output, config={"sample": str(DATA_SAMPLE)}) as handle:
                write_hetero_graph(handle, 0, graph)

            self.assertEqual(hetero_graph_count(output), 1)
            with h5py.File(output, "r") as handle:
                self.assertEqual(handle.attrs["format"], FORMAT_NAME)
                self.assertEqual(handle.attrs["graph_definition"], GRAPH_DEFINITION)
                self.assertEqual(handle.attrs["waveform_schema"], WAVEFORM_SCHEMA)
                event = handle["events"]["00000000"]
                self.assertIn("detector_features", event)
                self.assertIn("detector_context_features", event)
                self.assertIn("detector_waveforms", event)
                self.assertIn("pulse_features", event)
                self.assertIn("pulse_detector_index", event)
                self.assertEqual(set(event["edge_index_by_type"].keys()), set(EDGE_RELATIONS))
                self.assertEqual(set(event["edge_features_by_type"].keys()), set(EDGE_RELATIONS))
                self.assertIn("metadata_json", event.attrs)

            dataset = H5HeteroGraphDataset(output)
            try:
                sample = dataset[0]
                np.testing.assert_array_equal(sample["detector_lids"], graph.detector_lids)
                np.testing.assert_array_equal(sample["pulse_lids"], graph.pulse_lids)
                self.assertEqual(sample["detector_waveforms"].shape, graph.detector_waveforms.shape)
                self.assertEqual(set(sample["edge_index_by_type"]), set(EDGE_RELATIONS))
                self.assertEqual(sample["metadata"]["graph_definition"], GRAPH_DEFINITION)
                self.assertIsInstance(dataset.source_path(0), str)
                self.assertIsNone(dataset.target(0))
                self.assertIsNone(dataset.particle_label(0))
            finally:
                dataset.close()

    def test_tensor_conversion_and_minimal_model_forward(self) -> None:
        graph = next(tale_graph.iter_graphs(DATA_SAMPLE, kind="data", max_events=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hetero_graph.h5"
            with create_hetero_graph_file(output) as handle:
                write_hetero_graph(handle, 0, graph)

            dataset = H5HeteroGraphDataset(output)
            try:
                sample = dataset[0]
            finally:
                dataset.close()

        tensors = hetero_sample_to_tensors(sample)
        self.assertEqual(tensors["detector"]["x"].shape[1], DETECTOR_FEATURE_DIM)
        self.assertEqual(tensors["detector"]["context"].shape[1], DETECTOR_CONTEXT_DIM)
        self.assertEqual(tensors["pulse"]["x"].shape[1], PULSE_FEATURE_DIM)
        self.assertEqual(tensors["detector"]["waveform"].ndim, 3)

        model = MinimalHeteroTaleSdGNN.from_sample(
            sample,
            target_dim=6,
            classification_dim=1,
            quality_dim=1,
            error_dim=3,
            hidden_dim=24,
            num_layers=1,
            dropout=0.0,
            waveform_embedding_dim=12,
        )
        output_tensor = model(tensors)
        self.assertEqual(tuple(output_tensor.shape), (1, 11))
        self.assertTrue(torch.isfinite(output_tensor).all())
        loss = output_tensor.square().mean()
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters() if param.requires_grad))

    def test_optional_pyg_conversion_when_available(self) -> None:
        graph = next(tale_graph.iter_graphs(DATA_SAMPLE, kind="data", max_events=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hetero_graph.h5"
            with create_hetero_graph_file(output) as handle:
                write_hetero_graph(handle, 0, graph)

            dataset = H5HeteroGraphDataset(output)
            try:
                sample = dataset[0]
            finally:
                dataset.close()

        try:
            data = sample_to_hetero_data(sample)
        except TorchGeometricUnavailableError:
            self.skipTest("torch_geometric is not installed")
        self.assertEqual(data["detector"].x.shape[1], DETECTOR_FEATURE_DIM)
        self.assertEqual(data["pulse"].x.shape[1], PULSE_FEATURE_DIM)
        self.assertIn(("detector", "observes", "pulse"), data.edge_types)
        self.assertIn(("pulse", "observed_by", "detector"), data.edge_types)

    def test_pyg_batch_forward(self) -> None:
        from torch_geometric.loader import DataLoader

        graph = next(tale_graph.iter_graphs(DATA_SAMPLE, kind="data", max_events=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hetero_graph.h5"
            with create_hetero_graph_file(output) as handle:
                write_hetero_graph(handle, 0, graph)

            dataset = H5PyGHeteroGraphDataset(output)
            try:
                data = dataset[0]
            finally:
                dataset.close()
            sample_dataset = H5HeteroGraphDataset(output)
            try:
                sample = sample_dataset[0]
            finally:
                sample_dataset.close()

        loader = DataLoader([data, data], batch_size=2)
        batch = next(iter(loader))
        model = MinimalHeteroTaleSdGNN.from_sample(
            sample,
            target_dim=6,
            classification_dim=1,
            quality_dim=1,
            error_dim=3,
            hidden_dim=24,
            num_layers=1,
            dropout=0.0,
            waveform_embedding_dim=12,
        )
        output_tensor = model(batch)
        self.assertEqual(tuple(output_tensor.shape), (2, 11))
        self.assertTrue(torch.isfinite(output_tensor).all())
        loss = output_tensor.square().mean()
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters() if param.requires_grad))

    def test_invalid_detector_waveform_is_masked(self) -> None:
        graph = _synthetic_graph(0)
        if "detector_waveform_valid" not in DETECTOR_FEATURE_INDEX:
            self.skipTest("detector_waveform_valid is not available in this graph schema")
        invalid_detector = 0
        graph.detector_features[invalid_detector, DETECTOR_FEATURE_INDEX["detector_waveform_valid"]] = 0.0
        sample_a = {
            "detector_features": graph.detector_features,
            "detector_context_features": graph.detector_context_features,
            "detector_positions_km": graph.detector_positions_km,
            "detector_lids": graph.detector_lids,
            "detector_waveforms": graph.detector_waveforms,
            "pulse_features": graph.pulse_features,
            "pulse_positions_km": graph.pulse_positions_km,
            "pulse_lids": graph.pulse_lids,
            "pulse_detector_index": graph.pulse_detector_index,
            "pulse_bounds": graph.pulse_bounds,
            "edge_index_by_type": graph.edge_index_by_type,
            "edge_features_by_type": graph.edge_features_by_type,
            "target": graph.target,
            "particle_label": graph.particle_label,
            "metadata": graph.metadata,
        }
        sample_b = dict(sample_a)
        sample_b["detector_waveforms"] = np.array(sample_a["detector_waveforms"], copy=True)
        sample_b["detector_waveforms"][invalid_detector] += 1000.0

        model = MinimalHeteroTaleSdGNN.from_sample(
            sample_a,
            target_dim=6,
            classification_dim=1,
            hidden_dim=24,
            num_layers=1,
            dropout=0.0,
            waveform_embedding_dim=12,
        )
        model.eval()
        with torch.no_grad():
            output_a = model(hetero_sample_to_tensors(sample_a))
            output_b = model(hetero_sample_to_tensors(sample_b))

        self.assertTrue(torch.allclose(output_a, output_b, atol=1.0e-6, rtol=1.0e-6))

    def test_train_hetero_smoke_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(6):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            result = train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                gradient_accumulation_steps=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                quality_prediction=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                checkpoint_milestones=(1,),
                checkpoint_milestone_full_eval=True,
                show_progress=False,
            )

            self.assertEqual(result["checkpoint"], str(checkpoint_path))
            self.assertEqual(result["metrics_json"], str(checkpoint_path) + ".metrics.json")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["architecture"], "hetero_attention")
            self.assertEqual(checkpoint["runtime"]["model_architecture"], "hetero_attention")
            self.assertEqual(checkpoint["model_config"]["quality_dim"], 1)
            self.assertEqual(checkpoint["model_config"]["error_dim"], 0)
            self.assertEqual(checkpoint["runtime"]["gradient_accumulation_steps"], 2)
            self.assertEqual(checkpoint["runtime"]["effective_batch_size"], 4)
            self.assertTrue(checkpoint["runtime"]["completed"])
            self.assertEqual(checkpoint["runtime"]["checkpoint_kind"], "final")
            self.assertEqual(checkpoint["runtime"]["best_epoch"], 1)
            self.assertEqual(checkpoint["runtime"]["checkpoint_epoch"], 1)
            self.assertIn("hetero_scalers", checkpoint)
            self.assertIn("detector", checkpoint["hetero_scalers"])
            self.assertIn("target", checkpoint["hetero_scalers"])
            self.assertIn("metrics", checkpoint)
            self.assertIn("test", checkpoint["metrics"])
            self.assertEqual(len(checkpoint["history"]), 1)
            self.assertIn("train_quality_loss", checkpoint["history"][0])
            self.assertTrue(checkpoint["runtime"]["quality_prediction"])
            self.assertFalse(checkpoint["runtime"]["error_prediction"])
            self.assertTrue((Path(str(checkpoint_path) + ".metrics.json")).exists())
            milestone_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.best_through_epoch0001.pt")
            milestone_metrics = Path(f"{milestone_path}.metrics.json")
            self.assertTrue(milestone_path.exists())
            self.assertTrue(milestone_metrics.exists())
            milestone_payload = json.loads(milestone_metrics.read_text())
            self.assertEqual(milestone_payload["runtime"]["checkpoint_kind"], "milestone_epoch_1")
            self.assertIn("validation", milestone_payload["metrics"])
            self.assertIn("test", milestone_payload["metrics"])

    def test_validation_skip_does_not_update_best_without_explicit_train_loss_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            checkpoint_train_loss_path = Path(tmpdir) / "checkpoint_train_loss.pt"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(8):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                validate_every_n_epochs=0,
                show_progress=False,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["runtime"]["best_checkpoint_metric"], "none_no_validation")
            self.assertEqual(checkpoint["runtime"]["best_checkpoint_kind"], "none_no_validation")

            train_hetero_model(
                graph_path,
                checkpoint_train_loss_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                validate_every_n_epochs=0,
                allow_train_loss_checkpoint=True,
                show_progress=False,
            )
            checkpoint_train = torch.load(checkpoint_train_loss_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint_train["runtime"]["best_checkpoint_metric"], "train_loss_benchmark")
            self.assertEqual(checkpoint_train["runtime"]["best_checkpoint_kind"], "train_loss_benchmark")

    def test_train_hetero_reuses_matching_scaler_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_a = Path(tmpdir) / "checkpoint_a.pt"
            checkpoint_b = Path(tmpdir) / "checkpoint_b.pt"
            scaler_cache = Path(tmpdir) / "scalers.json"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(8):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            train_hetero_model(
                graph_path,
                checkpoint_a,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                validate_every_n_epochs=0,
                scaler_cache_path=scaler_cache,
                show_progress=False,
            )
            self.assertTrue(scaler_cache.exists())
            payload = json.loads(scaler_cache.read_text())
            self.assertGreater(payload["metadata"]["train_graph_count"], 0)
            self.assertEqual(payload["metadata"]["graph_format"], "grouped_hdf5")

            with mock.patch.object(hetero_training, "fit_hetero_scalers", side_effect=AssertionError):
                train_hetero_model(
                    graph_path,
                    checkpoint_b,
                    epochs=1,
                    batch_size=2,
                    hidden_dim=16,
                    num_layers=1,
                    dropout=0.0,
                    waveform_embedding_dim=8,
                    mass_classification=True,
                    split_mode="event",
                    device="cpu",
                    num_workers=0,
                    validate_every_n_epochs=0,
                    scaler_cache_path=scaler_cache,
                    reuse_scaler_cache=True,
                    show_progress=False,
                )

    def test_train_hetero_small_max_graphs_smoke_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(30):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            result = train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                max_graphs=20,
                show_progress=False,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            split = checkpoint["split"]
            self.assertGreaterEqual(split["n_train"], 1)
            self.assertGreaterEqual(split["n_val"], 1)
            self.assertGreaterEqual(split["n_test"], 1)
            self.assertEqual(result["checkpoint"], str(checkpoint_path))

    def test_train_hetero_from_flat_cache_smoke_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            flat_path = Path(tmpdir) / "synthetic_hetero.flat.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(6):
                    write_hetero_graph(handle, index, _synthetic_graph(index))
            convert_hetero_to_flat_cache(graph_path, flat_path, compression="none")

            train_hetero_model(
                flat_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                gradient_accumulation_steps=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                training_data_format="fast_tensor",
                show_progress=False,
            )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["runtime"]["training_data_format"], "fast_tensor")
            self.assertTrue(checkpoint["runtime"]["completed"])

    def test_train_hetero_minimal_architecture_smoke_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(6):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                model_architecture="minimal_hetero",
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                show_progress=False,
            )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["architecture"], "minimal_hetero")
            self.assertEqual(checkpoint["runtime"]["model_architecture"], "minimal_hetero")

    def test_train_hetero_error_head_smoke_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(6):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                quality_prediction=False,
                error_prediction=True,
                error_loss_weight=0.2,
                split_mode="event",
                device="cpu",
                num_workers=0,
                show_progress=False,
            )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["quality_dim"], 0)
            self.assertEqual(checkpoint["model_config"]["error_dim"], 3)
            self.assertFalse(checkpoint["runtime"]["quality_prediction"])
            self.assertTrue(checkpoint["runtime"]["error_prediction"])
            self.assertEqual(checkpoint["runtime"]["error_loss_weight"], 0.2)
            self.assertIn("train_error_loss", checkpoint["history"][0])
            self.assertNotIn("train_quality_loss", checkpoint["history"][0])

    def test_hetero_feature_importance_smoke_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            output_dir = Path(tmpdir) / "importance"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(6):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                show_progress=False,
            )

            result = save_hetero_feature_group_importance(
                graph_path,
                checkpoint_path,
                output_dir,
                split="validation",
                max_graphs=2,
                batch_size=2,
                device="cpu",
                show_progress=False,
            )

            summary_path = Path(result["summary_json"])
            self.assertTrue(summary_path.exists())
            self.assertTrue((output_dir / "feature_group_importance.pdf").exists())
            payload = json.loads(summary_path.read_text())
            self.assertEqual(payload["n_graphs"], 1)
            self.assertTrue(payload["groups"])
            self.assertIn("baseline", payload)
            redraw = payload["redraw_artifacts"]
            self.assertTrue(Path(redraw["plot_data_json"]).exists())
            plot_data = json.loads(Path(redraw["plot_data_json"]).read_text())
            self.assertIn("median_abs_relative_energy", payload["groups"][0]["reconstruction_delta"])
            self.assertIn("plot_specs", plot_data)

    def test_hetero_attention_maps_smoke_writes_json_and_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            output_dir = Path(tmpdir) / "attention"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(6):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                show_progress=False,
            )

            result = save_hetero_attention_maps(
                graph_path,
                checkpoint_path,
                output_dir,
                split="validation",
                max_graphs=1,
                device="cpu",
                show_progress=False,
            )

            summary_path = Path(result["summary_json"])
            array_path = Path(result["array_file"])
            self.assertTrue(summary_path.exists())
            self.assertTrue(array_path.exists())
            payload = json.loads(summary_path.read_text())
            self.assertEqual(payload["format"], "hetero_attention_maps_v1")
            self.assertEqual(payload["n_graphs"], 1)
            self.assertTrue(payload["events"][0]["relations"])
            self.assertIn("readout_detector_weights", payload["events"][0]["arrays"])
            self.assertIn("pulse_bounds", payload["events"][0]["arrays"])
            with np.load(array_path) as arrays:
                self.assertIn(payload["events"][0]["arrays"]["pulse_bounds"], arrays.files)
                weight_keys = [key for key in arrays.files if key.endswith("_attention_weights")]
                self.assertTrue(weight_keys)

    def test_hetero_split_distribution_summary_reads_counts_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(12):
                    graph = _synthetic_graph(index)
                    particle = "proton" if index < 6 else "iron"
                    graph.particle_label = 0.0 if particle == "proton" else 1.0
                    graph.target[0] = 17.0
                    graph.target[3:6] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
                    graph.metadata["source_path"] = f"/synthetic/{particle}/DAT{index:04d}16_gea_trg_000.dst.gz"
                    graph.metadata["source_index"] = 0
                    graph.metadata["parttype"] = 14 if particle == "proton" else 5626
                    write_hetero_graph(handle, index, graph)

            dataset = H5HeteroGraphDataset(graph_path, require_target=True, require_particle_label=True)
            try:
                plot_dir = Path(tmpdir) / "split_plots"
                payload = summarize(
                    dataset,
                    val_fraction=0.20,
                    test_fraction=0.20,
                    source_val_fraction=0.20,
                    source_test_fraction=0.20,
                    seed=123,
                    energy_bin_width=0.1,
                    split_workers=2,
                    show_progress=False,
                    plot_dir=plot_dir,
                )
            finally:
                dataset.close()
            redraw = payload["config"]["redraw_artifacts"]
            self.assertTrue(Path(redraw["split_distribution_plot_data_json"]).exists())
            plot_data = json.loads(Path(redraw["split_distribution_plot_data_json"]).read_text())
            self.assertIn("features", plot_data)
            self.assertIn("energy_bin_counts", plot_data)
            self.assertEqual(len(plot_data["features"]), 9)
            self.assertNotIn("core_radius_km", plot_data["features"])
            self.assertNotIn("event_time_hour", plot_data["features"])
            self.assertIn("independent_showers", plot_data["count_definitions"])
            for split_counts in plot_data["energy_bin_counts"]["splits"].values():
                self.assertIn("events", split_counts)
                self.assertIn("independent_showers", split_counts)
                self.assertNotIn("sources", split_counts)
            for pdf in ("split_parameter_distributions.pdf", "split_energy_bin_counts.pdf"):
                (plot_dir / pdf).unlink()
                self.assertFalse((plot_dir / pdf).exists())
            redraw_result = redraw_split_distribution_plots(Path(redraw["split_distribution_plot_data_json"]))
            self.assertEqual(len(redraw_result["plot_files"]), 2)
            self.assertTrue((plot_dir / "split_parameter_distributions.pdf").exists())
            self.assertTrue((plot_dir / "split_energy_bin_counts.pdf").exists())

        self.assertEqual(payload["config"]["graph_format"], "hetero")
        total_events = sum(split["events"] for split in payload["totals"].values())
        self.assertEqual(total_events, 12)
        for split in payload["totals"].values():
            self.assertEqual(split["sources"], split["independent_showers"])
            self.assertGreater(split["detector_nodes"]["n"], 0)
            self.assertGreater(split["pulse_nodes"]["n"], 0)
            self.assertGreater(split["event_time_hour"]["n"], 0)

    def test_hetero_balanced_entries_are_interleaved_before_write(self) -> None:
        inputs = [
            "/mc/proton/sel/DAT000016_gea_trg_000.dst.gz",
            "/mc/proton/sel/DAT000017_gea_trg_000.dst.gz",
            "/mc/iron/sel/DAT100016_gea_trg_000.dst.gz",
            "/mc/iron/sel/DAT100017_gea_trg_000.dst.gz",
        ]
        selected = {path: set(range(4)) for path in inputs}

        source_entries = _selected_entries_from_path_indices(inputs, selected, stratify_particle=True)
        ordered = _ordered_selected_entries(
            source_entries,
            output_order="interleaved",
            seed=12345,
            locality_run_size=1,
        )

        self.assertEqual(len(ordered), len(source_entries))
        self.assertEqual({entry[1] for entry in ordered}, {entry[1] for entry in source_entries})
        self.assertNotEqual([entry[1] for entry in ordered], [entry[1] for entry in source_entries])
        first_eight_particles = ["iron" if "/iron/" in entry[2] else "proton" for entry in ordered[:8]]
        self.assertIn("iron", first_eight_particles)
        self.assertIn("proton", first_eight_particles)

    def test_reshard_hetero_reorders_existing_h5_without_changing_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "source.h5"
            output_path = Path(tmpdir) / "reshuffled.h5"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(8):
                    graph = _synthetic_graph(index)
                    particle = "proton" if index < 4 else "iron"
                    dat_code = "16" if index % 2 == 0 else "17"
                    graph.metadata["source_path"] = f"/mc/{particle}/sel/DAT0000{dat_code}_gea_trg_{index:03d}.dst.gz"
                    graph.metadata["source_index"] = index
                    graph.metadata["event_id"] = f"event_{index:03d}"
                    graph.event_id = f"event_{index:03d}"
                    graph.particle_label = 0.0 if particle == "proton" else 1.0
                    write_hetero_graph(handle, index, graph)

            _cmd_reshard_hetero(
                SimpleNamespace(
                    graphs=[str(graph_path)],
                    graphs_list=[],
                    output=str(output_path),
                    output_order="interleaved",
                    output_locality_run_size=1,
                    seed=24680,
                    shard_size=0,
                    workers=1,
                    energy_sample_stratify_particle=True,
                    energy_sample_per_bin=None,
                    overwrite=False,
                )
            )

            with h5py.File(graph_path, "r") as source, h5py.File(output_path, "r") as target:
                source_ids = [source["metadata"]["event_id"][index].decode("utf-8") for index in range(8)]
                target_ids = [target["metadata"]["event_id"][index].decode("utf-8") for index in range(8)]
                self.assertEqual(set(target_ids), set(source_ids))
                self.assertNotEqual(target_ids, source_ids)
                target_particles = [float(target["metadata"]["particle_label"][index]) for index in range(8)]
                self.assertIn(0.0, target_particles[:4])
                self.assertIn(1.0, target_particles[:4])

    def test_reshard_hetero_parallel_writes_multiple_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_paths = [Path(tmpdir) / "source_a.h5", Path(tmpdir) / "source_b.h5"]
            output_path = Path(tmpdir) / "reshuffled.h5"
            for path_number, graph_path in enumerate(graph_paths):
                with create_hetero_graph_file(graph_path) as handle:
                    for local_index in range(3):
                        index = 3 * path_number + local_index
                        graph = _synthetic_graph(index)
                        particle = "proton" if index < 3 else "iron"
                        dat_code = "16" if index % 2 == 0 else "17"
                        graph.metadata["source_path"] = f"/mc/{particle}/sel/DAT0000{dat_code}_gea_trg_{index:03d}.dst.gz"
                        graph.metadata["source_index"] = index
                        graph.metadata["event_id"] = f"event_{index:03d}"
                        graph.event_id = f"event_{index:03d}"
                        graph.particle_label = 0.0 if particle == "proton" else 1.0
                        write_hetero_graph(handle, local_index, graph)

            _cmd_reshard_hetero(
                SimpleNamespace(
                    graphs=[str(path) for path in graph_paths],
                    graphs_list=[],
                    output=str(output_path),
                    output_order="interleaved",
                    output_locality_run_size=1,
                    seed=13579,
                    shard_size=2,
                    workers=2,
                    energy_sample_stratify_particle=True,
                    energy_sample_per_bin=None,
                    overwrite=False,
                )
            )

            output_shards = sorted(Path(tmpdir).glob("reshuffled_*.h5"))
            self.assertEqual(len(output_shards), 3)
            event_ids = []
            for shard in output_shards:
                with h5py.File(shard, "r") as handle:
                    self.assertEqual(len(handle["events"]), 2)
                    event_ids.extend(handle["metadata"]["event_id"][index].decode("utf-8") for index in range(2))
            self.assertEqual(set(event_ids), {f"event_{index:03d}" for index in range(6)})

    def test_reshard_hetero_can_downsample_existing_h5_by_filename_energy_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "source.h5"
            output_path = Path(tmpdir) / "downsampled.h5"
            with create_hetero_graph_file(graph_path) as handle:
                index = 0
                for particle in ("proton", "iron"):
                    for dat_code in ("16", "17"):
                        for local in range(5):
                            graph = _synthetic_graph(index)
                            graph.metadata["source_path"] = (
                                f"/mc/{particle}/sel/DAT{index:04d}{dat_code}_gea_trg_{local:03d}.dst.gz"
                            )
                            graph.metadata["source_index"] = local
                            graph.metadata["event_id"] = f"event_{index:03d}"
                            graph.event_id = f"event_{index:03d}"
                            graph.particle_label = 0.0 if particle == "proton" else 1.0
                            write_hetero_graph(handle, index, graph)
                            index += 1

            _cmd_reshard_hetero(
                SimpleNamespace(
                    graphs=[str(graph_path)],
                    graphs_list=[],
                    output=str(output_path),
                    output_order="interleaved",
                    output_locality_run_size=1,
                    seed=97531,
                    shard_size=0,
                    workers=1,
                    energy_sample_stratify_particle=True,
                    energy_sample_per_bin=2,
                    overwrite=False,
                )
            )

            with h5py.File(output_path, "r") as handle:
                self.assertEqual(len(handle["events"]), 8)
                config = json.loads(handle.attrs["config_json"])
                self.assertEqual(config["downsample_summary"]["selected_events"], 8)
                counts: dict[tuple[str, str], int] = {}
                for local_index in range(8):
                    source_path = handle["metadata"]["source_path"][local_index].decode("utf-8")
                    particle = "iron" if "/iron/" in source_path else "proton"
                    dat_code = "16" if "16_gea" in source_path else "17"
                    counts[(particle, dat_code)] = counts.get((particle, dat_code), 0) + 1
                self.assertEqual(set(counts.values()), {2})

    def test_hetero_input_distributions_write_summary_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            output_dir = Path(tmpdir) / "input_distributions"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(4):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            summary = save_hetero_input_distributions(
                graph_path,
                output_dir,
                max_graphs=4,
                max_values_per_feature=100,
                show_progress=False,
            )

            summary_path = Path(summary["summary_json"])
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text())
            self.assertEqual(payload["graph_format"], "hetero")
            self.assertEqual(payload["n_graphs_total"], 4)
            self.assertIn("detector", payload["features"])
            self.assertIn("pulse", payload["features"])
            self.assertIn("edge_features_by_type", payload["features"])
            artifacts = payload["redraw_artifacts"]
            self.assertTrue(Path(artifacts["sample_values_npz"]).exists())
            self.assertTrue(Path(artifacts["sample_values_manifest"]).exists())
            manifest = json.loads(Path(artifacts["sample_values_manifest"]).read_text())
            self.assertIn(["detector", "detector_trigger_usec_rel"], [item["path"] for item in manifest["arrays"]])
            self.assertTrue((output_dir / "detector_features.pdf").exists())
            self.assertTrue((output_dir / "pulse_features.pdf").exists())
            self.assertTrue((output_dir / "waveform_features.pdf").exists())

    @unittest.skipUnless(
        MC_SAMPLE.exists() and CONST_DST.exists() and MC_CALIB_DIR.exists(),
        "dstio TALE MC sample and calibration files are not available",
    )
    def test_train_checkpoint_reconstructs_dst_without_h5_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "mc_hetero.h5"
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            output_csv = Path(tmpdir) / "reco.csv"
            graphs = tale_graph.iter_graphs(
                MC_SAMPLE,
                kind="mc",
                const_dst=CONST_DST,
                mc_calib_dir=MC_CALIB_DIR,
                max_events=10,
                require_reference_core=True,
                skip_missing_mc_calibration=True,
            )
            with create_hetero_graph_file(graph_path) as handle:
                written = 0
                for written, graph in enumerate(graphs, start=1):
                    write_hetero_graph(handle, written - 1, graph)
            self.assertGreaterEqual(written, 2)

            train_hetero_model(
                graph_path,
                checkpoint_path,
                epochs=1,
                batch_size=2,
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                show_progress=False,
            )
            result = reconstruct_dst(
                MC_SAMPLE,
                checkpoint_path,
                output_csv,
                kind="mc",
                const_dst=CONST_DST,
                mc_calib_dir=MC_CALIB_DIR,
                max_events=10,
                batch_size=2,
                device="cpu",
                require_reference_core=True,
                skip_missing_mc_calibration=True,
            )

            self.assertGreater(result["events_written"], 0)
            with output_csv.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), result["events_written"])
            self.assertIn("event_id", rows[0])
            self.assertIn("log10_energy_eV", rows[0])
            self.assertIn("p_iron", rows[0])


if __name__ == "__main__":
    unittest.main()
