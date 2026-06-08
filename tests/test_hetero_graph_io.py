from __future__ import annotations

import csv
import json
import tempfile
import unittest
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
from talesd_gnn_reconstruction.hetero_feature_analysis import (
    save_hetero_feature_group_importance,
    save_hetero_input_distributions,
)
from talesd_gnn_reconstruction.hetero_graph_io import (
    EDGE_RELATIONS,
    FORMAT_NAME,
    GRAPH_DEFINITION,
    H5HeteroGraphDataset,
    H5PyGHeteroGraphDataset,
    WAVEFORM_SCHEMA,
    create_hetero_graph_file,
    hetero_graph_count,
    write_hetero_graph,
)
from talesd_gnn_reconstruction.hetero_model import MinimalHeteroTaleSdGNN
from talesd_gnn_reconstruction.hetero_predict import reconstruct_dst
from talesd_gnn_reconstruction.hetero_training import (
    _estimate_graph_bytes,
    _resolve_loader_settings,
    train_hetero_model,
)
from talesd_gnn_reconstruction.cli import (
    _cmd_reshard_hetero,
    _ordered_selected_entries,
    _selected_entries_from_path_indices,
)
from scripts.summarize_split_distributions import summarize


DATA_SAMPLE = Path("/Users/ikomae/TALE/dstio/test/data/tale_data_talesdcalibev_single_event.dst")
MC_SAMPLE = Path("/Users/ikomae/TALE/dstio/test/data/tale_mc_DAT000327_gea_sel_10_events.dst.gz")
CONST_DST = Path("/Users/ikomae/TALE/TASoft/development/data/SD/talesdconst_pass2.dst")
MC_CALIB_DIR = Path("/Users/ikomae/TALE/TASoft/development/data/SD")


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
    return SimpleNamespace(
        event_id=f"synthetic_{index:04d}",
        detector_features=rng.normal(size=(detector_count, 10)).astype(np.float32),
        detector_context_features=rng.normal(size=(detector_count, 7)).astype(np.float32),
        detector_positions_km=rng.normal(size=(detector_count, 3)).astype(np.float32),
        detector_lids=np.asarray([101, 102, 103], dtype=np.int64),
        detector_waveforms=rng.normal(size=(detector_count, 2, 16)).astype(np.float32),
        pulse_features=rng.normal(size=(pulse_count, 13)).astype(np.float32),
        pulse_positions_km=rng.normal(size=(pulse_count, 3)).astype(np.float32),
        pulse_lids=pulse_lids,
        pulse_detector_index=pulse_detector_index,
        pulse_bounds=np.asarray(
            [[1, 3, 5, 8], [2, 4, 6, 9], [3, 5, 7, 10], [4, 6, 8, 11]],
            dtype=np.float32,
        ),
        edge_index_by_type={
            "pulse__interacts__pulse": np.asarray([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
            "detector__near__detector": np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
            "detector__observes__pulse": np.asarray(
                [pulse_detector_index, np.arange(pulse_count, dtype=np.int64)],
                dtype=np.int64,
            ),
        },
        edge_features_by_type={
            "pulse__interacts__pulse": rng.normal(size=(4, 4)).astype(np.float32),
            "detector__near__detector": rng.normal(size=(3, 3)).astype(np.float32),
            "detector__observes__pulse": rng.normal(size=(pulse_count, 2)).astype(np.float32),
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
                self.assertEqual(scaler_sample["detector_features"].shape, (3, 10))
                self.assertEqual(scaler_sample["pulse_features"].shape, (4, 13))
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
        self.assertEqual(tensors["detector"]["x"].shape[1], 10)
        self.assertEqual(tensors["detector"]["context"].shape[1], 7)
        self.assertEqual(tensors["pulse"]["x"].shape[1], 13)
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
        self.assertEqual(data["detector"].x.shape[1], 10)
        self.assertEqual(data["pulse"].x.shape[1], 13)
        self.assertIn(("detector", "observes", "pulse"), data.edge_types)

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
                hidden_dim=16,
                num_layers=1,
                dropout=0.0,
                waveform_embedding_dim=8,
                mass_classification=True,
                quality_prediction=True,
                split_mode="event",
                device="cpu",
                num_workers=0,
                show_progress=False,
            )

            self.assertEqual(result["checkpoint"], str(checkpoint_path))
            self.assertEqual(result["metrics_json"], str(checkpoint_path) + ".metrics.json")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["architecture"], "hetero_attention")
            self.assertEqual(checkpoint["runtime"]["model_architecture"], "hetero_attention")
            self.assertEqual(checkpoint["model_config"]["quality_dim"], 1)
            self.assertEqual(checkpoint["model_config"]["error_dim"], 0)
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

    def test_hetero_split_distribution_summary_reads_counts_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "synthetic_hetero.h5"
            with create_hetero_graph_file(graph_path) as handle:
                for index in range(12):
                    write_hetero_graph(handle, index, _synthetic_graph(index))

            dataset = H5HeteroGraphDataset(graph_path, require_target=True, require_particle_label=True)
            try:
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
            )
            finally:
                dataset.close()

        self.assertEqual(payload["config"]["graph_format"], "hetero")
        total_events = sum(split["events"] for split in payload["totals"].values())
        self.assertEqual(total_events, 12)
        for split in payload["totals"].values():
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
