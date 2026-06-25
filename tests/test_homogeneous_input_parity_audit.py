from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_homogeneous_input_parity.py"
    spec = importlib.util.spec_from_file_location("audit_homogeneous_input_parity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_h5(path: Path, *, node_offset: float = 0.0, source_index: int = 7) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "talesd_gnn_graphs"
        events = handle.create_group("events")
        metadata = handle.create_group("metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        metadata.create_dataset("event_id", data=np.asarray(["event-a"], dtype=object), dtype=string_dtype)
        metadata.create_dataset(
            "source_path",
            data=np.asarray(["/mc/proton/DAT000123_gea_trg_001.dst.gz"], dtype=object),
            dtype=string_dtype,
        )
        metadata.create_dataset("source_index", data=np.asarray([source_index], dtype=np.int64))
        metadata.create_dataset("parttype", data=np.asarray([14], dtype=np.int32))
        metadata.create_dataset("particle_label", data=np.asarray([0.0], dtype=np.float32))

        event = events.create_group("00000000")
        event.attrs["event_id"] = "event-a"
        event.attrs["source_path"] = "/mc/proton/DAT000123_gea_trg_001.dst.gz"
        event.attrs["source_index"] = source_index
        event.create_dataset("node_features", data=np.asarray([[1.0 + node_offset, 2.0]], dtype=np.float32))
        event.create_dataset("node_positions_km", data=np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32))
        event.create_dataset("node_lids", data=np.asarray([101], dtype=np.int64))
        event.create_dataset("edge_index", data=np.asarray([[0], [0]], dtype=np.int64))
        event.create_dataset("edge_features", data=np.asarray([[0.5, 0.25]], dtype=np.float32))
        event.create_dataset("pulse_features", data=np.asarray([[0.0, 1.0]], dtype=np.float32))
        event.create_dataset("waveform_features", data=np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
        event.create_dataset("target", data=np.asarray([18.0, 0.1, -0.2, 0.0, 0.0, 1.0], dtype=np.float32))
        event.create_dataset("particle_label", data=np.asarray(0.0, dtype=np.float32))


class HomogeneousInputParityAuditTest(unittest.TestCase):
    def test_matching_inputs_report_no_array_mismatch(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.h5"
            candidate = Path(tmpdir) / "candidate.h5"
            _write_h5(reference)
            _write_h5(candidate)

            payload = module.run_audit(
                reference=[str(reference)],
                candidate=[str(candidate)],
                match_key="source_group_index",
                sample_size=1,
                seed=123,
                skip_waveforms=False,
                progress_interval_sec=0,
            )

        self.assertEqual(payload["overlap"]["matched_reference_keys"], 1)
        self.assertEqual(payload["comparison"]["sampled_pairs"], 1)
        for stats in payload["comparison"]["array_stats"].values():
            self.assertEqual(stats["shape_mismatch"], 0)
            self.assertEqual(stats["any_value_mismatch"], 0)

    def test_value_mismatch_is_reported_for_matched_event(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.h5"
            candidate = Path(tmpdir) / "candidate.h5"
            _write_h5(reference)
            _write_h5(candidate, node_offset=0.5)

            payload = module.run_audit(
                reference=[str(reference)],
                candidate=[str(candidate)],
                match_key="source_group_index",
                sample_size=1,
                seed=123,
                skip_waveforms=True,
                progress_interval_sec=0,
            )

        node_stats = payload["comparison"]["array_stats"]["node_features"]
        self.assertEqual(node_stats["any_value_mismatch"], 1)
        self.assertAlmostEqual(node_stats["max_abs_diff"], 0.5)
        self.assertTrue(payload["comparison"]["array_mismatch_examples"])


if __name__ == "__main__":
    unittest.main()
