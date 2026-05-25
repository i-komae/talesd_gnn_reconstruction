from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from talesd_gnn_reconstruction.constants import WAVEFORM_RISE_ANCHOR_BIN, WAVEFORM_TRACE_BINS
from talesd_gnn_reconstruction.dataset import H5GraphDataset
from talesd_gnn_reconstruction.event_graph import (
    _copy_accepted_gapped_pulses,
    _coincident_pulse_rise_anchor_bin,
    _waveform_features_for_pulse,
)


def _pulse(upper_rise: int, upper_fall: int, lower_rise: int, lower_fall: int) -> SimpleNamespace:
    return SimpleNamespace(
        upper_rise_bin=upper_rise,
        upper_fall_bin=upper_fall,
        lower_rise_bin=lower_rise,
        lower_fall_bin=lower_fall,
    )


class EventGraphWaveformTest(unittest.TestCase):
    def test_raw_waveform_window_aligns_earliest_layer_rise_bin(self) -> None:
        pulse = _pulse(upper_rise=40, upper_fall=52, lower_rise=44, lower_fall=56)
        anchor_bin = _coincident_pulse_rise_anchor_bin(pulse)
        self.assertEqual(anchor_bin, 40)

        waveform = np.arange(160, dtype=np.float32)
        features = _waveform_features_for_pulse(
            upper_wf=waveform,
            lower_wf=waveform,
            upper_ped=0.0,
            lower_ped=0.0,
            upper_mev2cnt=0.5,
            lower_mev2cnt=0.5,
            pulse=pulse,
            accepted_pulses=[pulse],
        )

        self.assertEqual(features.shape, (4, WAVEFORM_TRACE_BINS))
        self.assertEqual(features[0, WAVEFORM_RISE_ANCHOR_BIN], waveform[anchor_bin])
        self.assertEqual(features[1, WAVEFORM_RISE_ANCHOR_BIN], waveform[anchor_bin])
        self.assertEqual(features[1, WAVEFORM_RISE_ANCHOR_BIN + 4], waveform[pulse.lower_rise_bin])

    def test_accepted_gapped_waveform_preserves_pulse_time_gaps(self) -> None:
        pulses = [
            _pulse(upper_rise=10, upper_fall=12, lower_rise=11, lower_fall=13),
            _pulse(upper_rise=20, upper_fall=22, lower_rise=23, lower_fall=25),
        ]
        waveform = np.zeros(64, dtype=np.float32)
        waveform[10:13] = 1.0
        waveform[20:23] = 2.0

        accepted = _copy_accepted_gapped_pulses(waveform, pulses, channel="upper", length=32)

        np.testing.assert_array_equal(accepted[0:3], np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(accepted[3:10], np.zeros(7, dtype=np.float32))
        np.testing.assert_array_equal(accepted[10:13], np.full(3, 2.0, dtype=np.float32))

    def test_old_compact_waveform_hdf5_is_rejected(self) -> None:
        old_columns = {
            "waveform_features": [
                "upper_raw_window_vem",
                "lower_raw_window_vem",
                "upper_accepted_compact_vem",
                "lower_accepted_compact_vem",
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_waveform_schema.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["columns_json"] = json.dumps(old_columns)
                handle.create_group("events")

            with self.assertRaisesRegex(ValueError, "old compact accepted-pulse waveforms"):
                H5GraphDataset(path)

    def test_hdf5_handles_are_lru_limited(self) -> None:
        def write_graph(path: Path) -> None:
            with h5py.File(path, "w") as handle:
                event = handle.create_group("events").create_group("00000000")
                event.create_dataset("node_features", data=np.zeros((1, 1), dtype=np.float32))
                event.create_dataset("edge_index", data=np.zeros((2, 0), dtype=np.int64))
                event.create_dataset("edge_features", data=np.zeros((0, 1), dtype=np.float32))
                event.create_dataset("pulse_features", data=np.zeros((0, 1), dtype=np.float32))
                event.create_dataset("waveform_features", data=np.zeros((1, 0, 0), dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.h5"
            second = Path(tmpdir) / "second.h5"
            write_graph(first)
            write_graph(second)

            dataset = H5GraphDataset(
                [first, second],
                max_open_files=1,
                load_attrs=False,
                load_node_positions=False,
            )
            dataset[0]
            first_handle = dataset._handles[0]
            self.assertEqual(list(dataset._handles), [0])

            dataset[1]
            self.assertEqual(list(dataset._handles), [1])
            self.assertFalse(bool(first_handle.id.valid))
            dataset.close()


if __name__ == "__main__":
    unittest.main()
