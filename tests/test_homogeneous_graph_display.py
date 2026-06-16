import unittest

import numpy as np

from scripts.make_homogeneous_graph_event_display import _displayed_pulse_mask, _used_time_limits


class HomogeneousGraphDisplayTests(unittest.TestCase):
    def test_time_limits_ignore_rejected_pulses(self) -> None:
        arrival = np.asarray([0.0, 13.0, 18.0, 24.0], dtype=np.float64)
        used = np.asarray([False, True, True, False])

        self.assertEqual(_used_time_limits(arrival, used), (13.0, 18.0))

    def test_time_limits_expand_single_used_pulse(self) -> None:
        arrival = np.asarray([0.0, 12.5], dtype=np.float64)
        used = np.asarray([False, True])

        self.assertEqual(_used_time_limits(arrival, used), (12.5, 13.5))

    def test_time_limits_default_without_used_pulses(self) -> None:
        arrival = np.asarray([0.0, 12.5], dtype=np.float64)
        used = np.asarray([False, False])

        self.assertEqual(_used_time_limits(arrival, used), (0.0, 1.0))

    def test_kept_only_display_mask_drops_rejected_pulses(self) -> None:
        used = np.asarray([True, False, True, False])

        np.testing.assert_array_equal(
            _displayed_pulse_mask(used, drop_rejected_pulses=True),
            np.asarray([True, False, True, False]),
        )
        np.testing.assert_array_equal(
            _displayed_pulse_mask(used, drop_rejected_pulses=False),
            np.asarray([True, True, True, True]),
        )


if __name__ == "__main__":
    unittest.main()
