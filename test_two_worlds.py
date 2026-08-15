import unittest

import numpy as np

from two_worlds import accumulate_discrimination


class TwoWorldTests(unittest.TestCase):
    def test_identical_worlds_are_exactly_dark(self):
        result = accumulate_discrimination(
            0.5,
            0.5,
            horizon=12,
            resolution=32,
            readout="complex",
        )
        self.assertEqual(float(np.max(result.d2)), 0.0)
        self.assertEqual(float(np.max(result.fraction_curve)), 0.0)

    def test_threshold_crossing_fraction_is_monotone(self):
        result = accumulate_discrimination(
            0.42,
            0.48,
            horizon=16,
            resolution=32,
            readout="complex",
        )
        self.assertTrue(np.all(np.diff(result.fraction_curve) >= -1e-15))
        self.assertTrue(np.all(result.d2 >= 0.0))

    def test_all_readouts_execute(self):
        for readout in ("complex", "magnitude", "phase", "escape"):
            with self.subTest(readout=readout):
                result = accumulate_discrimination(
                    0.40,
                    0.55,
                    horizon=6,
                    resolution=24,
                    readout=readout,
                )
                self.assertEqual(result.d2.shape, (24, 24))
                self.assertTrue(np.all(np.isfinite(result.d2)))


if __name__ == "__main__":
    unittest.main()
