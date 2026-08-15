import unittest

import numpy as np

from commitment_dissolve import commitment_alpha, dissolve_frames
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
        self.assertTrue(np.isnan(result.commitment_time).all())

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
        finite = result.commitment_time[np.isfinite(result.commitment_time)]
        self.assertTrue(np.all((finite >= 1) & (finite <= 16)))

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
                self.assertEqual(result.commitment_time.shape, (24, 24))
                self.assertTrue(np.all(np.isfinite(result.d2)))

    def test_commitment_alpha_respects_unresolved_pixels(self):
        tstar = np.array([[2.0, 5.0], [np.nan, 9.0]], np.float32)
        early = commitment_alpha(tstar, 1.0, softness=0.25)
        late = commitment_alpha(tstar, 10.0, softness=0.25)
        self.assertLess(float(early[0, 1]), 0.01)
        self.assertGreater(float(late[0, 0]), 0.99)
        self.assertEqual(float(late[1, 0]), 0.0)

    def test_dissolve_finishes_on_b_with_tail(self):
        a = np.zeros((12, 16, 3), np.uint8)
        b = np.full((12, 16, 3), 255, np.uint8)
        tstar = np.full((6, 8), np.nan, np.float32)
        tstar[:, :4] = 3.0
        frames = dissolve_frames(
            a, b, tstar, horizon=8, frame_count=8,
            softness=0.25, finish_tail=0.25,
        )
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0].shape, a.shape)
        self.assertTrue(np.all(frames[-1] == 255))


if __name__ == "__main__":
    unittest.main()
