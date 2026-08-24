import unittest

import numpy as np

from windowing import make_windows


class WindowingTests(unittest.TestCase):
    def test_overlapping_windows_preserve_order(self) -> None:
        sequence = np.arange(12).reshape(6, 2)
        windows = make_windows(sequence, window_size=3, stride=2)
        np.testing.assert_array_equal(windows[1], sequence[2:5])
        self.assertEqual(windows.shape, (2, 3, 2))

    def test_incomplete_window_is_dropped(self) -> None:
        sequence = np.zeros((2, 4))
        self.assertEqual(make_windows(sequence, 3).shape, (0, 3, 4))

    def test_invalid_window_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_windows(np.zeros((4, 2)), 0)
        with self.assertRaises(ValueError):
            make_windows(np.zeros((4, 2)), 2, 0)


if __name__ == "__main__":
    unittest.main()
