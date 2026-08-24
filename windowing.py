import numpy as np


def make_windows(
    sequence: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Create overlapping windows from an event sequence.

    The input is ``(events, features)`` and the result is
    ``(windows, window_size, features)``. Incomplete trailing windows are
    discarded rather than padded with synthetic telemetry.
    """
    sequence = np.asarray(sequence)
    if sequence.ndim != 2:
        raise ValueError(f"sequence must be 2D, received {sequence.shape}")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if window_size > sequence.shape[0]:
        return np.empty((0, window_size, sequence.shape[1]), dtype=sequence.dtype)
    starts = np.arange(0, sequence.shape[0] - window_size + 1, stride)
    return np.stack([sequence[start:start + window_size] for start in starts])