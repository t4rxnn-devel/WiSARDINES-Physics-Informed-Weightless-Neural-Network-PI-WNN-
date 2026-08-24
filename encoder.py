import numpy as np

class ThermometerQuantizer:
    """Converts continuous telemetry inputs into linear, position-dependent bit arrays."""
    
    def __init__(self, min_val: float, max_val: float, bit_depth: int) -> None:
        if min_val >= max_val:
            raise ValueError("Minimum bounds cannot exceed or match maximum bounds.")
        if bit_depth <= 0:
            raise ValueError("Bit depth allocation must be a positive non-zero integer.")
            
        self.min_val: float = min_val
        self.max_val: float = max_val
        self.bit_depth: int = bit_depth
        self.thresholds: np.ndarray = np.linspace(min_val, max_val, bit_depth + 1)[1:-1]

    def process(self, matrix: np.ndarray) -> np.ndarray:
        """Transforms a 2D continuous array directly into a concatenated binary stream matrix."""
        if matrix.ndim != 2:
            raise ValueError(f"Input must be a 2D matrix, received shape: {matrix.shape}")

        n_samples, n_features = matrix.shape
        clipped: np.ndarray = np.clip(matrix, self.min_val, self.max_val)
        
        # Parallel element evaluation via array broadcasting
        expanded_clipped: np.ndarray = np.expand_dims(clipped, axis=-1)
        binary_activations: np.ndarray = expanded_clipped > self.thresholds
        
        output_bits: np.ndarray = np.zeros((n_samples, n_features, self.bit_depth), dtype=np.uint8)
        output_bits[:, :, :len(self.thresholds)] = binary_activations.astype(np.uint8)
        
        # Lock anchor bit settings
        output_bits[:, :, -1] = (clipped >= self.min_val).astype(np.uint8)
        
        return output_bits.reshape(n_samples, n_features * self.bit_depth)
