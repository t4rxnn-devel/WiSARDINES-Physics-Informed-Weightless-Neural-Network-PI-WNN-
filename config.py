from typing import Final

class WiSARDPhysicsConfig:
    BASE_FEATURES: Final[int] = 4

    def __init__(
        self,
        *,
        target_mass_a: float = 5.0,
        target_mass_b: float = 2.0,
        energy_threshold: float = 0.30,
        bit_depth: int = 32,
        tuple_size: int = 8,
        num_discriminators: int = 3,
        window_size: int = 1,
        window_stride: int = 1,
        quantizer_min: float = -15.0,
        quantizer_max: float = 15.0,
        soft_prior_alpha: float = 1.0,
        hard_physics_constraints: bool = True,
        storage_mode: str = "dense",
        hash_buckets: int = 4096,
        physical_partitioning: bool = True,
        input_features: int = 4,
    ) -> None:
        self.TARGET_MASS_A = float(target_mass_a)
        self.TARGET_MASS_B = float(target_mass_b)
        self.ENERGY_THRESHOLD = float(energy_threshold)
        self.BIT_DEPTH = int(bit_depth)
        self.TUPLE_SIZE = int(tuple_size)
        self.NUM_DISCRIMINATORS = int(num_discriminators)
        self.WINDOW_SIZE = int(window_size)
        self.WINDOW_STRIDE = int(window_stride)
        self.QUANTIZER_MIN = float(quantizer_min)
        self.QUANTIZER_MAX = float(quantizer_max)
        self.SOFT_PRIOR_ALPHA = float(soft_prior_alpha)
        self.HARD_PHYSICS_CONSTRAINTS = bool(hard_physics_constraints)
        self.STORAGE_MODE = storage_mode.lower()
        self.HASH_BUCKETS = int(hash_buckets)
        self.PHYSICAL_PARTITIONING = bool(physical_partitioning)
        self.INPUT_FEATURES = int(input_features)
        if self.BIT_DEPTH <= 0:
            raise ValueError("BIT_DEPTH and NUM_FEATURES must be positive")
        if self.TUPLE_SIZE <= 0 or self.TUPLE_SIZE > self.BIT_DEPTH:
            raise ValueError("TUPLE_SIZE must be between 1 and BIT_DEPTH")
        if self.WINDOW_SIZE <= 0 or self.WINDOW_STRIDE <= 0:
            raise ValueError("WINDOW_SIZE and WINDOW_STRIDE must be positive")
        if self.INPUT_FEATURES < self.BASE_FEATURES:
            raise ValueError("INPUT_FEATURES must include the four physical anchor features")
        self.NUM_FEATURES = self.INPUT_FEATURES * self.WINDOW_SIZE
        self.TOTAL_INPUT_BITS = self.NUM_FEATURES * self.BIT_DEPTH
        self.NUM_RAMS_PER_DISCRIMINATOR = self.TOTAL_INPUT_BITS // self.TUPLE_SIZE
        if self.TOTAL_INPUT_BITS % self.TUPLE_SIZE != 0:
            raise ValueError("TOTAL_INPUT_BITS must be divisible by TUPLE_SIZE")
        if self.NUM_RAMS_PER_DISCRIMINATOR % self.NUM_FEATURES != 0:
            raise ValueError("RAM count must be divisible by NUM_FEATURES")
        if self.NUM_DISCRIMINATORS <= 0:
            raise ValueError("NUM_DISCRIMINATORS must be positive")
        if not self.QUANTIZER_MIN < self.QUANTIZER_MAX:
            raise ValueError("QUANTIZER_MIN must be less than QUANTIZER_MAX")
        if self.ENERGY_THRESHOLD < 0:
            raise ValueError("ENERGY_THRESHOLD cannot be negative")
        if self.SOFT_PRIOR_ALPHA <= 0:
            raise ValueError("SOFT_PRIOR_ALPHA must be positive")
        if self.STORAGE_MODE not in {"dense", "sparse", "hashed"}:
            raise ValueError("STORAGE_MODE must be dense, sparse, or hashed")
        if self.HASH_BUCKETS <= 0:
            raise ValueError("HASH_BUCKETS must be positive")
