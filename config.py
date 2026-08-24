from typing import Final

class WiSARDPhysicsConfig:
    """Defines the strict hardware constraints and physical constants for the WiSARD system."""
    # Particle Trajectory Constants (Relativistic Invariant System)
    REST_MASS: Final[float] = 0.511    # MeV/c^2 (Electron Rest Mass equivalent)
    SPEED_OF_LIGHT: Final[float] = 1.0 # Normalized c (Natural Units)
    TARGET_INVARIANT: Final[float] = 2.5 # Targeted system mass invariant boundary shell (MeV)

    # WiSARD Structural Configurations
    ENERGY_THRESHOLD: Final[float] = 0.25 # Physical variance allowance window (Epsilon)
    BIT_DEPTH: Final[int] = 32            # Resolution depth for thermometer stream encoding
    TUPLE_SIZE: Final[int] = 4            # N-bits per RAM node address slot
    NUM_DISCRIMINATORS: Final[int] = 3    # Total independent target memory banks

    # Architectural Layout Derivations
    NUM_FEATURES: Final[int] = 2          # Independent tracking signals: [Momentum (p), Total Energy (E)]
    TOTAL_INPUT_BITS: Final[int] = NUM_FEATURES * BIT_DEPTH
    NUM_RAMS_PER_DISCRIMINATOR: Final[int] = TOTAL_INPUT_BITS // TUPLE_SIZE

    def __init__(self) -> None:
        """Validates structural bit alignment for clean RAM node slicing."""
        if self.TOTAL_INPUT_BITS % self.TUPLE_SIZE != 0:
            raise ValueError(
                f"WiSARD Structure Alignment Fault: Total bit length ({self.TOTAL_INPUT_BITS}) "
                f"must be perfectly divisible by target tuple_size ({self.TUPLE_SIZE})."
            )
