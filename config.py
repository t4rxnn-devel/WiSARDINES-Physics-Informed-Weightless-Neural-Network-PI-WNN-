from typing import Final

class WiSARDPhysicsConfig:
    """Defines the hardware constraints and physical constants for the WiSARD system."""
    REST_MASS: Final[float] = 0.511    
    SPEED_OF_LIGHT: Final[float] = 1.0 
    TARGET_INVARIANT: Final[float] = 2.5 

    # WiSARD Structural Configurations
    ENERGY_THRESHOLD: Final[float] = 0.25 
    BIT_DEPTH: Final[int] = 32            
    TUPLE_SIZE: Final[int] = 4            
    NUM_DISCRIMINATORS: Final[int] = 3    

    # Architectural Layout Derivations
    NUM_FEATURES: Final[int] = 2          
    TOTAL_INPUT_BITS: Final[int] = NUM_FEATURES * BIT_DEPTH
    NUM_RAMS_PER_DISCRIMINATOR: Final[int] = TOTAL_INPUT_BITS // TUPLE_SIZE

    def __init__(self) -> None:
        if self.TOTAL_INPUT_BITS % self.TUPLE_SIZE != 0:
            raise ValueError(
                f"WiSARD Structure Alignment Fault: Total bit length ({self.TOTAL_INPUT_BITS}) "
                f"must be perfectly divisible by target tuple_size ({self.TUPLE_SIZE})."
            )
