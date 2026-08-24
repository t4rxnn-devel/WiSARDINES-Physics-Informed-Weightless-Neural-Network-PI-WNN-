from typing import Tuple
import hashlib
import numpy as np
from config import WiSARDPhysicsConfig

class PurePhysicsInformedWiSARD:
    """A pure WiSARD RAM Discriminator Bank System with Axiomatic Address Partitioning."""
    
    def __init__(self, config: WiSARDPhysicsConfig) -> None:
        self.cfg: WiSARDPhysicsConfig = config
        
        # Allocate partitioned memory space: 2^(TUPLE_SIZE + 1) to accommodate the physics bit
        self.discriminator_banks: np.ndarray = np.zeros(
            (self.cfg.NUM_DISCRIMINATORS, self.cfg.NUM_RAMS_PER_DISCRIMINATOR, 2**(self.cfg.TUPLE_SIZE + 1)),
            dtype=np.uint8
        )
        
        self.address_mapping_table: np.ndarray = self._compile_nist_mapping()

    def _compile_nist_mapping(self) -> np.ndarray:
        indices = []
        counter = 0
        while len(indices) < self.cfg.TOTAL_INPUT_BITS:
            hasher = hashlib.sha256()
            hasher.update(f"PURE_WISARD_SEED_{counter}".encode('utf-8'))
            digest = hasher.digest()
            for i in range(0, len(digest), 4):
                val = int.from_bytes(digest[i:i+4], 'big') % self.cfg.TOTAL_INPUT_BITS
                if val not in indices and len(indices) < self.cfg.TOTAL_INPUT_BITS:
                    indices.append(val)
            counter += 1
        return np.array(indices, dtype=np.int32).reshape(self.cfg.NUM_RAMS_PER_DISCRIMINATOR, self.cfg.TUPLE_SIZE)

    def verify_kinematic_invariants(self, raw_telemetry: np.ndarray) -> np.ndarray:
        momentum: np.ndarray = raw_telemetry[:, 0]
        total_energy: np.ndarray = raw_telemetry[:, 1]
        
        momentum_sq = (momentum * self.cfg.SPEED_OF_LIGHT) ** 2
        energy_sq = total_energy ** 2
        
        computed_mass = np.sqrt(np.maximum(energy_sq - momentum_sq, 0.0))
        variance = np.abs(computed_mass - self.cfg.TARGET_INVARIANT)
        return variance <= self.cfg.ENERGY_THRESHOLD

    def _extract_physics_partitioned_addresses(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        n_samples = binary_stream.shape[0]
        bit_shifter_weights = 2 ** np.arange(self.cfg.TUPLE_SIZE - 1, -1, -1)
        
        # Vectorized generation of base addresses for all RAM nodes simultaneously
        # Shape: [n_samples, NUM_RAMS_PER_DISCRIMINATOR]
        ram_addresses = np.zeros((n_samples, self.cfg.NUM_RAMS_PER_DISCRIMINATOR), dtype=np.int32)
        for ram_idx in range(self.cfg.NUM_RAMS_PER_DISCRIMINATOR):
            target_bit_paths = self.address_mapping_table[ram_idx]
            selected_bits = binary_stream[:, target_bit_paths]
            ram_addresses[:, ram_idx] = np.dot(selected_bits, bit_shifter_weights)
            
        # Enforce Axiomatic Physics Partitioning
        # If physical -> offset is 0. If unphysical -> offset shifts address into upper partition.
        physical_valid_mask = self.verify_kinematic_invariants(raw_telemetry)
        physics_offset = np.where(physical_valid_mask[:, np.newaxis], 0, 2**self.cfg.TUPLE_SIZE)
        
        return ram_addresses + physics_offset

    def memorize(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray, discriminator_id: np.ndarray) -> None:
        """Trains the network with zero Python loops via advanced multi-dimensional matrix slicing."""
        n_samples = binary_stream.shape[0]
        ram_addresses = self._extract_physics_partitioned_addresses(binary_stream, raw_telemetry)
        
        # Create full coordinate broadcasting grids for advanced NumPy assignments
        ram_nodes_vector = np.arange(self.cfg.NUM_RAMS_PER_DISCRIMINATOR)[np.newaxis, :]
        discriminator_targets = discriminator_id[:, np.newaxis]
        
        # One-shot vectorized memory block register allocation write
        self.discriminator_banks[discriminator_targets, ram_nodes_vector, ram_addresses] = 1

    def evaluate(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        """Queries memory registers using vectorized lookups across all sample spaces."""
        n_samples = binary_stream.shape[0]
        ram_addresses = self._extract_physics_partitioned_addresses(binary_stream, raw_telemetry)
        tally_matrix = np.zeros((n_samples, self.cfg.NUM_DISCRIMINATORS), dtype=np.int32)
        
        for bank_idx in range(self.cfg.NUM_DISCRIMINATORS):
            for ram_idx in range(self.cfg.NUM_RAMS_PER_DISCRIMINATOR):
                target_slots = ram_addresses[:, ram_idx]
                tally_matrix[:, bank_idx] += self.discriminator_banks[bank_idx, ram_idx, target_slots]
                
        return tally_matrix
