from typing import Tuple
import hashlib
import numpy as np
from config import WiSARDPhysicsConfig

class PurePhysicsInformedWiSARD:
    """A pure WiSARD RAM Discriminator Bank System with hardware-blind physics addressing."""
    
    def __init__(self, config: WiSARDPhysicsConfig) -> None:
        self.cfg: WiSARDPhysicsConfig = config
        
        # Allocate pure WiSARD memory allocation registers
        # Structure Layout: [Independent Discriminator Banks, RAM Nodes per Bank, 2^N Memory Slots]
        self.discriminator_banks: np.ndarray = np.zeros(
            (self.cfg.NUM_DISCRIMINATORS, self.cfg.NUM_RAMS_PER_DISCRIMINATOR, 2**self.cfg.TUPLE_SIZE),
            dtype=np.uint8
        )
        
        # NIST SP 800-90A: Generate uniform address mapping paths via cryptographic SHA-256 expansion
        self.address_mapping_table: np.ndarray = self._compile_nist_mapping()

    def _compile_nist_mapping(self) -> np.ndarray:
        """Compiles standard, unbiased address routing matrices via deterministic entropy streams."""
        entropy_pool = bytearray()
        counter = 0
        needed_bytes = self.cfg.TOTAL_INPUT_BITS * 4
        
        while len(entropy_pool) < needed_bytes:
            hasher = hashlib.sha256()
            hasher.update(f"PURE_WISARD_SEED_{counter}".encode('utf-8'))
            entropy_pool.extend(hasher.digest())
            counter += 1
            
        raw_ints = np.frombuffer(entropy_pool[:needed_bytes], dtype=np.uint32)
        shuffled_indices = np.argsort(raw_ints)[:self.cfg.TOTAL_INPUT_BITS]
        return shuffled_indices.reshape(self.cfg.NUM_RAMS_PER_DISCRIMINATOR, self.cfg.TUPLE_SIZE)

    def verify_kinematic_invariants(self, raw_telemetry: np.ndarray) -> np.ndarray:
        """Vectorized execution check testing conformity against physical boundary conservation laws."""
        momentum: np.ndarray = raw_telemetry[:, 0]
        total_energy: np.ndarray = raw_telemetry[:, 1]
        
        # Relativistic invariant equation: mass^2 = Energy^2 - (momentum * c)^2
        momentum_sq = (momentum * self.cfg.SPEED_OF_LIGHT) ** 2
        energy_sq = total_energy ** 2
        
        computed_mass = np.sqrt(np.maximum(energy_sq - momentum_sq, 0.0))
        variance = np.abs(computed_mass - self.cfg.TARGET_INVARIANT)
        
        # Binary mask tracking samples that respect the physical laws
        return variance <= self.cfg.ENERGY_THRESHOLD

    def _extract_physics_gated_addresses(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        """
        Maps binary bit selections directly to memory addresses.
        Enforces a Hardware Blind (forcing target location to index 0) for any unphysical states.
        """
        n_samples = binary_stream.shape
        ram_addresses = np.zeros((n_samples, self.cfg.NUM_RAMS_PER_DISCRIMINATOR), dtype=np.int32)
        
        # Identify samples that satisfy the physics constraints
        physical_valid_mask = self.verify_kinematic_invariants(raw_telemetry)
        valid_indices = np.where(physical_valid_mask)
        
        if valid_indices.size == 0:
            return ram_addresses  # All addresses remain safely unmapped at index 0

        valid_binary_segments = binary_stream[valid_indices]
        bit_shifter_weights = 2 ** np.arange(self.cfg.TUPLE_SIZE - 1, -1, -1)

        # Map binary representations directly to designated RAM address integers
        for ram_idx in range(self.cfg.NUM_RAMS_PER_DISCRIMINATOR):
            target_bit_paths = self.address_mapping_table[ram_idx]
            selected_bits = valid_binary_segments[:, target_bit_paths]
            computed_addresses = np.dot(selected_bits, bit_shifter_weights)
            ram_addresses[valid_indices, ram_idx] = computed_addresses

        return ram_addresses

    def memorize(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray, discriminator_id: np.ndarray) -> None:
        """Writes binary patterns into specific discriminator banks."""
        assert binary_stream.shape == raw_telemetry.shape, "Input alignment sizing mismatch."
        
        physical_valid_mask = self.verify_kinematic_invariants(raw_telemetry)
        ram_addresses = self._extract_physics_gated_addresses(binary_stream, raw_telemetry)
        
        # Set bit tags to 1 inside the corresponding discriminator address configurations
        for idx in range(binary_stream.shape):
            if physical_valid_mask[idx]:
                bank_target = int(discriminator_id[idx])
                for ram_idx in range(self.cfg.NUM_RAMS_PER_DISCRIMINATOR):
                    addr_slot = ram_addresses[idx, ram_idx]
                    # Address 0 is reserved as the dead-zone for unphysical data
                    if addr_slot != 0:
                        self.discriminator_banks[bank_target, ram_idx, addr_slot] = 1

    def evaluate(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        """Queries all memory registers in parallel and computes total tally outputs."""
        n_samples = binary_stream.shape
        ram_addresses = self._extract_physics_gated_addresses(binary_stream, raw_telemetry)
        
        # Accumulate memory lookup hits across all discriminators
        tally_matrix = np.zeros((n_samples, self.cfg.NUM_DISCRIMINATORS), dtype=np.int32)
        
        for bank_idx in range(self.cfg.NUM_DISCRIMINATORS):
            for ram_idx in range(self.cfg.NUM_RAMS_PER_DISCRIMINATOR):
                target_slots = ram_addresses[:, ram_idx]
                # Vectorized lookup across memory spaces
                tally_matrix[:, bank_idx] += self.discriminator_banks[bank_idx, ram_idx, target_slots]
                
        return tally_matrix
