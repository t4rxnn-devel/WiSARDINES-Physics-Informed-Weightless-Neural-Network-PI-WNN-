from typing import Tuple
import hashlib
import numpy as np
from config import WiSARDPhysicsConfig

class PurePhysicsInformedWiSARD:
    def __init__(self, config: WiSARDPhysicsConfig) -> None:
        self.cfg = config

        # 3 distinct partition layers: 0=Mass A, 1=Mass B, 2=Noise Background
        total_slots_per_ram = (2 ** self.cfg.TUPLE_SIZE) * 3
        storage_slots_per_ram = {
            "dense": total_slots_per_ram,
            "hashed": self.cfg.HASH_BUCKETS,
            "sparse": 0,
        }[self.cfg.STORAGE_MODE]
        self.bytes_per_ram = storage_slots_per_ram // 8
        if storage_slots_per_ram % 8 != 0:
            self.bytes_per_ram += 1

        self.discriminator_banks = np.zeros(
            (self.cfg.NUM_DISCRIMINATORS, self.cfg.NUM_RAMS_PER_DISCRIMINATOR, self.bytes_per_ram),
            dtype=np.uint8
        )
        # Sparse counters provide soft likelihoods without allocating a second dense RAM.
        self.ram_counts: dict[tuple[int, int, int], int] = {}
        self.ram_totals = np.zeros(
            (self.cfg.NUM_DISCRIMINATORS, self.cfg.NUM_RAMS_PER_DISCRIMINATOR),
            dtype=np.int64,
        )
        self.discriminator_counts = np.zeros(self.cfg.NUM_DISCRIMINATORS, dtype=np.int64)
        self.sparse_ram_bits: set[tuple[int, int, int]] = set()
        self.address_mapping_table = self._compile_feature_isolated_mapping()

    def _compile_feature_isolated_mapping(self) -> np.ndarray:
        """Enforces feature isolation to prevent cross-feature bit collisions."""
        rams_per_feature = self.cfg.NUM_RAMS_PER_DISCRIMINATOR // self.cfg.NUM_FEATURES
        mapping = np.zeros((self.cfg.NUM_RAMS_PER_DISCRIMINATOR, self.cfg.TUPLE_SIZE), dtype=np.int32)

        hasher = hashlib.sha256()
        hasher.update(b"HARDENED_WISARD_SEED_PROD_10")
        seed_uint32 = np.frombuffer(hasher.digest()[:4], dtype=np.uint32)
        rng = np.random.default_rng(seed_uint32)

        for f_idx in range(self.cfg.NUM_FEATURES):
            start_bit = f_idx * self.cfg.BIT_DEPTH
            feature_bits = np.arange(start_bit, start_bit + self.cfg.BIT_DEPTH)

            for r_idx in range(rams_per_feature):
                global_ram_idx = f_idx * rams_per_feature + r_idx
                mapping[global_ram_idx] = rng.choice(feature_bits, size=self.cfg.TUPLE_SIZE, replace=False)

        return mapping

    @property
    def memory_bytes(self) -> int:
        """Report allocated hard-memory bytes for the selected backend."""
        if self.cfg.STORAGE_MODE == "sparse":
            return len(self.sparse_ram_bits) * 24
        return int(self.discriminator_banks.nbytes)

    @property
    def occupied_bits(self) -> int:
        if self.cfg.STORAGE_MODE == "sparse":
            return len(self.sparse_ram_bits)
        return int(np.unpackbits(self.discriminator_banks).sum())

    def _stored_address(self, discriminator: int, ram: int, address: int) -> int:
        if self.cfg.STORAGE_MODE != "hashed":
            return address
        mixed = int(address) ^ (int(ram) * 0x9E3779B1) ^ (int(discriminator) * 0x85EBCA77)
        return (mixed ^ (mixed >> 16)) % self.cfg.HASH_BUCKETS

    def _validate_inputs(
        self,
        binary_stream: np.ndarray,
        raw_telemetry: np.ndarray,
    ) -> None:
        if binary_stream.ndim != 2 or binary_stream.shape[1] != self.cfg.TOTAL_INPUT_BITS:
            raise ValueError(
                f"binary_stream must have shape (n, {self.cfg.TOTAL_INPUT_BITS}), "
                f"received {binary_stream.shape}"
            )
        if raw_telemetry.ndim != 2 or raw_telemetry.shape != (
            binary_stream.shape[0],
            self.cfg.BASE_FEATURES,
        ):
            raise ValueError(
                "raw_telemetry must have one row per sample and "
                f"{self.cfg.BASE_FEATURES} features, received {raw_telemetry.shape}"
            )
        if not np.issubdtype(binary_stream.dtype, np.number):
            raise TypeError("binary_stream must contain numeric bit values")
        if np.any((binary_stream != 0) & (binary_stream != 1)):
            raise ValueError("binary_stream may contain only 0 and 1")
        if not np.isfinite(raw_telemetry).all():
            raise ValueError("raw_telemetry must contain only finite values")

    def _validate_discriminator_ids(self, discriminator_id: np.ndarray, n_samples: int) -> None:
        if discriminator_id.ndim != 1 or discriminator_id.shape[0] != n_samples:
            raise ValueError(
                f"discriminator_id must have shape ({n_samples},), received {discriminator_id.shape}"
            )
        if not np.issubdtype(discriminator_id.dtype, np.integer):
            raise TypeError("discriminator_id must contain integers")
        if np.any((discriminator_id < 0) | (discriminator_id >= self.cfg.NUM_DISCRIMINATORS)):
            raise ValueError(
                f"discriminator_id values must be in [0, {self.cfg.NUM_DISCRIMINATORS})"
            )

    def calculate_invariant_mass(self, raw_telemetry: np.ndarray) -> np.ndarray:
        p1x, p1y, p2x, p2y = raw_telemetry[:, 0], raw_telemetry[:, 1], raw_telemetry[:, 2], raw_telemetry[:, 3]
        E1 = np.sqrt(p1x**2 + p1y**2)
        E2 = np.sqrt(p2x**2 + p2y**2)
        E_tot = E1 + E2
        px_tot = p1x + p2x
        py_tot = p1y + p2y
        invariant_mass_sq = E_tot**2 - px_tot**2 - py_tot**2
        return np.sqrt(np.maximum(invariant_mass_sq, 0.0))

    def physical_validity(self, raw_telemetry: np.ndarray) -> np.ndarray:
        """Return the hard admissibility mask used before inference tie-breaking."""
        telemetry = np.asarray(raw_telemetry)
        if telemetry.ndim != 2 or telemetry.shape[1] != self.cfg.BASE_FEATURES:
            raise ValueError("raw_telemetry must have shape (n, 4)")
        return np.isfinite(telemetry).all(axis=1) & np.isfinite(
            self.calculate_invariant_mass(telemetry)
        )

    def _extract_physics_partitioned_addresses(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        n_samples = binary_stream.shape[0]
        bit_shifter_weights = 2 ** np.arange(self.cfg.TUPLE_SIZE - 1, -1, -1)

        ram_addresses = np.zeros((n_samples, self.cfg.NUM_RAMS_PER_DISCRIMINATOR), dtype=np.int32)
        for ram_idx in range(self.cfg.NUM_RAMS_PER_DISCRIMINATOR):
            target_bit_paths = self.address_mapping_table[ram_idx]
            selected_bits = binary_stream[:, target_bit_paths]
            ram_addresses[:, ram_idx] = np.dot(selected_bits, bit_shifter_weights)

        partition_tier = np.zeros(n_samples, dtype=np.int32)
        if self.cfg.PHYSICAL_PARTITIONING:
            computed_masses = self.calculate_invariant_mass(raw_telemetry)
            mask_mass_a = np.abs(computed_masses - self.cfg.TARGET_MASS_A) <= self.cfg.ENERGY_THRESHOLD
            mask_mass_b = np.abs(computed_masses - self.cfg.TARGET_MASS_B) <= self.cfg.ENERGY_THRESHOLD
            partition_tier[mask_mass_b] = 1
            partition_tier[~mask_mass_a & ~mask_mass_b] = 2

        return ram_addresses + partition_tier[:, np.newaxis] * (2 ** self.cfg.TUPLE_SIZE)

    def memorize(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray, discriminator_id: np.ndarray) -> None:
        binary_stream = np.asarray(binary_stream)
        raw_telemetry = np.asarray(raw_telemetry)
        discriminator_id = np.asarray(discriminator_id)
        self._validate_inputs(binary_stream, raw_telemetry)
        n_samples = binary_stream.shape[0]
        self._validate_discriminator_ids(discriminator_id, n_samples)
        ram_addresses = self._extract_physics_partitioned_addresses(binary_stream, raw_telemetry)

        stored_addresses = np.vectorize(self._stored_address)(
            discriminator_id[:, np.newaxis],
            np.arange(self.cfg.NUM_RAMS_PER_DISCRIMINATOR)[np.newaxis, :],
            ram_addresses,
        ).astype(np.int32)
        if self.cfg.STORAGE_MODE == "sparse":
            for sample_index, discriminator in enumerate(discriminator_id):
                for ram_index, address in enumerate(ram_addresses[sample_index]):
                    self.sparse_ram_bits.add((int(discriminator), ram_index, int(address)))
        else:
            byte_indices = stored_addresses // 8
            bit_offsets = stored_addresses % 8
            bit_masks = (1 << bit_offsets).astype(np.uint8)

            targets_broad = np.broadcast_to(discriminator_id[:, np.newaxis], (n_samples, self.cfg.NUM_RAMS_PER_DISCRIMINATOR))
            rams_broad = np.broadcast_to(np.arange(self.cfg.NUM_RAMS_PER_DISCRIMINATOR)[np.newaxis, :], (n_samples, self.cfg.NUM_RAMS_PER_DISCRIMINATOR))

            np.bitwise_or.at(self.discriminator_banks, (targets_broad, rams_broad, byte_indices), bit_masks)
        for sample_index, discriminator in enumerate(discriminator_id):
            self.discriminator_counts[discriminator] += 1
            for ram_index, address in enumerate(ram_addresses[sample_index]):
                key = (int(discriminator), ram_index, int(address))
                self.ram_counts[key] = self.ram_counts.get(key, 0) + 1
                self.ram_totals[discriminator, ram_index] += 1

    def evaluate(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        binary_stream = np.asarray(binary_stream)
        raw_telemetry = np.asarray(raw_telemetry)
        self._validate_inputs(binary_stream, raw_telemetry)
        n_samples = binary_stream.shape[0]
        ram_addresses = self._extract_physics_partitioned_addresses(binary_stream, raw_telemetry)

        tally_matrix = np.zeros((n_samples, self.cfg.NUM_DISCRIMINATORS), dtype=np.int32)
        if self.cfg.STORAGE_MODE == "sparse":
            for sample_index in range(n_samples):
                for bank_idx in range(self.cfg.NUM_DISCRIMINATORS):
                    for ram_idx, address in enumerate(ram_addresses[sample_index]):
                        if (bank_idx, ram_idx, int(address)) in self.sparse_ram_bits:
                            tally_matrix[sample_index, bank_idx] += 1
            return tally_matrix

        stored_addresses = np.vectorize(self._stored_address)(
            np.arange(self.cfg.NUM_DISCRIMINATORS)[:, np.newaxis, np.newaxis],
            np.arange(self.cfg.NUM_RAMS_PER_DISCRIMINATOR)[np.newaxis, np.newaxis, :],
            ram_addresses[np.newaxis, :, :],
        )

        for bank_idx in range(self.cfg.NUM_DISCRIMINATORS):
            for ram_idx in range(self.cfg.NUM_RAMS_PER_DISCRIMINATOR):
                b_idx = stored_addresses[bank_idx, :, ram_idx] // 8
                mask = (1 << (stored_addresses[bank_idx, :, ram_idx] % 8)).astype(np.uint8)

                target_bytes = self.discriminator_banks[bank_idx, ram_idx, b_idx]
                hit_mask = (target_bytes & mask) > 0
                tally_matrix[:, bank_idx] += hit_mask.astype(np.int32)

        return tally_matrix

    def soft_scores(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        """Score addresses with sparse per-subspace frequencies and Laplace smoothing.

        Unlike hard RAM hits, this returns useful non-zero evidence for an unseen
        address by using the observed distribution for its RAM and physical tier.
        """
        binary_stream = np.asarray(binary_stream)
        raw_telemetry = np.asarray(raw_telemetry)
        self._validate_inputs(binary_stream, raw_telemetry)
        addresses = self._extract_physics_partitioned_addresses(binary_stream, raw_telemetry)
        slot_count = 3 * (2 ** self.cfg.TUPLE_SIZE)
        scores = np.zeros((binary_stream.shape[0], self.cfg.NUM_DISCRIMINATORS), dtype=np.float64)
        alpha = self.cfg.SOFT_PRIOR_ALPHA
        for sample_index, sample_addresses in enumerate(addresses):
            for discriminator in range(self.cfg.NUM_DISCRIMINATORS):
                for ram_index, address in enumerate(sample_addresses):
                    count = self.ram_counts.get((discriminator, ram_index, int(address)), 0)
                    denominator = self.discriminator_counts[discriminator] + alpha * slot_count
                    scores[sample_index, discriminator] += np.log((count + alpha) / denominator)
        return scores

    def bleach(self, scores: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        """Apply hard physical masking, then stable soft tie-breaking."""
        scores = np.asarray(scores, dtype=np.float64)
        telemetry = np.asarray(raw_telemetry)
        if scores.ndim != 2 or scores.shape[1] != self.cfg.NUM_DISCRIMINATORS:
            raise ValueError("scores have an invalid discriminator dimension")
        valid = self.physical_validity(telemetry)
        if scores.shape[0] != valid.shape[0]:
            raise ValueError("scores and raw_telemetry must have equal sample counts")
        filtered = scores.copy()
        if self.cfg.HARD_PHYSICS_CONSTRAINTS:
            filtered[~valid] = -np.inf
        return np.argmax(filtered, axis=1).astype(np.int32)

    def predict(self, binary_stream: np.ndarray, raw_telemetry: np.ndarray) -> np.ndarray:
        """Predict using soft generalization and hard physical bleaching."""
        binary_stream = np.asarray(binary_stream)
        raw_telemetry = np.asarray(raw_telemetry)
        self._validate_inputs(
            binary_stream,
            np.nan_to_num(raw_telemetry, nan=0.0, posinf=0.0, neginf=0.0),
        )
        valid = self.physical_validity(raw_telemetry)
        predictions = np.full(binary_stream.shape[0], -1, dtype=np.int32)
        if not self.cfg.HARD_PHYSICS_CONSTRAINTS:
            valid[:] = True
        if np.any(valid):
            scores = self.soft_scores(binary_stream[valid], raw_telemetry[valid])
            predictions[valid] = self.bleach(scores, raw_telemetry[valid])
        return predictions

    def reset(self) -> None:
        """Clear hard RAM state and sparse soft counters."""
        self.discriminator_banks.fill(0)
        self.ram_counts.clear()
        self.ram_totals.fill(0)
        self.discriminator_counts.fill(0)
        self.sparse_ram_bits.clear()