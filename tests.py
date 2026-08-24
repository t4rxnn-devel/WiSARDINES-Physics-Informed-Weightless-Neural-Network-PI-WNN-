import unittest
from typing import Tuple
import numpy as np
import scipy.special as spc
from config import WiSARDPhysicsConfig
from encoder import ThermometerQuantizer
from wisard_engine import PurePhysicsInformedWiSARD
from physics_simulators import EnergyConstrainedDoublePendulum, IsingTransitionSimulator

class NISTStatisticalSuite:
    """Small statistical smoke suite for the engine's populated RAM state."""

    def __init__(self, engine: PurePhysicsInformedWiSARD) -> None:
        self.engine = engine
        self.bit_stream = self._extract_engine_bitstream()

    def _extract_engine_bitstream(self) -> np.ndarray:
        """Extract the bits actually stored in the engine's RAM banks."""
        return np.unpackbits(self.engine.discriminator_banks).astype(np.int8)

    def run_monobit_frequency_test(self) -> Tuple[float, bool]:
        n = len(self.bit_stream)
        if n == 0: return 0.0, False
        transformed_stream = np.where(self.bit_stream == 1, 1, -1)
        absolute_sum = float(np.abs(np.sum(transformed_stream)))
        s_obs = absolute_sum / np.sqrt(n)
        p_value = float(spc.erfc(s_obs / np.sqrt(2.0)))
        return p_value, p_value >= 0.01

    def run_block_frequency_test(self, block_length: int = 128) -> Tuple[float, bool]:
        n = len(self.bit_stream)
        num_blocks = n // block_length
        if num_blocks == 0: return 0.0, False
        truncated_stream = self.bit_stream[:num_blocks * block_length]
        blocks = truncated_stream.reshape(num_blocks, block_length)
        block_proportions = np.sum(blocks, axis=1) / block_length
        chi_square_stat = float(4.0 * block_length * np.sum((block_proportions - 0.5) ** 2))
        p_value: float = float(spc.gammaincc(num_blocks / 2.0, chi_square_stat / 2.0))
        return p_value, p_value >= 0.01

    def execute_all_verification_checks(self) -> None:
        print("\n================= ENGINE STATISTICAL SMOKE CHECKS =================")
        print(f"Engine RAM Bit Stream Under Evaluation: {len(self.bit_stream)} total bits")

        mono_p, mono_pass = self.run_monobit_frequency_test()
        status_mono = "[ PASSED ]" if mono_pass else "[ FAILED ]"
        print(f"Test 01: Monobit Frequency Test  | P-Value: {mono_p:.6f} | Status: {status_mono}")

        block_p, block_pass = self.run_block_frequency_test(block_length=128)
        status_block = "[ PASSED ]" if block_pass else "[ FAILED ]"
        print(f"Test 02: Block Frequency Test    | P-Value: {block_p:.6f} | Status: {status_block}")

        print("--------------------------------------------------------------------------")
        if mono_pass and block_pass:
            print("VERDICT: STATISTICAL SMOKE CHECKS PASSED")
        else:
            print("VERDICT: STATISTICAL SMOKE CHECKS FAILED")
        print("==========================================================================\n")


class EngineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WiSARDPhysicsConfig()
        self.engine = PurePhysicsInformedWiSARD(self.config)

    def test_quantizer_preserves_configured_width(self) -> None:
        quantizer = ThermometerQuantizer(-1.0, 1.0, self.config.BIT_DEPTH)
        encoded = quantizer.process(np.zeros((3, self.config.NUM_FEATURES)))
        self.assertEqual(encoded.shape, (3, self.config.TOTAL_INPUT_BITS))

    def test_engine_rejects_invalid_discriminator(self) -> None:
        bits = np.zeros((1, self.config.TOTAL_INPUT_BITS), dtype=np.uint8)
        raw = np.zeros((1, self.config.NUM_FEATURES))
        with self.assertRaises(ValueError):
            self.engine.memorize(bits, raw, np.array([self.config.NUM_DISCRIMINATORS]))

    def test_mapping_is_deterministic_and_feature_isolated(self) -> None:
        other_engine = PurePhysicsInformedWiSARD(self.config)
        np.testing.assert_array_equal(
            self.engine.address_mapping_table, other_engine.address_mapping_table
        )
        feature_width = self.config.BIT_DEPTH
        for ram_index, tuple_bits in enumerate(self.engine.address_mapping_table):
            feature_index = ram_index // (self.config.NUM_RAMS_PER_DISCRIMINATOR // self.config.NUM_FEATURES)
            self.assertTrue(np.all(tuple_bits // feature_width == feature_index))

    def test_memorized_sample_scores_its_target_discriminator(self) -> None:
        bits = np.zeros((1, self.config.TOTAL_INPUT_BITS), dtype=np.uint8)
        raw = np.zeros((1, self.config.BASE_FEATURES))
        self.engine.memorize(bits, raw, np.array([1], dtype=np.int32))
        scores = self.engine.evaluate(bits, raw)
        self.assertEqual(scores[0, 1], self.config.NUM_RAMS_PER_DISCRIMINATOR)
        self.assertLess(scores[0, 0], scores[0, 1])

    def test_quantizer_clips_and_sets_anchor_bit(self) -> None:
        quantizer = ThermometerQuantizer(-1.0, 1.0, 4)
        encoded = quantizer.process(np.array([[-10.0], [10.0]]))
        np.testing.assert_array_equal(encoded[0], [0, 0, 0, 1])
        np.testing.assert_array_equal(encoded[1], [1, 1, 1, 1])

    def test_unseen_addresses_receive_soft_scores(self) -> None:
        bits = np.zeros((1, self.config.TOTAL_INPUT_BITS), dtype=np.uint8)
        raw = np.zeros((1, self.config.BASE_FEATURES))
        self.engine.memorize(bits, raw, np.array([0], dtype=np.int32))
        scores = self.engine.soft_scores(np.ones_like(bits), raw)
        self.assertTrue(np.isfinite(scores).all())

    def test_soft_scores_do_not_prefer_larger_class_prior(self) -> None:
        bits = np.zeros((2, self.config.TOTAL_INPUT_BITS), dtype=np.uint8)
        raw = np.zeros((2, self.config.BASE_FEATURES))
        self.engine.memorize(bits, raw, np.array([0, 1], dtype=np.int32))
        scores = self.engine.soft_scores(np.ones_like(bits), raw)
        self.assertAlmostEqual(float(scores[0, 0]), float(scores[0, 1]), places=6)

    def test_invalid_physics_is_masked_during_bleaching(self) -> None:
        bits = np.zeros((1, self.config.TOTAL_INPUT_BITS), dtype=np.uint8)
        raw = np.full((1, self.config.BASE_FEATURES), np.nan)
        self.assertEqual(self.engine.predict(bits, raw)[0], -1)

    def test_ising_uses_nine_bit_local_neighborhoods(self) -> None:
        simulator = IsingTransitionSimulator(size=5, seed=3)
        address = simulator.neighborhood_address(2, 2)
        self.assertGreaterEqual(address, 0)
        self.assertLess(address, 2 ** 9)
        simulator.run(20)
        self.assertGreater(len(simulator.transition_counts), 0)

    def test_pendulum_projection_preserves_energy_budget(self) -> None:
        simulator = EnergyConstrainedDoublePendulum(dt=0.005)
        simulator.trajectory(50)
        self.assertAlmostEqual(simulator.energy(), simulator.energy_budget, places=8)

    def test_tuple_sixteen_can_use_bounded_backends(self) -> None:
        raw = np.zeros((2, self.config.BASE_FEATURES))
        labels = np.array([0, 1], dtype=np.int32)
        for mode, maximum_bytes in (("sparse", 10000), ("hashed", 100000)):
            config = WiSARDPhysicsConfig(
                window_size=4, tuple_size=16, storage_mode=mode, hash_buckets=4096
            )
            bits = np.zeros((2, config.TOTAL_INPUT_BITS), dtype=np.uint8)
            engine = PurePhysicsInformedWiSARD(config)
            engine.memorize(bits, raw, labels)
            self.assertLess(engine.memory_bytes, maximum_bytes)
            self.assertEqual(engine.evaluate(bits, raw).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
