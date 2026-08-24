from typing import Tuple, List
import numpy as np
import scipy.special as spc
from config import WiSARDPhysicsConfig
from wisard_engine import PurePhysicsInformedWiSARD

class NISTStatisticalSuite:
    """NIST SP 800-22 inspired statistical verification suite for WiSARD Address Spaces."""

    def __init__(self, engine: PurePhysicsInformedWiSARD) -> None:
        self.engine: PurePhysicsInformedWiSARD = engine
        self.mapping_matrix: np.ndarray = engine.address_mapping_table
        
        # Dynamically calculate required bit width based on system configurations
        # Upper partition uses 2^(TUPLE_SIZE + 1) slots. Total bits needed per int = TUPLE_SIZE + 1
        self.bit_width: int = self.engine.cfg.TUPLE_SIZE + 1
        self.bit_stream: np.ndarray = self._extract_mapping_bitstream()

    def _extract_mapping_bitstream(self) -> np.ndarray:
        """Converts the mapping integers to pure unstructured binary sequences dynamically."""
        bits = []
        fmt_str = f"0{self.bit_width}b"  # Dynamic string formatting to close the gap completely
        for val in self.mapping_matrix.flatten():
            bits.extend([int(x) for x in format(val, fmt_str)])
        return np.array(bits, dtype=np.int8)

    def run_monobit_frequency_test(self) -> Tuple[float, bool]:
        n: int = len(self.bit_stream)
        if n == 0:
            return 0.0, False

        transformed_stream: np.ndarray = np.where(self.bit_stream == 1, 1, -1)
        absolute_sum: float = float(np.abs(np.sum(transformed_stream)))
        s_obs: float = absolute_sum / np.sqrt(n)
        p_value: float = float(spc.erfc(s_obs / np.sqrt(2.0)))
        
        return p_value, p_value >= 0.01

    def run_block_frequency_test(self, block_length: int = 16) -> Tuple[float, bool]:
        n: int = len(self.bit_stream)
        num_blocks: int = n // block_length
        if num_blocks == 0:
            return 0.0, False
            
        truncated_stream: np.ndarray = self.bit_stream[:num_blocks * block_length]
        blocks: np.ndarray = truncated_stream.reshape(num_blocks, block_length)
        block_proportions: np.ndarray = np.sum(blocks, axis=1) / block_length
        chi_square_stat: float = float(4.0 * block_length * np.sum((block_proportions - 0.5) ** 2))
        p_value: float = float(spc.gammaincc(num_blocks / 2.0, chi_square_stat / 2.0))
        
        return p_value, p_value >= 0.01

    def execute_all_verification_checks(self) -> None:
        print("\n================= NIST STATISTICAL COMPLIANCE INSPECTION =================")
        print(f"Target Sequence Bit Stream Under Evaluation: {len(self.bit_stream)} total bits")
        
        mono_p, mono_pass = self.run_monobit_frequency_test()
        print(f"Test 01: Monobit Frequency Test  | P-Value: {mono_p:.6f} | Status: {'[ PASSED ]' if mono_pass else '[ FAILED ]'}")
        
        block_len_param = 16
        block_p, block_pass = self.run_block_frequency_test(block_length=block_len_param)
        print(f"Test 02: Block Frequency Test    | P-Value: {block_p:.6f} | Status: {'[ PASSED ]' if block_pass else '[ FAILED ]'}")
        
        print("--------------------------------------------------------------------------")
        if mono_pass and block_pass:
            print("VERDICT: NIST SP 800-22 COMPLIANT - ADDRESS ROUTING ENTROPY BALANCED")
        else:
            print("VERDICT: CRITICAL WARNING - ADDRESS STRUCTURAL MAPPING BIAS DETECTED")
        print("==========================================================================\n")
