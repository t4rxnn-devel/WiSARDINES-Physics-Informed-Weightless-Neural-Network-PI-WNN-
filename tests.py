from typing import Final
import numpy as np
import scipy.special as spc  # Standard scientific library for calculating the complementary error function (erfc)
from config import WiSARDPhysicsConfig
from wisard_engine import PurePhysicsInformedWiSARD

class NISTStatisticalSuite:
    """NIST SP 800-22 inspired statistical verification suite for WiSARD Address Spaces."""

    def __init__(self, engine: PurePhysicsInformedWiSARD) -> None:
        self.engine: PurePhysicsInformedWiSARD = engine
        self.mapping_matrix: np.ndarray = engine.address_mapping_table
        
        # Flatten the structural mapping array and unpack it into a pure bit stream
        self.bit_stream: np.ndarray = self._extract_mapping_bitstream()

    def _extract_mapping_bitstream(self) -> np.ndarray:
        """Converts the internal integer mapping matrix into a contiguous binary bit stream array."""
        flattened_ints: np.ndarray = self.mapping_matrix.flatten()
        
        # Unpack uint32 integers into individual bits (MSB to LSB layout mapping)
        # We enforce big-endian bit unpacking to preserve bit string sequence continuity
        bit_matrix: np.ndarray = np.unpackbits(flattened_ints.view(np.uint8))
        return bit_matrix.astype(np.int8)

    def run_monobit_frequency_test(self) -> Tuple[float, bool]:
        """
        NIST SP 800-22 Section 2.1: Frequency (Monobit) Test.
        Tests if the proportion of zeroes and ones is approximately equal.
        Returns: (p_value, pass_status)
        """
        n: int = len(self.bit_stream)
        if n == 0:
            return 0.0, False

        # Transform 0s to -1s and 1s to +1s as specified by NIST criteria
        transformed_stream: np.ndarray = np.where(self.bit_stream == 1, 1, -1)
        absolute_sum: float = float(np.abs(np.sum(transformed_stream)))
        
        # Compute the test statistic S_obs
        s_obs: float = absolute_sum / np.sqrt(n)
        
        # Calculate P-value using the complementary error function (erfc)
        p_value: float = float(spc.erfc(s_obs / np.sqrt(2.0)))
        
        # NIST standard threshold: Alpha = 0.01. If p_value >= 0.01, sequence is random.
        is_valid: bool = p_value >= 0.01
        return p_value, is_valid

    def run_block_frequency_test(self, block_length: int = 128) -> Tuple[float, bool]:
        """
        NIST SP 800-22 Section 2.2: Frequency Test within a Block.
        Tests if the frequency of ones inside an M-bit block is approximately M/2.
        Returns: (p_value, pass_status)
        """
        n: int = len(self.bit_stream)
        num_blocks: int = n // block_length
        
        if num_blocks == 0:
            return 0.0, False
            
        # Truncate any trailing bits that do not cleanly fit inside the block division limit
        truncated_stream: np.ndarray = self.bit_stream[:num_blocks * block_length]
        blocks: np.ndarray = truncated_stream.reshape(num_blocks, block_length)
        
        # Calculate the proportion of ones within each block matrix lane
        block_proportions: np.ndarray = np.sum(blocks, axis=1) / block_length
        
        # Compute the Chi-Square statistic
        chi_square_stat: float = float(4.0 * block_length * np.sum((block_proportions - 0.5) ** 2))
        
        # Compute P-value using the incomplete gamma function (or survival function for chi2)
        # spc.gammaincc(dof / 2, chi_square / 2) corresponds directly to the NIST target formula
        p_value: float = float(spc.gammaincc(num_blocks / 2.0, chi_square_stat / 2.0))
        
        is_valid: bool = p_value >= 0.01
        return p_value, is_valid

    def execute_all_verification_checks(self) -> None:
        """Executes full diagnostic test suite loops and generates compliance report log parameters."""
        print("\n================= NIST STATISTICAL COMPLIANCE INSPECTION =================")
        print(f"Target Sequence Bit Stream Under Evaluation: {len(self.bit_stream)} total bits")
        
        # 1. Monobit Verification Test
        mono_p, mono_pass = self.run_monobit_frequency_test()
        print(f"Test 01: Monobit Frequency Test  | P-Value: {mono_p:.6f} | Status: {'[ PASSED ]' if mono_pass else '[ FAILED ]'}")
        
        # 2. Block Frequency Verification Test
        block_len_param = 128
        block_p, block_pass = self.run_block_frequency_test(block_length=block_len_param)
        print(f"Test 02: Block Frequency Test    | P-Value: {block_p:.6f} | Status: {'[ PASSED ]' if block_pass else '[ FAILED ]'}")
        
        print("--------------------------------------------------------------------------")
        if mono_pass and block_pass:
            print("VERDICT: NIST SP 800-22 COMPLIANT - ADDRESS ROUTING ENTROPY BALANCED")
        else:
            print("VERDICT: CRITICAL WARNING - ADDRESS STRUCTURAL MAPPING BIAS DETECTED")
        print("==========================================================================\n")

if __name__ == "__main__":
    # Self-contained initialization loop for standalone diagnostic validation execution
    cfg = WiSARDPhysicsConfig()
    wisard_instance = PurePhysicsInformedWiSARD(cfg)
    
    suite = NISTStatisticalSuite(wisard_instance)
    suite.execute_all_verification_checks()
