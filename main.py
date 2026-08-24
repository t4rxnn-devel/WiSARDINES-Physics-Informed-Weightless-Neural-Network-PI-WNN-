import sys
import time
import numpy as np
from config import WiSARDPhysicsConfig
from encoder import ThermometerQuantizer
from wisard_engine import PurePhysicsInformedWiSARD
from tests import NISTStatisticalSuite

def simulate_cern_stream(num_samples: int, cfg: WiSARDPhysicsConfig):
    """Generates a shuffled mixture of valid particle tracks and physical anomalies."""
    np.random.seed(888)
    telemetry_data = []
    labels = []
    
    # Discriminator Bank 0: Nominal Tracks matching mass invariants
    count_bank_0 = int(num_samples * 0.6)
    while len(labels) < count_bank_0:
        p = np.random.uniform(0.5, 12.0)
        e = np.sqrt(p**2 + cfg.TARGET_INVARIANT**2)
        telemetry_data.append([p, e])
        labels.append(0)

    # Discriminator Bank 1: Low-Mass Variant Decay tracks
    count_bank_1 = int(num_samples * 0.8)
    while len(labels) < count_bank_1:
        p = np.random.uniform(0.5, 6.0)
        e = p + np.random.uniform(0.02, 0.4)
        telemetry_data.append([p, e])
        labels.append(1)

    # Discriminator Bank 2: Scattered Beam-Halo/Pileup Noise Background
    while len(labels) < num_samples:
        p = np.random.uniform(10.0, 20.0)
        e = p + np.random.uniform(15.0, 35.0)
        telemetry_data.append([p, e])
        labels.append(2)

    X_raw = np.array(telemetry_data, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    
    # CRITICAL FIX: Distribute dataset across training/testing allocations uniformly
    shuffle_indices = np.random.permutation(num_samples)
    return X_raw[shuffle_indices], y[shuffle_indices]

def run_system_verification():
    cfg = WiSARDPhysicsConfig()
    print(">> Initializing CERN Invariant Track Simulator stream...")
    X_raw, y = simulate_cern_stream(2500, cfg)
    
    p_quantizer = ThermometerQuantizer(0.0, 25.0, cfg.BIT_DEPTH)
    e_quantizer = ThermometerQuantizer(0.0, 60.0, cfg.BIT_DEPTH)
    
    X_bin_p = p_quantizer.process(X_raw[:, 0, np.newaxis])
    X_bin_e = e_quantizer.process(X_raw[:, 1, np.newaxis])
    X_bin = np.concatenate((X_bin_p, X_bin_e), axis=1)
    
    split = int(len(y) * 0.80)
    X_train_bin, X_test_bin = X_bin[:split], X_bin[split:]
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train, y_test = y[:split], y[split:]
    
    wisard = PurePhysicsInformedWiSARD(cfg)
    
    print(">> Commencing pattern memorization phase across RAM Discriminator Banks...")
    wisard.memorize(X_train_bin, X_train_raw, y_train)
    
    print(">> Benchmarking real-time streaming inference latency metrics...")
    latencies = []
    for idx in range(min(200, len(y_test))):
        s_bin = X_test_bin[idx:idx+1]
        s_raw = X_test_raw[idx:idx+1]
        
        start = time.perf_counter_ns()
        _ = wisard.evaluate(s_bin, s_raw)
        end = time.perf_counter_ns()
        latencies.append(end - start)
        
    mean_ns = np.mean(latencies)
    jitter_ns = np.max(latencies) - mean_ns
    
    tally_scores = wisard.evaluate(X_test_bin, X_test_raw)
    final_predictions = np.argmax(tally_scores, axis=1)
    accuracy = float(np.mean(final_predictions == y_test) * 100)
    
    print(f"\n================ WISARD MEMORY REGISTER REPORT ================")
    print(f"Total Test Stream Sample Pool: {len(y_test)}")
    print(f"Discriminator Resolution Accuracy: {accuracy:.2f}%")
    print(f"Mean Latency Window Profile: {mean_ns:.2f} ns")
    print(f"Hardware-Emulated Processing Jitter: {jitter_ns:.2f} ns")
    print(f"===============================================================\n")

def main() -> None:
    print("=== Start Pure WiSARD Framework Execution ===")
    try:
        run_system_verification()
        cfg = WiSARDPhysicsConfig()
        wisard_instance = PurePhysicsInformedWiSARD(cfg)
        validator = NISTStatisticalSuite(wisard_instance)
        validator.execute_all_verification_checks()
    except Exception as error:
        print(f"Critical System Crash within WiSARD engine: {str(error)}", file=sys.stderr)
        sys.exit(1)
    print("=== System Environment Gracefully Terminated ===")

if __name__ == "__main__":
    main()

