import time
from typing import Tuple
import numpy as np
from config import WiSARDPhysicsConfig
from encoder import ThermometerQuantizer
from wisard_engine import PurePhysicsInformedWiSARD

class NaturalTrainingPipeline:
    """Production stream manager executing high-volume continuous data ingestion loops."""

    def __init__(self, cfg: WiSARDPhysicsConfig) -> None:
        self.cfg: WiSARDPhysicsConfig = cfg
        self.engine: PurePhysicsInformedWiSARD = PurePhysicsInformedWiSARD(cfg)
        self.p_quantizer = ThermometerQuantizer(0.0, 25.0, self.cfg.BIT_DEPTH)
        self.e_quantizer = ThermometerQuantizer(0.0, 60.0, self.cfg.BIT_DEPTH)

    def generate_streaming_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulates live data streams coming straight from raw particle detectors."""
        telemetry_data = []
        labels = []
        
        c0 = int(batch_size * 0.4)
        c1 = int(batch_size * 0.7)
        
        while len(labels) < c0:
            p = np.random.uniform(0.5, 12.0)
            e = np.sqrt(p**2 + self.cfg.TARGET_INVARIANT**2)
            telemetry_data.append([p, e])
            labels.append(0)

        while len(labels) < c1:
            p = np.random.uniform(0.5, 6.0)
            e = p + np.random.uniform(0.02, 0.4)
            telemetry_data.append([p, e])
            labels.append(1)

        while len(labels) < batch_size:
            p = np.random.uniform(10.0, 20.0)
            e = p + np.random.uniform(15.0, 35.0)
            telemetry_data.append([p, e])
            labels.append(2)

        X_raw = np.array(telemetry_data, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        
        shuffle = np.random.permutation(batch_size)
        X_raw, y = X_raw[shuffle], y[shuffle]
        
        X_bin_p = self.p_quantizer.process(X_raw[:, 0, np.newaxis])
        X_bin_e = self.e_quantizer.process(X_raw[:, 1, np.newaxis])
        X_bin = np.concatenate((X_bin_p, X_bin_e), axis=1)
        
        return X_bin, X_raw, y

    def execute_natural_stream(self, total_epochs: int = 5, samples_per_epoch: int = 15000) -> None:
        """Feeds streaming matrices to the hardware-blind RAM registers over time."""
        print(f"\n>> Commencing Live Training Pipeline ({total_epochs} Stream Eras, {samples_per_epoch:,} frames/era)...")
        
        for epoch in range(1, total_epochs + 1):
            X_train_bin, X_train_raw, y_train = self.generate_streaming_batch(samples_per_epoch)
            X_test_bin, X_test_raw, y_test = self.generate_streaming_batch(3000)
            
            start_time = time.perf_counter()
            self.engine.memorize(X_train_bin, X_train_raw, y_train)
            end_time = time.perf_counter()
            
            tally_scores = self.engine.evaluate(X_test_bin, X_test_raw)
            predictions = np.argmax(tally_scores, axis=1)
            accuracy = float(np.mean(predictions == y_test) * 100)
            
            occupied_cells = np.sum(self.engine.discriminator_banks == 1)
            total_cells = self.engine.discriminator_banks.size
            saturation_ratio = (occupied_cells / total_cells) * 100
            
            print(f"Era {epoch:02d} | Ingested: {samples_per_epoch:,} | "
                  f"Delta: {(end_time - start_time)*1000:.1f}ms | "
                  f"Test Accuracy: {accuracy:.2f}% | "
                  f"RAM Saturation: {saturation_ratio:.2f}%")
