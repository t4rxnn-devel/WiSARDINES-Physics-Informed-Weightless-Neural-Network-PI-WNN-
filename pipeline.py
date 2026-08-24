import time
from typing import Tuple
import numpy as np
from config import WiSARDPhysicsConfig
from encoder import ThermometerQuantizer
from wisard_engine import PurePhysicsInformedWiSARD
from windowing import make_windows

class NaturalTrainingPipeline:
    def __init__(self, cfg: WiSARDPhysicsConfig) -> None:
        self.cfg = cfg
        self.engine = PurePhysicsInformedWiSARD(cfg)
        self.quantizer = ThermometerQuantizer(
            self.cfg.QUANTIZER_MIN, self.cfg.QUANTIZER_MAX, self.cfg.BIT_DEPTH
        )

    def _generate_decay(self, count: int, parent_mass: float) -> np.ndarray:
        p_parent_x = np.random.normal(0, 2.0, count)
        p_parent_y = np.random.normal(0, 2.0, count)
        E_parent = np.sqrt(p_parent_x**2 + p_parent_y**2 + parent_mass**2)

        p_star = parent_mass / 2.0
        phi_star = np.random.uniform(0, 2 * np.pi, count)
        p1x_star = p_star * np.cos(phi_star)
        p1y_star = p_star * np.sin(phi_star)

        bx = p_parent_x / E_parent
        by = p_parent_y / E_parent
        gamma = 1.0 / np.sqrt(1.0 - bx**2 - by**2)
        bp1 = bx * p1x_star + by * p1y_star
        factor1 = (gamma - 1.0) * bp1 / np.maximum(bx**2 + by**2, 1e-10) + gamma * p_star

        data = np.zeros((count, 4), dtype=np.float64)
        data[:, 0] = p1x_star + bx * factor1 + np.random.normal(0, 0.02, count)
        data[:, 1] = p1y_star + by * factor1 + np.random.normal(0, 0.02, count)
        data[:, 2] = -p1x_star + bx * factor1 + np.random.normal(0, 0.02, count)
        data[:, 3] = -p1y_star + by * factor1 + np.random.normal(0, 0.02, count)
        return data

    def generate_streaming_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        event_count = batch_size * self.cfg.WINDOW_STRIDE + self.cfg.WINDOW_SIZE - 1
        c0, c1 = int(event_count * 0.4), int(event_count * 0.7)
        telemetry_data = np.zeros((event_count, self.cfg.BASE_FEATURES), dtype=np.float64)
        labels = np.zeros(event_count, dtype=np.int32)

        telemetry_data[:c0] = self._generate_decay(c0, self.cfg.TARGET_MASS_A)
        labels[:c0] = 0
        telemetry_data[c0:c1] = self._generate_decay(c1 - c0, self.cfg.TARGET_MASS_B)
        labels[c0:c1] = 1
        telemetry_data[c1:] = np.random.normal(0, 3.5, size=(event_count - c1, self.cfg.BASE_FEATURES))
        labels[c1:] = 2

        windows = make_windows(telemetry_data, self.cfg.WINDOW_SIZE, self.cfg.WINDOW_STRIDE)
        label_windows = make_windows(labels[:, np.newaxis], self.cfg.WINDOW_SIZE, self.cfg.WINDOW_STRIDE)
        if windows.shape[0] < batch_size:
            raise RuntimeError("window generation produced fewer samples than requested")
        window_order = np.random.permutation(windows.shape[0])[:batch_size]
        windows = windows[window_order]
        y = label_windows[window_order, -1, 0]
        X_raw = windows[:, -1, :]
        X_bin = self.quantizer.process(windows.reshape(batch_size, -1))
        return X_bin, X_raw, y

    def execute_natural_stream(self, total_epochs: int = 5, samples_per_epoch: int = 15000) -> None:
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

            occupied_bits = self.engine.occupied_bits
            total_bits = max(self.engine.memory_bytes * 8, 1)
            saturation_ratio = (occupied_bits / total_bits) * 100

            print(f"Era {epoch:02d} | Ingested: {samples_per_epoch:,} | Delta: {(end_time - start_time)*1000:.1f}ms | Test Accuracy: {accuracy:.2f}% | RAM Saturation: {saturation_ratio:.2f}%")
