"""Measure model accuracy as telemetry noise increases."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WiSARDPhysicsConfig
from pipeline import NaturalTrainingPipeline


def run(noise_levels: list[float], batch_size: int, window_size: int, seed: int) -> list[dict[str, float | int]]:
    results = []
    for noise in noise_levels:
        np.random.seed(seed)
        config = WiSARDPhysicsConfig(window_size=window_size)
        pipeline = NaturalTrainingPipeline(config)
        train_bits, train_raw, train_labels = pipeline.generate_streaming_batch(batch_size)
        test_bits, test_raw, test_labels = pipeline.generate_streaming_batch(batch_size)
        noisy_train = train_raw + np.random.normal(0.0, noise, train_raw.shape)
        noisy_test = test_raw + np.random.normal(0.0, noise, test_raw.shape)
        train_bits = pipeline.quantizer.process(
            np.repeat(noisy_train[:, np.newaxis, :], window_size, axis=1).reshape(batch_size, -1)
        )
        test_bits = pipeline.quantizer.process(
            np.repeat(noisy_test[:, np.newaxis, :], window_size, axis=1).reshape(batch_size, -1)
        )
        pipeline.engine.memorize(train_bits, noisy_train, train_labels)
        scores = pipeline.engine.evaluate(test_bits, noisy_test)
        results.append({
            "noise_std": noise,
            "window_size": window_size,
            "batch_size": batch_size,
            "accuracy": float(np.mean(np.argmax(scores, axis=1) == test_labels)),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.05, 0.2, 0.5, 1.0])
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/noise_sensitivity.json"))
    args = parser.parse_args()
    results = run(args.noise, args.batch_size, args.window_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
