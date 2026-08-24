"""Measure accuracy, throughput, and memory growth as temporal windows grow."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WiSARDPhysicsConfig
from pipeline import NaturalTrainingPipeline


def run(window_sizes: list[int], batch_size: int, seed: int) -> list[dict[str, float | int]]:
    results = []
    for window_size in window_sizes:
        np.random.seed(seed)
        config = WiSARDPhysicsConfig(window_size=window_size)
        pipeline = NaturalTrainingPipeline(config)
        train_bits, train_raw, train_labels = pipeline.generate_streaming_batch(batch_size)
        test_bits, test_raw, test_labels = pipeline.generate_streaming_batch(batch_size)
        start = time.perf_counter()
        pipeline.engine.memorize(train_bits, train_raw, train_labels)
        elapsed = time.perf_counter() - start
        scores = pipeline.engine.evaluate(test_bits, test_raw)
        accuracy = float(np.mean(np.argmax(scores, axis=1) == test_labels))
        results.append({
            "window_size": window_size,
            "batch_size": batch_size,
            "accuracy": accuracy,
            "memorize_samples_per_second": batch_size / elapsed,
            "encoded_bits": config.TOTAL_INPUT_BITS,
            "rams_per_discriminator": config.NUM_RAMS_PER_DISCRIMINATOR,
            "model_bytes": int(pipeline.engine.discriminator_banks.nbytes),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/window_scaling.json"))
    args = parser.parse_args()
    results = run(args.windows, args.batch_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
