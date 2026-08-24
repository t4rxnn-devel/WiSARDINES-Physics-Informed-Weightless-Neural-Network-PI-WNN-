"""Stream and train on a million or more UCI HIGGS physics events."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WiSARDPhysicsConfig
from encoder import ThermometerQuantizer
from real_data import iter_higgs_chunks
from wisard_engine import PurePhysicsInformedWiSARD


def split_mask(start: int, size: int, test_percent: int, seed: int) -> np.ndarray:
    indices = np.arange(start, start + size, dtype=np.uint64)
    mixed = (indices + np.uint64(seed)) * np.uint64(2654435761)
    return (mixed % 100) < test_percent


def run(archive: Path, limit: int, chunk_size: int, test_percent: int, seed: int, quantizer_max: float) -> dict:
    if limit <= 0 or not 0 < test_percent < 100:
        raise ValueError("limit must be positive and test_percent must be between 0 and 100")
    config = WiSARDPhysicsConfig(
        num_discriminators=2,
        physical_partitioning=False,
        hard_physics_constraints=False,
        quantizer_min=-10.0,
        quantizer_max=quantizer_max,
        input_features=28,
    )
    quantizer = ThermometerQuantizer(-10.0, quantizer_max, config.BIT_DEPTH)
    engine = PurePhysicsInformedWiSARD(config)
    row_offset = 0
    train_count = test_count = 0
    class_counts = np.zeros(2, dtype=np.int64)
    for features, labels in iter_higgs_chunks(archive, chunk_size, limit):
        mask = split_mask(row_offset, labels.size, test_percent, seed)
        train_mask = ~mask
        if train_mask.any():
            engine.memorize(
                quantizer.process(features[train_mask]),
                features[train_mask, :4],
                labels[train_mask],
            )
            class_counts += np.bincount(labels[train_mask], minlength=2)
            train_count += int(train_mask.sum())
        test_count += int(mask.sum())
        row_offset += labels.size

    correct = 0
    evaluated = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    row_offset = 0
    for features, labels in iter_higgs_chunks(archive, chunk_size, limit):
        mask = split_mask(row_offset, labels.size, test_percent, seed)
        if mask.any():
            predictions = engine.predict(
                quantizer.process(features[mask]), features[mask, :4]
            )
            correct += int(np.sum(predictions == labels[mask]))
            evaluated += int(mask.sum())
            for actual, predicted in zip(labels[mask], predictions):
                if predicted in (0, 1):
                    confusion[int(actual), int(predicted)] += 1
        row_offset += labels.size
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "dataset": "UCI HIGGS (Monte Carlo physics benchmark)",
        "archive": str(archive),
        "rows_processed": limit,
        "train_samples": train_count,
        "test_samples": test_count,
        "evaluated_samples": evaluated,
        "accuracy": correct / max(evaluated, 1),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix": confusion.tolist(),
        "train_class_counts": class_counts.tolist(),
        "chunk_size": chunk_size,
        "test_percent": test_percent,
        "seed": seed,
        "model_memory_bytes": engine.memory_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="UCI higgs.zip archive")
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--test-percent", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quantizer-max", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("results/higgs_million.json"))
    args = parser.parse_args()
    result = run(args.archive, args.limit, args.chunk_size, args.test_percent, args.seed, args.quantizer_max)
    serialized = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
