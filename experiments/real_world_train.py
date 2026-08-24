"""Train/evaluate PI-WNN on an explicit real CSV or NPZ dataset."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WiSARDPhysicsConfig
from real_data import load_csv, load_npz, stratified_split
from wisard_engine import PurePhysicsInformedWiSARD
from encoder import ThermometerQuantizer


def run(dataset_path: Path, file_format: str, features: list[str], label: str, group: str | None, test_fraction: float, seed: int) -> dict:
    dataset = load_npz(dataset_path) if file_format == "npz" else load_csv(dataset_path, features, label, group)
    train, test = stratified_split(dataset, test_fraction, seed)
    labels = np.unique(dataset.labels)
    if labels.size > 3:
        raise ValueError("the current engine supports at most three discriminator classes")
    label_to_id = {value: index for index, value in enumerate(labels)}
    train_ids = np.asarray([label_to_id[value] for value in train.labels], dtype=np.int32)
    test_ids = np.asarray([label_to_id[value] for value in test.labels], dtype=np.int32)
    config = WiSARDPhysicsConfig(
        num_discriminators=labels.size,
        hard_physics_constraints=False,
        physical_partitioning=False,
        input_features=train.features.shape[1],
    )
    quantizer = ThermometerQuantizer(config.QUANTIZER_MIN, config.QUANTIZER_MAX, config.BIT_DEPTH)
    engine = PurePhysicsInformedWiSARD(config)
    engine.memorize(quantizer.process(train.features), train.features, train_ids)
    predictions = engine.predict(quantizer.process(test.features), test.features)
    per_class = {}
    for class_id, class_name in enumerate(labels):
        actual = test_ids == class_id
        predicted = predictions == class_id
        per_class[str(class_name)] = {
            "support": int(actual.sum()),
            "recall": float((predicted & actual).sum() / max(actual.sum(), 1)),
            "precision": float((predicted & actual).sum() / max(predicted.sum(), 1)),
        }
    return {
        "dataset": str(dataset_path),
        "train_samples": int(train.features.shape[0]),
        "test_samples": int(test.features.shape[0]),
        "classes": [str(value) for value in labels],
        "accuracy": float(np.mean(predictions == test_ids)),
        "per_class": per_class,
        "group_aware_split": dataset.groups is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--format", choices=["csv", "npz"], default="csv")
    parser.add_argument("--features", nargs="+", default=["p1x", "p1y", "p2x", "p2y"])
    parser.add_argument("--label", default="label")
    parser.add_argument("--group")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.dataset, args.format, args.features, args.label, args.group, args.test_fraction, args.seed)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
