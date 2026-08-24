"""Rigid volume-scaling comparison: ordinary versus symmetry-seeded WiSARD."""
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


def collect_fixed_split(archive: Path, max_train: int, test_size: int, chunk_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    offset = 0
    for features, labels in iter_higgs_chunks(archive, chunk_size, max_train * 2 + test_size * 2):
        mask = split_mask(offset, labels.size, 20, seed)
        train_needed = max_train - sum(block.shape[0] for block in train_features)
        test_needed = test_size - sum(block.shape[0] for block in test_features)
        if train_needed > 0:
            selected = features[~mask][:train_needed]
            selected_labels = labels[~mask][:train_needed]
            if selected.size:
                train_features.append(selected)
                train_labels.append(selected_labels)
        if test_needed > 0:
            selected = features[mask][:test_needed]
            selected_labels = labels[mask][:test_needed]
            if selected.size:
                test_features.append(selected)
                test_labels.append(selected_labels)
        offset += labels.size
        if sum(block.shape[0] for block in train_features) >= max_train and sum(block.shape[0] for block in test_features) >= test_size:
            break
    return (
        np.concatenate(train_features)[:max_train],
        np.concatenate(train_labels)[:max_train],
        np.concatenate(test_features)[:test_size],
        np.concatenate(test_labels)[:test_size],
    )


def evaluate_mode(config: WiSARDPhysicsConfig, quantizer: ThermometerQuantizer, train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray, test_labels: np.ndarray, symmetric: bool) -> float:
    engine = PurePhysicsInformedWiSARD(config)
    train_bits = quantizer.process(train_features)
    if symmetric:
        engine.memorize_symmetric(train_bits, train_features[:, :4], train_labels)
    else:
        engine.memorize(train_bits, train_features[:, :4], train_labels)
    predictions = engine.predict(quantizer.process(test_features), test_features[:, :4])
    recalls = []
    for label in (0, 1):
        actual = test_labels == label
        recalls.append(float(np.mean(predictions[actual] == label)))
    return float(np.mean(recalls))


def write_svg(results: list[dict[str, float | int]], path: Path) -> None:
    volumes = sorted({int(row["volume"]) for row in results})
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 40, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = np.log10(min(volumes)), np.log10(max(volumes))

    def point(volume: int, accuracy: float) -> tuple[float, float]:
        x = left + (np.log10(volume) - x_min) / max(x_max - x_min, 1.0) * plot_width
        y = top + (1.0 - accuracy) * plot_height
        return x, y

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family: sans-serif; fill:#263238} .grid{stroke:#cfd8dc;stroke-width:1} .baseline{fill:none;stroke:#546e7a;stroke-width:4} .symmetric{fill:none;stroke:#e65100;stroke-width:4} </style>', '<text x="90" y="25" font-size="20" font-weight="bold">Rigid HIGGS split: symmetry prior vs baseline</text>']
    for tick in np.linspace(0, 1, 6):
        y = top + (1.0 - tick) * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" font-size="12" text-anchor="end">{tick:.1f}</text>')
    for mode, css, label in (("baseline", "baseline", "Baseline"), ("symmetric", "symmetric", "Symmetry seeded")):
        rows = [row for row in results if row["mode"] == mode]
        points = " ".join(f"{point(int(row['volume']), float(row['balanced_accuracy']))[0]:.1f},{point(int(row['volume']), float(row['balanced_accuracy']))[1]:.1f}" for row in rows)
        parts.append(f'<polyline class="{css}" points="{points}"/>')
        for row in rows:
            x, y = point(int(row["volume"]), float(row["balanced_accuracy"]))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="white" stroke="currentColor" stroke-width="3"/>')
        parts.append(f'<text x="{width-170}" y="{top + (0 if mode == "baseline" else 28)}" font-size="14">{label}</text>')
    for volume in volumes:
        x, _ = point(volume, 0)
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+25}" font-size="12" text-anchor="middle">{volume:,}</text>')
    parts.extend([f'<text x="{width/2}" y="{height-18}" font-size="15" text-anchor="middle">Training volume (log scale)</text>', f'<text transform="translate(20 {height/2}) rotate(-90)" font-size="15" text-anchor="middle">Balanced accuracy</text>', '</svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def run(archive: Path, volumes: list[int], test_size: int, chunk_size: int, seed: int, quantizer_max: float) -> list[dict[str, float | int]]:
    max_volume = max(volumes)
    train_features, train_labels, test_features, test_labels = collect_fixed_split(archive, max_volume, test_size, chunk_size, seed)
    results = []
    for volume in volumes:
        config = WiSARDPhysicsConfig(
            num_discriminators=2,
            physical_partitioning=False,
            hard_physics_constraints=False,
            input_features=28,
            quantizer_min=-10.0,
            quantizer_max=quantizer_max,
            storage_mode="dense",
        )
        quantizer = ThermometerQuantizer(-10.0, quantizer_max, config.BIT_DEPTH)
        for symmetric in (False, True):
            balanced_accuracy = evaluate_mode(
                config, quantizer, train_features[:volume], train_labels[:volume],
                test_features, test_labels, symmetric
            )
            results.append({
                "volume": volume,
                "mode": "symmetric" if symmetric else "baseline",
                "balanced_accuracy": balanced_accuracy,
                "test_size": test_size,
                "seed": seed,
                "input_features": 28,
                "tuple_size": config.TUPLE_SIZE,
                "symmetry_orbit": 4,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--volumes", nargs="+", type=int, default=[1000, 5000, 20000, 100000])
    parser.add_argument("--test-size", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quantizer-max", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("results/symmetry_scaling.json"))
    parser.add_argument("--plot", type=Path, default=Path("results/symmetry_scaling.png"))
    args = parser.parse_args()
    results = run(args.archive, args.volumes, args.test_size, args.chunk_size, args.seed, args.quantizer_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    try:
        write_svg(results, args.plot.with_suffix(".svg"))
    except OSError as error:
        print(f"plot unavailable: {error}; JSON results were still written")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
