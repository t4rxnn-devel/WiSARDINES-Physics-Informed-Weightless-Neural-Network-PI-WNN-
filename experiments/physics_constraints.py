"""Exercise hard/soft constraints, Ising transitions, and pendulum energy."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WiSARDPhysicsConfig
from physics_simulators import EnergyConstrainedDoublePendulum, IsingTransitionSimulator
from pipeline import NaturalTrainingPipeline


def run(seed: int, batch_size: int) -> dict[str, float | int | bool]:
    np.random.seed(seed)
    config = WiSARDPhysicsConfig(window_size=4)
    pipeline = NaturalTrainingPipeline(config)
    bits, raw, labels = pipeline.generate_streaming_batch(batch_size)
    pipeline.engine.memorize(bits, raw, labels)
    unknown_bits = np.bitwise_xor(bits, np.uint8(1))
    unseen_scores = pipeline.engine.soft_scores(unknown_bits, raw)
    invalid_raw = raw[:1].copy()
    invalid_raw[0, 0] = np.nan
    hard_masked = bool(pipeline.engine.predict(bits[:1], invalid_raw[:1])[0] == -1)

    ising = IsingTransitionSimulator(size=12, seed=seed)
    acceptance = ising.run(1000)
    pendulum = EnergyConstrainedDoublePendulum(dt=0.005)
    _, valid = pendulum.trajectory(500)
    return {
        "window_size": config.WINDOW_SIZE,
        "unseen_score_is_finite": bool(np.isfinite(unseen_scores).all()),
        "invalid_physics_is_hard_masked": hard_masked,
        "ising_acceptance_rate": float(acceptance),
        "ising_transition_patterns": len(ising.transition_counts),
        "pendulum_all_steps_valid": bool(valid.all()),
        "pendulum_energy_error": float(abs(pendulum.energy() - pendulum.energy_budget)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/physics_constraints.json"))
    args = parser.parse_args()
    result = run(args.seed, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
