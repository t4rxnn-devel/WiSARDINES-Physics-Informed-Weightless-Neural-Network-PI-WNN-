"""Measure symmetry seeding and emit exact RAM saliency records."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WiSARDPhysicsConfig
from encoder import ThermometerQuantizer
from physics_simulators import IsingTransitionSimulator
from wisard_engine import PurePhysicsInformedWiSARD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/symmetry_saliency.json"))
    args = parser.parse_args()
    config = WiSARDPhysicsConfig(storage_mode="sparse")
    engine = PurePhysicsInformedWiSARD(config)
    quantizer = ThermometerQuantizer(-15.0, 15.0, config.BIT_DEPTH)
    raw = np.zeros((1, config.BASE_FEATURES), dtype=np.float64)
    bits = quantizer.process(raw)
    seed_result = engine.memorize_symmetric(bits, raw, np.array([0], dtype=np.int32))
    saliency = engine.explain(bits, raw, target_discriminator=0)[0]
    ising = IsingTransitionSimulator(size=8, seed=7)
    ising_seeded = ising.seed_symmetric_transition(0b100010001, 0b111010111)
    result = {
        "address_orbit_size_for_zero": len(engine.address_orbit(0)),
        "symmetry_seed": seed_result,
        "saliency_records": saliency[:4],
        "ising_transition_orbit_entries": ising_seeded,
        "saliency_is_exact": all(record["attribution_status"] == "exact" for record in saliency),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
