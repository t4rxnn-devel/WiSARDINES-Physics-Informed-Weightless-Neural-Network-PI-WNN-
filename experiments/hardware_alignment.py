"""Profile RAM growth and analytical address-logic cost."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WiSARDPhysicsConfig
from hardware_profile import profile_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuple-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--storage-modes", nargs="+", default=["dense", "sparse", "hashed"])
    parser.add_argument("--hash-buckets", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/hardware_alignment.json"))
    args = parser.parse_args()
    results = []
    for storage_mode in args.storage_modes:
        for tuple_size in args.tuple_sizes:
            config = WiSARDPhysicsConfig(
                window_size=args.window_size,
                tuple_size=tuple_size,
                storage_mode=storage_mode,
                hash_buckets=args.hash_buckets,
            )
            results.append(profile_runtime(config, args.batch_size, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
