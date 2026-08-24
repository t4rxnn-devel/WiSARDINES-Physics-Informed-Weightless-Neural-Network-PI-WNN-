# PI-WNN

Physics-informed weightless learning with bit-packed memory, bounded hardware growth, and enough experimental receipts to keep the claims honest.

> The RAMs are tiny. The caveats are not.

## Release 1.1.0

This release adds configurable temporal windows, sparse and hashed tuple storage, hard and soft physical constraints, Ising and double-pendulum simulators, leakage-aware real-data ingestion, and a one-million-event HIGGS benchmark.

## Quick Start

```bash
python -m pip install -e .
python -m unittest discover -v
python experiments/higgs_million_train.py data/higgs.zip --limit 1000000 --chunk-size 8192 --quantizer-max 10
```

The HIGGS archive is intentionally ignored by Git because it is multi-gigabyte. Download it from the documented UCI source before running the benchmark.

## What is implemented

The model quantizes telemetry with thermometer coding, routes deterministic tuples into bit-packed RAM, and partitions addresses into target-mass A, target-mass B, and background tiers. Temporal windows are supported without changing the physics contract:

- Raw event: `(4,)` = `p1x, p1y, p2x, p2y`
- Raw window: `(window_size, 4)`
- Encoded model input: `(window_size * 4 * bit_depth,)`
- Physics partition: invariant mass of the final event in each window
- Window stride: configurable; incomplete trailing windows are dropped

The default `window_size=1` retains the original event-level behavior. A larger window increases address capacity linearly. For example, `window_size=8` produces 1,024 encoded bits and 128 RAMs per discriminator.

## Install And Test

```bash
python -m pip install -e .
python -m unittest discover -v
```

## Run The Application

```bash
python main.py
```

For quick experiments, use the separate scripts:

```bash
python experiments/window_scaling.py --windows 1 4 8 16 --batch-size 512
python experiments/noise_sensitivity.py --window-size 4 --batch-size 512
python experiments/real_world_train.py data/telemetry.csv --features p1x p1y p2x p2y --label label --group device_id
python experiments/higgs_million_train.py data/higgs.zip --limit 1000000 --chunk-size 8192 --quantizer-max 10
```

Both scripts print results and write JSON to `results/`. Set `--seed` for reproducibility.

## Real-World Training

The checked-in real-world example is the UCI Iris dataset in `data/iris.csv`, licensed CC BY 4.0; provenance is recorded in `data/iris_source.txt`. Run `experiments/real_world_train.py` with it or another CSV/NPZ file. CSV requires four numeric feature columns and a label column; NPZ requires `features` with shape `(n, 4)` and `labels`, plus optional `groups`. The command uses class-stratified splitting when groups are absent and whole-group splitting when groups are supplied, preventing subject/device leakage. It reports overall accuracy, precision, recall, and support per class. Physics hard masking and mass partitioning are disabled for arbitrary real-world labels by default because the default particle-mass thresholds would otherwise impose an uncalibrated dataset bias; calibrate the physical thresholds before enabling those constraints.

For million-scale physics training, `experiments/higgs_million_train.py` streams the UCI HIGGS archive twice in bounded chunks. It uses all 28 published features, keeps the first four as the physical anchor for compatibility, applies a deterministic row-hash split, and creates no synthetic rows. HIGGS is a Monte Carlo physics benchmark rather than detector data; its provenance and license must be reviewed before redistribution.

The checked-in million-row run with all 28 features achieved 62.378% accuracy and 61.508% balanced accuracy. The earlier four-feature run achieved 52.089% and 52.739%, demonstrating that the expanded representation materially improves the benchmark without changing the split.

## Parameters

`WiSARDPhysicsConfig` accepts `window_size`, `window_stride`, `bit_depth`, `tuple_size`, `num_discriminators`, target masses, energy threshold, and quantizer bounds. Tuple size must divide the encoded input width. The engine rejects malformed shapes, non-binary input, non-finite telemetry, and invalid discriminator IDs.

## Physical Constraints And Unseen Addresses

The implementation uses both constraint types:

- **Hard constraint:** mass-tier routing selects exactly one admissible physical tier. During `predict`/bleaching, invalid telemetry is masked and returns `-1`; it cannot win a tie. The pendulum simulator also refuses to retain a step that cannot fit its energy budget.
- **Soft constraint:** `ram_counts` stores sparse per-discriminator, per-RAM, per-tier frequencies. `soft_scores` applies Laplace smoothing, so an unseen binary address receives a finite prior based on the observed subspace instead of an all-zero score. This is logical smoothing, not dense RAM filling, so the RAM footprint does not multiply.

Thermometer quantization preserves monotonic ordering and clips outside values to configured bounds. It does not, by itself, conserve a physical quantity; conservation is enforced at the physics boundary. The final event in a temporal window supplies the mass constraint.

`physics_simulators.py` contains two reference systems. `IsingTransitionSimulator` maps periodic local 3x3 neighborhoods to 9-bit addresses and learns Metropolis transition frequencies. `EnergyConstrainedDoublePendulum` integrates a chaotic double pendulum and projects velocity back to the initial energy budget after every step. `hardware_profile.py` reports analytical address-gate estimates, exact dense RAM bytes, sparse dictionary usage, and hash collisions. Set `storage_mode="sparse"` for exact accessed-address storage or `storage_mode="hashed"` for a fixed bucket budget. Increasing tuple size has exponential dense RAM cost: in the checked-in window-4 profile, tuple sizes 4, 8, and 16 use 2,304, 18,432, and 2,359,296 bytes respectively; tuple size 16 is 128x tuple size 8. Sparse and hashed modes avoid that dense allocation.

## Scope Of The Diagnostics

The statistical class in `tests.py` is an engine-RAM smoke diagnostic, not a cryptographic randomness certification or a complete NIST SP 800-22 implementation. Its bits are the actual populated RAM state, so sparse-memory failures are expected at low training volume.

## Experiments

- `experiments/window_scaling.py` measures accuracy, memorization throughput, encoded width, RAM count, and model bytes as windows grow.
- `experiments/noise_sensitivity.py` measures accuracy under increasing telemetry noise for a selected window size.
- `experiments/physics_constraints.py` verifies hard masking, soft unseen-address scores, Ising transitions, and pendulum energy.
- `experiments/hardware_alignment.py` profiles RAM growth and address logic as tuple size changes.
- JSON outputs in `results/` are generated artifacts and should be regenerated when changing Python, NumPy, configuration, or random seed.

## Benchmark Receipt

The checked-in one-million-row HIGGS run used all 28 published features, an 80/20 deterministic split, seed 7, and a `-10..10` thermometer range. It achieved `62.378%` accuracy and `61.508%` balanced accuracy. The earlier four-feature run achieved `52.089%` and `52.739%` under the same split. HIGGS is Monte Carlo physics data, not detector data; these numbers are benchmark evidence, not a claim of production scientific validity.

## Release Notes

- Python `>=3.9`, NumPy, and SciPy.
- CI runs the contract suite and compilation checks before any long experiment.
- Dataset archives are excluded from source control; provenance files and compact JSON results are included.
- No model weights, labels, or benchmark results are silently fabricated.

The tiny Easter egg: if a tuple grows from 8 to 16 bits, the lookup table does not become twice as dramatic. It becomes `128x` the dense RAM footprint. Hardware has a sense of humor; exponential growth is the punchline.
