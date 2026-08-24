# Experiment Results

The JSON files in this directory are reference runs from the checked-in experiment scripts. They use small batches so they remain quick to reproduce in development. Results are sensitive to Python, NumPy, machine speed, and random seed.

Regenerate them from the repository root with:

```bash
python experiments/window_scaling.py
python experiments/noise_sensitivity.py
```

The real-world reference run is `iris_real_world.json`, generated from the CC BY 4.0 UCI Iris data in `data/iris.csv` with seed 7 and a 20% stratified test split.

`higgs_million.json` is a one-million-row streaming benchmark from the UCI HIGGS Monte Carlo dataset using all 28 published features, an 80/20 deterministic split, seed 7, and a `-10..10` thermometer range. It achieved 62.378% accuracy and 61.508% balanced accuracy. This is a benchmark result, not detector-data performance.

`symmetry_saliency.json` records the orbit-seeding and dense/sparse reverse-lookup smoke test. Exact attribution is guaranteed only for dense and sparse storage; hashed storage can introduce bucket ambiguity.

`symmetry_scaling.json` compares baseline and symmetry-seeded models on one fixed HIGGS split. `symmetry_scaling.svg` is the corresponding plot. `symmetry_robustness.json` repeats the comparison for seeds 3, 7, and 19; the mean balanced-accuracy lift is positive at every tested volume, but the effect narrows as training volume grows.
