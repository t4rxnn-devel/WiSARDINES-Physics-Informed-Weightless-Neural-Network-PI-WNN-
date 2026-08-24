"""Compare PI-WNN with PyTorch PINN and XGBoost on a rigid HIGGS split.

Energy is reported as an optional measured proxy: wall_seconds * configured
power_watts. It is not a hardware power measurement unless the user supplies
power estimates from an external meter.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from higgs_million_train import collect_fixed_split


def run(archive: Path, train_size: int, test_size: int, seed: int, epochs: int, power_watts: float) -> dict:
    try:
        import torch
        import torch.nn as nn
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("Install optional baselines with: pip install torch xgboost") from error

    train_x, train_y, test_x, test_y = collect_fixed_split(archive, train_size, test_size, 8192, seed)
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-8
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    results = {}

    tree = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=1)
    start = time.perf_counter(); tree.fit(train_x, train_y); elapsed = time.perf_counter() - start
    results["xgboost"] = {"accuracy": float(np.mean(tree.predict(test_x) == test_y)), "seconds": elapsed, "energy_proxy_joules": elapsed * power_watts}

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(28, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.tensor(train_x, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor(train_y[:, None], dtype=torch.float32)
    start = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(features)
        data_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        physics_residual = torch.autograd.functional.jacobian(lambda value: model(value).sum(), features[:1]).pow(2).mean()
        loss = data_loss + 1e-4 * physics_residual
        loss.backward(); optimizer.step()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        predictions = (torch.sigmoid(model(torch.tensor(test_x, dtype=torch.float32))) >= 0.5).numpy().ravel().astype(np.int32)
    results["pytorch_pinn"] = {"accuracy": float(np.mean(predictions == test_y)), "seconds": elapsed, "energy_proxy_joules": elapsed * power_watts, "physics_loss_weight": 1e-4}
    results["protocol"] = {"train_size": train_size, "test_size": test_size, "seed": seed, "features": 28, "power_watts_assumption": power_watts, "energy_is_proxy": True}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--power-watts", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("results/continuous_baselines.json"))
    args = parser.parse_args()
    result = run(args.archive, args.train_size, args.test_size, args.seed, args.epochs, args.power_watts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
