"""Leakage-aware loading and evaluation utilities for real telemetry datasets."""
from dataclasses import dataclass
import csv
import gzip
from pathlib import Path
from typing import Iterator
import zipfile

import numpy as np


@dataclass(frozen=True)
class RealDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray | None = None


def _validate(features: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None) -> RealDataset:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[1] < 4:
        raise ValueError("real data must contain at least four features")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must contain one value per feature row")
    if not np.isfinite(features).all():
        raise ValueError("real features must contain only finite values")
    if np.unique(labels).size < 2:
        raise ValueError("real data must contain at least two classes")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or groups.shape[0] != features.shape[0]:
            raise ValueError("groups must contain one value per feature row")
    return RealDataset(features, labels, groups)


def load_csv(path: str | Path, feature_columns: list[str], label_column: str, group_column: str | None = None) -> RealDataset:
    """Load at least four named features and a label from a CSV file."""
    if len(feature_columns) < 4:
        raise ValueError("feature_columns must contain at least four names")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(feature_columns + [label_column] + ([group_column] if group_column else []))
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV is missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        features, labels, groups = [], [], [] if group_column else None
        for row in reader:
            features.append([float(row[column]) for column in feature_columns])
            labels.append(row[label_column])
            if group_column:
                groups.append(row[group_column])
    return _validate(np.asarray(features), np.asarray(labels), groups)


def load_npz(path: str | Path) -> RealDataset:
    """Load an NPZ containing ``features``, ``labels``, and optional ``groups``."""
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data or "labels" not in data:
            raise ValueError("NPZ must contain features and labels arrays")
        return _validate(data["features"], data["labels"], data.get("groups"))


def stratified_split(dataset: RealDataset, test_fraction: float = 0.2, seed: int = 7) -> tuple[RealDataset, RealDataset]:
    """Split by class, or by whole groups when groups are supplied."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    if dataset.groups is not None:
        unique_groups = np.unique(dataset.groups)
        rng.shuffle(unique_groups)
        test_groups = set(unique_groups[:max(1, int(len(unique_groups) * test_fraction))])
        test_mask = np.isin(dataset.groups, list(test_groups))
    else:
        test_mask = np.zeros(dataset.labels.shape[0], dtype=bool)
        for label in np.unique(dataset.labels):
            indices = np.flatnonzero(dataset.labels == label)
            rng.shuffle(indices)
            test_mask[indices[:max(1, int(len(indices) * test_fraction))]] = True
    train = RealDataset(dataset.features[~test_mask], dataset.labels[~test_mask], None if dataset.groups is None else dataset.groups[~test_mask])
    test = RealDataset(dataset.features[test_mask], dataset.labels[test_mask], None if dataset.groups is None else dataset.groups[test_mask])
    return train, test


def iter_higgs_chunks(path: str | Path, chunk_size: int = 8192, limit: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream all 28 UCI HIGGS features without loading the 11M rows.

    HIGGS rows contain a label followed by 28 features. The source is Monte
    Carlo physics data, not detector observations.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rows: list[list[float]] = []
    labels: list[int] = []
    count = 0
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith((".csv", ".csv.gz"))]
        if not names:
            raise ValueError("HIGGS archive does not contain a CSV or CSV.GZ file")
        compressed_handle = archive.open(names[0])
        raw_handle = gzip.GzipFile(fileobj=compressed_handle) if names[0].endswith(".gz") else compressed_handle
        with raw_handle:
            for raw_line in raw_handle:
                if limit is not None and count >= limit:
                    break
                values = raw_line.decode("ascii").strip().split(",")
                if len(values) != 29:
                    raise ValueError("HIGGS rows must contain one label and 28 features")
                labels.append(int(float(values[0])))
                rows.append([float(value) for value in values[1:]])
                count += 1
                if len(rows) == chunk_size:
                    yield np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int32)
                    rows, labels = [], []
    if rows:
        yield np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int32)
