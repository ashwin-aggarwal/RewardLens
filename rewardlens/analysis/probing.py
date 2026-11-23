"""Linear probing: fit logistic regression on hidden activations vs behavioral targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)


@dataclass
class ProbeResult:
    name: str
    auroc: float | None       # None if single-class in test set
    balanced_accuracy: float
    confusion: np.ndarray     # 2x2
    class_counts: dict[int, int]
    n_train: int
    n_test: int
    train_prevalence: float   # fraction of positive class in train
    test_prevalence: float


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "probe",
    max_iter: int = 2000,
    C: float = 1.0,
    seed: int = 42,
) -> ProbeResult:
    """Fit a logistic regression probe and evaluate on held-out data."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, max_iter=max_iter, random_state=seed, solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    unique_test = set(y_test)
    if len(unique_test) >= 2:
        y_prob = clf.predict_proba(X_test_s)
        if y_prob.shape[1] == 2:
            auroc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auroc = None
    else:
        auroc = None

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    unique, counts = np.unique(y_test, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    train_prev = float(y_train.mean())
    test_prev = float(y_test.mean())

    return ProbeResult(
        name=name,
        auroc=auroc,
        balanced_accuracy=bal_acc,
        confusion=cm,
        class_counts=class_counts,
        n_train=len(y_train),
        n_test=len(y_test),
        train_prevalence=train_prev,
        test_prevalence=test_prev,
    )


def build_observable_baselines(
    metadata: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Construct baseline feature sets from observable state information."""
    n = len(metadata["agent_x"])
    position_features = np.column_stack([
        metadata["agent_x"],
        metadata["agent_y"],
        metadata["time_remaining"],
    ])
    distance_features = np.column_stack([
        metadata["dist_to_goal"],
        metadata["dist_to_reward"],
    ])
    combined_observable = np.column_stack([
        position_features,
        distance_features,
    ])
    return {
        "position_time": position_features,
        "distances": distance_features,
        "combined_observable": combined_observable,
    }


def spatial_train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    grid_width: int = 8,
    grid_height: int = 8,
    test_fraction: float = 0.25,
    seed: int = 42,
    observations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by spatial position to avoid leakage from correlated time steps.

    Holds out entire grid positions, not individual time-step records,
    to prevent the probe from memorizing position-specific patterns
    at training positions.

    If `observations` is provided, positions are extracted from its first
    two columns (the normalized agent coordinates). Otherwise, positions
    are extracted from `features` columns 0 and 1 — only valid when
    `features` is the raw observation array.
    """
    rng = np.random.RandomState(seed)

    pos_source = observations if observations is not None else features
    ax = (pos_source[:, 0] * grid_width).round().astype(int)
    ay = (pos_source[:, 1] * grid_height).round().astype(int)
    positions = set(zip(ax.tolist(), ay.tolist()))
    positions = sorted(positions)
    rng.shuffle(positions)  # type: ignore[arg-type]

    n_test = max(1, int(len(positions) * test_fraction))
    test_positions = set(positions[:n_test])

    test_mask = np.array([
        (int(ax[i]), int(ay[i])) in test_positions for i in range(len(ax))
    ])
    train_mask = ~test_mask

    return (
        features[train_mask],
        labels[train_mask],
        features[test_mask],
        labels[test_mask],
    )


def check_label_quality(labels: np.ndarray) -> dict[str, float | int | bool]:
    """Basic checks on label distribution and viability."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    if total == 0:
        return {"n_classes": 0, "total": 0, "positive_count": 0,
                "negative_count": 0, "positive_fraction": 0.0, "viable": False}
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    return {
        "n_classes": int(len(unique)),
        "total": int(total),
        "positive_count": int(count_map.get(1, 0)),
        "negative_count": int(count_map.get(0, total)),
        "positive_fraction": float(count_map.get(1, 0) / total),
        "viable": bool(len(unique) >= 2 and min(counts) >= 5),
    }
