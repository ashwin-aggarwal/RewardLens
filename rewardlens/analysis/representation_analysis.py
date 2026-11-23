"""Representation analysis: within-run checkpoint comparison and descriptive statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from rewardlens.analysis.activation_extraction import (
    ActivationRecord,
    extract_activations,
)


def activation_statistics(record: ActivationRecord) -> dict[str, dict[str, float]]:
    """Per-layer descriptive statistics."""
    stats = {}
    for name, arr in [("hidden1", record.hidden1), ("hidden2", record.hidden2)]:
        active_frac = (arr > 0).mean()
        stats[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "sparsity": float(1.0 - active_frac),
            "active_fraction": float(active_frac),
            "l2_norm_mean": float(np.linalg.norm(arr, axis=1).mean()),
        }
    return stats


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two matrices of the same number of rows."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return (A_norm * B_norm).sum(axis=1)


def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices (N x d1, N x d2).

    Measures representational similarity that is invariant to orthogonal
    transformation and isotropic scaling, appropriate for comparing
    representations across checkpoints within the same training run.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def fit_pca_projection(
    activations: np.ndarray,
    n_components: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, PCA]:
    """Fit PCA and return 2D projections."""
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(activations)
    return projected, pca


def project_with_fitted_pca(
    activations: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Project new activations using an already-fitted PCA."""
    return pca.transform(activations)


def extract_checkpoint_sequence(
    checkpoint_dir: Path,
    observations: np.ndarray,
    seed: int,
    condition: str,
    device=None,
) -> list[ActivationRecord]:
    """Load all checkpoints in order and extract activations from each."""
    records = []
    ckpt_files = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    for ckpt_path in ckpt_files:
        name = ckpt_path.stem
        if name == "checkpoint_final":
            step = -1
        else:
            step = int(name.split("_")[1])
        rec = extract_activations(
            ckpt_path, observations, seed, condition, step, device,
        )
        if step == -1:
            rec.step = 100000
        records.append(rec)
    records.sort(key=lambda r: r.step)
    return records


def track_representation_evolution(
    records: list[ActivationRecord],
    reference_idx: int = 0,
) -> dict[str, list]:
    """Track how representations change across a checkpoint sequence.

    Uses the first record as reference for CKA and cosine similarity.
    """
    steps = []
    cka_h1 = []
    cka_h2 = []
    sparsity_h1 = []
    sparsity_h2 = []
    l2_h1 = []
    l2_h2 = []
    greedy_action_entropy = []

    ref = records[reference_idx]
    for rec in records:
        steps.append(rec.step)
        cka_h1.append(centered_kernel_alignment(ref.hidden1, rec.hidden1))
        cka_h2.append(centered_kernel_alignment(ref.hidden2, rec.hidden2))

        stats = activation_statistics(rec)
        sparsity_h1.append(stats["hidden1"]["sparsity"])
        sparsity_h2.append(stats["hidden2"]["sparsity"])
        l2_h1.append(stats["hidden1"]["l2_norm_mean"])
        l2_h2.append(stats["hidden2"]["l2_norm_mean"])

        _, counts = np.unique(rec.greedy_actions, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        greedy_action_entropy.append(float(entropy))

    return {
        "steps": steps,
        "cka_h1_vs_first": cka_h1,
        "cka_h2_vs_first": cka_h2,
        "sparsity_h1": sparsity_h1,
        "sparsity_h2": sparsity_h2,
        "l2_norm_h1": l2_h1,
        "l2_norm_h2": l2_h2,
        "greedy_action_entropy": greedy_action_entropy,
    }
