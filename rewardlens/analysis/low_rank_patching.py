"""Experimental extension: low-rank activation patching.

Phase 4.7's full-layer patch replaces the recipient's post-ReLU layer-2
activation with the donor's (donor time input clamped to 0.2). Phase 4.8
showed that no small subset of UNITS reproduces it. This module asks the
same question in a different basis: can a few DIRECTIONS do it?

    delta(s)     = h_donor(s') - h_rec(s)
    h_patched(s) = h_rec(s) + P_k(delta(s))

Two projection variants are provided:
  "pca"  centered PCA fitted on dev deltas; P_k(d) = mu + U_k U_k^T (d - mu).
         k = 0 is the mean delta alone (a constant edit, i.e. the constant
         Q-bias control); k = 128 is the full patch.
  "svd"  uncentered SVD; P_k(d) = V_k V_k^T d. k = 0 is the unmodified
         recipient; k = 128 is the full patch.

Random-subspace controls use random orthonormal bases of the same rank,
optionally rescaled per state to the norm of the low-rank projection.
All patches run through the existing Phase 4.6 hook (matched mode, alpha 1),
so the applied delta is exactly P_k(delta(s)) and no weights change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.analysis.behavior_classification import DEFAULT_VISIT_THRESHOLD
from rewardlens.analysis.sparse_patching import h2_hookfree
from rewardlens.analysis.steering import donor_activation, evaluate_steering, schedule_constant, summarize
from rewardlens.envs.reward_grid import RewardGridConfig

DIM = 128


# ---------------------------------------------------------------------------
# Deltas and bases
# ---------------------------------------------------------------------------

def collect_deltas(recipient: QNetwork, donor: QNetwork, obs: np.ndarray,
                   donor_obs_transform: Callable | None = None) -> np.ndarray:
    ot = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        src = donor_obs_transform(ot) if donor_obs_transform is not None else ot
        d = donor_activation(donor, src, 2) - h2_hookfree(recipient, ot)
    return d.numpy().astype(np.float32)


@dataclass
class LowRankBasis:
    components: np.ndarray        # (DIM, DIM) rows = directions, descending importance
    mean: np.ndarray | None       # (DIM,) for the centered variant, None for uncentered
    singular_values: np.ndarray   # (DIM,)
    explained_variance_ratio: np.ndarray  # (DIM,)
    variant: str                  # "pca" | "svd"
    n_fit: int

    def project(self, d: torch.Tensor, k: int) -> torch.Tensor:
        """P_k applied to a batch of deltas (torch)."""
        U = torch.as_tensor(self.components[:k], dtype=torch.float32)  # (k, DIM)
        if self.mean is not None:
            mu = torch.as_tensor(self.mean, dtype=torch.float32)
            c = d - mu
            return mu + (c @ U.T) @ U if k > 0 else mu.expand_as(d)
        return (d @ U.T) @ U if k > 0 else torch.zeros_like(d)

    def project_np(self, d: np.ndarray, k: int) -> np.ndarray:
        return self.project(torch.as_tensor(d, dtype=torch.float32), k).numpy()


def fit_basis(deltas: np.ndarray, variant: str = "pca") -> LowRankBasis:
    """Fit a full orthonormal basis ordered by variance (pca, centered) or by
    energy (svd, uncentered) on the dev deltas."""
    X = np.asarray(deltas, dtype=np.float64)
    if variant == "pca":
        mu = X.mean(0)
        Xc = X - mu
    elif variant == "svd":
        mu = None
        Xc = X
    else:
        raise ValueError("variant must be 'pca' or 'svd'")
    _, s, vt = np.linalg.svd(Xc, full_matrices=True)
    s_full = np.zeros(DIM)
    s_full[:len(s)] = s
    ev = s_full ** 2
    ratio = ev / ev.sum() if ev.sum() > 0 else ev
    return LowRankBasis(components=vt[:DIM].astype(np.float32),
                        mean=None if mu is None else mu.astype(np.float32),
                        singular_values=s_full.astype(np.float32),
                        explained_variance_ratio=ratio.astype(np.float32),
                        variant=variant, n_fit=len(X))


def reconstruction_error(basis: LowRankBasis, deltas: np.ndarray, k: int) -> dict:
    """Relative squared reconstruction error of P_k on a delta set, plus norms."""
    d = np.asarray(deltas, dtype=np.float32)
    p = basis.project_np(d, k)
    num = float(((d - p) ** 2).sum())
    den = float((d ** 2).sum())
    return {"relative_error": num / den if den > 0 else 0.0,
            "mean_projection_norm": float(np.linalg.norm(p, axis=1).mean()),
            "mean_delta_norm": float(np.linalg.norm(d, axis=1).mean())}


def random_orthonormal_bases(k: int, n: int, seed: int, dim: int = DIM) -> list[np.ndarray]:
    """n reproducible random orthonormal bases of rank k; each (k, dim)."""
    rng = np.random.RandomState(seed * 1000 + k)
    out = []
    for _ in range(n):
        q, _ = np.linalg.qr(rng.randn(dim, k))
        out.append(q.T.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Donor functions for the hook (matched mode, alpha 1)
# ---------------------------------------------------------------------------

def make_low_rank_donor(
    recipient: QNetwork, donor: QNetwork, basis: LowRankBasis, k: int,
    donor_obs_transform: Callable | None = None,
    pool_obs: np.ndarray | None = None, seed: int = 0,
) -> Callable:
    """Target activation h_rec + P_k(delta). With `pool_obs`, delta uses the
    donor activation of a random pool state (shuffled-state control)."""
    rng = np.random.RandomState(seed)
    pool = torch.as_tensor(pool_obs, dtype=torch.float32) if pool_obs is not None else None

    def fn(obs: torch.Tensor) -> torch.Tensor:
        src = pool[rng.randint(0, len(pool), size=obs.shape[0])] if pool is not None else obs
        src = donor_obs_transform(src) if donor_obs_transform is not None else src
        h_rec = h2_hookfree(recipient, obs)
        d = donor_activation(donor, src, 2) - h_rec
        return h_rec + basis.project(d, k)
    return fn


def make_subspace_donor(
    recipient: QNetwork, donor: QNetwork, components: np.ndarray, mean: np.ndarray | None,
    donor_obs_transform: Callable | None = None,
    norm_match_basis: LowRankBasis | None = None, norm_match_k: int = 0,
) -> Callable:
    """Random-subspace control: h_rec + [mu +] R R^T (delta [- mu]).

    With `norm_match_basis`, the subspace projection is rescaled per state to
    the norm of that basis's rank-k projection of the same delta, so the
    control differs from the low-rank patch only in direction."""
    R = torch.as_tensor(components, dtype=torch.float32)
    mu = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)

    def fn(obs: torch.Tensor) -> torch.Tensor:
        src = donor_obs_transform(obs) if donor_obs_transform is not None else obs
        h_rec = h2_hookfree(recipient, obs)
        d = donor_activation(donor, src, 2) - h_rec
        c = d - mu if mu is not None else d
        p = (c @ R.T) @ R
        if norm_match_basis is not None:
            target = norm_match_basis.project(d, norm_match_k)
            tn = torch.linalg.norm(target - (mu if mu is not None else 0), dim=1, keepdim=True)
            pn = torch.linalg.norm(p, dim=1, keepdim=True) + 1e-8
            p = p * (tn / pn)
        if mu is not None:
            p = p + mu
        return h_rec + p
    return fn


def evaluate_donor_fn(recipient: QNetwork, positions, donor_fn: Callable | None,
                      env_cfg: RewardGridConfig | None = None, threshold: int = DEFAULT_VISIT_THRESHOLD,
                      keep_traces: bool = False, mode: str = "matched", constant_delta=None) -> dict:
    env_cfg = env_cfg or RewardGridConfig()
    r = evaluate_steering(recipient, env_cfg, positions, donor_fn, schedule_constant(1.0), 2,
                          mode if donor_fn is not None or mode == "constant" else "none",
                          constant_delta=constant_delta, threshold=threshold, keep_traces=keep_traces)
    return r if keep_traces else summarize(r)
