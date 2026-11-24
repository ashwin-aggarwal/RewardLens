"""Experiment 2, Phase 2: activations on matched states and representation comparisons.

All comparisons here are correlational. Similar representations do not show
that a representation causes a behavior; the patching module tests that
separately.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge

from rewardlens.analysis.representation_analysis import centered_kernel_alignment
from rewardlens.exp2.evaluation import observations
from rewardlens.exp2.layouts import Layout, layouts_to_array
from rewardlens.exp2.procedural_env import ProceduralEnvParams

LAYERS = ("layer1", "layer2")


def sample_matched_states(layouts: list[Layout], states_per_layout: int, seed: int,
                          params: ProceduralEnvParams) -> dict:
    """Observations that do not depend on any policy.

    For each layout, draws `states_per_layout` agent positions (any cell except
    the goal) and elapsed-step counts uniformly. Every network is then run on
    exactly these observations.
    """
    rng = np.random.RandomState(seed)
    lay = np.repeat(layouts_to_array(layouts), states_per_layout, axis=0)
    n = len(lay)
    pos = np.empty((n, 2), dtype=np.int64)
    todo = np.ones(n, dtype=bool)
    while todo.any():
        k = int(todo.sum())
        cand = np.stack([rng.randint(0, params.width, k), rng.randint(0, params.height, k)], axis=1)
        pos[todo] = cand
        todo[todo] = (cand == lay[todo, 2:4]).all(axis=1)
    steps = rng.randint(0, params.max_steps, n)
    return {"obs": observations(pos, lay, steps, params), "layouts": lay, "pos": pos, "steps": steps,
            "layout_index": np.repeat(np.arange(len(layouts)), states_per_layout)}


def forward_all(net: torch.nn.Module, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Post-ReLU layer-1 and layer-2 activations and Q-values, without hooks."""
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        h1 = torch.relu(net.fc1(x))
        h2 = torch.relu(net.fc2(h1))
        q = net.fc3(h2)
    return {"layer1": h1.numpy().astype(np.float64), "layer2": h2.numpy().astype(np.float64),
            "q": q.numpy().astype(np.float64)}


def state_features(states: dict, params: ProceduralEnvParams) -> tuple[np.ndarray, list[str]]:
    """Human-readable descriptors of each state, used to ask what a direction encodes."""
    pos, lay = states["pos"].astype(float), states["layouts"].astype(float)
    dgx, dgy = lay[:, 2] - pos[:, 0], lay[:, 3] - pos[:, 1]
    dtx, dty = lay[:, 4] - pos[:, 0], lay[:, 5] - pos[:, 1]
    time_left = (params.max_steps - states["steps"]) / params.max_steps
    feats = {
        "agent_x": pos[:, 0], "agent_y": pos[:, 1], "goal_x": lay[:, 2], "goal_y": lay[:, 3],
        "tile_x": lay[:, 4], "tile_y": lay[:, 5], "time_left": time_left,
        "goal_dx": dgx, "goal_dy": dgy, "tile_dx": dtx, "tile_dy": dty,
        "goal_distance": np.abs(dgx) + np.abs(dgy), "tile_distance": np.abs(dtx) + np.abs(dty),
        "on_tile": ((dtx == 0) & (dty == 0)).astype(float),
    }
    return np.stack(list(feats.values()), axis=1), list(feats)


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    return float(centered_kernel_alignment(a, b))


def relative_change(reference: np.ndarray, other: np.ndarray) -> float:
    """Mean per-state distance between two activation sets, relative to the reference norm."""
    return float(np.linalg.norm(other - reference, axis=1).mean() / np.linalg.norm(reference, axis=1).mean())


class DifferencePCA:
    """PCA of matched-state activation differences, fitted on one state set and scored on another."""

    def __init__(self, diff_fit: np.ndarray) -> None:
        self.mean = diff_fit.mean(axis=0)
        _, s, vt = np.linalg.svd(diff_fit - self.mean, full_matrices=False)
        self.components = vt
        self.fit_variance = s ** 2 / (len(diff_fit) - 1)

    def scores(self, diff: np.ndarray, k: int) -> np.ndarray:
        return (diff - self.mean) @ self.components[:k].T

    def summary(self, diff_eval: np.ndarray, top_k: int, threshold: float) -> dict:
        centered = diff_eval - self.mean
        var = (centered @ self.components.T).var(axis=0)
        total = centered.var(axis=0).sum()
        ratio = var / total
        cum = np.cumsum(ratio)
        energy = (diff_eval ** 2).sum(axis=1).mean()
        return {
            "explained_variance_ratio": [float(v) for v in ratio[:32]],
            "cumulative_top_k": float(cum[top_k - 1]),
            "n_components_for_threshold": int(np.searchsorted(cum, threshold) + 1),
            "participation_ratio": float(var.sum() ** 2 / (var ** 2).sum()),
            # Share of the difference's squared norm that is one constant offset, the same in every state.
            "constant_offset_share": float((self.mean ** 2).sum() / energy),
            "mean_difference_norm": float(np.linalg.norm(diff_eval, axis=1).mean()),
        }


def held_out_r2(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray,
                alpha: float = 1e-3) -> float:
    """Variance-weighted R² of a linear map from x to y, fitted on one state set and scored on another.

    This asks whether y is a linear function of x across states. It needs no
    correspondence between the neurons of two networks.
    """
    mu, sd = x_fit.mean(axis=0), x_fit.std(axis=0) + 1e-8
    model = Ridge(alpha=alpha).fit((x_fit - mu) / sd, y_fit)
    resid = y_eval - model.predict((x_eval - mu) / sd)
    return float(1.0 - (resid ** 2).sum() / ((y_eval - y_fit.mean(axis=0)) ** 2).sum())


def top_pc_scores(x_fit: np.ndarray, x_eval: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores on the top-k principal components of `x_fit`, for both sets."""
    pca = DifferencePCA(x_fit)
    return pca.scores(x_fit, k), pca.scores(x_eval, k)
