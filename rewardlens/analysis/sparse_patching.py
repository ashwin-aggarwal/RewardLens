"""Phase 4.8 (experimental extension): sparse causal activation patching.

Masked replacement of post-ReLU layer-2 units:

    h_patched(s) = h_rec(s) + mask * (h_donor(s') - h_rec(s))

where `mask` is a 0/1 vector over the 128 units and s' is the donor's input
(the current observation with its remaining-time feature clamped, as in the
Phase 4.6/4.7 intervention). k = 0 is the unmodified recipient; k = 128 is
the full-layer patch of Phase 4.7.

Everything runs through the existing `scheduled_patching` hook in "matched"
mode with alpha = 1, so the applied delta is exactly mask * (h_donor - h_rec)
and all Phase 4.6 metrics (categories, path efficiency, applied norm) apply.

Three subset-selection methods are provided:
  A. activation difference  rank units by mean |h_donor - h_rec| on dev states
  B. causal ablation        start from the full patch, restore one unit at a
                            time to its recipient value, rank by the damage
  C. greedy forward         add the unit that most improves dev direct-goal
                            completion, under a fixed evaluation budget
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.analysis.behavior_classification import DEFAULT_VISIT_THRESHOLD
from rewardlens.analysis.steering import (
    PatchController,
    donor_activation,
    evaluate_steering,
    scheduled_patching,
    schedule_constant,
    summarize,
)
from rewardlens.envs.reward_grid import RewardGridConfig

DIM = 128
F = torch.nn.functional


# ---------------------------------------------------------------------------
# Masks and donor functions
# ---------------------------------------------------------------------------

def make_mask(indices, dim: int = DIM) -> np.ndarray:
    m = np.zeros(dim, dtype=np.float32)
    idx = np.asarray(list(indices), dtype=int)
    if len(idx):
        if idx.min() < 0 or idx.max() >= dim:
            raise ValueError("unit index out of range")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("duplicate unit indices")
        m[idx] = 1.0
    return m


def h2_hookfree(net: QNetwork, obs: torch.Tensor) -> torch.Tensor:
    """Post-ReLU layer-2 activation computed without invoking the net's modules
    (safe to call from inside a hook on those modules)."""
    h1 = F.relu(F.linear(obs, net.fc1.weight, net.fc1.bias))
    return F.relu(F.linear(h1, net.fc2.weight, net.fc2.bias))


def make_masked_donor(
    recipient: QNetwork,
    donor: QNetwork,
    mask: np.ndarray,
    donor_obs_transform: Callable | None = None,
    pool_obs: np.ndarray | None = None,
    seed: int = 0,
) -> Callable:
    """Donor function whose matched patch (alpha = 1) applies mask * (h_donor - h_rec).

    With `pool_obs`, the donor activation is taken for a random pool state
    (shuffled-state control) instead of the current one.
    """
    m = torch.as_tensor(mask, dtype=torch.float32)
    rng = np.random.RandomState(seed)
    pool = torch.as_tensor(pool_obs, dtype=torch.float32) if pool_obs is not None else None

    def fn(obs: torch.Tensor) -> torch.Tensor:
        src = pool[rng.randint(0, len(pool), size=obs.shape[0])] if pool is not None else obs
        src = donor_obs_transform(src) if donor_obs_transform is not None else src
        h_rec = h2_hookfree(recipient, obs)
        h_don = donor_activation(donor, src, 2)
        return h_rec + m * (h_don - h_rec)
    return fn


def masked_delta(recipient: QNetwork, donor: QNetwork, mask: np.ndarray, obs: np.ndarray,
                 donor_obs_transform: Callable | None = None) -> np.ndarray:
    """mask * (h_donor(s') - h_rec(s)) for a batch of observations; (N, 128)."""
    ot = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        src = donor_obs_transform(ot) if donor_obs_transform is not None else ot
        d = donor_activation(donor, src, 2) - h2_hookfree(recipient, ot)
    return (d.numpy() * mask[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_mask(
    recipient: QNetwork, donor: QNetwork, positions, mask: np.ndarray,
    donor_obs_transform: Callable | None, env_cfg: RewardGridConfig | None = None,
    threshold: int = DEFAULT_VISIT_THRESHOLD, keep_traces: bool = False,
    pool_obs: np.ndarray | None = None, seed: int = 0,
) -> dict:
    env_cfg = env_cfg or RewardGridConfig()
    fn = make_masked_donor(recipient, donor, mask, donor_obs_transform, pool_obs, seed)
    r = evaluate_steering(recipient, env_cfg, positions, fn, schedule_constant(1.0), 2, "matched",
                          threshold=threshold, keep_traces=keep_traces)
    r["n_patched_units"] = int(mask.sum())
    return r if keep_traces else summarize(r)


def greedy_actions_under_mask(recipient: QNetwork, donor: QNetwork, mask: np.ndarray, obs: np.ndarray,
                              donor_obs_transform: Callable | None) -> np.ndarray:
    fn = make_masked_donor(recipient, donor, mask, donor_obs_transform)
    ot = torch.as_tensor(obs, dtype=torch.float32)
    with scheduled_patching(recipient, fn, schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
        return recipient(ot).argmax(1).numpy()


def random_subsets(k: int, n: int, seed: int, dim: int = DIM) -> list[np.ndarray]:
    """n reproducible random subsets of size k (sorted indices)."""
    rng = np.random.RandomState(seed * 1000 + k)
    return [np.sort(rng.choice(dim, size=k, replace=False)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Method A: activation difference
# ---------------------------------------------------------------------------

def rank_by_activation_difference(recipient: QNetwork, donor: QNetwork, obs_dev: np.ndarray,
                                  donor_obs_transform: Callable | None) -> dict:
    d = masked_delta(recipient, donor, np.ones(DIM, dtype=np.float32), obs_dev, donor_obs_transform)
    score = np.abs(d).mean(0)
    order = np.argsort(-score, kind="stable")
    return {"order": order.tolist(), "score": score.tolist(), "n_states": int(len(obs_dev))}


# ---------------------------------------------------------------------------
# Method B: causal ablation from the full patch
# ---------------------------------------------------------------------------

def ablation_importance(
    recipient: QNetwork, donor: QNetwork, dev_positions, obs_dev: np.ndarray,
    donor_obs_transform: Callable | None, env_cfg: RewardGridConfig | None = None,
    threshold: int = DEFAULT_VISIT_THRESHOLD, group_size: int = 1,
) -> dict:
    """Restore units (one group at a time) from the full patch to their recipient
    values; score each by direct-goal drop, then greedy disagreement with the
    full patch on dev observations, then increase in mean episode steps."""
    env_cfg = env_cfg or RewardGridConfig()
    full = np.ones(DIM, dtype=np.float32)
    base = evaluate_mask(recipient, donor, dev_positions, full, donor_obs_transform, env_cfg, threshold)
    base_actions = greedy_actions_under_mask(recipient, donor, full, obs_dev, donor_obs_transform)
    groups = [list(range(i, min(i + group_size, DIM))) for i in range(0, DIM, group_size)]
    rows = []
    for g in groups:
        m = full.copy()
        m[g] = 0.0
        r = evaluate_mask(recipient, donor, dev_positions, m, donor_obs_transform, env_cfg, threshold)
        acts = greedy_actions_under_mask(recipient, donor, m, obs_dev, donor_obs_transform)
        rows.append({
            "units": g,
            "direct_goal_drop": base["direct_goal_rate"] - r["direct_goal_rate"],
            "greedy_disagreement_with_full": float((acts != base_actions).mean()),
            "steps_increase": r["mean_episode_steps"] - base["mean_episode_steps"],
            "visits_increase": r["mean_reward_tile_visits"] - base["mean_reward_tile_visits"],
            "direct_goal_rate": r["direct_goal_rate"],
        })
    ranked = sorted(range(len(rows)), key=lambda i: (-rows[i]["direct_goal_drop"],
                                                     -rows[i]["greedy_disagreement_with_full"],
                                                     -rows[i]["steps_increase"], i))
    order = [u for i in ranked for u in rows[i]["units"]]
    return {"order": order, "rows": rows, "full_patch_dev": base, "n_evaluations": len(groups),
            "group_size": group_size}


# ---------------------------------------------------------------------------
# Method C: greedy forward selection under a budget
# ---------------------------------------------------------------------------

def greedy_selection(
    recipient: QNetwork, donor: QNetwork, dev_positions, donor_obs_transform: Callable | None,
    candidates, max_k: int, eval_budget: int, env_cfg: RewardGridConfig | None = None,
    threshold: int = DEFAULT_VISIT_THRESHOLD, stop_when_perfect: bool = True,
) -> dict:
    """Add, at each round, the candidate unit that most improves dev direct-goal
    rate (tie-break: fewer tile visits, fewer steps). Stops at max_k, when the
    budget is exhausted, or when the dev direct-goal rate reaches 1.0 and a
    further round would not improve it."""
    env_cfg = env_cfg or RewardGridConfig()
    chosen: list[int] = []
    remaining = list(candidates)
    history = []
    used = 0
    current = evaluate_mask(recipient, donor, dev_positions, make_mask([]), donor_obs_transform, env_cfg, threshold)
    key = lambda r: (r["direct_goal_rate"], -r["mean_reward_tile_visits"], -r["mean_episode_steps"])
    while remaining and len(chosen) < max_k and used < eval_budget:
        best_u, best_r = None, None
        for u in remaining:
            if used >= eval_budget:
                break
            r = evaluate_mask(recipient, donor, dev_positions, make_mask(chosen + [u]), donor_obs_transform,
                              env_cfg, threshold)
            used += 1
            if best_r is None or key(r) > key(best_r):
                best_u, best_r = u, r
        if best_u is None:
            break
        improved = key(best_r) > key(current)
        chosen.append(int(best_u))
        remaining.remove(best_u)
        history.append({"k": len(chosen), "added": int(best_u), "dev": best_r, "improved": bool(improved)})
        current = best_r
        if stop_when_perfect and current["direct_goal_rate"] >= 1.0 and not improved:
            break
    return {"order": chosen, "history": history, "n_evaluations": used, "budget": eval_budget,
            "candidates": [int(c) for c in candidates]}
