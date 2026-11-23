"""Phase 4.6: does goal preference depend on remaining episode time?

Phase 4.5's patched agent farmed for ~90 steps and then reached the goal at
step 95-96. One hypothesis is that the patch makes the goal the preferred
target only when little time remains. This module tests that directly with
Q-values rather than trajectory timing:

  * the SAME grid positions are evaluated at many remaining-time values, so
    time is isolated from agent location;
  * for each (position, time) we record the greedy action, whether it moves
    the agent closer to the goal and/or the reward tile, and the Q-value gap
    between the best goal-approaching and best reward-approaching action;
  * the patched Q-values are decomposed exactly into their two affine parts,
    Q_patched = (1 - alpha) * Q_recipient + alpha * fc3_rec(h_donor),
    which shows WHICH part changes with time.

A complementary input-clamp diagnostic (`clamp_time`) feeds the network a
fixed remaining-time value while the environment runs normally.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.envs.reward_grid import ACTION_DELTAS, RewardGridConfig


def observations_for_positions(
    positions: list, time_fractions: list, config: RewardGridConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observations for every (position, time) pair; returns (obs, pos_index, time)."""
    c = config or RewardGridConfig()
    nx, ny = c.width - 1, c.height - 1
    obs, pidx, tvals = [], [], []
    for i, (x, y) in enumerate(positions):
        for t in time_fractions:
            obs.append([x / nx, y / ny, c.goal_pos[0] / nx, c.goal_pos[1] / ny,
                        c.reward_pos[0] / nx, c.reward_pos[1] / ny, float(t)])
            pidx.append(i)
            tvals.append(float(t))
    return (np.asarray(obs, dtype=np.float32), np.asarray(pidx), np.asarray(tvals, dtype=np.float32))


def action_geometry(position, config: RewardGridConfig | None = None) -> dict:
    """For each action: does it move closer to the goal / the reward tile?"""
    c = config or RewardGridConfig()
    x, y = position
    gd = abs(x - c.goal_pos[0]) + abs(y - c.goal_pos[1])
    rd = abs(x - c.reward_pos[0]) + abs(y - c.reward_pos[1])
    out = {}
    for a, (dx, dy) in ACTION_DELTAS.items():
        nx_, ny_ = x + dx, y + dy
        if not (0 <= nx_ < c.width and 0 <= ny_ < c.height):
            nx_, ny_ = x, y
        out[a] = {
            "goal_closer": abs(nx_ - c.goal_pos[0]) + abs(ny_ - c.goal_pos[1]) < gd,
            "reward_closer": abs(nx_ - c.reward_pos[0]) + abs(ny_ - c.reward_pos[1]) < rd,
            "no_move": (nx_, ny_) == (x, y),
        }
    return out


def goal_preference_by_time(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    positions: list,
    time_fractions: list,
    config: RewardGridConfig | None = None,
) -> dict:
    """Greedy-action geometry and goal-vs-reward Q gaps at matched positions across time.

    `forward_fn` maps an observation batch to Q-values (any patch should be
    active inside it). For each time value we report:
      goal_approach_fraction   greedy action moves closer to the goal
      reward_approach_fraction greedy action moves closer to the reward tile
                               and NOT closer to the goal (farming-type move)
      mean_goal_minus_reward_q max_a in Goal Q(a) - max_a in Reward Q(a)
                               (Goal = goal-closer actions; Reward = reward-
                               closer-not-goal-closer actions), averaged over
                               positions where both sets are non-empty
      mean_max_q               mean of max_a Q(a)
    """
    c = config or RewardGridConfig()
    obs, pidx, tvals = observations_for_positions(positions, time_fractions, c)
    with torch.no_grad():
        q = forward_fn(torch.as_tensor(obs)).numpy()
    geoms = [action_geometry(p, c) for p in positions]
    greedy = q.argmax(1)

    per_time = {k: [] for k in ("goal_approach_fraction", "reward_approach_fraction",
                                "mean_goal_minus_reward_q", "mean_max_q", "n_gap_positions")}
    gap_matrix = np.full((len(positions), len(time_fractions)), np.nan, dtype=np.float32)
    greedy_goal_matrix = np.zeros((len(positions), len(time_fractions)), dtype=bool)
    greedy_matrix = np.zeros((len(positions), len(time_fractions)), dtype=int)
    q_matrix = np.zeros((len(positions), len(time_fractions), 4), dtype=np.float32)
    for j, t in enumerate(time_fractions):
        sel = np.where(np.isclose(tvals, float(t)))[0]
        ga, ra, gaps, qmax = [], [], [], []
        for k in sel:
            i = int(pidx[k])
            g = geoms[i]
            a = int(greedy[k])
            ga.append(g[a]["goal_closer"])
            ra.append(g[a]["reward_closer"] and not g[a]["goal_closer"])
            greedy_goal_matrix[i, j] = g[a]["goal_closer"]
            greedy_matrix[i, j] = a
            q_matrix[i, j] = q[k]
            goal_set = [x for x in range(4) if g[x]["goal_closer"]]
            rew_set = [x for x in range(4) if g[x]["reward_closer"] and not g[x]["goal_closer"]]
            qmax.append(float(q[k].max()))
            if goal_set and rew_set:
                gap = float(q[k, goal_set].max() - q[k, rew_set].max())
                gaps.append(gap)
                gap_matrix[i, j] = gap
        per_time["goal_approach_fraction"].append(float(np.mean(ga)))
        per_time["reward_approach_fraction"].append(float(np.mean(ra)))
        per_time["mean_goal_minus_reward_q"].append(float(np.mean(gaps)) if gaps else None)
        per_time["mean_max_q"].append(float(np.mean(qmax)))
        per_time["n_gap_positions"].append(len(gaps))
    return {
        "time_fractions": [float(t) for t in time_fractions],
        "positions": [list(p) for p in positions],
        **per_time,
        "gap_matrix": gap_matrix.tolist(),
        "greedy_goal_matrix": greedy_goal_matrix.tolist(),
        "greedy_matrix": greedy_matrix.tolist(),
        "q_matrix": q_matrix.tolist(),
    }


def compare_to_reference(analysis: dict, reference: dict) -> dict:
    """Per-time greedy flip rate and mean |dQ| of `analysis` relative to `reference`
    (same positions and time fractions), plus the action histogram per time."""
    g = np.asarray(analysis["greedy_matrix"])
    r = np.asarray(reference["greedy_matrix"])
    qa = np.asarray(analysis["q_matrix"])
    qr = np.asarray(reference["q_matrix"])
    n_t = g.shape[1]
    return {
        "greedy_flip_rate": [float((g[:, j] != r[:, j]).mean()) for j in range(n_t)],
        "mean_abs_dq": [float(np.abs(qa[:, j] - qr[:, j]).mean()) for j in range(n_t)],
        "action_fractions": [[float((g[:, j] == a).mean()) for a in range(4)] for j in range(n_t)],
        "dominant_action_fraction": [float(max((g[:, j] == a).mean() for a in range(4))) for j in range(n_t)],
    }


def patched_q_components(
    recipient: QNetwork, donor: QNetwork, obs: torch.Tensor, alpha: float,
) -> dict:
    """Exact affine decomposition of a layer-2 matched patch:

        Q_patched = (1 - alpha) * Q_rec + alpha * fc3_rec(h2_donor)

    Returns all three plus the donor's own Q-values for reference.
    """
    with torch.no_grad():
        h2_rec = recipient.relu2(recipient.fc2(recipient.relu1(recipient.fc1(obs))))
        h2_don = donor.relu2(donor.fc2(donor.relu1(donor.fc1(obs))))
        q_rec = recipient.fc3(h2_rec)
        q_don_via_rec = recipient.fc3(h2_don)
        q_patched = recipient.fc3(h2_rec + alpha * (h2_don - h2_rec))
        q_donor = donor.fc3(h2_don)
    return {"q_recipient": q_rec, "q_donor_via_recipient_head": q_don_via_rec,
            "q_patched": q_patched, "q_donor_own_head": q_donor,
            "q_affine_reconstruction": (1 - alpha) * q_rec + alpha * q_don_via_rec}


def clamp_time(value: float) -> Callable[[np.ndarray], np.ndarray]:
    """Observation transform that fixes the remaining-time feature the network sees."""
    def fn(obs: np.ndarray) -> np.ndarray:
        o = np.array(obs, dtype=np.float32, copy=True)
        o[6] = float(value)
        return o
    return fn


def earliest_goal_preferring_time(analysis: dict, min_fraction: float = 0.5) -> float | None:
    """Largest remaining-time value at which goal-approach fraction >= min_fraction,
    scanning from high remaining time to low. None if never reached."""
    ts = analysis["time_fractions"]
    fr = analysis["goal_approach_fraction"]
    order = sorted(range(len(ts)), key=lambda i: -ts[i])
    for i in order:
        if fr[i] >= min_fraction:
            return float(ts[i])
    return None
