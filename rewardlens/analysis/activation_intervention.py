"""Causal activation interventions: edit post-ReLU hidden-layer-2 activations at inference.

No network weights are modified. Edits are applied through a temporary forward
hook on the `relu2` module and removed afterward. Directions are always defined
in the raw (unstandardized) activation coordinates of the specific network
being intervened on.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rewardlens.agents.dqn import QNetwork
from rewardlens.envs.reward_grid import RewardGridEnv, RewardGridConfig


# ---------------------------------------------------------------------------
# Direction construction
# ---------------------------------------------------------------------------

@dataclass
class ProbeDirection:
    """A unit direction in raw layer-2 activation space, with provenance."""
    vector: np.ndarray            # (128,), unit L2 norm, raw coordinates
    source: str                   # "probe" | "contrastive" | "random"
    checkpoint: str               # checkpoint path the direction was fitted on
    sign_verified: bool           # +vector scores higher on farming states
    fit_n_states: int
    fit_positive_fraction: float


def fit_probe_direction(
    hidden: np.ndarray,
    labels: np.ndarray,
    checkpoint: str,
    C: float = 1.0,
    seed: int = 42,
) -> ProbeDirection:
    """Fit a logistic probe and return its coefficient as a raw-space unit direction.

    The probe is fitted on standardized features; coefficients are transformed
    back to raw coordinates via w_raw = w_std / scaler.scale_ so that
    w_raw . h differs from the standardized decision function only by a constant.
    The sign convention is: +vector points toward higher predicted farming.
    """
    scaler = StandardScaler().fit(hidden)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")
    clf.fit(scaler.transform(hidden), labels)

    w_raw = clf.coef_[0] / scaler.scale_
    v = w_raw / np.linalg.norm(w_raw)

    scores = hidden @ v
    sign_ok = bool(scores[labels == 1].mean() > scores[labels == 0].mean())
    if not sign_ok:
        v = -v
        sign_ok = True  # flipped to enforce the convention

    return ProbeDirection(
        vector=v.astype(np.float32),
        source="probe",
        checkpoint=checkpoint,
        sign_verified=sign_ok,
        fit_n_states=len(labels),
        fit_positive_fraction=float(labels.mean()),
    )


def fit_contrastive_direction(
    hidden: np.ndarray,
    labels: np.ndarray,
    checkpoint: str,
) -> ProbeDirection:
    """Mean-difference direction: mean(h | farming) - mean(h | non-farming)."""
    diff = hidden[labels == 1].mean(axis=0) - hidden[labels == 0].mean(axis=0)
    v = diff / np.linalg.norm(diff)
    return ProbeDirection(
        vector=v.astype(np.float32),
        source="contrastive",
        checkpoint=checkpoint,
        sign_verified=True,  # true by construction
        fit_n_states=len(labels),
        fit_positive_fraction=float(labels.mean()),
    )


def random_unit_directions(dim: int, n: int, seed: int = 0) -> np.ndarray:
    """n random unit vectors in R^dim with a fixed seed. Returns (n, dim)."""
    rng = np.random.RandomState(seed)
    v = rng.randn(n, dim)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Hook-based intervention
# ---------------------------------------------------------------------------

@dataclass
class InterventionStats:
    """Measured effects of the edit on the activations it touched."""
    n_calls: int = 0
    applied_norms: list = field(default_factory=list)       # L2 norm of the edit per call
    negative_fraction: list = field(default_factory=list)   # fraction of units pushed below 0

    def summary(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "mean_applied_norm": float(np.mean(self.applied_norms)) if self.applied_norms else 0.0,
            "mean_negative_fraction": float(np.mean(self.negative_fraction)) if self.negative_fraction else 0.0,
        }


@contextmanager
def layer2_intervention(
    net: QNetwork,
    direction: np.ndarray | None,
    alpha: float,
    mode: str = "subtract",
    stats: InterventionStats | None = None,
):
    """Temporarily edit post-ReLU layer-2 activations during forward passes.

    Modes:
      "subtract":    h' = h - alpha * v
      "add":         h' = h + alpha * v
      "project_out": h' = h - (h . v) v        (alpha ignored)
      "none":        h' = h                    (no hook registered)

    Edits may push activations negative (outside the natural post-ReLU
    distribution). This is intentional and measured via `stats`, not clipped.
    """
    if mode == "none" or direction is None or (alpha == 0.0 and mode != "project_out"):
        yield
        return

    v = torch.as_tensor(direction, dtype=torch.float32)

    def hook(module, inputs, output):
        if mode == "subtract":
            edited = output - alpha * v
        elif mode == "add":
            edited = output + alpha * v
        elif mode == "project_out":
            coeff = output @ v
            edited = output - coeff.unsqueeze(-1) * v
        else:
            raise ValueError(f"Unknown intervention mode: {mode}")
        if stats is not None:
            stats.n_calls += 1
            delta = (edited - output).reshape(-1, output.shape[-1])
            stats.applied_norms.append(float(torch.linalg.norm(delta, dim=-1).mean()))
            stats.negative_fraction.append(float((edited < 0).float().mean()))
        return edited

    handle = net.relu2.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Intervened greedy rollout
# ---------------------------------------------------------------------------

def greedy_rollout_with_intervention(
    net: QNetwork,
    env_config: RewardGridConfig,
    direction: np.ndarray | None = None,
    alpha: float = 0.0,
    mode: str = "none",
    seed: int = 0,
) -> dict:
    """Roll out the greedy policy for one full episode with an optional edit active.

    Returns trajectory-level behavioral metrics plus the action sequence and
    the measured perturbation stats. Deterministic given (net, config, seed).
    """
    env = RewardGridEnv(config=env_config)
    obs, info = env.reset(seed=seed)
    net.eval()

    stats = InterventionStats()
    actions: list[int] = []
    positions = [tuple(info["agent_pos"])]
    total_reward = 0.0
    reward_tile_visits = 0
    terminated = truncated = False
    no_move_steps = 0

    with layer2_intervention(net, direction, alpha, mode, stats), torch.no_grad():
        while not (terminated or truncated):
            q = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = int(q.argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            new_pos = tuple(info["agent_pos"])
            if new_pos == positions[-1]:
                no_move_steps += 1
            positions.append(new_pos)
            total_reward += reward
            reward_tile_visits = info["reward_tile_visits"]

    goal_reached = bool(terminated)
    episode_steps = len(actions)
    is_hacking = bool(
        reward_tile_visits >= env_config.exploit_visit_threshold
        and not goal_reached
        and (terminated or truncated)
    )
    unique_positions = len(set(positions))

    return {
        "goal_reached": goal_reached,
        "is_reward_hacking": is_hacking,
        "reward_tile_visits": int(reward_tile_visits),
        "proxy_reward": round(float(total_reward), 4),
        "episode_steps": episode_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "unique_positions": unique_positions,
        "no_move_steps": no_move_steps,
        "actions": actions,
        "start_pos": list(env_config.agent_start),
        "intervention": {
            "mode": mode,
            "alpha": alpha,
            **stats.summary(),
        },
    }


def evaluate_on_start_states(
    net: QNetwork,
    base_config: RewardGridConfig,
    start_positions: list[tuple[int, int]],
    direction: np.ndarray | None = None,
    alpha: float = 0.0,
    mode: str = "none",
) -> dict:
    """Run one greedy episode from each start position and aggregate metrics.

    The environment and policy are deterministic, so one rollout per distinct
    start state is exhaustive; rates are fractions over distinct start states,
    not samples from a stochastic process.
    """
    episodes = []
    for pos in start_positions:
        cfg = RewardGridConfig(
            width=base_config.width,
            height=base_config.height,
            agent_start=tuple(pos),
            goal_pos=base_config.goal_pos,
            reward_pos=base_config.reward_pos,
            max_steps=base_config.max_steps,
            step_penalty=base_config.step_penalty,
            reward_tile_bonus=base_config.reward_tile_bonus,
            goal_bonus=base_config.goal_bonus,
            exploit_visit_threshold=base_config.exploit_visit_threshold,
        )
        episodes.append(
            greedy_rollout_with_intervention(net, cfg, direction, alpha, mode)
        )

    n = len(episodes)
    return {
        "n_start_states": n,
        "goal_rate": sum(e["goal_reached"] for e in episodes) / n,
        "hacking_rate": sum(e["is_reward_hacking"] for e in episodes) / n,
        "mean_reward_tile_visits": sum(e["reward_tile_visits"] for e in episodes) / n,
        "mean_proxy_reward": sum(e["proxy_reward"] for e in episodes) / n,
        "mean_episode_steps": sum(e["episode_steps"] for e in episodes) / n,
        "mean_no_move_steps": sum(e["no_move_steps"] for e in episodes) / n,
        "mean_unique_positions": sum(e["unique_positions"] for e in episodes) / n,
        "mean_applied_norm": float(np.mean([
            e["intervention"]["mean_applied_norm"] for e in episodes
        ])),
        "mean_negative_fraction": float(np.mean([
            e["intervention"]["mean_negative_fraction"] for e in episodes
        ])),
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# State partitioning
# ---------------------------------------------------------------------------

def partition_start_positions(
    config: RewardGridConfig,
    n_dev: int = 16,
    n_test: int = 16,
    seed: int = 7,
) -> dict[str, list[tuple[int, int]]]:
    """Fixed train/dev/test partition of valid start positions.

    Excludes the goal and reward tiles. Train positions are used for direction
    fitting (via the matched-state probe dataset restricted to them), dev for
    strength selection, test for final held-out evaluation.
    """
    all_positions = [
        (x, y)
        for x in range(config.width)
        for y in range(config.height)
        if (x, y) != tuple(config.goal_pos) and (x, y) != tuple(config.reward_pos)
    ]
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(all_positions))
    shuffled = [all_positions[i] for i in order]
    return {
        "test": shuffled[:n_test],
        "dev": shuffled[n_test:n_test + n_dev],
        "train": shuffled[n_test + n_dev:],
    }
