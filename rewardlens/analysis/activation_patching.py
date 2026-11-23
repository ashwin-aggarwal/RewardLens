"""Goal-directed activation patching: inject state-dependent layer-2 activations
from a donor network into a recipient network at inference time.

    h_patched(s) = h_recipient(s) + alpha * (h_donor'(s) - h_recipient(s))

where h_donor'(s) is the donor's post-ReLU layer-2 activation for the SAME
observation s, optionally passed through a linear mapping M into the
recipient's coordinates. The recipient's own output layer (fc3) is always used.
No weights are modified; edits run through temporary hooks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import Ridge

from rewardlens.agents.dqn import QNetwork
from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv


# ---------------------------------------------------------------------------
# Linear alignment between activation spaces
# ---------------------------------------------------------------------------

@dataclass
class LinearMapping:
    weight: np.ndarray    # (128, 128): maps donor -> recipient
    bias: np.ndarray      # (128,)
    fit_r2: float
    fit_n: int
    name: str

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        W = torch.as_tensor(self.weight, dtype=torch.float32)
        b = torch.as_tensor(self.bias, dtype=torch.float32)
        return h @ W.T + b


def fit_linear_mapping(
    h_donor: np.ndarray, h_recipient: np.ndarray, ridge_alpha: float = 10.0, name: str = "ridge",
) -> LinearMapping:
    """Ridge regression from donor to recipient activations on matched observations."""
    model = Ridge(alpha=ridge_alpha).fit(h_donor, h_recipient)
    return LinearMapping(
        weight=model.coef_.astype(np.float32),
        bias=model.intercept_.astype(np.float32),
        fit_r2=float(model.score(h_donor, h_recipient)),
        fit_n=len(h_donor),
        name=name,
    )


def identity_mapping() -> LinearMapping:
    return LinearMapping(np.eye(128, dtype=np.float32), np.zeros(128, dtype=np.float32),
                         fit_r2=float("nan"), fit_n=0, name="identity")


# ---------------------------------------------------------------------------
# Donor activation functions
# ---------------------------------------------------------------------------

def donor_layer2(donor: QNetwork, obs: torch.Tensor) -> torch.Tensor:
    return donor.relu2(donor.fc2(donor.relu1(donor.fc1(obs))))


def make_matched_donor(donor: QNetwork, mapping: LinearMapping | None = None) -> Callable:
    """Donor activation for the SAME observation the recipient is processing."""
    def fn(obs: torch.Tensor) -> torch.Tensor:
        h = donor_layer2(donor, obs)
        return mapping.apply(h) if mapping is not None else h
    return fn


def make_shuffled_donor(
    donor: QNetwork, pool_obs: np.ndarray, seed: int, mapping: LinearMapping | None = None,
) -> Callable:
    """Donor activation for a RANDOM pool observation instead of the current one.

    Tests whether matching the donor activation to the current state matters.
    """
    rng = np.random.RandomState(seed)
    pool = torch.as_tensor(pool_obs, dtype=torch.float32)

    def fn(obs: torch.Tensor) -> torch.Tensor:
        idx = rng.randint(0, len(pool), size=obs.shape[0])
        h = donor_layer2(donor, pool[idx])
        return mapping.apply(h) if mapping is not None else h
    return fn


# ---------------------------------------------------------------------------
# Patching hook
# ---------------------------------------------------------------------------

@dataclass
class PatchStats:
    n_calls: int = 0
    applied_norms: list = field(default_factory=list)
    negative_fraction: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "mean_applied_norm": float(np.mean(self.applied_norms)) if self.applied_norms else 0.0,
            "mean_negative_fraction": float(np.mean(self.negative_fraction)) if self.negative_fraction else 0.0,
        }


@contextmanager
def layer2_patching(
    recipient: QNetwork,
    alpha: float,
    donor_fn: Callable | None = None,
    mode: str = "matched",
    random_seed: int = 0,
    constant_delta: np.ndarray | None = None,
    stats: PatchStats | None = None,
):
    """Temporarily patch the recipient's post-ReLU layer-2 output.

    Modes:
      "matched":  h + alpha * (donor_fn(s) - h)         (donor_fn decides matched/shuffled)
      "random":   h + alpha * r(s), r random direction with ||r(s)|| = ||donor_fn(s) - h||
      "constant": h + alpha * constant_delta             (state-independent; Q-bias control)
      "none":     no hook
    """
    if mode == "none" or alpha == 0.0:
        yield
        return

    captured: dict = {}
    gen = torch.Generator().manual_seed(random_seed)

    def pre_hook(module, inputs):
        captured["obs"] = inputs[0]

    def hook(module, inputs, output):
        h = output
        if mode == "constant":
            delta = torch.as_tensor(constant_delta, dtype=torch.float32).expand_as(h)
        else:
            with torch.no_grad():
                h_donor = donor_fn(captured["obs"])
            delta = h_donor - h
            if mode == "random":
                r = torch.randn(delta.shape, generator=gen)
                r = r / (torch.linalg.norm(r, dim=-1, keepdim=True) + 1e-8)
                delta = r * torch.linalg.norm(delta, dim=-1, keepdim=True)
            elif mode != "matched":
                raise ValueError(f"Unknown patching mode: {mode}")
        edited = h + alpha * delta
        if stats is not None:
            stats.n_calls += 1
            stats.applied_norms.append(float(torch.linalg.norm(alpha * delta, dim=-1).mean()))
            stats.negative_fraction.append(float((edited < 0).float().mean()))
        return edited

    h_pre = recipient.fc1.register_forward_pre_hook(pre_hook)
    h_post = recipient.relu2.register_forward_hook(hook)
    try:
        yield
    finally:
        h_pre.remove()
        h_post.remove()


# ---------------------------------------------------------------------------
# Full-episode evaluation
# ---------------------------------------------------------------------------

def patched_rollout(
    recipient: QNetwork,
    env_config: RewardGridConfig,
    alpha: float,
    donor_fn: Callable | None = None,
    mode: str = "none",
    random_seed: int = 0,
    constant_delta: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    env = RewardGridEnv(config=env_config)
    obs, info = env.reset(seed=seed)
    recipient.eval()
    stats = PatchStats()
    actions: list[int] = []
    positions = [tuple(info["agent_pos"])]
    total_reward = 0.0
    reward_tile_visits = 0
    terminated = truncated = False
    no_move_steps = 0

    with layer2_patching(recipient, alpha, donor_fn, mode, random_seed, constant_delta, stats), \
            torch.no_grad():
        while not (terminated or truncated):
            q = recipient(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
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
    is_hacking = bool(reward_tile_visits >= env_config.exploit_visit_threshold
                      and not goal_reached and (terminated or truncated))
    return {
        "goal_reached": goal_reached,
        "is_reward_hacking": is_hacking,
        "reward_tile_visits": int(reward_tile_visits),
        "proxy_reward": round(float(total_reward), 4),
        "episode_steps": len(actions),
        "unique_positions": len(set(positions)),
        "no_move_steps": no_move_steps,
        "actions": actions,
        "start_pos": list(env_config.agent_start),
        "patch": {"mode": mode, "alpha": alpha, **stats.summary()},
    }


def evaluate_patching(
    recipient: QNetwork,
    base_config: RewardGridConfig,
    start_positions: list,
    alpha: float,
    donor_fn: Callable | None = None,
    mode: str = "none",
    random_seed: int = 0,
    constant_delta: np.ndarray | None = None,
    keep_actions: bool = False,
) -> dict:
    """One deterministic greedy episode per start state; rates are exhaustive fractions."""
    episodes = []
    for pos in start_positions:
        cfg = RewardGridConfig(
            width=base_config.width, height=base_config.height, agent_start=tuple(pos),
            goal_pos=base_config.goal_pos, reward_pos=base_config.reward_pos,
            max_steps=base_config.max_steps, step_penalty=base_config.step_penalty,
            reward_tile_bonus=base_config.reward_tile_bonus, goal_bonus=base_config.goal_bonus,
            exploit_visit_threshold=base_config.exploit_visit_threshold,
        )
        ep = patched_rollout(recipient, cfg, alpha, donor_fn, mode, random_seed, constant_delta)
        if not keep_actions:
            ep = {k: v for k, v in ep.items() if k != "actions"}
        episodes.append(ep)
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
        "mean_applied_norm": float(np.mean([e["patch"]["mean_applied_norm"] for e in episodes])),
        "mean_negative_fraction": float(np.mean([e["patch"]["mean_negative_fraction"] for e in episodes])),
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# Scientific checks on matched observation sets
# ---------------------------------------------------------------------------

def q_change_analysis(
    recipient: QNetwork,
    observations: np.ndarray,
    alpha: float,
    donor_fn: Callable | None = None,
    mode: str = "matched",
    random_seed: int = 0,
    constant_delta: np.ndarray | None = None,
) -> dict:
    """How the patch changes Q-values on a fixed observation set, and how
    state-dependent that change is (fraction of dQ energy explained by its mean)."""
    obs_t = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        q_base = recipient(obs_t).numpy()
    with layer2_patching(recipient, alpha, donor_fn, mode, random_seed, constant_delta), torch.no_grad():
        q_patch = recipient(obs_t).numpy()
    dq = q_patch - q_base
    mean_dq = dq.mean(axis=0)
    energy_total = float((dq ** 2).sum(axis=1).mean())
    energy_const = float((mean_dq ** 2).sum())
    return {
        "greedy_flip_rate": float((q_base.argmax(1) != q_patch.argmax(1)).mean()),
        "mean_abs_q_change": float(np.abs(dq).mean()),
        "mean_dq_vector": mean_dq.tolist(),
        "constant_fraction_of_dq_energy": energy_const / energy_total if energy_total > 0 else 0.0,
        "patched_action_counts": {int(a): int(c) for a, c in zip(*np.unique(q_patch.argmax(1), return_counts=True))},
        "base_action_counts": {int(a): int(c) for a, c in zip(*np.unique(q_base.argmax(1), return_counts=True))},
    }


def activation_distribution_comparison(h_donor: np.ndarray, h_recipient: np.ndarray) -> dict:
    """Compare donor and recipient layer-2 activation distributions on matched states."""
    def stats(h):
        return {
            "mean_norm": float(np.linalg.norm(h, axis=1).mean()),
            "mean_activation": float(h.mean()),
            "active_fraction": float((h > 0).mean()),
        }
    delta = h_donor - h_recipient
    hn = np.linalg.norm(h_donor, axis=1) * np.linalg.norm(h_recipient, axis=1) + 1e-8
    return {
        "donor": stats(h_donor),
        "recipient": stats(h_recipient),
        "mean_delta_norm": float(np.linalg.norm(delta, axis=1).mean()),
        "mean_cosine_donor_recipient": float(((h_donor * h_recipient).sum(1) / hn).mean()),
    }
