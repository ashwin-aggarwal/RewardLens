"""Phase 4.6: scheduled and layer-selectable goal-activation patching.

Extends Phase 4.5's patching in three scientifically motivated directions:

  A. Time-dependent schedules: the patch strength alpha is a function of the
     episode step, so the patch can be concentrated early (before farming
     starts), decayed, or applied only late (diagnostic contrast).
  B. Strengths above 1: because fc3 is affine, a layer-2 patch with strength
     alpha gives Q = (1 - alpha) * Q_recipient + alpha * fc3_rec(h_donor).
     alpha > 1 therefore puts NEGATIVE weight on the recipient's own
     Q-values, actively opposing its farming preference. Activation norms and
     the fraction of units pushed negative are monitored.
  C. Layer choice: patch post-ReLU layer 1 instead of layer 2, so the
     recipient's own fc2/relu2/fc3 process the donor's earlier representation.

All edits are applied through temporary hooks; weights are never modified.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.analysis.behavior_classification import (
    DEFAULT_VISIT_THRESHOLD,
    action_fractions,
    category_rates,
    classify_episode,
    efficiency_summary,
)
from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv


# ---------------------------------------------------------------------------
# Schedules: alpha as a function of the episode step (0-indexed)
# ---------------------------------------------------------------------------

def schedule_constant(alpha: float) -> Callable[[int], float]:
    return lambda step: float(alpha)


def schedule_early_window(alpha: float, n_steps: int) -> Callable[[int], float]:
    """Full strength for the first n_steps decisions, then no patch."""
    return lambda step: float(alpha) if step < n_steps else 0.0


def schedule_early_decay(alpha: float, n_steps: int) -> Callable[[int], float]:
    """Linear decay from alpha at step 0 to zero at n_steps."""
    return lambda step: float(alpha) * max(0.0, 1.0 - step / n_steps)


def schedule_late_window(alpha: float, start_step: int) -> Callable[[int], float]:
    """No patch until start_step, then full strength (diagnostic contrast)."""
    return lambda step: float(alpha) if step >= start_step else 0.0


def make_schedule(name: str, alpha: float, **params) -> Callable[[int], float]:
    if name == "constant":
        return schedule_constant(alpha)
    if name == "early_window":
        return schedule_early_window(alpha, int(params["n_steps"]))
    if name == "early_decay":
        return schedule_early_decay(alpha, int(params["n_steps"]))
    if name == "late_window":
        return schedule_late_window(alpha, int(params["start_step"]))
    raise ValueError(f"Unknown schedule: {name}")


# ---------------------------------------------------------------------------
# Donor activations at a chosen layer
# ---------------------------------------------------------------------------

def donor_activation(donor: QNetwork, obs: torch.Tensor, layer: int) -> torch.Tensor:
    h1 = donor.relu1(donor.fc1(obs))
    if layer == 1:
        return h1
    if layer == 2:
        return donor.relu2(donor.fc2(h1))
    raise ValueError("layer must be 1 or 2")


def recipient_activation(recipient: QNetwork, obs: torch.Tensor, layer: int) -> torch.Tensor:
    return donor_activation(recipient, obs, layer)


def clamp_time_tensor(value: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Batch observation transform that fixes the remaining-time feature (column 6)."""
    def fn(obs: torch.Tensor) -> torch.Tensor:
        o = obs.clone()
        o[:, 6] = float(value)
        return o
    return fn


def make_matched_donor_layer(
    donor: QNetwork, layer: int, donor_obs_transform: Callable | None = None,
) -> Callable:
    """Donor activation for the SAME observation the recipient is processing.

    `donor_obs_transform` (Phase 4.6 block D) alters only what the DONOR sees,
    e.g. clamps its remaining-time feature; the recipient still sees the true
    observation. The injected quantity is still a donor activation.
    """
    def fn(obs: torch.Tensor) -> torch.Tensor:
        o = donor_obs_transform(obs) if donor_obs_transform is not None else obs
        return donor_activation(donor, o, layer)
    return fn


def make_time_delta_donor(
    recipient: QNetwork, donor: QNetwork, layer: int, time_value: float,
    pool_obs: np.ndarray | None = None, seed: int = 0,
) -> Callable:
    """Phase 4.7 time-only transfer control.

    Returns a donor function whose "matched" patch adds ONLY the part of the
    donor's activation change that is caused by changing its time input:

        target(s) = h_rec(s, t_real) + [h_donor(s, t_clamp) - h_donor(s, t_real)]

    so that with alpha = 1 the recipient keeps its own representation of the
    current state and receives the donor's time-related difference. Together
    with the matched-time patch this gives an exact decomposition of the
    Phase 4.6 full replacement:

        h_donor(s, t_clamp) - h_rec(s, t_real)
          = [h_donor(s, t_real) - h_rec(s, t_real)]   (matched-time patch)
          + [h_donor(s, t_clamp) - h_donor(s, t_real)] (time-only delta)

    With `pool_obs`, the time delta is computed for a RANDOM pool state
    instead of the current one (shuffled-state control for this condition).
    """
    rng = np.random.RandomState(seed)
    pool = torch.as_tensor(pool_obs, dtype=torch.float32) if pool_obs is not None else None
    clamp = clamp_time_tensor(time_value)
    F = torch.nn.functional

    def h_rec_hookfree(obs: torch.Tensor) -> torch.Tensor:
        # Functional forward: this function is called from inside a hook on the
        # recipient's own ReLU module, so it must not invoke that module again.
        h1 = F.relu(F.linear(obs, recipient.fc1.weight, recipient.fc1.bias))
        if layer == 1:
            return h1
        return F.relu(F.linear(h1, recipient.fc2.weight, recipient.fc2.bias))

    def fn(obs: torch.Tensor) -> torch.Tensor:
        src = pool[rng.randint(0, len(pool), size=obs.shape[0])] if pool is not None else obs
        delta = donor_activation(donor, clamp(src), layer) - donor_activation(donor, src, layer)
        return h_rec_hookfree(obs) + delta
    return fn


def make_shuffled_donor_layer(
    donor: QNetwork, pool_obs: np.ndarray, seed: int, layer: int,
    donor_obs_transform: Callable | None = None,
) -> Callable:
    rng = np.random.RandomState(seed)
    pool = torch.as_tensor(pool_obs, dtype=torch.float32)

    def fn(obs: torch.Tensor) -> torch.Tensor:
        idx = rng.randint(0, len(pool), size=obs.shape[0])
        o = pool[idx]
        o = donor_obs_transform(o) if donor_obs_transform is not None else o
        return donor_activation(donor, o, layer)
    return fn


# ---------------------------------------------------------------------------
# Hook with a step-aware controller
# ---------------------------------------------------------------------------

@dataclass
class PatchController:
    """Mutable step counter shared between the rollout loop and the hook."""
    step: int = 0


@dataclass
class SteeringStats:
    n_calls: int = 0
    alphas: list = field(default_factory=list)
    applied_norms: list = field(default_factory=list)
    base_norms: list = field(default_factory=list)
    edited_norms: list = field(default_factory=list)
    negative_fraction: list = field(default_factory=list)

    def summary(self) -> dict:
        active = [a for a in self.alphas if a > 0]
        return {
            "n_calls": self.n_calls,
            "n_active_steps": len(active),
            "mean_alpha": float(np.mean(self.alphas)) if self.alphas else 0.0,
            "mean_applied_norm": float(np.mean(self.applied_norms)) if self.applied_norms else 0.0,
            "mean_base_norm": float(np.mean(self.base_norms)) if self.base_norms else 0.0,
            "mean_edited_norm": float(np.mean(self.edited_norms)) if self.edited_norms else 0.0,
            "mean_norm_ratio": (float(np.mean(self.edited_norms)) / float(np.mean(self.base_norms))
                                if self.base_norms and np.mean(self.base_norms) > 0 else 0.0),
            "mean_negative_fraction": float(np.mean(self.negative_fraction)) if self.negative_fraction else 0.0,
        }


@contextmanager
def scheduled_patching(
    recipient: QNetwork,
    donor_fn: Callable | None,
    schedule: Callable[[int], float],
    controller: PatchController,
    layer: int = 2,
    mode: str = "matched",
    random_seed: int = 0,
    constant_delta: np.ndarray | None = None,
    stats: SteeringStats | None = None,
):
    """Temporarily patch post-ReLU layer `layer` with alpha = schedule(controller.step).

    Modes (as in Phase 4.5):
      "matched":  h + a * (donor_fn(s) - h)
      "random":   h + a * r(s), ||r(s)|| = ||donor_fn(s) - h||
      "constant": h + a * constant_delta
      "none":     no hook
    """
    if mode == "none":
        yield
        return
    if layer not in (1, 2):
        raise ValueError("layer must be 1 or 2")

    captured: dict = {}
    gen = torch.Generator().manual_seed(random_seed)
    target = recipient.relu1 if layer == 1 else recipient.relu2

    def pre_hook(module, inputs):
        captured["obs"] = inputs[0]

    def hook(module, inputs, output):
        h = output
        a = float(schedule(controller.step))
        if a == 0.0:
            if stats is not None:
                stats.n_calls += 1
                stats.alphas.append(0.0)
                stats.applied_norms.append(0.0)
                bn = float(torch.linalg.norm(h, dim=-1).mean())
                stats.base_norms.append(bn)
                stats.edited_norms.append(bn)
                stats.negative_fraction.append(0.0)
            return h
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
        edited = h + a * delta
        if stats is not None:
            stats.n_calls += 1
            stats.alphas.append(a)
            stats.applied_norms.append(float(torch.linalg.norm(a * delta, dim=-1).mean()))
            stats.base_norms.append(float(torch.linalg.norm(h, dim=-1).mean()))
            stats.edited_norms.append(float(torch.linalg.norm(edited, dim=-1).mean()))
            stats.negative_fraction.append(float((edited < 0).float().mean()))
        return edited

    h_pre = recipient.fc1.register_forward_pre_hook(pre_hook)
    h_post = target.register_forward_hook(hook)
    try:
        yield
    finally:
        h_pre.remove()
        h_post.remove()


# ---------------------------------------------------------------------------
# Rollouts and evaluation
# ---------------------------------------------------------------------------

def steered_rollout(
    recipient: QNetwork,
    env_config: RewardGridConfig,
    donor_fn: Callable | None = None,
    schedule: Callable[[int], float] | None = None,
    layer: int = 2,
    mode: str = "none",
    random_seed: int = 0,
    constant_delta: np.ndarray | None = None,
    seed: int = 0,
    obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    threshold: int = DEFAULT_VISIT_THRESHOLD,
) -> dict:
    """One deterministic greedy episode with a scheduled patch active.

    `obs_transform` (diagnostic only) alters what the NETWORK sees, e.g. to
    clamp the remaining-time feature; the environment itself is unchanged.
    """
    env = RewardGridEnv(config=env_config)
    obs, info = env.reset(seed=seed)
    recipient.eval()
    schedule = schedule or schedule_constant(0.0)
    controller = PatchController()
    stats = SteeringStats()
    actions: list[int] = []
    positions = [tuple(info["agent_pos"])]
    visits_trace: list[int] = [0]
    total_reward = 0.0
    reward_tile_visits = 0
    terminated = truncated = False
    no_move_steps = 0

    with scheduled_patching(recipient, donor_fn, schedule, controller, layer, mode,
                            random_seed, constant_delta, stats), torch.no_grad():
        while not (terminated or truncated):
            controller.step = len(actions)
            net_obs = obs_transform(obs) if obs_transform is not None else obs
            q = recipient(torch.as_tensor(net_obs, dtype=torch.float32).unsqueeze(0))
            action = int(q.argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            new_pos = tuple(info["agent_pos"])
            if new_pos == positions[-1]:
                no_move_steps += 1
            positions.append(new_pos)
            total_reward += reward
            reward_tile_visits = info["reward_tile_visits"]
            visits_trace.append(int(reward_tile_visits))

    goal_reached = bool(terminated)
    complete = bool(terminated or truncated)
    is_hacking = bool(reward_tile_visits >= env_config.exploit_visit_threshold
                      and not goal_reached and complete)
    return {
        "goal_reached": goal_reached,
        "is_reward_hacking": is_hacking,
        "episode_complete": complete,
        "category": classify_episode(goal_reached, int(reward_tile_visits), complete, threshold),
        "reward_tile_visits": int(reward_tile_visits),
        "proxy_reward": round(float(total_reward), 4),
        "episode_steps": len(actions),
        "goal_step": len(actions) if goal_reached else None,
        "unique_positions": len(set(positions)),
        "no_move_steps": no_move_steps,
        "actions": actions,
        "positions": [list(p) for p in positions],
        "visits_trace": visits_trace,
        "start_pos": list(env_config.agent_start),
        "patch": {"mode": mode, "layer": layer, **stats.summary()},
    }


def _config_with_start(base: RewardGridConfig, pos) -> RewardGridConfig:
    return RewardGridConfig(
        width=base.width, height=base.height, agent_start=tuple(pos),
        goal_pos=base.goal_pos, reward_pos=base.reward_pos, max_steps=base.max_steps,
        step_penalty=base.step_penalty, reward_tile_bonus=base.reward_tile_bonus,
        goal_bonus=base.goal_bonus, exploit_visit_threshold=base.exploit_visit_threshold,
    )


def evaluate_steering(
    recipient: QNetwork,
    base_config: RewardGridConfig,
    start_positions: list,
    donor_fn: Callable | None = None,
    schedule: Callable[[int], float] | None = None,
    layer: int = 2,
    mode: str = "none",
    random_seed: int = 0,
    constant_delta: np.ndarray | None = None,
    obs_transform: Callable | None = None,
    threshold: int = DEFAULT_VISIT_THRESHOLD,
    keep_traces: bool = False,
) -> dict:
    """One greedy episode per start state; Phase 4.6 category metrics plus patch stats."""
    full = [steered_rollout(recipient, _config_with_start(base_config, pos), donor_fn, schedule,
                            layer, mode, random_seed, constant_delta, obs_transform=obs_transform,
                            threshold=threshold)
            for pos in start_positions]
    episodes = full if keep_traces else [
        {k: v for k, v in ep.items() if k not in ("actions", "positions", "visits_trace")} for ep in full]
    rates = category_rates(episodes, threshold)
    rates.pop("categories")
    rates.update(efficiency_summary(full, base_config.goal_pos, threshold))
    rates["action_fractions"] = action_fractions(full)
    rates.update({
        "mean_applied_norm": float(np.mean([e["patch"]["mean_applied_norm"] for e in episodes])),
        "mean_norm_ratio": float(np.mean([e["patch"]["mean_norm_ratio"] for e in episodes])),
        "mean_negative_fraction": float(np.mean([e["patch"]["mean_negative_fraction"] for e in episodes])),
        "mean_active_steps": float(np.mean([e["patch"]["n_active_steps"] for e in episodes])),
        "n_start_states": len(episodes),
        "episodes": episodes,
    })
    return rates


def summarize(result: dict) -> dict:
    """Drop per-episode detail for compact JSON output."""
    return {k: v for k, v in result.items() if k != "episodes"}


def mean_layer_delta(recipient: QNetwork, donor_fn: Callable, obs: np.ndarray, layer: int) -> np.ndarray:
    """Mean (donor - recipient) activation over a state set: the constant Q-bias control vector."""
    ot = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        h_rec = recipient_activation(recipient, ot, layer)
        h_don = donor_fn(ot)
    return (h_don - h_rec).mean(0).numpy()
