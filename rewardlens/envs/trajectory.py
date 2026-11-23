"""Trajectory recording and replay for environment episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from rewardlens.envs.reward_grid import RewardGridEnv, RewardGridConfig, ACTION_NAMES


@dataclass
class StepRecord:
    step: int
    agent_x: int
    agent_y: int
    action: int | None  # None for the initial observation
    action_name: str | None
    reward: float
    cumulative_reward: float
    reward_tile_visits: int
    goal_reached: bool
    terminated: bool
    truncated: bool


@dataclass
class Trajectory:
    episode_id: str
    policy_name: str
    seed: int | None
    config: dict[str, Any]
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        if not self.steps:
            return 0.0
        return self.steps[-1].cumulative_reward

    @property
    def total_steps(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].step

    @property
    def goal_reached(self) -> bool:
        if not self.steps:
            return False
        return self.steps[-1].goal_reached

    @property
    def final_reward_tile_visits(self) -> int:
        if not self.steps:
            return 0
        return self.steps[-1].reward_tile_visits

    @property
    def terminated(self) -> bool:
        if not self.steps:
            return False
        return self.steps[-1].terminated

    @property
    def truncated(self) -> bool:
        if not self.steps:
            return False
        return self.steps[-1].truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "policy_name": self.policy_name,
            "seed": self.seed,
            "config": self.config,
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        with open(path) as f:
            data = json.load(f)
        traj = cls(
            episode_id=data["episode_id"],
            policy_name=data["policy_name"],
            seed=data["seed"],
            config=data["config"],
        )
        for s in data["steps"]:
            traj.steps.append(StepRecord(**s))
        return traj


def record_episode(
    env: RewardGridEnv,
    actions: list[int],
    policy_name: str,
    episode_id: str,
    seed: int | None = None,
) -> Trajectory:
    """Run an episode with the given action sequence and record every step."""
    obs, info = env.reset(seed=seed)

    traj = Trajectory(
        episode_id=episode_id,
        policy_name=policy_name,
        seed=seed,
        config=env.config.to_dict(),
    )

    traj.steps.append(
        StepRecord(
            step=0,
            agent_x=info["agent_pos"][0],
            agent_y=info["agent_pos"][1],
            action=None,
            action_name=None,
            reward=0.0,
            cumulative_reward=0.0,
            reward_tile_visits=0,
            goal_reached=False,
            terminated=False,
            truncated=False,
        )
    )

    cumulative = 0.0
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        traj.steps.append(
            StepRecord(
                step=info["episode_steps"],
                agent_x=info["agent_pos"][0],
                agent_y=info["agent_pos"][1],
                action=action,
                action_name=ACTION_NAMES.get(action, str(action)),
                reward=reward,
                cumulative_reward=round(cumulative, 6),
                reward_tile_visits=info["reward_tile_visits"],
                goal_reached=info["goal_reached"],
                terminated=terminated,
                truncated=truncated,
            )
        )
        if terminated or truncated:
            break

    return traj
