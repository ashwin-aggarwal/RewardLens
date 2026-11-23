"""DQN agent: Q-network, replay buffer, and training utilities."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    # Environment
    env_width: int = 8
    env_height: int = 8
    agent_start: tuple[int, int] = (1, 6)
    goal_pos: tuple[int, int] = (6, 1)
    reward_pos: tuple[int, int] = (3, 5)
    max_steps: int = 100
    step_penalty: float = -0.01
    reward_tile_bonus: float = 2.0
    goal_bonus: float = 10.0
    exploit_visit_threshold: int = 5

    # Network
    obs_dim: int = 7
    n_actions: int = 4
    hidden_dim: int = 128

    # Training
    learning_rate: float = 0.00025
    gamma: float = 0.99
    buffer_capacity: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_frequency: int = 4
    target_update_frequency: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.5
    total_timesteps: int = 100_000
    seed: int = 42

    # Evaluation
    eval_interval: int = 5_000
    eval_episodes: int = 1

    # Checkpointing
    checkpoint_interval: int = 10_000

    # Device
    device: str = "auto"

    # Output
    experiment_id: str = "baseline_seed42"
    output_dir: str = "results/phase2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> DQNConfig:
        with open(path) as f:
            data = json.load(f)
        # Convert lists back to tuples for position fields
        for key in ("agent_start", "goal_pos", "reward_pos"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_id


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Two-hidden-layer feedforward Q-network.

    Layers are named for future activation extraction:
      fc1 (7 -> 128), relu1, fc2 (128 -> 128), relu2, fc3 (128 -> 4)
    """

    def __init__(self, obs_dim: int = 7, n_actions: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform random sampling."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.rng = np.random.RandomState(seed)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.rng.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------------------------

def linear_epsilon(step: int, total: int, start: float, end: float, fraction: float) -> float:
    decay_steps = int(total * fraction)
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    step: int,
    config: DQNConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_net_state_dict": q_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    path = Path(path)
    map_location = device if device else "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)


def load_q_network_for_eval(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[QNetwork, dict[str, Any]]:
    """Load a Q-network from a checkpoint for inference."""
    ckpt = load_checkpoint(path, device)
    cfg = ckpt["config"]
    net = QNetwork(
        obs_dim=cfg.get("obs_dim", 7),
        n_actions=cfg.get("n_actions", 4),
        hidden_dim=cfg.get("hidden_dim", 128),
    )
    net.load_state_dict(ckpt["q_net_state_dict"])
    if device:
        net = net.to(device)
    net.eval()
    return net, ckpt
