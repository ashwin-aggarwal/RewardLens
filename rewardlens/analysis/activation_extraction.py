"""Extract post-ReLU hidden activations and Q-values from trained DQN checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rewardlens.agents.dqn import QNetwork, load_checkpoint


@dataclass
class ActivationRecord:
    observation: np.ndarray       # (N, 7)
    hidden1: np.ndarray           # (N, 128) post-ReLU
    hidden2: np.ndarray           # (N, 128) post-ReLU
    q_values: np.ndarray          # (N, 4)
    greedy_actions: np.ndarray    # (N,)
    checkpoint_path: str
    seed: int
    condition: str                # "flawed" or "corrected"
    step: int                     # training step of checkpoint


def extract_activations(
    checkpoint_path: str | Path,
    observations: np.ndarray,
    seed: int,
    condition: str,
    step: int,
    device: torch.device | None = None,
) -> ActivationRecord:
    """Run observations through a checkpoint and capture post-ReLU activations.

    The QNetwork architecture is fc1 -> relu1 -> fc2 -> relu2 -> fc3.
    We capture the output of relu1 (hidden layer 1 post-ReLU) and
    relu2 (hidden layer 2 post-ReLU) via explicit forward pass.
    """
    if device is None:
        device = torch.device("cpu")

    ckpt = load_checkpoint(checkpoint_path, device)
    cfg = ckpt["config"]
    net = QNetwork(
        obs_dim=cfg.get("obs_dim", 7),
        n_actions=cfg.get("n_actions", 4),
        hidden_dim=cfg.get("hidden_dim", 128),
    )
    net.load_state_dict(ckpt["q_net_state_dict"])
    net.to(device)
    net.eval()

    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)

    with torch.no_grad():
        h1 = net.relu1(net.fc1(obs_tensor))
        h2 = net.relu2(net.fc2(h1))
        q = net.fc3(h2)

    q_np = q.cpu().numpy()
    return ActivationRecord(
        observation=observations.copy(),
        hidden1=h1.cpu().numpy(),
        hidden2=h2.cpu().numpy(),
        q_values=q_np,
        greedy_actions=q_np.argmax(axis=1),
        checkpoint_path=str(checkpoint_path),
        seed=seed,
        condition=condition,
        step=step,
    )


def verify_extraction_preserves_predictions(
    checkpoint_path: str | Path,
    observations: np.ndarray,
    device: torch.device | None = None,
) -> bool:
    """Confirm extraction produces identical Q-values to a standard forward pass."""
    if device is None:
        device = torch.device("cpu")

    ckpt = load_checkpoint(checkpoint_path, device)
    cfg = ckpt["config"]
    net = QNetwork(
        obs_dim=cfg.get("obs_dim", 7),
        n_actions=cfg.get("n_actions", 4),
        hidden_dim=cfg.get("hidden_dim", 128),
    )
    net.load_state_dict(ckpt["q_net_state_dict"])
    net.to(device)
    net.eval()

    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_standard = net(obs_tensor).cpu().numpy()

        h1 = net.relu1(net.fc1(obs_tensor))
        h2 = net.relu2(net.fc2(h1))
        q_explicit = net.fc3(h2).cpu().numpy()

    return np.allclose(q_standard, q_explicit, atol=1e-6)


def build_matched_state_dataset(
    grid_width: int = 8,
    grid_height: int = 8,
    goal_pos: tuple[int, int] = (6, 1),
    reward_pos: tuple[int, int] = (3, 5),
    time_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> np.ndarray:
    """Enumerate gridworld observations for matched-state analysis.

    Returns normalized observations: [agent_x/w, agent_y/h, goal_x/w, goal_y/h,
    reward_x/w, reward_y/h, remaining_time_fraction].
    """
    rows = []
    gx, gy = goal_pos
    rx, ry = reward_pos
    for x in range(grid_width):
        for y in range(grid_height):
            for t in time_fractions:
                rows.append([
                    x / grid_width,
                    y / grid_height,
                    gx / grid_width,
                    gy / grid_height,
                    rx / grid_width,
                    ry / grid_height,
                    t,
                ])
    return np.array(rows, dtype=np.float32)


def compute_state_metadata(
    observations: np.ndarray,
    grid_width: int = 8,
    grid_height: int = 8,
) -> dict[str, np.ndarray]:
    """Compute per-observation metadata: positions, distances, time remaining."""
    ax = (observations[:, 0] * grid_width).astype(int)
    ay = (observations[:, 1] * grid_height).astype(int)
    gx = (observations[:, 2] * grid_width).astype(int)
    gy = (observations[:, 3] * grid_height).astype(int)
    rx = (observations[:, 4] * grid_width).astype(int)
    ry = (observations[:, 5] * grid_height).astype(int)
    time_remaining = observations[:, 6]

    dist_to_goal = np.abs(ax - gx) + np.abs(ay - gy)
    dist_to_reward = np.abs(ax - rx) + np.abs(ay - ry)

    return {
        "agent_x": ax,
        "agent_y": ay,
        "goal_x": gx,
        "goal_y": gy,
        "reward_x": rx,
        "reward_y": ry,
        "time_remaining": time_remaining,
        "dist_to_goal": dist_to_goal.astype(float),
        "dist_to_reward": dist_to_reward.astype(float),
    }
