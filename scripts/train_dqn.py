"""Phase 2: Train a DQN on RewardGridEnv and evaluate periodically."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rewardlens.agents.dqn import (
    DQNConfig,
    QNetwork,
    ReplayBuffer,
    linear_epsilon,
    save_checkpoint,
)
from rewardlens.envs.reward_grid import RewardGridEnv, RewardGridConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config: DQNConfig) -> RewardGridEnv:
    env_config = RewardGridConfig(
        width=config.env_width,
        height=config.env_height,
        agent_start=config.agent_start,
        goal_pos=config.goal_pos,
        reward_pos=config.reward_pos,
        max_steps=config.max_steps,
        step_penalty=config.step_penalty,
        reward_tile_bonus=config.reward_tile_bonus,
        goal_bonus=config.goal_bonus,
        exploit_visit_threshold=config.exploit_visit_threshold,
    )
    return RewardGridEnv(config=env_config)


def greedy_eval(
    q_net: QNetwork,
    config: DQNConfig,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Run one deterministic greedy evaluation episode."""
    env = make_env(config)
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    positions = [(info["agent_pos"][0], info["agent_pos"][1])]

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(obs_t)
        action = int(q_values.argmax(dim=1).item())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        positions.append((info["agent_pos"][0], info["agent_pos"][1]))
        done = terminated or truncated

    return {
        "proxy_reward": round(total_reward, 4),
        "goal_reached": info["goal_reached"],
        "reward_tile_visits": info["reward_tile_visits"],
        "is_reward_hacking": info["is_reward_hacking"],
        "episode_steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "positions": positions,
    }


def train(config: DQNConfig) -> None:
    seed_everything(config.seed)
    device = config.resolve_device()
    run_dir = config.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config.save(run_dir / "config.json")

    print(f"Device: {device}")
    print(f"Output: {run_dir}")
    print(f"Total timesteps: {config.total_timesteps}")

    env = make_env(config)
    q_net = QNetwork(config.obs_dim, config.n_actions, config.hidden_dim).to(device)
    target_net = QNetwork(config.obs_dim, config.n_actions, config.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.buffer_capacity, config.obs_dim, seed=config.seed)

    # CSV loggers
    train_csv_path = run_dir / "training_metrics.csv"
    eval_csv_path = run_dir / "evaluation_metrics.csv"

    train_fields = [
        "timestep", "episode", "episode_reward", "episode_length",
        "goal_reached", "reward_tile_visits", "is_reward_hacking",
        "epsilon", "td_loss",
    ]
    eval_fields = [
        "timestep", "proxy_reward", "goal_reached", "reward_tile_visits",
        "is_reward_hacking", "episode_steps", "terminated", "truncated",
    ]

    with open(train_csv_path, "w", newline="") as f:
        csv.DictWriter(f, train_fields).writeheader()
    with open(eval_csv_path, "w", newline="") as f:
        csv.DictWriter(f, eval_fields).writeheader()

    # Training state
    obs, info = env.reset(seed=config.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    last_td_loss = 0.0
    start_time = time.time()

    # Initial evaluation
    eval_result = greedy_eval(q_net, config, device, seed=0)
    _log_eval(eval_csv_path, eval_fields, 0, eval_result)
    print(f"[Step 0] Eval: reward={eval_result['proxy_reward']:.2f} "
          f"goal={eval_result['goal_reached']} visits={eval_result['reward_tile_visits']}")

    for step in range(1, config.total_timesteps + 1):
        epsilon = linear_epsilon(
            step, config.total_timesteps,
            config.epsilon_start, config.epsilon_end, config.epsilon_fraction,
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(obs_t)
            action = int(q_values.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Treat finite horizon as terminal for DQN value target
        replay.add(obs, action, reward, next_obs, done)

        episode_reward += reward
        episode_steps += 1
        obs = next_obs

        if done:
            episode_num += 1
            train_row = {
                "timestep": step,
                "episode": episode_num,
                "episode_reward": round(episode_reward, 4),
                "episode_length": episode_steps,
                "goal_reached": info["goal_reached"],
                "reward_tile_visits": info["reward_tile_visits"],
                "is_reward_hacking": info["is_reward_hacking"],
                "epsilon": round(epsilon, 4),
                "td_loss": round(last_td_loss, 6),
            }
            with open(train_csv_path, "a", newline="") as f:
                csv.DictWriter(f, train_fields).writerow(train_row)

            if episode_num % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[Step {step}] Ep {episode_num}: reward={episode_reward:.2f} "
                      f"len={episode_steps} goal={info['goal_reached']} "
                      f"visits={info['reward_tile_visits']} eps={epsilon:.3f} "
                      f"loss={last_td_loss:.4f} ({elapsed:.0f}s)")

            obs, info = env.reset(seed=config.seed + episode_num)
            episode_reward = 0.0
            episode_steps = 0

        # Training update
        if step >= config.learning_starts and step % config.train_frequency == 0:
            batch = replay.sample(config.batch_size)

            obs_b = torch.tensor(batch["obs"], device=device)
            act_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
            rew_b = torch.tensor(batch["rewards"], device=device)
            next_b = torch.tensor(batch["next_obs"], device=device)
            done_b = torch.tensor(batch["dones"], device=device)

            with torch.no_grad():
                next_q = target_net(next_b).max(dim=1).values
                targets = rew_b + config.gamma * (1.0 - done_b) * next_q

            current_q = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(current_q, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_td_loss = loss.item()

        # Target network update
        if step % config.target_update_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Periodic evaluation
        if step % config.eval_interval == 0:
            eval_result = greedy_eval(q_net, config, device, seed=0)
            _log_eval(eval_csv_path, eval_fields, step, eval_result)
            print(f"[Step {step}] Eval: reward={eval_result['proxy_reward']:.2f} "
                  f"goal={eval_result['goal_reached']} visits={eval_result['reward_tile_visits']} "
                  f"hack={eval_result['is_reward_hacking']}")

        # Checkpoint
        if step % config.checkpoint_interval == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_{step}.pt",
                q_net, target_net, optimizer, step, config,
            )

    # Final checkpoint and evaluation
    save_checkpoint(ckpt_dir / "checkpoint_final.pt", q_net, target_net, optimizer, config.total_timesteps, config)
    final_eval = greedy_eval(q_net, config, device, seed=0)
    _log_eval(eval_csv_path, eval_fields, config.total_timesteps, final_eval)

    # Save final eval trajectory
    _save_eval_trajectory(q_net, config, device, run_dir / "evaluation_trajectories")

    elapsed = time.time() - start_time
    metadata = {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "total_timesteps": config.total_timesteps,
        "episodes": episode_num,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "python_version": sys.version,
        "elapsed_seconds": round(elapsed, 1),
        "final_eval": final_eval,
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete: {episode_num} episodes in {elapsed:.1f}s")
    print(f"Final eval: reward={final_eval['proxy_reward']:.2f} "
          f"goal={final_eval['goal_reached']} visits={final_eval['reward_tile_visits']} "
          f"hack={final_eval['is_reward_hacking']}")


def _log_eval(path: Path, fields: list[str], step: int, result: dict) -> None:
    row = {"timestep": step, **{k: result[k] for k in fields if k != "timestep" and k in result}}
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fields).writerow(row)


def _save_eval_trajectory(
    q_net: QNetwork,
    config: DQNConfig,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Save a full evaluation trajectory as JSON for visualization."""
    from rewardlens.envs.trajectory import Trajectory, StepRecord
    from rewardlens.envs.reward_grid import ACTION_NAMES

    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(config)
    obs, info = env.reset(seed=0)

    traj = Trajectory(
        episode_id=f"dqn_eval_{config.experiment_id}",
        policy_name="Trained DQN (greedy)",
        seed=0,
        config=env.config.to_dict(),
    )
    traj.steps.append(StepRecord(
        step=0, agent_x=info["agent_pos"][0], agent_y=info["agent_pos"][1],
        action=None, action_name=None, reward=0.0, cumulative_reward=0.0,
        reward_tile_visits=0, goal_reached=False, terminated=False, truncated=False,
    ))

    cumulative = 0.0
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(obs_t)
        action = int(q_values.argmax(dim=1).item())
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        done = terminated or truncated
        traj.steps.append(StepRecord(
            step=info["episode_steps"],
            agent_x=info["agent_pos"][0], agent_y=info["agent_pos"][1],
            action=action, action_name=ACTION_NAMES.get(action, str(action)),
            reward=reward, cumulative_reward=round(cumulative, 6),
            reward_tile_visits=info["reward_tile_visits"],
            goal_reached=info["goal_reached"],
            terminated=terminated, truncated=truncated,
        ))

    traj.save(out_dir / "final_greedy_trajectory.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN on RewardGridEnv")
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        config = DQNConfig.from_json(args.config)
    else:
        config = DQNConfig()

    if args.seed is not None:
        config.seed = args.seed
        if args.experiment_id is None:
            config.experiment_id = f"baseline_seed{args.seed}"
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.device is not None:
        config.device = args.device
    if args.experiment_id is not None:
        config.experiment_id = args.experiment_id

    train(config)


if __name__ == "__main__":
    main()
