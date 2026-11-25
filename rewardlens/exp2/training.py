"""DQN training on procedural layouts for Experiment 2.

One loop serves three stages:
  pretrain  corrected reward, random initialization
  good      corrected reward, initialized from the pretrained checkpoint
  hacking   flawed reward, initialized from the same pretrained checkpoint

The network, replay buffer, epsilon schedule, loss and checkpoint format are
the Experiment 1 ones (`rewardlens.agents.dqn`). Training only ever sees the
training split. Periodic evaluation uses the development split and a fixed
subsample of the training split; the test split is never loaded here.
"""

from __future__ import annotations

import csv
import json
import random
import resource
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rewardlens.agents.dqn import QNetwork, ReplayBuffer, linear_epsilon, load_checkpoint, save_checkpoint
from rewardlens.exp2.evaluation import greedy_policy, rollout, summarize
from rewardlens.exp2.layouts import load_manifest
from rewardlens.exp2.procedural_env import ProceduralEnvParams, ProceduralRewardGridEnv

EVAL_KEYS = ("direct_goal", "farm_then_finish", "farm_only", "other", "farming", "goal_completion",
             "mean_reward_tile_visits", "mean_steps", "mean_steps_to_goal", "path_efficiency_direct",
             "mean_return_flawed", "mean_return_corrected")


@dataclass
class Exp2TrainConfig:
    # Stage
    stage: str = "pretrain"            # pretrain | good | hacking
    reward_mode: str = "corrected"     # corrected | flawed
    init_checkpoint: str | None = None
    manifest_path: str = "experiments/02_generalizable_reward_hacking/layouts/manifest.json"
    env: dict = field(default_factory=lambda: ProceduralEnvParams().to_dict())

    # Network (Experiment 1 architecture)
    obs_dim: int = 7
    n_actions: int = 4
    hidden_dim: int = 128

    # DQN (Experiment 1 values unless a config overrides them)
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
    total_timesteps: int = 300_000
    seed: int = 42

    # Evaluation and checkpoints
    eval_interval: int = 10_000
    checkpoint_interval: int = 10_000
    n_train_eval_layouts: int = 1_500

    # Output
    run_dir: str = "results/exp2/phase1/seed42/pretrain"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def env_params(self) -> ProceduralEnvParams:
        return ProceduralEnvParams(**self.env)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_networks(config: Exp2TrainConfig) -> tuple[QNetwork, QNetwork, int]:
    """Create the online and target networks, loading `init_checkpoint` when given.

    Seeding happens before construction, so a run without a checkpoint is
    reproducible, and two runs given the same checkpoint start bit-identical.
    """
    seed_everything(config.seed)
    q_net = QNetwork(config.obs_dim, config.n_actions, config.hidden_dim)
    init_step = 0
    if config.init_checkpoint:
        ckpt = load_checkpoint(config.init_checkpoint)
        q_net.load_state_dict(ckpt["q_net_state_dict"])
        init_step = int(ckpt["step"])
    target_net = QNetwork(config.obs_dim, config.n_actions, config.hidden_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    return q_net, target_net, init_step


def evaluate_splits(q_net: QNetwork, eval_sets: dict, params: ProceduralEnvParams, gamma: float) -> dict:
    q_net.eval()
    out = {name: summarize(rollout(greedy_policy(q_net), arr, params, gamma), params.exploit_visit_threshold)
           for name, arr in eval_sets.items()}
    q_net.train()
    return out


def train(config: Exp2TrainConfig, verbose: bool = True) -> dict:
    torch.set_num_threads(1)
    params = config.env_params()
    splits, manifest_meta = load_manifest(config.manifest_path)
    train_layouts = splits["train"]
    from rewardlens.exp2.layouts import layouts_to_array
    eval_sets = {
        "dev": layouts_to_array(splits["dev"]),
        "train_sample": layouts_to_array(train_layouts[:config.n_train_eval_layouts]),
    }

    run_dir = Path(config.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    q_net, target_net, init_step = build_networks(config)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.buffer_capacity, config.obs_dim, seed=config.seed)
    action_rng = np.random.RandomState(config.seed + 1)

    env = ProceduralRewardGridEnv(train_layouts, config.reward_mode, params)
    obs, info = env.reset(seed=config.seed)

    eval_rows: list[dict] = []
    episode_rows: list[dict] = []

    def log_eval(step: int) -> dict:
        t0 = time.time()
        res = evaluate_splits(q_net, eval_sets, params, config.gamma)
        row = {"timestep": step, "eval_seconds": round(time.time() - t0, 3)}
        for split, summary in res.items():
            for k in EVAL_KEYS:
                row[f"{split}_{k}"] = summary[k]
        eval_rows.append(row)
        if verbose:
            d = res["dev"]
            print(f"[{config.stage} seed {config.seed} step {step}] dev direct={d['direct_goal']:.3f} "
                  f"farming={d['farming']:.3f} other={d['other']:.3f} visits={d['mean_reward_tile_visits']:.1f}",
                  flush=True)
        return res

    save_checkpoint(ckpt_dir / "checkpoint_0.pt", q_net, target_net, optimizer, 0, config)
    log_eval(0)

    ep_reward, ep_steps, ep_num, last_loss = 0.0, 0, 0, 0.0
    start = time.time()
    eval_time = 0.0

    for step in range(1, config.total_timesteps + 1):
        epsilon = linear_epsilon(step, config.total_timesteps, config.epsilon_start,
                                 config.epsilon_end, config.epsilon_fraction)
        if action_rng.random_sample() < epsilon:
            action = int(action_rng.randint(config.n_actions))
        else:
            with torch.no_grad():
                action = int(q_net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # The remaining-time feature is observed, so the time limit is treated as terminal (as in Experiment 1).
        replay.add(obs, action, reward, next_obs, done)
        ep_reward += reward
        ep_steps += 1
        obs = next_obs

        if done:
            ep_num += 1
            episode_rows.append({"timestep": step, "episode": ep_num, "episode_reward": round(ep_reward, 4),
                                 "episode_length": ep_steps, "goal_reached": int(info["goal_reached"]),
                                 "reward_tile_visits": info["reward_tile_visits"], "epsilon": round(epsilon, 4)})
            obs, info = env.reset()
            ep_reward, ep_steps = 0.0, 0

        if step >= config.learning_starts and step % config.train_frequency == 0:
            batch = replay.sample(config.batch_size)
            obs_b = torch.from_numpy(batch["obs"])
            act_b = torch.from_numpy(batch["actions"])
            rew_b = torch.from_numpy(batch["rewards"])
            next_b = torch.from_numpy(batch["next_obs"])
            done_b = torch.from_numpy(batch["dones"])
            with torch.no_grad():
                targets = rew_b + config.gamma * (1.0 - done_b) * target_net(next_b).max(dim=1).values
            current = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(current, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        if step % config.target_update_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % config.checkpoint_interval == 0:
            save_checkpoint(ckpt_dir / f"checkpoint_{step}.pt", q_net, target_net, optimizer, step, config)

        if step % config.eval_interval == 0:
            t0 = time.time()
            log_eval(step)
            eval_time += time.time() - t0

    save_checkpoint(ckpt_dir / "checkpoint_final.pt", q_net, target_net, optimizer, config.total_timesteps, config)
    elapsed = time.time() - start

    if eval_rows:
        with open(run_dir / "eval_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(eval_rows[0].keys()))
            w.writeheader()
            w.writerows(eval_rows)
    if episode_rows:
        with open(run_dir / "training_episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(episode_rows[0].keys()))
            w.writeheader()
            w.writerows(episode_rows)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    train_seconds = elapsed - eval_time
    metadata = {
        "stage": config.stage, "reward_mode": config.reward_mode, "seed": config.seed,
        "init_checkpoint": config.init_checkpoint, "init_checkpoint_step": init_step,
        "total_timesteps": config.total_timesteps, "episodes": ep_num,
        "elapsed_seconds": round(elapsed, 1), "train_seconds": round(train_seconds, 1),
        "eval_seconds_total": round(eval_time, 1),
        "env_steps_per_second": round(config.total_timesteps / max(train_seconds, 1e-9), 1),
        "peak_memory_mb": round(rss_mb, 1), "final_td_loss": last_loss,
        "manifest_sha256": manifest_meta["sha256"], "n_train_layouts": len(train_layouts),
        "pytorch_version": torch.__version__, "python_version": sys.version.split()[0],
        "final_dev": {k: eval_rows[-1][f"dev_{k}"] for k in EVAL_KEYS},
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
