"""Generate Phase 2.5 visualizations: flawed-reward vs. corrected-reward DQN."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rewardlens.envs.trajectory import Trajectory
from rewardlens.visualization.gridworld_viz import (
    render_trajectory_gif,
    BG_COLOR, TEXT_COLOR, GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR,
    _draw_grid, _draw_agent, _positions,
)

FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")

FLAWED_COLOR = EXPLOIT_PATH_COLOR
CORRECTED_COLOR = GOAL_PATH_COLOR


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _smooth(values: list[float], window: int = 50) -> list[float]:
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


def render_goal_dqn_gif(seed: int = 42) -> None:
    traj_path = CORRECTED_DIR / f"goal_seed{seed}" / "evaluation_trajectories" / "final_greedy_trajectory.json"
    traj = Trajectory.load(traj_path)
    render_trajectory_gif(
        traj, GIFS / "phase2_5_goal_dqn.gif",
        max_frames=30, frame_duration_ms=250, last_frame_duration_ms=2000,
    )


def render_policy_comparison(seed: int = 42) -> None:
    flawed_traj = Trajectory.load(FLAWED_DIR / f"baseline_seed{seed}" / "evaluation_trajectories" / "final_greedy_trajectory.json")
    corrected_traj = Trajectory.load(CORRECTED_DIR / f"goal_seed{seed}" / "evaluation_trajectories" / "final_greedy_trajectory.json")

    with open(CORRECTED_DIR / "comparison.json") as f:
        comparison = json.load(f)[str(seed)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    _draw_grid(ax1, flawed_traj.config, title="Learned: Flawed-Reward DQN")
    pos = _positions(flawed_traj)
    seen = set()
    for i in range(1, len(pos)):
        seg = (pos[i - 1], pos[i])
        if seg not in seen:
            seen.add(seg)
            ax1.annotate("", xy=pos[i], xytext=pos[i - 1],
                         arrowprops=dict(arrowstyle="->", color=FLAWED_COLOR, lw=1.5, alpha=0.6))
    _draw_agent(ax1, pos[-1][0], pos[-1][1], color=FLAWED_COLOR)
    f = comparison["flawed_reward_dqn"]
    ax1.text(
        flawed_traj.config["width"] / 2 - 0.5, flawed_traj.config["height"] + 0.4,
        f"Native (flawed) reward: {f['native_reward']:+.2f}\n"
        f"Scored under corrected reward: {f['scored_under_corrected_reward']:+.2f}\n"
        f"Goal: {'Yes' if f['goal_reached'] else 'No'} | Reward-tile visits: {f['reward_tile_visits']} | "
        f"Reward hacking: {'Yes' if f['is_reward_hacking'] else 'No'}",
        ha="center", va="top", fontsize=8, color=TEXT_COLOR, fontfamily="monospace",
    )

    _draw_grid(ax2, corrected_traj.config, title="Learned: Corrected-Reward DQN")
    pos2 = _positions(corrected_traj)
    for i in range(1, len(pos2)):
        ax2.annotate("", xy=pos2[i], xytext=pos2[i - 1],
                     arrowprops=dict(arrowstyle="->", color=CORRECTED_COLOR, lw=2, alpha=0.7))
    _draw_agent(ax2, pos2[-1][0], pos2[-1][1], color=CORRECTED_COLOR)
    c = comparison["corrected_reward_dqn"]
    ax2.text(
        corrected_traj.config["width"] / 2 - 0.5, corrected_traj.config["height"] + 0.4,
        f"Native (corrected) reward: {c['native_reward']:+.2f}\n"
        f"Scored under flawed reward: {c['scored_under_flawed_reward']:+.2f}\n"
        f"Goal: {'Yes' if c['goal_reached'] else 'No'} | Reward-tile visits: {c['reward_tile_visits']} | "
        f"Reward hacking: {'Yes' if c['is_reward_hacking'] else 'No'}",
        ha="center", va="top", fontsize=8, color=TEXT_COLOR, fontfamily="monospace",
    )

    fig.suptitle(
        "Same Environment, Different Reward Functions, Different Agents",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.08,
    )
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase2_5_policy_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_training_comparison(seed: int = 42) -> None:
    flawed_rows = _load_csv(FLAWED_DIR / f"baseline_seed{seed}" / "training_metrics.csv")
    corrected_rows = _load_csv(CORRECTED_DIR / f"goal_seed{seed}" / "training_metrics.csv")

    flawed_steps = [int(r["timestep"]) for r in flawed_rows]
    flawed_reward = [float(r["episode_reward"]) for r in flawed_rows]
    flawed_goal = [1.0 if r["goal_reached"] == "True" else 0.0 for r in flawed_rows]
    flawed_hack = [1.0 if r["is_reward_hacking"] == "True" else 0.0 for r in flawed_rows]

    corrected_steps = [int(r["timestep"]) for r in corrected_rows]
    corrected_reward = [float(r["episode_reward"]) for r in corrected_rows]
    corrected_goal = [1.0 if r["goal_reached"] == "True" else 0.0 for r in corrected_rows]
    corrected_hack = [1.0 if r["is_reward_hacking"] == "True" else 0.0 for r in corrected_rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    # Top-left: flawed-reward training reward (own scale)
    ax = axes[0, 0]
    ax.set_facecolor(BG_COLOR)
    ax.scatter(flawed_steps, flawed_reward, alpha=0.1, s=3, color=FLAWED_COLOR)
    ax.plot(flawed_steps, _smooth(flawed_reward), color=FLAWED_COLOR, lw=2)
    ax.set_title("Flawed-Reward DQN: Training Reward\n(own reward scale, max ~97)", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Episode Reward (flawed objective)", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: corrected-reward training reward (own scale, NOT shared with left)
    ax = axes[0, 1]
    ax.set_facecolor(BG_COLOR)
    ax.scatter(corrected_steps, corrected_reward, alpha=0.1, s=3, color=CORRECTED_COLOR)
    ax.plot(corrected_steps, _smooth(corrected_reward), color=CORRECTED_COLOR, lw=2)
    ax.set_title("Corrected-Reward DQN: Training Reward\n(own reward scale, max ~9.9 — note the different y-axis)", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Episode Reward (corrected objective)", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: goal completion rate, both conditions overlaid (comparable, 0-1 scale)
    ax = axes[1, 0]
    ax.set_facecolor(BG_COLOR)
    ax.plot(flawed_steps, _smooth(flawed_goal), color=FLAWED_COLOR, lw=2, label="Flawed-reward DQN")
    ax.plot(corrected_steps, _smooth(corrected_goal), color=CORRECTED_COLOR, lw=2, label="Corrected-reward DQN")
    ax.set_title("Goal Completion Rate (directly comparable)", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Environment Steps", fontsize=9)
    ax.set_ylabel("Goal Completion Rate", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: reward-hacking rate, both conditions overlaid (comparable, 0-1 scale)
    ax = axes[1, 1]
    ax.set_facecolor(BG_COLOR)
    ax.plot(flawed_steps, _smooth(flawed_hack), color=FLAWED_COLOR, lw=2, label="Flawed-reward DQN")
    ax.plot(corrected_steps, _smooth(corrected_hack), color=CORRECTED_COLOR, lw=2, label="Corrected-reward DQN")
    ax.set_title("Reward-Hacking Rate (directly comparable)", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Environment Steps", fontsize=9)
    ax.set_ylabel("Reward-Hacking Rate", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for a in axes.flat:
        a.tick_params(colors=TEXT_COLOR)

    fig.suptitle(
        "Training Outcomes: Flawed vs. Corrected Reward (seed 42)\n"
        "Top row uses each model's own reward scale — do not compare magnitudes directly",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.04,
    )
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase2_5_training_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    print("Generating Phase 2.5 visualizations...")
    render_goal_dqn_gif(seed=42)
    print(f"  ✓ {GIFS / 'phase2_5_goal_dqn.gif'}")
    render_policy_comparison(seed=42)
    print(f"  ✓ {FIGURES / 'phase2_5_policy_comparison.png'}")
    render_training_comparison(seed=42)
    print(f"  ✓ {FIGURES / 'phase2_5_training_comparison.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
