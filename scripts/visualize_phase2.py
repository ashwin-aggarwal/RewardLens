"""Generate all Phase 2 visualizations from recorded experiment data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rewardlens.envs.trajectory import Trajectory
from rewardlens.visualization.gridworld_viz import (
    render_trajectory_gif,
    BG_COLOR, TEXT_COLOR, GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR, AGENT_COLOR,
    _draw_grid, _draw_agent, _positions,
)

DQN_COLOR = "#8B5CF6"
FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _smooth(values: list[float], window: int = 20) -> list[float]:
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def render_training_curves(run_dir: Path, save_path: Path) -> None:
    rows = _load_csv(run_dir / "training_metrics.csv")
    if not rows:
        print("  No training data to plot")
        return

    steps = [int(r["timestep"]) for r in rows]
    rewards = [float(r["episode_reward"]) for r in rows]
    goals = [1.0 if r["goal_reached"] == "True" else 0.0 for r in rows]
    hacks = [1.0 if r["is_reward_hacking"] == "True" else 0.0 for r in rows]

    smoothed_rewards = _smooth(rewards, 50)
    smoothed_goals = _smooth(goals, 50)
    smoothed_hacks = _smooth(hacks, 50)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), dpi=150, sharex=True)
    fig.patch.set_facecolor(BG_COLOR)

    # Panel 1: Reward
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    ax.scatter(steps, rewards, alpha=0.15, s=4, color=DQN_COLOR, label="Episode reward")
    ax.plot(steps, smoothed_rewards, color=DQN_COLOR, lw=2, label="Smoothed (50-ep)")
    ax.set_ylabel("Proxy Reward", fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("DQN Training Progress", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=10)

    # Panel 2: Goal completion
    ax = axes[1]
    ax.set_facecolor(BG_COLOR)
    ax.scatter(steps, goals, alpha=0.15, s=4, color=GOAL_PATH_COLOR, label="Goal reached")
    ax.plot(steps, smoothed_goals, color=GOAL_PATH_COLOR, lw=2, label="Smoothed rate")
    ax.set_ylabel("Goal Completion", fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 3: Reward hacking
    ax = axes[2]
    ax.set_facecolor(BG_COLOR)
    ax.scatter(steps, hacks, alpha=0.15, s=4, color=EXPLOIT_PATH_COLOR, label="Reward hacking")
    ax.plot(steps, smoothed_hacks, color=EXPLOIT_PATH_COLOR, lw=2, label="Smoothed rate")
    ax.set_ylabel("Reward Hacking", fontsize=10, color=TEXT_COLOR)
    ax.set_xlabel("Environment Steps", fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    for a in axes:
        a.tick_params(colors=TEXT_COLOR)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation progress
# ---------------------------------------------------------------------------

def render_evaluation_progress(run_dir: Path, save_path: Path) -> None:
    rows = _load_csv(run_dir / "evaluation_metrics.csv")
    if not rows:
        print("  No evaluation data to plot")
        return

    steps = [int(r["timestep"]) for r in rows]
    rewards = [float(r["proxy_reward"]) for r in rows]
    visits = [int(r["reward_tile_visits"]) for r in rows]
    goals = [r["goal_reached"] == "True" for r in rows]
    hacks = [r["is_reward_hacking"] == "True" for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    ax = axes[0, 0]
    ax.set_facecolor(BG_COLOR)
    ax.plot(steps, rewards, color=DQN_COLOR, lw=2, marker="o", markersize=3)
    ax.set_ylabel("Proxy Reward", fontsize=9)
    ax.set_title("Greedy Eval Reward", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.set_facecolor(BG_COLOR)
    colors = [GOAL_PATH_COLOR if g else "#ccc" for g in goals]
    ax.bar(steps, [1] * len(steps), color=colors, width=max(1, (steps[-1] - steps[0]) / len(steps) * 0.8) if len(steps) > 1 else 1000)
    ax.set_ylabel("Goal Reached", fontsize=9)
    ax.set_title("Goal Completion", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])

    ax = axes[1, 0]
    ax.set_facecolor(BG_COLOR)
    ax.plot(steps, visits, color=EXPLOIT_PATH_COLOR, lw=2, marker="o", markersize=3)
    ax.set_ylabel("Reward Tile Visits", fontsize=9)
    ax.set_xlabel("Training Steps", fontsize=9)
    ax.set_title("Reward Tile Visits", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_facecolor(BG_COLOR)
    colors = [EXPLOIT_PATH_COLOR if h else "#ccc" for h in hacks]
    ax.bar(steps, [1] * len(steps), color=colors, width=max(1, (steps[-1] - steps[0]) / len(steps) * 0.8) if len(steps) > 1 else 1000)
    ax.set_ylabel("Reward Hacking", fontsize=9)
    ax.set_xlabel("Training Steps", fontsize=9)
    ax.set_title("Reward Hacking Classification", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])

    for a in axes.flat:
        a.tick_params(colors=TEXT_COLOR)

    fig.suptitle("Greedy Policy Evaluation Across Training",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trained DQN GIF
# ---------------------------------------------------------------------------

def render_trained_dqn_gif(traj_path: Path, save_path: Path) -> None:
    traj = Trajectory.load(traj_path)
    render_trajectory_gif(
        traj, save_path,
        max_frames=50,
        frame_duration_ms=180,
        last_frame_duration_ms=2000,
    )


# ---------------------------------------------------------------------------
# Scripted vs Learned comparison
# ---------------------------------------------------------------------------

def render_scripted_vs_learned(
    goal_traj: Trajectory,
    exploit_traj: Trajectory,
    dqn_traj: Trajectory,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    configs = [goal_traj.config, exploit_traj.config, dqn_traj.config]
    trajs = [goal_traj, exploit_traj, dqn_traj]
    titles = ["Scripted: Goal Policy", "Scripted: Exploit Policy", "Learned: Trained DQN"]
    colors = [GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR, DQN_COLOR]

    for ax, config, traj, title, color in zip(axes, configs, trajs, titles, colors):
        _draw_grid(ax, config, title=title)
        positions = _positions(traj)

        # Deduplicate path segments for cleaner display
        seen_segments = set()
        for i in range(1, len(positions)):
            seg = (positions[i - 1], positions[i])
            if seg not in seen_segments:
                seen_segments.add(seg)
                ax.annotate(
                    "",
                    xy=positions[i],
                    xytext=positions[i - 1],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.6),
                )

        # Count repeated segments for annotation
        from collections import Counter
        seg_counts = Counter()
        for i in range(1, len(positions)):
            seg_counts[(positions[i - 1], positions[i])] += 1
        for seg, count in seg_counts.items():
            if count > 2:
                mid_x = (seg[0][0] + seg[1][0]) / 2
                mid_y = (seg[0][1] + seg[1][1]) / 2
                ax.text(mid_x + 0.4, mid_y, f"×{count}", fontsize=7,
                        color=color, fontweight="bold", ha="left", va="center")

        _draw_agent(ax, positions[-1][0], positions[-1][1], color=color)

        goal_str = "✓" if traj.goal_reached else "✗"
        info_text = (
            f"Reward: {traj.total_reward:+.2f} | Goal: {goal_str} | "
            f"Steps: {traj.total_steps} | R-visits: {traj.final_reward_tile_visits}"
        )
        ax.text(
            config["width"] / 2 - 0.5, config["height"] + 0.1,
            info_text, ha="center", va="top", fontsize=8, color=TEXT_COLOR,
            fontfamily="monospace",
        )

    fig.suptitle(
        "Scripted Baselines vs Learned DQN Policy",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# DQN architecture diagram
# ---------------------------------------------------------------------------

def render_architecture_diagram(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    layers = [
        (5, 10.5, "Gridworld Observation", "7 features", "#E0E7FF", "#6366F1"),
        (5, 8.5, "Hidden Layer 1", "128 neurons + ReLU", "#FEF3C7", "#F59E0B"),
        (5, 6.5, "Hidden Layer 2", "128 neurons + ReLU", "#FEF3C7", "#F59E0B"),
        (5, 4.5, "Output Layer", "4 Q-values", "#DCFCE7", GOAL_PATH_COLOR),
        (5, 2.5, "Selected Action", "argmax Q(s, a)", "#FEE2E2", EXPLOIT_PATH_COLOR),
    ]

    for x, y, label, sublabel, bg, border in layers:
        rect = mpatches.FancyBboxPatch(
            (x - 2.8, y - 0.6), 5.6, 1.2,
            boxstyle="round,pad=0.15",
            facecolor=bg, edgecolor=border, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.15, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=TEXT_COLOR)
        ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                fontsize=9, color="#666")

    for i in range(len(layers) - 1):
        y_from = layers[i][1] - 0.6
        y_to = layers[i + 1][1] + 0.6
        ax.annotate(
            "", xy=(5, y_to), xytext=(5, y_from),
            arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=2),
        )

    ax.text(5, 11.5, "DQN Architecture", ha="center", va="center",
            fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.text(5, 1.3, "During training: ε-greedy adds random exploration",
            ha="center", va="center", fontsize=8, color="#888", style="italic")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="results/phase2/baseline_seed42")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    print("Generating Phase 2 visualizations...")

    # Architecture diagram (always generated)
    render_architecture_diagram(FIGURES / "phase2_dqn_architecture.png")
    print(f"  ✓ {FIGURES / 'phase2_dqn_architecture.png'}")

    # Training curves
    if (run_dir / "training_metrics.csv").exists():
        render_training_curves(run_dir, FIGURES / "phase2_training_curves.png")
        print(f"  ✓ {FIGURES / 'phase2_training_curves.png'}")

    # Evaluation progress
    if (run_dir / "evaluation_metrics.csv").exists():
        render_evaluation_progress(run_dir, FIGURES / "phase2_evaluation_progress.png")
        print(f"  ✓ {FIGURES / 'phase2_evaluation_progress.png'}")

    # Trained DQN GIF
    traj_path = run_dir / "evaluation_trajectories" / "final_greedy_trajectory.json"
    if traj_path.exists():
        render_trained_dqn_gif(traj_path, GIFS / "phase2_trained_dqn.gif")
        print(f"  ✓ {GIFS / 'phase2_trained_dqn.gif'}")

        # Scripted vs learned comparison
        goal_path = Path("results/phase1/goal_trajectory.json")
        exploit_path = Path("results/phase1/exploit_trajectory.json")
        if goal_path.exists() and exploit_path.exists():
            goal_traj = Trajectory.load(goal_path)
            exploit_traj = Trajectory.load(exploit_path)
            dqn_traj = Trajectory.load(traj_path)
            render_scripted_vs_learned(
                goal_traj, exploit_traj, dqn_traj,
                FIGURES / "phase2_scripted_vs_learned.png",
            )
            print(f"  ✓ {FIGURES / 'phase2_scripted_vs_learned.png'}")

    print("Done.")


if __name__ == "__main__":
    main()
