"""Pre-training sanity check: compare scripted policies under discounted returns."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rewardlens.envs.trajectory import Trajectory
from rewardlens.cli import script_args

BG_COLOR = "#F7F7F7"
GOAL_COLOR = "#16A34A"
EXPLOIT_COLOR = "#EF4444"
TEXT_COLOR = "#1E293B"

GAMMAS = [0.95, 0.99, 0.995, 1.00]


def discounted_return(rewards: list[float], gamma: float) -> float:
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def main() -> None:
    goal_traj = Trajectory.load("results/phase1/goal_trajectory.json")
    exploit_traj = Trajectory.load("results/phase1/exploit_trajectory.json")

    goal_rewards = [s.reward for s in goal_traj.steps[1:]]
    exploit_rewards = [s.reward for s in exploit_traj.steps[1:]]

    results = {}
    print(f"\n{'Gamma':>8} {'Goal Return':>14} {'Exploit Return':>16} {'Exploit > Goal':>16}")
    print("-" * 58)
    for gamma in GAMMAS:
        g_goal = discounted_return(goal_rewards, gamma)
        g_exploit = discounted_return(exploit_rewards, gamma)
        favors = g_exploit > g_goal
        print(f"{gamma:>8.3f} {g_goal:>14.4f} {g_exploit:>16.4f} {str(favors):>16}")
        results[str(gamma)] = {
            "goal_return": round(g_goal, 4),
            "exploit_return": round(g_exploit, 4),
            "exploit_favored": favors,
        }

    # Save results
    out_dir = Path("results/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "discounted_returns.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualization
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(len(GAMMAS))
    width = 0.35

    goal_returns = [results[str(g)]["goal_return"] for g in GAMMAS]
    exploit_returns = [results[str(g)]["exploit_return"] for g in GAMMAS]

    bars1 = ax.bar(x - width / 2, goal_returns, width, label="Goal policy", color=GOAL_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width / 2, exploit_returns, width, label="Exploit policy", color=EXPLOIT_COLOR, alpha=0.85)

    ax.set_xlabel("Discount Factor (γ)", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Discounted Return", fontsize=11, color=TEXT_COLOR)
    ax.set_title("Exploit Policy Dominates at All Tested Discount Factors",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"γ = {g}" for g in GAMMAS])
    ax.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(colors=TEXT_COLOR)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8, color=GOAL_COLOR)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8, color=EXPLOIT_COLOR)

    plt.tight_layout()
    fig_path = Path("assets/figures/phase2_discounted_return.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {out_dir / 'discounted_returns.json'}")


if __name__ == "__main__":
    script_args(__doc__)
    main()
