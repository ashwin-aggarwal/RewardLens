"""Visualization for RewardGridEnv: static figures, animated GIFs, and comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from rewardlens.envs.trajectory import Trajectory

# -- Visual identity ----------------------------------------------------------

BG_COLOR = "#F7F7F7"
GRID_LINE_COLOR = "#D0D0D0"
EMPTY_COLOR = "#FFFFFF"
AGENT_COLOR = "#3B82F6"
GOAL_COLOR = "#22C55E"
REWARD_COLOR = "#F97316"
GOAL_PATH_COLOR = "#16A34A"
EXPLOIT_PATH_COLOR = "#EF4444"
TEXT_COLOR = "#1E293B"
HIGHLIGHT_COLOR = "#FBBF24"

CELL_SIZE = 0.9


def _draw_grid(
    ax: plt.Axes,
    config: dict[str, Any],
    *,
    title: str = "",
    show_coords: bool = False,
) -> None:
    w, h = config["width"], config["height"]
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)

    for x in range(w):
        for y in range(h):
            rect = mpatches.FancyBboxPatch(
                (x - CELL_SIZE / 2, y - CELL_SIZE / 2),
                CELL_SIZE,
                CELL_SIZE,
                boxstyle="round,pad=0.02",
                facecolor=EMPTY_COLOR,
                edgecolor=GRID_LINE_COLOR,
                linewidth=0.8,
            )
            ax.add_patch(rect)

    gx, gy = config["goal_pos"]
    goal_rect = mpatches.FancyBboxPatch(
        (gx - CELL_SIZE / 2, gy - CELL_SIZE / 2),
        CELL_SIZE,
        CELL_SIZE,
        boxstyle="round,pad=0.02",
        facecolor=GOAL_COLOR,
        edgecolor=GOAL_COLOR,
        alpha=0.3,
        linewidth=1.5,
    )
    ax.add_patch(goal_rect)
    ax.text(gx, gy, "★", ha="center", va="center", fontsize=16, color=GOAL_COLOR, fontweight="bold")

    rx, ry = config["reward_pos"]
    reward_rect = mpatches.FancyBboxPatch(
        (rx - CELL_SIZE / 2, ry - CELL_SIZE / 2),
        CELL_SIZE,
        CELL_SIZE,
        boxstyle="round,pad=0.02",
        facecolor=REWARD_COLOR,
        edgecolor=REWARD_COLOR,
        alpha=0.3,
        linewidth=1.5,
    )
    ax.add_patch(reward_rect)
    ax.text(rx, ry, "$", ha="center", va="center", fontsize=16, color=REWARD_COLOR, fontweight="bold")

    if show_coords:
        for x in range(w):
            ax.text(x, -0.85, str(x), ha="center", va="center", fontsize=7, color="#999")
        for y in range(h):
            ax.text(-0.85, y, str(y), ha="center", va="center", fontsize=7, color="#999")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=10)


def _draw_agent(ax: plt.Axes, x: int, y: int, color: str = AGENT_COLOR) -> None:
    circle = plt.Circle((x, y), 0.3, color=color, zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, "●", ha="center", va="center", fontsize=8, color="white", zorder=11)


def _positions(traj: Trajectory) -> list[tuple[int, int]]:
    return [(s.agent_x, s.agent_y) for s in traj.steps]


# -- Gridworld overview -------------------------------------------------------

def render_gridworld_overview(config: dict[str, Any], save_path: str | Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    _draw_grid(ax, config, title="One Agent, Two Ways to Earn Reward", show_coords=True)

    sx, sy = config["agent_start"]
    _draw_agent(ax, sx, sy)

    gx, gy = config["goal_pos"]
    rx, ry = config["reward_pos"]

    legend_elements = [
        mpatches.Patch(facecolor=AGENT_COLOR, label=f"Agent start ({sx}, {sy})"),
        mpatches.Patch(facecolor=GOAL_COLOR, alpha=0.5, label=f"Goal  +{config['goal_bonus']:.0f}  ({gx}, {gy})"),
        mpatches.Patch(facecolor=REWARD_COLOR, alpha=0.5, label=f"Reward tile  +{config['reward_tile_bonus']:.0f}  ({rx}, {ry})"),
        mpatches.Patch(facecolor="white", edgecolor="#999", label=f"Step penalty  {config['step_penalty']}"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=8,
        frameon=True,
        fancybox=True,
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# -- Trajectory animation (GIF) -----------------------------------------------

def render_trajectory_gif(
    traj: Trajectory,
    save_path: str | Path,
    *,
    max_frames: int | None = None,
    frame_duration_ms: int = 200,
    last_frame_duration_ms: int = 1500,
) -> None:
    config = traj.config
    positions = _positions(traj)

    if max_frames and len(positions) > max_frames:
        # Sample: keep first, last, and evenly spaced middle
        indices = [0] + list(np.linspace(1, len(positions) - 2, max_frames - 2, dtype=int)) + [len(positions) - 1]
        indices = sorted(set(indices))
    else:
        indices = list(range(len(positions)))

    frames: list[Image.Image] = []

    for idx in indices:
        step_rec = traj.steps[idx]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5.5), dpi=100)
        fig.patch.set_facecolor(BG_COLOR)

        status = traj.policy_name
        if step_rec.terminated:
            status += "  ✓ Goal reached!"
        elif step_rec.truncated:
            status += "  ✗ Truncated"

        _draw_grid(ax, config, title=status)

        # Draw path so far
        path_color = GOAL_PATH_COLOR if "goal" in traj.policy_name.lower() else EXPLOIT_PATH_COLOR
        for j in range(1, idx + 1):
            prev = positions[j - 1] if j - 1 < len(positions) else positions[-1]
            curr = positions[j] if j < len(positions) else positions[-1]
            ax.annotate(
                "",
                xy=curr,
                xytext=prev,
                arrowprops=dict(arrowstyle="-", color=path_color, alpha=0.3, lw=1.5),
            )

        # Highlight reward tile on entry
        if step_rec.action is not None:
            if (step_rec.agent_x, step_rec.agent_y) == tuple(config["reward_pos"]):
                prev_step = traj.steps[idx - 1] if idx > 0 else step_rec
                if (prev_step.agent_x, prev_step.agent_y) != tuple(config["reward_pos"]):
                    highlight = mpatches.FancyBboxPatch(
                        (step_rec.agent_x - CELL_SIZE / 2, step_rec.agent_y - CELL_SIZE / 2),
                        CELL_SIZE, CELL_SIZE,
                        boxstyle="round,pad=0.02",
                        facecolor=HIGHLIGHT_COLOR, alpha=0.6, linewidth=2, edgecolor=REWARD_COLOR,
                    )
                    ax.add_patch(highlight)

        _draw_agent(ax, step_rec.agent_x, step_rec.agent_y)

        info_text = (
            f"Step: {step_rec.step}/{config['max_steps']}    "
            f"Reward: {step_rec.cumulative_reward:+.2f}    "
            f"R-visits: {step_rec.reward_tile_visits}"
        )
        ax.text(
            config["width"] / 2 - 0.5, config["height"] + 0.1,
            info_text, ha="center", va="top", fontsize=9, color=TEXT_COLOR,
            fontfamily="monospace",
        )

        fig.tight_layout()
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        img = Image.frombytes("RGBA", (w_px, h_px), fig.canvas.buffer_rgba())
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig)

    if not frames:
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    durations = [frame_duration_ms] * len(frames)
    durations[-1] = last_frame_duration_ms

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


# -- Side-by-side trajectory comparison ----------------------------------------

def render_trajectory_comparison(
    goal_traj: Trajectory,
    exploit_traj: Trajectory,
    save_path: str | Path,
) -> None:
    config = goal_traj.config
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    # Left panel: goal policy
    _draw_grid(ax1, config, title="Intended Goal Completion")
    goal_pos = _positions(goal_traj)
    for i in range(1, len(goal_pos)):
        ax1.annotate(
            "",
            xy=goal_pos[i],
            xytext=goal_pos[i - 1],
            arrowprops=dict(arrowstyle="->", color=GOAL_PATH_COLOR, lw=2, alpha=0.7),
        )
    sx, sy = goal_pos[0]
    ax1.text(sx, sy + 0.5, "Start", ha="center", va="top", fontsize=8, color=TEXT_COLOR)
    _draw_agent(ax1, goal_pos[-1][0], goal_pos[-1][1], color=GOAL_PATH_COLOR)
    ax1.text(
        config["width"] / 2 - 0.5, config["height"] + 0.1,
        f"Reward: {goal_traj.total_reward:+.2f} | Goal: ✓ | Steps: {goal_traj.total_steps}",
        ha="center", va="top", fontsize=9, color=TEXT_COLOR, fontfamily="monospace",
    )

    # Right panel: exploit policy
    _draw_grid(ax2, config, title="Proxy Reward Exploitation")
    exploit_pos = _positions(exploit_traj)

    # Draw initial path to reward tile
    # Find first time agent reaches reward tile
    first_reward_idx = None
    rpos = tuple(config["reward_pos"])
    for i, pos in enumerate(exploit_pos):
        if pos == rpos:
            first_reward_idx = i
            break

    if first_reward_idx:
        for i in range(1, first_reward_idx + 1):
            ax2.annotate(
                "",
                xy=exploit_pos[i],
                xytext=exploit_pos[i - 1],
                arrowprops=dict(arrowstyle="->", color=EXPLOIT_PATH_COLOR, lw=2, alpha=0.7),
            )

    # Draw farming loop annotation
    rx, ry = config["reward_pos"]
    adj_y = ry + 1 if ry + 1 < config["height"] else ry - 1
    loop_style = dict(arrowstyle="->", color=EXPLOIT_PATH_COLOR, lw=2.5, connectionstyle="arc3,rad=0.3")
    ax2.annotate("", xy=(rx, adj_y), xytext=(rx, ry), arrowprops=loop_style)
    ax2.annotate("", xy=(rx, ry), xytext=(rx, adj_y), arrowprops={**loop_style, "connectionstyle": "arc3,rad=-0.3"})
    ax2.text(
        rx + 0.65, (ry + adj_y) / 2,
        f"×{exploit_traj.final_reward_tile_visits}",
        ha="left", va="center", fontsize=11, fontweight="bold", color=EXPLOIT_PATH_COLOR,
    )

    sx, sy = exploit_pos[0]
    ax2.text(sx, sy + 0.5, "Start", ha="center", va="top", fontsize=8, color=TEXT_COLOR)
    _draw_agent(ax2, exploit_pos[-1][0], exploit_pos[-1][1], color=EXPLOIT_PATH_COLOR)
    ax2.text(
        config["width"] / 2 - 0.5, config["height"] + 0.1,
        f"Reward: {exploit_traj.total_reward:+.2f} | Goal: ✗ | Steps: {exploit_traj.total_steps}",
        ha="center", va="top", fontsize=9, color=TEXT_COLOR, fontfamily="monospace",
    )

    fig.suptitle(
        "Same Reward Function, Different Outcomes",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# -- Cumulative reward comparison chart ----------------------------------------

def render_cumulative_reward(
    goal_traj: Trajectory,
    exploit_traj: Trajectory,
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    goal_steps = [s.step for s in goal_traj.steps]
    goal_rewards = [s.cumulative_reward for s in goal_traj.steps]
    exploit_steps = [s.step for s in exploit_traj.steps]
    exploit_rewards = [s.cumulative_reward for s in exploit_traj.steps]

    ax.plot(goal_steps, goal_rewards, color=GOAL_PATH_COLOR, linewidth=2.5, label="Goal policy", zorder=5)
    ax.plot(exploit_steps, exploit_rewards, color=EXPLOIT_PATH_COLOR, linewidth=2.5, label="Exploit policy", zorder=5)

    # Mark goal completion
    if goal_traj.goal_reached:
        final = goal_traj.steps[-1]
        ax.scatter([final.step], [final.cumulative_reward], color=GOAL_PATH_COLOR, s=100, zorder=6, edgecolors="white", linewidth=2)
        ax.annotate(
            f"Goal reached\n{final.cumulative_reward:+.2f}",
            xy=(final.step, final.cumulative_reward),
            xytext=(final.step + 8, final.cumulative_reward - 5),
            fontsize=9, color=GOAL_PATH_COLOR,
            arrowprops=dict(arrowstyle="->", color=GOAL_PATH_COLOR, lw=1.5),
        )

    # Mark exploit truncation
    exploit_final = exploit_traj.steps[-1]
    ax.scatter([exploit_final.step], [exploit_final.cumulative_reward], color=EXPLOIT_PATH_COLOR, s=100, zorder=6, edgecolors="white", linewidth=2)
    ax.annotate(
        f"Truncated (no goal)\n{exploit_final.cumulative_reward:+.2f}",
        xy=(exploit_final.step, exploit_final.cumulative_reward),
        xytext=(exploit_final.step - 30, exploit_final.cumulative_reward + 10),
        fontsize=9, color=EXPLOIT_PATH_COLOR,
        arrowprops=dict(arrowstyle="->", color=EXPLOIT_PATH_COLOR, lw=1.5),
    )

    ax.set_xlabel("Environment Steps", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Cumulative Proxy Reward", fontsize=11, color=TEXT_COLOR)
    ax.set_title(
        "More Reward ≠ Better Behavior",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12,
    )
    ax.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors=TEXT_COLOR)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
