"""Generate Phase 4 visualizations from recorded intervention results.

Usage:
    python -m scripts.visualize_phase4
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_extraction import (
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
)
from rewardlens.analysis.activation_intervention import (
    fit_probe_direction,
    greedy_rollout_with_intervention,
    partition_start_positions,
    random_unit_directions,
)
from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv, ACTION_NAMES
from rewardlens.envs.trajectory import record_episode
from rewardlens.visualization.gridworld_viz import (
    BG_COLOR, TEXT_COLOR, GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR,
    _draw_grid, _draw_agent, _positions, render_trajectory_gif,
)
from rewardlens.cli import script_args
from scripts.probe_activations import compute_farming_labels
from scripts.run_interventions import _position_mask

FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
RESULTS = Path("results/phase4")
FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")
CONFIG_PATH = Path("experiments/01_fixed_gridworld_repair/configs/phase4_interventions.json")

BASELINE_COLOR = EXPLOIT_PATH_COLOR      # red (farming baseline)
TARGETED_COLOR = "#8B5CF6"               # violet (targeted edit)
RANDOM_COLOR = "#94A3B8"                 # gray (random controls)
GOAL_COLOR = GOAL_PATH_COLOR             # green


def _load_results() -> dict:
    with open(RESULTS / "intervention_results.json") as f:
        return json.load(f)


def _style(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 1. Intervention mechanism diagram (conceptual, labeled as such)
# ---------------------------------------------------------------------------

def render_intervention_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.5, "Observation\n(7 features)", "#94A3B8"),
        (2.7, "Hidden 1\n(128, ReLU)", "#F59E0B"),
        (4.9, "Hidden 2\n(128, ReLU)", "#F59E0B"),
        (7.6, "ACTIVATION EDIT\nh' = h − α·v", TARGETED_COLOR),
        (10.0, "Q-values\n(4 actions)", "#3B82F6"),
    ]
    for x, label, color in boxes:
        rect = plt.Rectangle((x, 1.8), 1.8, 1.4, facecolor=color, alpha=0.25,
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.9, 2.5, label, ha="center", va="center", fontsize=9,
                color=TEXT_COLOR, fontweight="bold")
    for x1, x2 in [(2.3, 2.7), (4.5, 4.9), (6.7, 7.6), (9.4, 10.0)]:
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#64748B", lw=2))

    ax.text(8.5, 0.9,
            "v = probe-derived farming direction (unit norm, raw activation space)\n"
            "α = intervention strength, scaled to the typical layer-2 activation norm",
            ha="center", fontsize=8, color=TEXT_COLOR)
    ax.text(8.5, 4.1, "Weights are never modified —\nonly the activations flowing through",
            ha="center", fontsize=8, color=TARGETED_COLOR, fontstyle="italic")
    ax.text(0.5, 0.3, "Conceptual diagram (not generated from experimental data)",
            fontsize=7, color="#94A3B8", fontstyle="italic")

    ax.set_title("Phase 4: Editing Hidden Activations at Inference Time",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase4_intervention_diagram.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Strength sweep (dev states, all seeds)
# ---------------------------------------------------------------------------

def render_strength_sweep() -> None:
    data = _load_results()
    seeds = list(data["flawed"].keys())
    rels = [float(r) for r in data["config"]["relative_strengths"]]

    fig, axes = plt.subplots(2, len(seeds), figsize=(4.2 * len(seeds), 7), dpi=150,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)

    for col, seed in enumerate(seeds):
        sweep = data["flawed"][seed]["dev_sweep"]
        sub_goal = [sweep["probe_subtract"][str(r)]["goal_rate"] for r in rels]
        sub_hack = [sweep["probe_subtract"][str(r)]["hacking_rate"] for r in rels]
        add_goal = [sweep["probe_add"][str(r)]["goal_rate"] for r in rels]
        add_hack = [sweep["probe_add"][str(r)]["hacking_rate"] for r in rels]
        rnd_goal = np.array([[x["goal_rate"] for x in sweep["random_subtract"][str(r)]] for r in rels])
        rnd_hack = np.array([[x["hacking_rate"] for x in sweep["random_subtract"][str(r)]] for r in rels])

        ax = axes[0, col]
        _style(ax, f"Seed {seed}: Goal Completion")
        ax.plot(rels, sub_goal, "-o", color=TARGETED_COLOR, label="Probe subtract")
        ax.plot(rels, add_goal, "--s", color=TARGETED_COLOR, alpha=0.5, label="Probe add")
        ax.plot(rels, rnd_goal.mean(1), "-", color=RANDOM_COLOR, label="Random mean (n=20)")
        ax.fill_between(rels, rnd_goal.min(1), rnd_goal.max(1), color=RANDOM_COLOR, alpha=0.25,
                        label="Random range")
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel("Goal completion rate", fontsize=9)
            ax.legend(fontsize=7)

        ax = axes[1, col]
        _style(ax, f"Seed {seed}: Reward Hacking")
        ax.plot(rels, sub_hack, "-o", color=TARGETED_COLOR, label="Probe subtract")
        ax.plot(rels, add_hack, "--s", color=TARGETED_COLOR, alpha=0.5, label="Probe add")
        ax.plot(rels, rnd_hack.mean(1), "-", color=RANDOM_COLOR)
        ax.fill_between(rels, rnd_hack.min(1), rnd_hack.max(1), color=RANDOM_COLOR, alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Relative intervention strength (α / typical ‖h₂‖)", fontsize=8)
        if col == 0:
            ax.set_ylabel("Reward-hacking rate", fontsize=9)

    fig.suptitle(
        "Intervention Strength Sweep (development start states, 16 per seed)\n"
        "Targeted probe-direction edits eliminate farming at small strengths, "
        "but goal completion is NOT restored",
        fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "phase4_strength_sweep.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Targeted vs random controls (held-out test states)
# ---------------------------------------------------------------------------

def render_random_controls() -> None:
    data = _load_results()
    seeds = list(data["flawed"].keys())

    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 4.5), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor(BG_COLOR)

    rng = np.random.RandomState(0)
    for col, seed in enumerate(seeds):
        test = data["flawed"][seed]["test"]
        rel = data["flawed"][seed]["selected_relative_strength"]
        rnd_hack = [r["hacking_rate"] for r in test["random_subtract"]]
        rnd_goal = [r["goal_rate"] for r in test["random_subtract"]]

        ax = axes[col]
        _style(ax, f"Seed {seed} (α = {rel}·‖h₂‖, test states)")
        jitter = rng.uniform(-0.08, 0.08, len(rnd_hack))
        ax.scatter(np.zeros(len(rnd_hack)) + jitter, rnd_hack, color=RANDOM_COLOR,
                   alpha=0.7, s=40, label="Random directions (n=20)")
        ax.scatter([0], [np.mean(rnd_hack)], marker="_", s=600, color="#475569", lw=2)
        ax.scatter([1], [test["baseline"]["hacking_rate"]], marker="D", s=90,
                   color=BASELINE_COLOR, label="No intervention")
        ax.scatter([2], [test["probe_subtract"]["hacking_rate"]], marker="*", s=220,
                   color=TARGETED_COLOR, label="Probe subtract")
        ax.scatter([3], [test["probe_subtract"]["goal_rate"]], marker="*", s=220,
                   color=GOAL_COLOR, label="Probe subtract: goal rate")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Random\n(hack rate)", "Baseline\n(hack rate)",
                            "Targeted\n(hack rate)", "Targeted\n(goal rate)"], fontsize=7)
        ax.set_ylim(-0.05, 1.1)
        if col == 0:
            ax.set_ylabel("Rate over 16 held-out start states", fontsize=9)
            ax.legend(fontsize=7, loc="center right")

    fig.suptitle(
        "Targeted vs Random-Direction Interventions (held-out evaluation)\n"
        "The targeted direction suppresses farming more reliably than norm-matched random edits,\n"
        "but goal completion stays near zero — disruption, not redirection",
        fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.06)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "phase4_random_controls.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4 + 5. Behavior comparison figure and before/after GIFs
# ---------------------------------------------------------------------------

def _rollout_trajectory(net, start_pos, direction, alpha, mode, policy_name, episode_id):
    cfg = RewardGridConfig(agent_start=tuple(start_pos))
    roll = greedy_rollout_with_intervention(net, cfg, direction, alpha, mode)
    env = RewardGridEnv(config=RewardGridConfig(agent_start=tuple(start_pos)))
    traj = record_episode(env, roll["actions"], policy_name, episode_id, seed=0)
    return roll, traj


def render_behavior_comparison_and_gifs() -> None:
    data = _load_results()
    cfg_all = data["config"]
    seed = str(cfg_all["primary_seed"])
    entry = data["flawed"][seed]
    rel = entry["selected_relative_strength"]
    alpha = entry["selected_alpha"]

    ckpt = Path(entry["checkpoint"])
    net, _ = load_q_network_for_eval(ckpt)

    # Refit the direction exactly as run_interventions did (deterministic)
    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)
    parts = partition_start_positions(RewardGridConfig(), n_dev=cfg_all["n_dev_positions"],
                                      n_test=cfg_all["n_test_positions"],
                                      seed=cfg_all["partition_seed"])
    corrected_ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    fl = extract_activations(ckpt, observations, int(seed), "flawed", 100000)
    co = extract_activations(corrected_ckpt, observations, int(seed), "corrected", 100000)
    labels = compute_farming_labels(fl.greedy_actions, co.greedy_actions, metadata)
    train_mask = _position_mask(observations, parts["train"])
    direction = fit_probe_direction(fl.hidden2[train_mask], labels[train_mask],
                                    str(ckpt), C=cfg_all["probe_C"], seed=cfg_all["probe_seed"])

    # Median-outcome random control (by test hacking rate) — no cherry-picking
    randoms = random_unit_directions(128, cfg_all["n_random_directions"],
                                     cfg_all["random_direction_seed"])
    rnd_hacks = [r["hacking_rate"] for r in entry["test"]["random_subtract"]]
    median_idx = int(np.argsort(rnd_hacks)[len(rnd_hacks) // 2])

    start_pos = tuple(data["partition"]["test"][0])

    roll_base, traj_base = _rollout_trajectory(net, start_pos, None, 0.0, "none",
                                               "Original DQN", "phase4_base")
    roll_edit, traj_edit = _rollout_trajectory(net, start_pos, direction.vector, alpha,
                                               "subtract", "Probe-subtract", "phase4_edit")
    roll_rand, traj_rand = _rollout_trajectory(net, start_pos, randoms[median_idx], alpha,
                                               "subtract", "Random control", "phase4_rand")

    # --- Figure: side-by-side trajectories ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    panels = [
        (axes[0], traj_base, roll_base, "Original DQN (no edit)", BASELINE_COLOR),
        (axes[1], traj_edit, roll_edit, f"Probe subtract (α = {rel}·‖h₂‖)", TARGETED_COLOR),
        (axes[2], traj_rand, roll_rand, "Median random control", RANDOM_COLOR),
    ]
    for ax, traj, roll, title, color in panels:
        _draw_grid(ax, traj.config, title=title)
        pos = _positions(traj)
        seen = set()
        for i in range(1, len(pos)):
            seg = (pos[i - 1], pos[i])
            if seg not in seen:
                seen.add(seg)
                ax.annotate("", xy=pos[i], xytext=pos[i - 1],
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.65))
        _draw_agent(ax, pos[-1][0], pos[-1][1], color=color)
        ax.text(traj.config["width"] / 2 - 0.5, traj.config["height"] + 0.4,
                f"Goal: {'Yes' if roll['goal_reached'] else 'No'} | "
                f"Hacking: {'Yes' if roll['is_reward_hacking'] else 'No'} | "
                f"Tile visits: {roll['reward_tile_visits']} | "
                f"Reward: {roll['proxy_reward']:+.2f} | Steps: {roll['episode_steps']}",
                ha="center", va="top", fontsize=8, color=TEXT_COLOR, fontfamily="monospace")

    fig.suptitle(
        f"Same Start State ({start_pos}), Same Trained Weights, Different Activation Edits "
        f"(flawed-reward DQN, seed {seed})\n"
        "The targeted edit stops the farming loop but does not produce goal completion — "
        "the agent wanders instead",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.06)
    plt.tight_layout()
    fig.savefig(FIGURES / "phase4_behavior_comparison.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- GIFs ---
    GIFS.mkdir(parents=True, exist_ok=True)
    render_trajectory_gif(traj_base, GIFS / "phase4_original_policy.gif",
                          max_frames=30, frame_duration_ms=250, last_frame_duration_ms=2000)
    render_trajectory_gif(traj_edit, GIFS / "phase4_intervened_policy.gif",
                          max_frames=30, frame_duration_ms=250, last_frame_duration_ms=2000)

    return {
        "start_pos": list(start_pos),
        "baseline": {k: roll_base[k] for k in ("goal_reached", "is_reward_hacking",
                                               "reward_tile_visits", "proxy_reward")},
        "intervened": {k: roll_edit[k] for k in ("goal_reached", "is_reward_hacking",
                                                 "reward_tile_visits", "proxy_reward")},
    }


# ---------------------------------------------------------------------------
# 6. Q-value shift figure
# ---------------------------------------------------------------------------

def render_q_value_shift() -> None:
    """Show that subtracting a constant vector at layer 2 shifts Q-values by a
    state-independent constant: fc3 is linear, so Q'(s) = Q(s) - alpha * W v."""
    import torch
    data = _load_results()
    seed = str(data["config"]["primary_seed"])
    entry = data["flawed"][seed]
    net, _ = load_q_network_for_eval(Path(entry["checkpoint"]))

    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)
    parts = {k: [tuple(p) for p in v] for k, v in data["partition"].items()}

    # Direction (refit, deterministic)
    corrected_ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    fl = extract_activations(Path(entry["checkpoint"]), observations, int(seed), "flawed", 100000)
    co = extract_activations(corrected_ckpt, observations, int(seed), "corrected", 100000)
    labels = compute_farming_labels(fl.greedy_actions, co.greedy_actions, metadata)
    train_mask = _position_mask(observations, parts["train"])
    direction = fit_probe_direction(fl.hidden2[train_mask], labels[train_mask], entry["checkpoint"])

    alpha = entry["selected_alpha"]
    with torch.no_grad():
        shift = -alpha * torch.as_tensor(direction.vector) @ net.fc3.weight.T
    shift = shift.numpy()

    # Representative states: 4 test states near the reward tile and near the goal
    dr, dg = metadata["dist_to_reward"], metadata["dist_to_goal"]
    t = metadata["time_remaining"]
    test_mask = _position_mask(observations, parts["test"])
    idx_pool = np.where(test_mask & (np.abs(t - 0.5) < 0.01))[0]
    idx_near_reward = idx_pool[np.argsort(dr[idx_pool])[:2]]
    idx_near_goal = idx_pool[np.argsort(dg[idx_pool])[:2]]
    chosen = list(idx_near_reward) + list(idx_near_goal)
    titles = ["Near reward tile", "Near reward tile (2)", "Near goal", "Near goal (2)"]

    fig, axes = plt.subplots(1, len(chosen) + 1, figsize=(3.3 * (len(chosen) + 1), 4), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    for ax, idx, title in zip(axes[:-1], chosen, titles):
        _style(ax, f"{title}\npos=({metadata['agent_x'][idx]},{metadata['agent_y'][idx]})")
        q = fl.q_values[idx]
        q_edit = q + shift
        x = np.arange(4)
        ax.bar(x - 0.2, q, 0.4, color=BASELINE_COLOR, alpha=0.8, label="Original Q")
        ax.bar(x + 0.2, q_edit, 0.4, color=TARGETED_COLOR, alpha=0.8, label="Edited Q")
        ax.set_xticks(x)
        ax.set_xticklabels([ACTION_NAMES[i] for i in range(4)], fontsize=7)
        ax.axhline(0, color="#CBD5E1", lw=0.8)
    axes[0].legend(fontsize=7)

    ax = axes[-1]
    _style(ax, "The uniform Q-shift\n(same at every state)")
    ax.bar(np.arange(4), shift, color=TARGETED_COLOR, alpha=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([ACTION_NAMES[i] for i in range(4)], fontsize=7)
    ax.axhline(0, color="#CBD5E1", lw=0.8)

    fig.suptitle(
        "Why the Edit Acts Globally: Subtracting a Constant Vector Before a Linear Layer\n"
        f"= Adding a Fixed Per-Action Q Bias at Every State (seed {seed}, α = "
        f"{entry['selected_relative_strength']}·‖h₂‖)",
        fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.08)
    plt.tight_layout()
    fig.savefig(FIGURES / "phase4_q_value_shift.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating Phase 4 visualizations...")
    render_intervention_diagram()
    print(f"  ✓ {FIGURES / 'phase4_intervention_diagram.png'}")
    render_strength_sweep()
    print(f"  ✓ {FIGURES / 'phase4_strength_sweep.png'}")
    render_random_controls()
    print(f"  ✓ {FIGURES / 'phase4_random_controls.png'}")
    info = render_behavior_comparison_and_gifs()
    print(f"  ✓ {FIGURES / 'phase4_behavior_comparison.png'}")
    print(f"  ✓ {GIFS / 'phase4_original_policy.gif'} / phase4_intervened_policy.gif "
          f"(start {info['start_pos']})")
    render_q_value_shift()
    print(f"  ✓ {FIGURES / 'phase4_q_value_shift.png'}")
    print("Done.")


if __name__ == "__main__":
    script_args(__doc__)
    main()
