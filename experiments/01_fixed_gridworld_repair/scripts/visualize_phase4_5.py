"""Generate Phase 4.5 visualizations from recorded patching results.

Usage:
    python -m scripts.visualize_phase4_5
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_patching import (
    evaluate_patching,
    make_matched_donor,
    patched_rollout,
)
from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv
from rewardlens.envs.trajectory import record_episode
from rewardlens.visualization.gridworld_viz import (
    BG_COLOR, TEXT_COLOR, GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR,
    _draw_grid, _draw_agent, _positions, render_trajectory_gif,
)
from rewardlens.cli import script_args

FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
RESULTS = Path("results/phase4_5")
FLAWED_DIR = Path("results/phase2")

HACK_COLOR = EXPLOIT_PATH_COLOR
DONOR_COLOR = GOAL_PATH_COLOR
PATCH_COLOR = "#0EA5E9"      # sky blue
RANDOM_COLOR = "#94A3B8"
SHUFFLE_COLOR = "#F59E0B"
CONST_COLOR = "#8B5CF6"


def _load():
    with open(RESULTS / "patching_results.json") as f:
        return json.load(f)


def _style(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 1. Patching diagram (conceptual)
# ---------------------------------------------------------------------------

def render_patching_diagram():
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, label, color):
        ax.add_patch(plt.Rectangle((x, y), 1.9, 1.0, facecolor=color, alpha=0.25,
                                   edgecolor=color, linewidth=2))
        ax.text(x + 0.95, y + 0.5, label, ha="center", va="center", fontsize=8.5,
                color=TEXT_COLOR, fontweight="bold")

    # Donor row (top)
    ax.text(0.2, 5.2, "GOAL-FOLLOWING DONOR", fontsize=9, color=DONOR_COLOR, fontweight="bold")
    box(0.3, 3.9, "Observation s", "#94A3B8")
    box(2.7, 3.9, "Donor H1", DONOR_COLOR)
    box(5.1, 3.9, "Donor H2\nh_goal(s)", DONOR_COLOR)
    ax.text(7.3, 4.4, "(donor's own Q head\nis NOT used)", fontsize=7.5, color="#94A3B8",
            fontstyle="italic", va="center")

    # Recipient row (bottom)
    ax.text(0.2, 2.4, "REWARD-HACKING RECIPIENT", fontsize=9, color=HACK_COLOR, fontweight="bold")
    box(0.3, 1.1, "Observation s\n(same s)", "#94A3B8")
    box(2.7, 1.1, "Recipient H1", HACK_COLOR)
    box(5.1, 1.1, "Recipient H2\nh_hack(s)", HACK_COLOR)
    box(7.6, 1.1, "PATCH\nh + α(h_goal − h)", PATCH_COLOR)
    box(10.0, 1.1, "Recipient's\nQ head (fc3)", "#3B82F6")

    for x1, x2, y in [(2.2, 2.7, 4.4), (4.6, 5.1, 4.4), (2.2, 2.7, 1.6), (4.6, 5.1, 1.6),
                      (7.0, 7.6, 1.6), (9.5, 10.0, 1.6)]:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#64748B", lw=2))
    ax.annotate("", xy=(8.3, 2.1), xytext=(6.4, 3.9),
                arrowprops=dict(arrowstyle="->", color=DONOR_COLOR, lw=2.5))
    ax.text(8.1, 3.1, "state-dependent\ninjection", fontsize=8, color=DONOR_COLOR, fontstyle="italic")
    ax.text(6.0, 0.35, "Weights of both networks untouched. Cross-model donors pass through a "
            "fitted linear map M before injection.", ha="center", fontsize=8, color=TEXT_COLOR)
    ax.text(0.2, 0.05, "Conceptual diagram (not generated from experimental data)",
            fontsize=7, color="#94A3B8", fontstyle="italic")
    ax.set_title("Phase 4.5: Patching Goal-Following Activations into a Reward-Hacking Network",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=10)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase4_5_patching_diagram.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Strength sweep (dev states): same-run and cross-model
# ---------------------------------------------------------------------------

def render_strength_sweep():
    d = _load()
    alphas = [float(a) for a in d["config"]["alphas"]]
    blocks = [("Same run: 20k → 30k", d["same_run"]["30000"]),
              ("Same run: 20k → 100k", d["same_run"]["100000"]),
              ("Cross-model: identity map", d["cross_model"]["identity"]),
              ("Cross-model: ridge (agreement states)", d["cross_model"]["ridge_agreement"])]

    fig, axes = plt.subplots(2, len(blocks), figsize=(4.1 * len(blocks), 7), dpi=150,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for col, (title, blk) in enumerate(blocks):
        sw = blk["dev_sweep"]
        mg = [sw[str(a)]["matched"]["goal_rate"] for a in alphas]
        mh = [sw[str(a)]["matched"]["hacking_rate"] for a in alphas]
        mv = [sw[str(a)]["matched"]["mean_reward_tile_visits"] / 49 for a in alphas]
        rg = [np.mean([r["goal_rate"] for r in sw[str(a)]["random"]]) if sw[str(a)]["random"] else mg[i]
              for i, a in enumerate(alphas)]
        sg = [np.mean([r["goal_rate"] for r in sw[str(a)]["shuffled"]]) if sw[str(a)]["shuffled"] else mg[i]
              for i, a in enumerate(alphas)]
        rh = [np.mean([r["hacking_rate"] for r in sw[str(a)]["random"]]) if sw[str(a)]["random"] else mh[i]
              for i, a in enumerate(alphas)]
        sh = [np.mean([r["hacking_rate"] for r in sw[str(a)]["shuffled"]]) if sw[str(a)]["shuffled"] else mh[i]
              for i, a in enumerate(alphas)]

        ax = axes[0, col]
        _style(ax, f"{title}\nGoal completion")
        ax.plot(alphas, mg, "-o", color=PATCH_COLOR, lw=2, label="Matched goal patch")
        ax.plot(alphas, rg, "--s", color=RANDOM_COLOR, label="Random (mean of 20)")
        ax.plot(alphas, sg, "--^", color=SHUFFLE_COLOR, label="Shuffled-state donor (mean of 20)")
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel("Goal completion rate", fontsize=9)
            ax.legend(fontsize=7)

        ax = axes[1, col]
        _style(ax, "Reward hacking / tile visits")
        ax.plot(alphas, mh, "-o", color=PATCH_COLOR, lw=2, label="Matched: hacking rate")
        ax.plot(alphas, mv, ":o", color=PATCH_COLOR, alpha=0.6, label="Matched: tile visits / 49")
        ax.plot(alphas, rh, "--s", color=RANDOM_COLOR, label="Random: hacking rate")
        ax.plot(alphas, sh, "--^", color=SHUFFLE_COLOR, label="Shuffled: hacking rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Patch strength α", fontsize=9)
        if col == 0:
            ax.set_ylabel("Rate", fontsize=9)
            ax.legend(fontsize=7)

    fig.suptitle("Patching Strength Sweep (development start states, 16 per condition)\n"
                 "Same-run patches restore goal completion; cross-model patches do not. "
                 "Dotted line: the patched agent still visits the reward tile before finishing.",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "phase4_5_strength_sweep.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Controls (held-out test states)
# ---------------------------------------------------------------------------

def render_controls():
    d = _load()
    blocks = [("Same run\n20k → 30k (α=%s)" % d["same_run"]["30000"]["selected_alpha"], d["same_run"]["30000"]),
              ("Same run\n20k → 100k (α=%s)" % d["same_run"]["100000"]["selected_alpha"], d["same_run"]["100000"]),
              ("Cross-model\nidentity (α=%s)" % d["cross_model"]["identity"]["selected_alpha"], d["cross_model"]["identity"]),
              ("Cross-model\nridge-all (α=%s)" % d["cross_model"]["ridge_all"]["selected_alpha"], d["cross_model"]["ridge_all"]),
              ("Cross-model\nridge-agreement (α=%s)" % d["cross_model"]["ridge_agreement"]["selected_alpha"], d["cross_model"]["ridge_agreement"])]

    fig, axes = plt.subplots(2, len(blocks), figsize=(3.6 * len(blocks), 7.5), dpi=150, sharey="row")
    fig.patch.set_facecolor(BG_COLOR)
    rng = np.random.RandomState(0)
    for col, (title, blk) in enumerate(blocks):
        t = blk["test"]
        for row, metric in enumerate(("goal_rate", "mean_reward_tile_visits")):
            ax = axes[row, col]
            _style(ax, title if row == 0 else "")
            scale = 1.0 if metric == "goal_rate" else 1.0
            rnd = [r[metric] * scale for r in t["random"]]
            shf = [r[metric] * scale for r in t["shuffled"]]
            ax.scatter(np.zeros(len(rnd)) + rng.uniform(-0.1, 0.1, len(rnd)), rnd,
                       color=RANDOM_COLOR, alpha=0.7, s=28, label="Random (n=20)")
            ax.scatter(np.ones(len(shf)) + rng.uniform(-0.1, 0.1, len(shf)), shf,
                       color=SHUFFLE_COLOR, alpha=0.7, s=28, label="Shuffled donor (n=20)")
            ax.scatter([2], [t["constant_qbias"][metric] * scale], marker="s", s=90,
                       color=CONST_COLOR, label="Constant Q-bias")
            ax.scatter([3], [t["baseline"][metric] * scale], marker="D", s=80,
                       color=HACK_COLOR, label="Unmodified recipient")
            ax.scatter([4], [t["matched"][metric] * scale], marker="*", s=240,
                       color=PATCH_COLOR, label="Matched goal patch")
            ax.scatter([5], [t["donor_independent"][metric] * scale], marker="P", s=110,
                       color=DONOR_COLOR, label="Donor alone")
            ax.set_xticks(range(6))
            ax.set_xticklabels(["Rand", "Shuf", "Q-bias", "Base", "Patch", "Donor"], fontsize=7)
            if col == 0:
                ax.set_ylabel("Goal completion rate" if row == 0 else "Mean reward-tile visits", fontsize=9)
                if row == 0:
                    ax.legend(fontsize=6, loc="upper left")
            if row == 0:
                ax.set_ylim(-0.05, 1.1)

    fig.suptitle("Patching vs Controls on Held-Out Test States (16 start states each)\n"
                 "Top: goal completion. Bottom: reward-tile visits — the same-run patch reaches the goal "
                 "but still farms first.",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "phase4_5_controls.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4 + 5. Behavior comparison and GIFs (same start state, actual recorded actions)
# ---------------------------------------------------------------------------

def _traj(net, start_pos, alpha, donor_fn, mode, name, eid):
    cfg = RewardGridConfig(agent_start=tuple(start_pos))
    roll = patched_rollout(net, cfg, alpha, donor_fn, mode)
    env = RewardGridEnv(config=RewardGridConfig(agent_start=tuple(start_pos)))
    return roll, record_episode(env, roll["actions"], name, eid, seed=0)


def render_behavior_and_gifs():
    d = _load()
    blk = d["same_run"]["30000"]
    alpha = blk["selected_alpha"]
    ck = FLAWED_DIR / "baseline_seed123" / "checkpoints"
    donor, _ = load_q_network_for_eval(ck / "checkpoint_20000.pt")
    recipient, _ = load_q_network_for_eval(ck / "checkpoint_30000.pt")
    donor_fn = make_matched_donor(donor)
    start = tuple(d["partition"]["test"][0])

    roll_h, tr_h = _traj(recipient, start, 0.0, None, "none", "Hacking recipient (30k)", "p45_hack")
    roll_d, tr_d = _traj(donor, start, 0.0, None, "none", "Goal donor (20k)", "p45_donor")
    roll_p, tr_p = _traj(recipient, start, alpha, donor_fn, "matched", "Patched recipient", "p45_patch")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, tr, roll, title, color in [
        (axes[0], tr_h, roll_h, "Original hacking DQN (30k)", HACK_COLOR),
        (axes[1], tr_d, roll_d, "Goal-following donor (20k)", DONOR_COLOR),
        (axes[2], tr_p, roll_p, f"Patched recipient (α={alpha})", PATCH_COLOR),
    ]:
        _draw_grid(ax, tr.config, title=title)
        pos = _positions(tr)
        seen = set()
        for i in range(1, len(pos)):
            seg = (pos[i - 1], pos[i])
            if seg not in seen:
                seen.add(seg)
                ax.annotate("", xy=pos[i], xytext=pos[i - 1],
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.65))
        _draw_agent(ax, pos[-1][0], pos[-1][1], color=color)
        ax.text(tr.config["width"] / 2 - 0.5, tr.config["height"] + 0.4,
                f"Goal: {'Yes' if roll['goal_reached'] else 'No'} | Hacking: "
                f"{'Yes' if roll['is_reward_hacking'] else 'No'} | Tile visits: "
                f"{roll['reward_tile_visits']} | Reward: {roll['proxy_reward']:+.2f} | Steps: {roll['episode_steps']}",
                ha="center", va="top", fontsize=8, color=TEXT_COLOR, fontfamily="monospace")
    fig.suptitle(f"Same Start State {start}: Hacking Recipient vs Goal Donor vs Patched Recipient (seed 123)\n"
                 "The patched agent eventually reaches the goal — after visiting the reward tile repeatedly",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.06)
    plt.tight_layout()
    fig.savefig(FIGURES / "phase4_5_behavior_comparison.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    GIFS.mkdir(parents=True, exist_ok=True)
    for tr, name in [(tr_h, "phase4_5_original_hacking.gif"), (tr_d, "phase4_5_goal_donor.gif"),
                     (tr_p, "phase4_5_patched.gif")]:
        render_trajectory_gif(tr, GIFS / name, max_frames=40, frame_duration_ms=200,
                              last_frame_duration_ms=2000)

    # Machine-readable per-episode detail for the matched test condition
    test_pos = [tuple(p) for p in d["partition"]["test"]]
    per_ep = evaluate_patching(recipient, RewardGridConfig(), test_pos, alpha, donor_fn, "matched")
    with open(RESULTS / "same_run_30k_matched_test_episodes.json", "w") as f:
        json.dump({"alpha": alpha, "episodes": per_ep["episodes"]}, f, indent=2)

    return {"start": start, "hack": roll_h, "donor": roll_d, "patched": roll_p}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating Phase 4.5 visualizations...")
    render_patching_diagram(); print("  ✓ phase4_5_patching_diagram.png")
    render_strength_sweep(); print("  ✓ phase4_5_strength_sweep.png")
    render_controls(); print("  ✓ phase4_5_controls.png")
    info = render_behavior_and_gifs()
    print(f"  ✓ phase4_5_behavior_comparison.png + 3 GIFs (start {info['start']})")
    for k in ("hack", "donor", "patched"):
        r = info[k]
        print(f"     {k:8s} goal={r['goal_reached']} visits={r['reward_tile_visits']} "
              f"reward={r['proxy_reward']} steps={r['episode_steps']}")
    print("Done.")


if __name__ == "__main__":
    script_args(__doc__)
    main()
