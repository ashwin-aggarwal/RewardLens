"""Generate Phase 4.7 visualizations from results/phase4_7/time_control_results.json.

Usage:
    python -m scripts.visualize_phase4_7
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rewardlens.analysis.behavior_classification import CATEGORIES
from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv
from rewardlens.envs.trajectory import record_episode
from rewardlens.visualization.gridworld_viz import (
    BG_COLOR, TEXT_COLOR, GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR,
    _draw_grid, _draw_agent, _positions, render_trajectory_gif,
)

FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
RESULTS = Path("results/phase4_7")
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm-then-finish",
              "farm_only": "Farm-only", "other": "Other / failure"}
HACK, DONOR, PATCH, NEW, RANDOM, SHUF, CONST, TIME = (
    EXPLOIT_PATH_COLOR, GOAL_PATH_COLOR, "#0EA5E9", "#7C3AED", "#94A3B8", "#F59E0B", "#EC4899", "#0D9488")


def _load():
    return json.load(open(RESULTS / "time_control_results.json"))


def _style(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 1. Time sweep
# ---------------------------------------------------------------------------

def render_time_sweep():
    d = _load()
    cfg = d["config"]
    a46, a45 = cfg["phase4_6_alpha"], cfg["phase4_5_alpha"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for col, label in enumerate(("phase4_6_test_16", "all_62")):
        r = d[label]
        sweep = [float(t) for t in cfg["donor_time_sweep"]]
        clamps = [float(t) for t in cfg["clamp_sweep"]]
        n = len(r["position_set"]) if "position_set" in r else (16 if "16" in label else 62)
        # Top: direct-goal rate vs the time value the DONOR (E, F) or the network (B, C) sees
        ax = axes[0, col]
        for key, a, color, ls, name in (("E_time_shifted", a46, NEW, "-", f"E: time-shifted patch α={a46}"),
                                        ("E_time_shifted", a45, PATCH, "--", f"E: time-shifted patch α={a45}"),
                                        ("F_time_only", a46, TIME, "-", f"F: time-only delta α={a46}"),
                                        ("F_time_only", a45, TIME, "--", f"F: time-only delta α={a45}")):
            ax.plot(sweep, [r[key][f"a={a},t={t}"]["direct_goal_rate"] for t in sweep], ls, marker="o", color=color,
                    lw=2, ms=4, label=name)
        ax.plot(clamps, [r["B_recipient_clamp"][str(t)]["direct_goal_rate"] for t in clamps], ":", marker="s",
                color=HACK, lw=2, ms=4, label="B: recipient, time input clamped (no patch)")
        ax.plot(clamps, [r["C_donor"][str(t)]["direct_goal_rate"] for t in clamps], ":", marker="P", color=DONOR,
                lw=2, ms=5, label="C: donor alone, time input clamped")
        ax.axhline(r["D_matched_time"][str(a46)]["direct_goal_rate"], color=NEW, lw=1, alpha=0.5)
        ax.text(0.98, r["D_matched_time"][str(a46)]["direct_goal_rate"] + 0.02, f"D: matched-time patch α={a46}",
                fontsize=7, color=NEW, ha="left")
        ax.axhline(r["A_recipient"]["direct_goal_rate"], color=HACK, lw=1, alpha=0.5)
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("Direct-goal rate", fontsize=9)
        _style(ax, f"Direct-goal rate vs time input ({label.replace('_', ' ')}, {n} starts)")
        if col == 0:
            ax.legend(fontsize=6.5, loc="lower left")
        # Bottom: category stack for E at alpha 1.0 across donor time + visits
        ax = axes[1, col]
        x = np.arange(len(sweep))
        left = np.zeros(len(sweep))
        for c in CATEGORIES:
            vals = np.array([r["E_time_shifted"][f"a={a46},t={t}"][f"{c}_rate"] for t in sweep])
            ax.bar(x, vals, 0.6, bottom=left, color=CAT_COLORS[c], label=CAT_LABELS[c])
            left += vals
        ax2 = ax.twinx()
        ax2.plot(x, [r["E_time_shifted"][f"a={a46},t={t}"]["mean_reward_tile_visits"] for t in sweep], "-o",
                 color=TEXT_COLOR, lw=1.5, ms=4, label="Tile visits (right axis)")
        ax2.set_ylabel("Mean reward-tile visits", fontsize=8)
        ax2.set_ylim(0, 50)
        ax.set_xticks(x)
        ax.set_xticklabels([f"donor t={t}" for t in sweep], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of start states", fontsize=9)
        _style(ax, f"E: behavior categories vs donor time input (α={a46})")
        if col == 0:
            ax.legend(fontsize=7, loc="upper left")
            ax2.legend(fontsize=7, loc="upper right")
    fig.suptitle("Phase 4.7: what the time input does. Changing the hacking network's own time input (B) does nothing;\n"
                 "the donor is time-insensitive (C); the patched behavior depends on the time the DONOR sees (E).",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "phase4_7_time_sweep.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Control comparison
# ---------------------------------------------------------------------------

def render_control_comparison():
    d = _load()
    cfg = d["config"]
    a46, t46, a45 = cfg["phase4_6_alpha"], cfg["phase4_6_donor_time"], cfg["phase4_5_alpha"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for col, label in enumerate(("phase4_6_test_16", "all_62")):
        r = d[label]
        conds = [
            ("A\nrecipient", r["A_recipient"], HACK),
            (f"B\nrecipient\nclamp {t46}", r["B_recipient_clamp"][str(t46)], HACK),
            ("C\ndonor", r["C_donor"]["real"], DONOR),
            (f"C\ndonor\nclamp {t46}", r["C_donor"][str(t46)], DONOR),
            (f"D\nmatched-time\nα={a45}", r["D_matched_time"][str(a45)], PATCH),
            (f"D\nmatched-time\nα={a46}", r["D_matched_time"][str(a46)], PATCH),
            (f"E\ntime-shifted\nα={a46}, t={t46}", r["E_time_shifted"][f"a={a46},t={t46}"], NEW),
            (f"F\ntime-only\nα={a46}, t={t46}", r["F_time_only"][f"a={a46},t={t46}"], TIME),
            ("G\nhead swap\nreal time", r["G_head_swap"]["recipient_h2_real_time"], "#64748B"),
        ]
        ctrl_at = {6: r["controls"]["E"], 7: r["controls"]["F"]}
        rng = np.random.RandomState(0)
        for row, (metric, ylabel, ylim) in enumerate((("direct_goal_rate", "Direct-goal rate", (0, 1.08)),
                                                       ("mean_reward_tile_visits", "Mean reward-tile visits", None),
                                                       ("mean_path_efficiency_direct", "Path efficiency (direct-goal episodes)", (0, 1.08)))):
            ax = axes[row, col]
            for i, (name, res, color) in enumerate(conds):
                v = res.get(metric)
                ax.bar(i, 0 if v is None else v, 0.55, color=color, alpha=0.9, zorder=2)
                if v is None:
                    ax.text(i, 0.02, "n/a", ha="center", fontsize=7, color=TEXT_COLOR)
                if i in ctrl_at:
                    c = ctrl_at[i]
                    rnd = [x[metric] for x in c["random"]["per_control"] if x.get(metric) is not None]
                    shf = [x[metric] for x in c["shuffled"]["per_control"] if x.get(metric) is not None]
                    ax.scatter(i - 0.3 + rng.uniform(-0.04, 0.04, len(rnd)), rnd, s=12, color=RANDOM, alpha=0.8, zorder=3,
                               label="Random norm-matched (20)" if (i == 6 and row == 0) else None)
                    ax.scatter(i + 0.3 + rng.uniform(-0.04, 0.04, len(shf)), shf, s=12, color=SHUF, alpha=0.8, zorder=3,
                               label="Shuffled-state (20)" if (i == 6 and row == 0) else None)
                    cq = c["constant_qbias"].get(metric)
                    if cq is not None:
                        ax.scatter([i], [cq], marker="s", s=50, color=CONST, zorder=4,
                                   label="Constant Q-bias" if (i == 6 and row == 0) else None)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([c[0] for c in conds], fontsize=6.5)
            ax.set_ylabel(ylabel, fontsize=8)
            if ylim:
                ax.set_ylim(*ylim)
            _style(ax, f"{ylabel} ({label.replace('_', ' ')})" if row == 0 else "")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Phase 4.7 control comparison: full time-shifted patch (E) vs matched-time patch (D), time-only delta (F),\n"
                 "recipient/donor time clamps (B, C) and head swap (G). Controls shown as dots at E and F.",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "phase4_7_control_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Q-value analysis
# ---------------------------------------------------------------------------

def render_q_value_analysis():
    d = _load()
    cfg = d["config"]
    t46 = cfg["phase4_6_donor_time"]
    q = d["q_analysis_all_62"]
    tf = q["A_recipient"]["time_fractions"]
    styles = {
        "A_recipient": ("A: recipient", HACK, "-"),
        f"B_recipient_clamp_{t46}": (f"B: recipient, clamp {t46}", HACK, "--"),
        "C_donor": ("C: donor", DONOR, "-"),
        f"C_donor_clamp_{t46}": (f"C: donor, clamp {t46}", DONOR, "--"),
        "D_matched_time": ("D: matched-time patch", PATCH, "-"),
        f"E_time_shifted_{t46}": (f"E: time-shifted patch (donor t={t46})", NEW, "-"),
        f"F_time_only_{t46}": (f"F: time-only delta (t={t46})", TIME, "-"),
        "G_head_swap": ("G: head swap", "#64748B", "-"),
        f"E_shuffled_{t46}": ("E shuffled-state control", SHUF, ":"),
        f"F_shuffled_{t46}": ("F shuffled-state control", SHUF, "-."),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax = axes[0, 0]
    for k, (lab, col, ls) in styles.items():
        ax.plot(tf, q[k]["goal_approach_fraction"], ls, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.invert_xaxis(); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Remaining time the RECIPIENT sees (real)", fontsize=9)
    ax.set_ylabel("Fraction of positions whose greedy action\nmoves closer to the goal", fontsize=9)
    _style(ax, "Greedy action geometry, 62 positions × 11 time values")
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)

    ax = axes[0, 1]
    for k, (lab, col, ls) in styles.items():
        vals = [np.nan if v is None else v for v in q[k]["mean_goal_minus_reward_q"]]
        ax.plot(tf, vals, ls, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.axhline(0, color=TEXT_COLOR, lw=1)
    ax.invert_xaxis()
    ax.set_xlabel("Remaining time the recipient sees", fontsize=9)
    ax.set_ylabel("Q(best goal action) − Q(best reward-tile action)", fontsize=9)
    _style(ax, "Goal-minus-reward Q gap")

    ax = axes[1, 0]
    for k, (lab, col, ls) in styles.items():
        if k == "A_recipient":
            continue
        ax.plot(tf, q[k]["vs_recipient"]["greedy_flip_rate"], ls, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.invert_xaxis(); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Remaining time the recipient sees", fontsize=9)
    ax.set_ylabel("Fraction of positions whose greedy action\ndiffers from the unmodified recipient", fontsize=9)
    _style(ax, "Greedy action changes relative to A")

    ax = axes[1, 1]
    keys = list(styles.keys())
    j = tf.index(1.0)
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    names = ["Up", "Right", "Down", "Left"]
    acols = ["#3B82F6", "#16A34A", "#F97316", "#A855F7"]
    for a in range(4):
        vals = np.array([q[k]["vs_recipient"]["action_fractions"][j][a] for k in keys])
        ax.bar(x, vals, 0.6, bottom=bottom, color=acols[a], label=names[a])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([styles[k][0].split(":")[0] if ":" in styles[k][0] else styles[k][0] for k in keys], fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of the 62 positions", fontsize=9)
    _style(ax, "Greedy action histogram at full remaining time (t=1.0)")
    ax.legend(fontsize=7, loc="upper right", ncol=4)
    fig.suptitle("Phase 4.7 Q-value analysis at identical grid positions under different time inputs\n"
                 "A state-dependent policy spreads its actions across the four directions; a constant drift piles onto one.",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "phase4_7_q_value_analysis.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Trajectories (recorded actions) and optional GIFs
# ---------------------------------------------------------------------------

def _replay(ep, name, eid):
    cfg = RewardGridConfig(agent_start=tuple(ep["start_pos"]))
    return record_episode(RewardGridEnv(config=cfg), ep["actions"], name, eid, seed=0)


def render_trajectories(make_gifs: bool):
    d = _load()
    cfg = d["config"]
    a46, t46 = cfg["phase4_6_alpha"], cfg["phase4_6_donor_time"]
    r = d["phase4_6_test_16"]
    idx = 0
    for i, (u, dn) in enumerate(zip(r["A_recipient"]["episodes"], r["C_donor"]["real"]["episodes"])):
        if u["category"] == "farm_only" and dn["category"] == "direct_goal":
            idx = i
            break
    start = r["A_recipient"]["episodes"][idx]["start_pos"]
    panels = [
        (r["A_recipient"]["episodes"][idx], "A: original recipient (30k)", HACK, "phase4_7_recipient.gif"),
        (r["B_recipient_clamp"][str(t46)]["episodes"][idx], f"B: recipient, time input clamped to {t46}", HACK, "phase4_7_recipient_time_clamped.gif"),
        (r["D_matched_time"][str(a46)]["episodes"][idx], f"D: matched-time patch (α={a46})", PATCH, None),
        (r["F_time_only"][f"a={a46},t={t46}"]["episodes"][idx], f"F: time-only delta (t={t46})", TIME, None),
        (r["E_time_shifted"][f"a={a46},t={t46}"]["episodes"][idx], f"E: time-shifted patch (α={a46}, donor t={t46})", NEW, "phase4_7_time_shifted_patch.gif"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    trajs = []
    for ax, (ep, title, color, gif) in zip(axes, panels):
        tr = _replay(ep, title, f"p47_{title[:1]}")
        trajs.append((tr, gif))
        _draw_grid(ax, tr.config, title=title)
        pos = _positions(tr)
        seen = set()
        for i in range(1, len(pos)):
            seg = (pos[i - 1], pos[i])
            if seg not in seen:
                seen.add(seg)
                ax.annotate("", xy=pos[i], xytext=pos[i - 1], arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.65))
        _draw_agent(ax, pos[-1][0], pos[-1][1], color=color)
        ax.text(tr.config["width"] / 2 - 0.5, tr.config["height"] + 0.4,
                f"{CAT_LABELS[ep['category']]} | visits {ep['reward_tile_visits']} | steps {ep['episode_steps']} | reward {ep['proxy_reward']:+.1f}",
                ha="center", va="top", fontsize=8, color=CAT_COLORS[ep["category"]], fontfamily="monospace", fontweight="bold")
    fig.suptitle(f"Phase 4.7, start state {tuple(start)}: recorded episodes (seed 123, donor 20k, recipient 30k)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES / "phase4_7_trajectory_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    written = []
    if make_gifs:
        GIFS.mkdir(parents=True, exist_ok=True)
        for tr, gif in trajs:
            if gif:
                render_trajectory_gif(tr, GIFS / gif, max_frames=40, frame_duration_ms=200, last_frame_duration_ms=2000)
                written.append(gif)
    return {"start": start, "gifs": written, "categories": [p[0]["category"] for p in panels]}


def main(make_gifs: bool = False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating Phase 4.7 visualizations...")
    render_time_sweep(); print("  ✓ phase4_7_time_sweep.png")
    render_control_comparison(); print("  ✓ phase4_7_control_comparison.png")
    render_q_value_analysis(); print("  ✓ phase4_7_q_value_analysis.png")
    info = render_trajectories(make_gifs)
    print(f"  ✓ phase4_7_trajectory_comparison.png (start {info['start']}, categories {info['categories']}, gifs {info['gifs']})")
    print("Done.")


if __name__ == "__main__":
    import sys
    main(make_gifs="--gifs" in sys.argv)
