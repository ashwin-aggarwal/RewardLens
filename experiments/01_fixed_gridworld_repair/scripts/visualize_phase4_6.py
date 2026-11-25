"""Generate Phase 4.6 visualizations from recorded results.

Reads results/phase4_6/steering_results.json and reclassification.json.
GIFs replay the recorded action sequences through the environment, so they
show exactly the episodes that produced the reported numbers.

Usage:
    python -m scripts.visualize_phase4_6
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
from rewardlens.cli import script_args

FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
RESULTS = Path("results/phase4_6")

CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B",
              "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm-then-finish",
              "farm_only": "Farm-only", "other": "Other / failure"}
HACK_COLOR = EXPLOIT_PATH_COLOR
DONOR_COLOR = GOAL_PATH_COLOR
PATCH_COLOR = "#0EA5E9"
NEW_COLOR = "#7C3AED"
RANDOM_COLOR = "#94A3B8"
SHUFFLE_COLOR = "#F59E0B"
CONST_COLOR = "#EC4899"


def _load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def _style(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.3)


def _short(label: str) -> str:
    return (label.replace("constant@a=", "const α=").replace("early_window(", "early-win(")
            .replace("early_decay(", "early-decay(").replace("late_window(", "late-win(")
            .replace("n_steps=", "K=").replace("start_step=", "from=").replace(")@a=", ") α=")
            .replace("+donor_t=", ", donor sees t="))


def _stacked(ax, rows, title):
    """rows: list of (label, {category: rate})"""
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for c in CATEGORIES:
        vals = np.array([r[1].get(f"{c}_rate", r[1].get(f"mean_{c}_rate", 0.0)) for r in rows])
        ax.barh(y, vals, left=left, color=CAT_COLORS[c], label=CAT_LABELS[c], height=0.7)
        for i, v in enumerate(vals):
            if v >= 0.12:
                ax.text(left[i] + v / 2, y[i], f"{v:.2f}", ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of start states", fontsize=9)
    _style(ax, title)


# ---------------------------------------------------------------------------
# 1. Behavior category comparison
# ---------------------------------------------------------------------------

def render_behavior_categories():
    d = _load("steering_results.json")
    rc = _load("reclassification.json")
    b = d["baseline_comparison"]
    a45 = d["config"]["phase4_5_alpha"]

    rows = [("Unmodified recipient (30k)", b["unmodified"]),
            ("Goal donor alone (20k)", b["donor"]),
            (f"Phase 4.5 patch (layer 2, α={a45})", b["phase4_5_patch"]),
            (f"  random control, α={a45} (mean of 20)", b["random"]),
            (f"  shuffled-state donor, α={a45} (mean of 20)", b["shuffled"]),
            (f"  constant Q-bias, α={a45}", b["constant_qbias"])]
    for name, blk in d["blocks"].items():
        rows.append((f"{name}: {_short(blk['selected'])}", blk["test"]["matched"]))
        rows.append(("  shuffled-state donor (mean of 20)", blk["test"]["shuffled"]))

    retro = [("Phase 1 scripted goal policy", {f"{r['category']}_rate": 1.0}) for r in rc["stored_trajectories"]["rows"]
             if r["label"] == "phase1_scripted_goal"]
    retro += [("Phase 1 scripted exploit policy", {f"{r['category']}_rate": 1.0}) for r in rc["stored_trajectories"]["rows"]
              if r["label"] == "phase1_scripted_exploit"]
    for cond, pretty in (("flawed", "Phase 2 flawed DQN"), ("corrected", "Phase 2.5 corrected DQN")):
        for seed in (42, 123, 456):
            k = f"{cond}_seed{seed}"
            if k in rc["checkpoints"] and rc["checkpoints"][k]["per_step"]:
                ps = rc["checkpoints"][k]["per_step"]
                last = ps[str(max(int(s) for s in ps))]["all_62_starts"]
                retro.append((f"{pretty} seed {seed}, final (62 starts)", last))
    if rc["phase4"].get("available"):
        for seed, blk in rc["phase4"]["flawed"].items():
            if "probe_subtract" in blk["test"]:
                retro.append((f"Phase 4 probe-subtract, seed {seed} (16 test)", blk["test"]["probe_subtract"]))
    if rc["phase4_5"].get("available"):
        for rstep in ("30000", "100000"):
            if rstep in rc["phase4_5"]["same_run"]:
                blk = rc["phase4_5"]["same_run"][rstep]
                retro.append((f"Phase 4.5 same-run 20k→{int(rstep)//1000}k patch, α={blk['selected_alpha']}", blk["matched"]))
        for m in ("identity", "ridge_all", "ridge_agreement"):
            if m in rc["phase4_5"]["cross_model"]:
                retro.append((f"Phase 4.5 cross-model [{m}]", rc["phase4_5"]["cross_model"][m]["matched"]))

    fig, axes = plt.subplots(1, 2, figsize=(17, 0.42 * max(len(rows), len(retro)) + 2.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    _stacked(axes[0], rows, "Phase 4.6 interventions on fresh held-out test states (16 each)")
    _stacked(axes[1], retro, "Retrospective reclassification of Phase 1–4.5 trajectories")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Behavior Categories: 'reached the goal' is not the same as 'pursued the goal'\n"
                 "Direct goal = goal reached with < 5 reward-tile visits. Farm-then-finish = goal reached after ≥ 5 visits.",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "phase4_6_behavior_categories.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Reward-tile visits vs goal-completion step
# ---------------------------------------------------------------------------

def render_visits_vs_completion():
    d = _load("steering_results.json")
    b = d["baseline_comparison"]
    mp = d["most_promising"]
    best = d["blocks"][mp["block"]]
    thr = d["config"]["visit_threshold"]
    series = [
        ("Unmodified recipient", b["unmodified"]["episodes"], HACK_COLOR, "D"),
        ("Goal donor alone", b["donor"]["episodes"], DONOR_COLOR, "P"),
        (f"Phase 4.5 patch (α={d['config']['phase4_5_alpha']})", b["phase4_5_patch"]["episodes"], PATCH_COLOR, "*"),
        (f"Most promising new: {_short(mp['selected'])}", best["test"]["matched"]["episodes"], NEW_COLOR, "o"),
        ("Shuffled-state donor (same α, one control)", best["test"]["shuffled_example_episodes"], SHUFFLE_COLOR, "^"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    rng = np.random.RandomState(0)
    for label, eps, color, marker in series:
        xs = [e["episode_steps"] if e["goal_reached"] else 100 for e in eps]
        ys = [e["reward_tile_visits"] for e in eps]
        reached = [bool(e["goal_reached"]) for e in eps]
        jx = rng.uniform(-1.2, 1.2, len(xs))
        jy = rng.uniform(-0.6, 0.6, len(ys))
        ax.scatter(np.array(xs)[reached] + jx[reached], np.array(ys)[reached] + jy[reached], s=70, marker=marker,
                   color=color, alpha=0.85, label=f"{label} (goal reached)", edgecolor="white", linewidth=0.5)
        nr = [not r for r in reached]
        if any(nr):
            ax.scatter(np.array(xs)[nr] + jx[nr], np.array(ys)[nr] + jy[nr], s=70, marker=marker,
                       facecolor="none", edgecolor=color, linewidth=1.5, label=f"{label} (not reached)")
    ax.axhline(thr - 0.5, color=TEXT_COLOR, ls="--", lw=1)
    ax.text(2, thr + 0.6, f"farming threshold ({thr} visits)", fontsize=8, color=TEXT_COLOR)
    ax.axvspan(99, 101.5, color="#E2E8F0", alpha=0.6)
    ax.text(100, 52, "time limit\n(100 steps)", ha="center", fontsize=8, color=TEXT_COLOR)
    ax.text(1, -2.5, "direct goal: bottom-left", fontsize=8, color=CAT_COLORS["direct_goal"], fontweight="bold")
    ax.text(70, 44, "farm-then-finish: top-right", fontsize=8, color=CAT_COLORS["farm_then_finish"], fontweight="bold")
    ax.set_xlabel("Step at which the goal was reached (100 = never)", fontsize=10)
    ax.set_ylabel("Reward-tile visits in the episode", fontsize=10)
    ax.set_xlim(-2, 103)
    ax.set_ylim(-4, 55)
    _style(ax, "Reward-tile visits vs goal-completion step (16 held-out start states per condition)")
    ax.legend(fontsize=7, loc="center left")
    fig.savefig(FIGURES / "phase4_6_visits_vs_completion.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Time-dependent Q-values
# ---------------------------------------------------------------------------

def render_time_dependent_q():
    d = _load("steering_results.json")
    td = d["time_dependence"]
    a45 = d["config"]["phase4_5_alpha"]
    conds = td["conditions"]
    tf = conds["recipient_unmodified"]["time_fractions"]
    styles = {
        "recipient_unmodified": ("Unmodified recipient", HACK_COLOR, "-"),
        "donor_own_head": ("Goal donor (own head)", DONOR_COLOR, "-"),
        "donor_via_recipient_head": ("Donor H2 → recipient head (α=1)", DONOR_COLOR, ":"),
        f"patched_a{a45}": (f"Patched α={a45}", PATCH_COLOR, "-"),
        "patched_a1.5": ("Patched α=1.5", NEW_COLOR, "-"),
        f"patched_a{a45}_donor_t0.05": (f"Patched α={a45}, donor time clamped 0.05", NEW_COLOR, "--"),
        f"random_a{a45}": (f"Random control α={a45}", RANDOM_COLOR, "--"),
        f"shuffled_a{a45}": (f"Shuffled-state donor α={a45}", SHUFFLE_COLOR, "--"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    ax = axes[0, 0]
    for k, (lab, col, ls) in styles.items():
        if k in conds:
            ax.plot(tf, conds[k]["goal_approach_fraction"], ls, color=col, lw=2, label=lab, marker="o", ms=3)
    ax.invert_xaxis()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Remaining time fraction (episode runs right → left)", fontsize=9)
    ax.set_ylabel("Fraction of positions whose greedy action\nmoves closer to the goal", fontsize=9)
    _style(ax, "Greedy action geometry at 16 matched positions")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[0, 1]
    for k, (lab, col, ls) in styles.items():
        if k in conds:
            vals = [np.nan if v is None else v for v in conds[k]["mean_goal_minus_reward_q"]]
            ax.plot(tf, vals, ls, color=col, lw=2, label=lab, marker="o", ms=3)
    ax.axhline(0, color=TEXT_COLOR, lw=1)
    ax.invert_xaxis()
    ax.set_xlabel("Remaining time fraction", fontsize=9)
    ax.set_ylabel("Q(best goal-approaching action) −\nQ(best reward-tile-approaching action)", fontsize=9)
    _style(ax, "Goal-minus-reward Q gap (positive = prefers goal direction)")

    ax = axes[1, 0]
    bt = td["affine_decomposition"]["by_time"]
    ts = [float(t) for t in bt]
    ax.plot(ts, [bt[t]["mean_max_q_recipient"] for t in bt], "-o", color=HACK_COLOR, lw=2, ms=3,
            label="(1) recipient's own Q: max_a Q_rec(s,a)")
    ax.plot(ts, [bt[t]["mean_max_q_donor_via_recipient_head"] for t in bt], "-o", color=DONOR_COLOR, lw=2, ms=3,
            label="(2) donor H2 read by recipient head: max_a")
    ax.plot(ts, [bt[t]["mean_max_q_patched"] for t in bt], "-o", color=PATCH_COLOR, lw=2, ms=3,
            label=f"patched α={a45} = (1−α)·(1) + α·(2)")
    ax.invert_xaxis()
    ax.set_xlabel("Remaining time fraction", fontsize=9)
    ax.set_ylabel("Mean max-Q over 16 positions", fontsize=9)
    _style(ax, f"Exact affine decomposition of the patched Q-values (max error {td['affine_decomposition']['max_reconstruction_error']:.0e})")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    cl = td["time_clamp_rollouts"]
    taus = [float(t) for t in cl]
    names = {"recipient": ("Unmodified recipient", HACK_COLOR), "donor": ("Goal donor", DONOR_COLOR),
             f"patched_a{a45}": (f"Patched α={a45}", PATCH_COLOR), "patched_a1.0": ("Patched α=1.0", NEW_COLOR)}
    w = 0.2
    for i, (k, (lab, col)) in enumerate(names.items()):
        ax.bar(np.arange(len(taus)) + (i - 1.5) * w, [cl[t][k]["direct_goal_rate"] for t in cl], w, color=col, label=lab)
    ax.set_xticks(np.arange(len(taus)))
    ax.set_xticklabels([f"t={t}" for t in taus], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Direct-goal rate (16 start states)", fontsize=9)
    ax.set_xlabel("Remaining-time value the NETWORK sees (clamped; environment unchanged)", fontsize=9)
    _style(ax, "Input-clamp test: fix the time feature, run a full episode")
    ax.legend(fontsize=7)

    fig.suptitle("Why the patched agent farms first and finishes late: goal preference is gated by remaining time\n"
                 "Same 16 held-out positions at every time value, so time is isolated from location (seed 123, 20k donor → 30k recipient)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "phase4_6_time_dependent_q.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Steering comparison
# ---------------------------------------------------------------------------

def render_steering_comparison():
    d = _load("steering_results.json")
    b = d["baseline_comparison"]
    a45 = d["config"]["phase4_5_alpha"]
    cols = [(f"Phase 4.5 patch\nlayer 2, α={a45}", b)]
    for name, blk in d["blocks"].items():
        cols.append((f"{name}\n{_short(blk['selected'])}", blk["test"]))

    metrics = [("direct_goal_rate", "Direct-goal rate (primary metric)", (0, 1.08)),
               ("mean_reward_tile_visits", "Mean reward-tile visits", None),
               ("mean_path_efficiency_direct", "Path efficiency of direct-goal episodes\n(Manhattan distance / steps; 1 = optimal)", (0, 1.08))]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(2.3 * len(cols) + 2, 11), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    x = np.arange(len(cols))
    rng = np.random.RandomState(0)
    for row, (metric, ylabel, ylim) in enumerate(metrics):
        ax = axes[row]
        _style(ax, ylabel if row == 0 else "")
        for i, (label, t) in enumerate(cols):
            key = "phase4_5_patch" if "matched" not in t else "matched"
            mv = t[key][metric]
            ax.bar(i, 0 if mv is None else mv, 0.5, color=PATCH_COLOR if i == 0 else NEW_COLOR, alpha=0.9,
                   label="Matched goal patch" if i == 0 else None, zorder=2)
            per_key = "per_control"
            rnd = [r[metric] for r in t["random"][per_key] if r.get(metric) is not None]
            shf = [r[metric] for r in t["shuffled"][per_key] if r.get(metric) is not None]
            ax.scatter(i - 0.33 + rng.uniform(-0.04, 0.04, len(rnd)), rnd, s=14, color=RANDOM_COLOR, alpha=0.8,
                       label="Random control (20)" if i == 0 else None, zorder=3)
            ax.scatter(i + 0.33 + rng.uniform(-0.04, 0.04, len(shf)), shf, s=14, color=SHUFFLE_COLOR, alpha=0.8,
                       label="Shuffled-state donor (20)" if i == 0 else None, zorder=3)
            cq = t["constant_qbias"].get(metric)
            if cq is not None:
                ax.scatter([i], [cq], marker="s", s=60, color=CONST_COLOR, label="Constant Q-bias" if i == 0 else None, zorder=4)
            if metric != "mean_path_efficiency_direct" or b["unmodified"].get(metric) is not None:
                ax.scatter([i - 0.12], [b["unmodified"][metric] or 0], marker="D", s=40, color=HACK_COLOR,
                           label="Unmodified recipient" if i == 0 else None, zorder=4)
            if b["donor"].get(metric) is not None:
                ax.scatter([i + 0.12], [b["donor"][metric]], marker="P", s=60, color=DONOR_COLOR,
                           label="Goal donor alone" if i == 0 else None, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in cols], fontsize=7.5)
        ax.set_ylabel(ylabel.split("\n")[0], fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        if row == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=3)
    fig.suptitle("Steering comparison on fresh held-out test states (16 start states, selection done on separate dev states)\n"
                 "A successful intervention must beat the shuffled-state donor control, not only the unmodified recipient",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "phase4_6_steering_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Trajectories and GIFs (replayed from recorded actions)
# ---------------------------------------------------------------------------

def _replay(ep, name, eid):
    cfg = RewardGridConfig(agent_start=tuple(ep["start_pos"]))
    return record_episode(RewardGridEnv(config=cfg), ep["actions"], name, eid, seed=0)


def render_trajectories_and_gifs():
    d = _load("steering_results.json")
    b = d["baseline_comparison"]
    mp = d["most_promising"]
    best = d["blocks"][mp["block"]]
    a45 = d["config"]["phase4_5_alpha"]
    test_pos = [tuple(p) for p in d["partition"]["test"]]
    # Pick the first test start where the unmodified recipient farms and the donor goes direct
    idx = 0
    for i, (u, dn) in enumerate(zip(b["unmodified"]["episodes"], b["donor"]["episodes"])):
        if u["category"] == "farm_only" and dn["category"] == "direct_goal":
            idx = i
            break
    start = test_pos[idx]
    new_ep = best["test"]["matched"]["episodes"][idx]
    new_label = f"{mp['block']}: {_short(mp['selected'])}"
    panels = [
        (b["unmodified"]["episodes"][idx], "Original hacking DQN (30k)", HACK_COLOR, "phase4_6_original_hacking.gif"),
        (b["donor"]["episodes"][idx], "Goal donor (20k)", DONOR_COLOR, None),
        (b["phase4_5_patch"]["episodes"][idx], f"Phase 4.5 patch (α={a45})", PATCH_COLOR, "phase4_6_farm_then_finish.gif"),
        (new_ep, new_label, NEW_COLOR, f"phase4_6_new_intervention_{new_ep['category']}.gif"),
        (best["test"]["shuffled_example_episodes"][idx], "Shuffled-state donor control\n(same α, control seed 0)", SHUFFLE_COLOR,
         "phase4_6_shuffled_control.gif"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    trajs = []
    for ax, (ep, title, color, gif) in zip(axes, panels):
        tr = _replay(ep, title.split("\n")[0], f"p46_{gif or 'donor'}")
        trajs.append((tr, gif))
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
                f"{CAT_LABELS[ep['category']]} | visits {ep['reward_tile_visits']} | steps {ep['episode_steps']}"
                f" | reward {ep['proxy_reward']:+.1f}",
                ha="center", va="top", fontsize=8, color=CAT_COLORS[ep["category"]], fontfamily="monospace",
                fontweight="bold")
    fig.suptitle(f"Same held-out start state {start}: recorded episodes behind the Phase 4.6 numbers (seed 123)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES / "phase4_6_trajectory_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    GIFS.mkdir(parents=True, exist_ok=True)
    written = []
    for tr, gif in trajs:
        if gif:
            render_trajectory_gif(tr, GIFS / gif, max_frames=40, frame_duration_ms=200, last_frame_duration_ms=2000)
            written.append(gif)
    return {"start": start, "gifs": written, "new_category": new_ep["category"]}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating Phase 4.6 visualizations...")
    render_behavior_categories(); print("  ✓ phase4_6_behavior_categories.png")
    render_visits_vs_completion(); print("  ✓ phase4_6_visits_vs_completion.png")
    render_time_dependent_q(); print("  ✓ phase4_6_time_dependent_q.png")
    render_steering_comparison(); print("  ✓ phase4_6_steering_comparison.png")
    info = render_trajectories_and_gifs()
    print(f"  ✓ phase4_6_trajectory_comparison.png + GIFs {info['gifs']} (start {info['start']}, "
          f"new intervention category: {info['new_category']})")
    print("Done.")


if __name__ == "__main__":
    script_args(__doc__)
    main()
