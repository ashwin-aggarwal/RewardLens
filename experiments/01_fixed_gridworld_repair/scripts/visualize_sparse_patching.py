"""Generate sparse-patching (Phase 4.8) visualizations from recorded results.

Usage:
    python -m scripts.visualize_sparse_patching [--gif]
"""

from __future__ import annotations

import json
import sys
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
RESULTS = Path("results/sparse_patching")
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm-then-finish",
              "farm_only": "Farm-only", "other": "Other / failure"}
HACK, DONOR, FULL, RANDOM, SHUF, CONST = EXPLOIT_PATH_COLOR, GOAL_PATH_COLOR, "#7C3AED", "#94A3B8", "#F59E0B", "#EC4899"
METHOD_COLORS = {"A": "#0EA5E9", "B": "#DC2626", "C": "#059669"}
METHOD_LABELS = {"A": "A: activation difference", "B": "B: causal ablation", "C": "C: greedy selection"}


def _load():
    return json.load(open(RESULTS / "sparse_patching_results.json"))


def _style(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.3)


def render_sparsity_vs_success():
    d = _load()
    ks = [int(k) for k in d["config"]["subset_sizes"]]
    rnd = d["random_subsets"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, (metric, ylabel, ylim) in zip(axes, (("direct_goal_rate", "Direct-goal rate (46 eval starts)", (-0.03, 1.05)),
                                                   ("mean_path_efficiency_direct", "Path efficiency of direct-goal episodes", (-0.03, 1.05)),
                                                   ("mean_applied_norm", "Applied perturbation L2 norm", None))):
        rk = "mean_" + metric if metric == "direct_goal_rate" else metric
        rmean = [rnd[str(k)][rk if rk in rnd[str(k)] else metric] for k in ks]
        rmean = [np.nan if v is None else v for v in rmean]
        ax.plot(ks, rmean, "--s", color=RANDOM, lw=1.5, ms=4, label="Random subsets (mean of 20)")
        if metric == "direct_goal_rate":
            rmax = [rnd[str(k)]["max_direct_goal_rate"] for k in ks]
            ax.fill_between(ks, 0, rmax, color=RANDOM, alpha=0.15, label="Random subsets (max)")
        for name in ("A", "B", "C"):
            sub = d["subsets"][name]
            kk = [k for k in ks if str(k) in sub]
            vals = [sub[str(k)]["eval"][metric] for k in kk]
            if name == "A" and d.get("fine_sweep_A"):
                for k, row in d["fine_sweep_A"].items():
                    kk.append(int(k)); vals.append(row["eval"][metric])
                order = np.argsort(kk); kk = [kk[i] for i in order]; vals = [vals[i] for i in order]
            vals = [np.nan if v is None else v for v in vals]
            ax.plot(kk, vals, "-o", color=METHOD_COLORS[name], lw=2, ms=5, label=METHOD_LABELS[name])
        if metric == "direct_goal_rate" and d.get("fine_sweep_A"):
            for k, row in d["fine_sweep_A"].items():
                ax.scatter([int(k)], [row["random"]["mean_direct_goal_rate"]], marker="s", s=14, color=RANDOM)
        if metric == "direct_goal_rate":
            ax.axhline(d["baseline_eval"]["unmodified"]["direct_goal_rate"], color=HACK, lw=1, ls=":", label="Unmodified recipient (k=0)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Number of patched layer-2 units (k)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        _style(ax, ylabel)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Sparse patching: direct-goal completion vs number of replaced layer-2 units\n"
                 "Subsets chosen on 16 dev starts, evaluated on the other 46 starts (exploratory; all positions were analyzed in earlier phases)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "sparse_patching_sparsity_vs_success.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_neuron_importance():
    d = _load()
    A = d["selection"]["A_activation_difference"]
    B = d["selection"]["B_ablation"]
    C = d["selection"]["C_greedy"]
    ks = [int(k) for k in d["config"]["subset_sizes"]]
    score = np.array(A["score"])
    drop = np.zeros(128)
    dis = np.zeros(128)
    for r in B["rows"]:
        for u in r["units"]:
            drop[u] = r["direct_goal_drop"]
            dis[u] = r["greedy_disagreement_with_full"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    ax = axes[0, 0]
    order = np.array(A["order"])
    ax.bar(range(128), score[order], color=METHOD_COLORS["A"], width=1.0)
    for i, u in enumerate(order[:10]):
        ax.text(i, score[u], str(u), fontsize=6, ha="center", va="bottom", color=TEXT_COLOR)
    ax.set_xlabel("Units sorted by mean |h_donor − h_recipient| on dev states", fontsize=9)
    ax.set_ylabel("Mean absolute activation difference", fontsize=9)
    _style(ax, "A: activation-difference profile (top-10 unit ids labeled)")

    ax = axes[0, 1]
    ax.bar(range(128), drop, color=METHOD_COLORS["B"], width=1.0, label="Dev direct-goal drop when the unit is restored")
    ax2 = ax.twinx()
    ax2.plot(range(128), dis, ".", color="#64748B", ms=3, label="Greedy disagreement with full patch (dev obs)")
    ax2.set_ylabel("Greedy disagreement with full patch", fontsize=8)
    ax.set_xlabel("Unit index", fontsize=9)
    ax.set_ylabel("Direct-goal drop (dev)", fontsize=9)
    _style(ax, f"B: single-unit ablation from the full patch ({sum(x > 0 for x in drop)} of 128 units hurt dev direct-goal)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")

    ax = axes[1, 0]
    hist = C["history"]
    ax.plot([h["k"] for h in hist], [h["dev"]["direct_goal_rate"] for h in hist], "-o", color=METHOD_COLORS["C"], lw=2, ms=5)
    for h in hist:
        ax.text(h["k"], h["dev"]["direct_goal_rate"] + 0.02, f"+{h['added']}", fontsize=7, ha="center", color=TEXT_COLOR)
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("Units selected (k)", fontsize=9)
    ax.set_ylabel("Dev direct-goal rate", fontsize=9)
    _style(ax, f"C: greedy forward selection on dev ({C['n_evaluations']} of {C['budget']} evaluations used; labels = unit added)")

    ax = axes[1, 1]
    kn = d["knockout"]
    for name in ("A", "B"):
        ax.plot(ks, [kn[name][str(k)]["direct_goal_rate"] for k in ks], "-o", color=METHOD_COLORS[name], lw=2, ms=5,
                label=f"Full patch minus top-k by {METHOD_LABELS[name].split(': ')[1]}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Units restored to recipient values (k)", fontsize=9)
    ax.set_ylabel("Direct-goal rate (46 eval starts)", fontsize=9)
    _style(ax, "Knockout: how much of the full patch can be removed")
    ax.legend(fontsize=7)
    fig.suptitle("Which layer-2 units matter? Three views of unit importance (seed 123, donor 20k at t=0.2, recipient 30k)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "sparse_patching_neuron_importance.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_sparse_vs_random():
    d = _load()
    ks = [int(k) for k in d["config"]["subset_sizes"]]
    rnd = d["random_subsets"]
    shuf = d["shuffled_controls"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    rng = np.random.RandomState(0)
    x = np.arange(len(ks))
    for ax, (metric, ylabel) in zip(axes, (("direct_goal_rate", "Direct-goal rate (46 eval starts)"),
                                           ("mean_reward_tile_visits", "Mean reward-tile visits"))):
        for i, k in enumerate(ks):
            vals = [r[metric] for r in rnd[str(k)]["per_subset"]]
            ax.scatter(x[i] - 0.3 + rng.uniform(-0.05, 0.05, len(vals)), vals, s=14, color=RANDOM, alpha=0.8,
                       label="Random subsets (20 per size)" if i == 0 else None, zorder=3)
            for j, name in enumerate(("A", "B", "C")):
                sub = d["subsets"][name]
                if str(k) in sub:
                    ax.scatter([x[i] - 0.1 + 0.15 * j], [sub[str(k)]["eval"][metric]], s=70, marker="o",
                               color=METHOD_COLORS[name], edgecolor="white", linewidth=0.6, zorder=4,
                               label=METHOD_LABELS[name] if i == 0 else None)
                    cq = sub[str(k)]["constant_qbias_eval"][metric]
                    ax.scatter([x[i] - 0.1 + 0.15 * j], [cq], s=30, marker="s", color=CONST, zorder=4,
                               label="Constant Q-bias (same mean delta)" if (i == 0 and j == 0) else None)
                key = f"{name}:{k}"
                if key in shuf:
                    sv = [r[metric] for r in shuf[key]["per_subset"]]
                    ax.scatter(x[i] + 0.35 + rng.uniform(-0.04, 0.04, len(sv)), sv, s=12, color=SHUF, alpha=0.8, zorder=3,
                               label="Shuffled-state donors (20)" if not ax.get_legend_handles_labels()[1].count("Shuffled-state donors (20)") else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylabel(ylabel, fontsize=9)
        _style(ax, ylabel)
    axes[0].set_ylim(-0.03, 1.08)
    axes[0].legend(fontsize=7, loc="upper left", ncol=3)
    fig.suptitle("Selected sparse subsets vs random subsets of the same size (same donor clamp, same 46 eval starts)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "sparse_patching_sparse_vs_random.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _replay(ep, name, eid):
    cfg = RewardGridConfig(agent_start=tuple(ep["start_pos"]))
    return record_episode(RewardGridEnv(config=cfg), ep["actions"], name, eid, seed=0)


def render_trajectories(make_gif: bool):
    d = _load()
    best = d["best_sparse"]
    if best is None:
        print("  (no best sparse subset; trajectory figure skipped)")
        return {}
    unm = d["baseline_eval"]["unmodified"]["episodes"]
    full = d["baseline_eval"]["full_patch"]["episodes"]
    idx = next((i for i, e in enumerate(unm) if e["category"] == "farm_only"), 0)
    panels = [
        (unm[idx], "Unmodified hacking DQN (30k)", HACK, None),
        (full[idx], "Full-layer patch (k=128)", FULL, None),
        (best["eval_traces"]["episodes"][idx], f"Best sparse patch ({best['method']}, k={best['k']})", METHOD_COLORS[best["method"]],
         "sparse_patching_best_sparse.gif"),
        (best["random_same_size_example_traces"]["episodes"][idx], f"Random subset, k={best['k']} (example)", RANDOM, None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    out = []
    for ax, (ep, title, color, gif) in zip(axes, panels):
        tr = _replay(ep, title, f"sp_{title[:4]}")
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
        if gif and make_gif:
            GIFS.mkdir(parents=True, exist_ok=True)
            render_trajectory_gif(tr, GIFS / gif, max_frames=40, frame_duration_ms=200, last_frame_duration_ms=2000)
            out.append(gif)
    start = tuple(unm[idx]["start_pos"])
    fig.suptitle(f"Start state {start}: recorded episodes (best sparse subset = method {best['method']}, "
                 f"{best['k']} of 128 units; unit ids in results/sparse_patching/sparse_patching_results.json)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES / "sparse_patching_trajectory_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"start": start, "gifs": out, "categories": [p[0]["category"] for p in panels]}


def main(make_gif: bool = False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating sparse-patching visualizations...")
    render_sparsity_vs_success(); print("  ✓ sparse_patching_sparsity_vs_success.png")
    render_neuron_importance(); print("  ✓ sparse_patching_neuron_importance.png")
    render_sparse_vs_random(); print("  ✓ sparse_patching_sparse_vs_random.png")
    info = render_trajectories(make_gif)
    print(f"  ✓ sparse_patching_trajectory_comparison.png {info}")
    print("Done.")


if __name__ == "__main__":
    main(make_gif=script_args(__doc__, "--gif").gif)
