"""Generate low-rank patching visualizations from recorded results.

Usage:
    python -m scripts.visualize_low_rank_patching [--gif]
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
RESULTS = Path("results/low_rank_patching")
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm-then-finish",
              "farm_only": "Farm-only", "other": "Other / failure"}
HACK, DONOR, FULL, RANDOM, RANDOM2, SHUF = EXPLOIT_PATH_COLOR, GOAL_PATH_COLOR, "#7C3AED", "#94A3B8", "#475569", "#F59E0B"
VAR_COLORS = {"pca": "#0EA5E9", "svd": "#DC2626"}
VAR_LABELS = {"pca": "PCA (centered, mean included)", "svd": "SVD (uncentered)"}


def _load():
    return json.load(open(RESULTS / "low_rank_patching_results.json"))


def _style(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.3)


def _ks(d):
    return [int(k) for k in d["config"]["ranks"]]


def render_success_vs_rank():
    d = _load()
    ks = _ks(d)
    kpos = [k for k in ks if k > 0]
    rnd = d["random_subspaces"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, (metric, ylabel, ylim) in zip(axes, (("direct_goal_rate", "Direct-goal rate (46 eval starts)", (-0.03, 1.05)),
                                                   ("mean_path_efficiency_direct", "Path efficiency of direct-goal episodes", (-0.03, 1.05)),
                                                   ("mean_applied_norm", "Applied perturbation L2 norm", None))):
        for v in d["config"]["variants"]:
            vals = [d["sweep"][v][str(k)]["eval"][metric] for k in ks]
            vals = [np.nan if x is None else x for x in vals]
            ax.plot([max(k, 0.5) for k in ks], vals, "-o", color=VAR_COLORS[v], lw=2, ms=5, label=VAR_LABELS[v])
        for key, col, lab in (("plain", RANDOM, "Random subspace (mean of 20)"), ("norm_matched", RANDOM2, "Random subspace, norm-matched (mean of 20)")):
            mk = "mean_" + metric if metric == "direct_goal_rate" else metric
            vals = [rnd[str(k)][key][mk] for k in kpos]
            vals = [np.nan if x is None else x for x in vals]
            ax.plot(kpos, vals, "--s", color=col, lw=1.5, ms=4, label=lab)
            if metric == "direct_goal_rate":
                ax.fill_between(kpos, 0, [rnd[str(k)][key]["max_direct_goal_rate"] for k in kpos], color=col, alpha=0.08)
        if metric == "direct_goal_rate":
            ax.axhline(d["baselines"]["eval"]["unmodified"]["direct_goal_rate"], color=HACK, ls=":", lw=1, label="Unmodified recipient")
            ax.axhline(d["baselines"]["eval"]["donor_alone"]["direct_goal_rate"], color=DONOR, ls=":", lw=1, label="Donor alone")
        ax.set_xscale("log", base=2)
        ax.set_xticks([max(k, 0.5) for k in ks])
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Retained directions (k); 0 = mean only (PCA) / unmodified (SVD)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        _style(ax, ylabel)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Low-rank patching: direct-goal completion vs number of retained activation directions\n"
                 "Bases fitted on 16 dev positions × 21 time values; evaluated on the other 46 starts (exploratory: all positions analyzed in earlier phases)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "low_rank_patching_success_vs_rank.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_explained_variance():
    d = _load()
    ks = _ks(d)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax = axes[0]
    for v in d["config"]["variants"]:
        cum = d["fit"][v]["cumulative_explained_variance"]
        ax.plot(range(1, 129), cum, "-", color=VAR_COLORS[v], lw=2, label=f"{VAR_LABELS[v]}: cumulative explained variance (dev)")
        err_dev = [d["fit"][v]["reconstruction"][str(k)]["dev"]["relative_error"] for k in ks]
        err_eval = [d["fit"][v]["reconstruction"][str(k)]["eval"]["relative_error"] for k in ks]
        ax.plot([max(k, 0.5) for k in ks], [1 - e for e in err_dev], "o", color=VAR_COLORS[v], ms=5)
        ax.plot([max(k, 0.5) for k in ks], [1 - e for e in err_eval], "s", color=VAR_COLORS[v], ms=5, mfc="none",
                label=f"{v}: 1 − relative reconstruction error on eval deltas")
    ax.set_xscale("log", base=2)
    ax.set_xticks([max(k, 0.5) for k in ks])
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Retained directions (k)", fontsize=9)
    ax.set_ylabel("Fraction of delta variance / energy captured", fontsize=9)
    _style(ax, "How much of delta(s) the top-k directions capture")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    for v in d["config"]["variants"]:
        cap = [1 - d["fit"][v]["reconstruction"][str(k)]["eval"]["relative_error"] for k in ks]
        succ = [d["sweep"][v][str(k)]["eval"]["direct_goal_rate"] for k in ks]
        ax.plot(cap, succ, "-o", color=VAR_COLORS[v], lw=2, ms=5, label=VAR_LABELS[v])
        for k, c, s in zip(ks, cap, succ):
            ax.annotate(f"k={k}", (c, s), fontsize=7, color=TEXT_COLOR, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Fraction of eval delta captured by P_k", fontsize=9)
    ax.set_ylabel("Direct-goal rate (46 eval starts)", fontsize=9)
    _style(ax, "Reconstruction vs behavior: capturing most of the delta is not enough")
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Explained variance and reconstruction error vs behavioral success", fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "low_rank_patching_explained_variance.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_direction_impact():
    d = _load()
    imp = d["direction_impact_pca"]
    idx = [r["direction"] for r in imp]
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    w = 0.27
    ax.bar(np.array(idx) - w, [r["explained_variance_ratio"] for r in imp], w, color=VAR_COLORS["pca"], label="Explained variance ratio (dev)")
    ax.bar(np.array(idx), [r["single_direction_plus_mean"]["direct_goal_rate"] for r in imp], w, color="#16A34A",
           label="Direct-goal rate: this direction + mean only")
    ax.bar(np.array(idx) + w, [1 - r["full_minus_direction"]["direct_goal_rate"] for r in imp], w, color="#DC2626",
           label="Direct-goal DROP when this direction is removed from the full patch")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"PC{i + 1}" for i in idx], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction", fontsize=9)
    _style(ax, "Top PCA directions: variance captured vs behavioral impact (46 eval starts)")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("High explained variance does not imply causal importance", fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "low_rank_patching_direction_impact.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_vs_random():
    d = _load()
    ks = [k for k in _ks(d) if k > 0]
    rnd = d["random_subspaces"]
    shuf = d["shuffled_controls"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    rng = np.random.RandomState(0)
    x = np.arange(len(ks))
    for ax, (metric, ylabel) in zip(axes, (("direct_goal_rate", "Direct-goal rate (46 eval starts)"),
                                           ("mean_reward_tile_visits", "Mean reward-tile visits"))):
        seen = set()
        for i, k in enumerate(ks):
            for off, key, col, lab in ((-0.32, "plain", RANDOM, "Random subspace (20)"), (-0.18, "norm_matched", RANDOM2, "Random subspace, norm-matched (20)")):
                vals = [r[metric] for r in rnd[str(k)][key]["per_control"]]
                ax.scatter(x[i] + off + rng.uniform(-0.03, 0.03, len(vals)), vals, s=12, color=col, alpha=0.8, zorder=3,
                           label=lab if lab not in seen else None); seen.add(lab)
            for j, v in enumerate(d["config"]["variants"]):
                lab = VAR_LABELS[v]
                ax.scatter([x[i] + 0.02 + 0.14 * j], [d["sweep"][v][str(k)]["eval"][metric]], s=70, color=VAR_COLORS[v],
                           edgecolor="white", linewidth=0.6, zorder=4, label=lab if lab not in seen else None); seen.add(lab)
                key = f"{v}:{k}"
                if key in shuf:
                    sv = [r[metric] for r in shuf[key]["per_control"]]
                    lab2 = "Shuffled-state donors (20)"
                    ax.scatter(x[i] + 0.36 + rng.uniform(-0.03, 0.03, len(sv)), sv, s=12, color=SHUF, alpha=0.8, zorder=3,
                               label=lab2 if lab2 not in seen else None); seen.add(lab2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylabel(ylabel, fontsize=9)
        _style(ax, ylabel)
    axes[0].set_ylim(-0.03, 1.08)
    axes[0].legend(fontsize=7, loc="upper left", ncol=3)
    fig.suptitle("Low-rank patches vs random subspaces of the same rank (same donor clamp, same 46 eval starts)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "low_rank_patching_vs_random.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _replay(ep, name, eid):
    cfg = RewardGridConfig(agent_start=tuple(ep["start_pos"]))
    return record_episode(RewardGridEnv(config=cfg), ep["actions"], name, eid, seed=0)


def render_trajectories(make_gif: bool):
    d = _load()
    best = d["best_low_rank"]
    unm = d["baselines"]["eval"]["unmodified"]["episodes"]
    full = d["sweep"]["pca"]["128"]["eval_episodes"]
    idx = next((i for i, e in enumerate(unm) if e["category"] == "farm_only"), 0)
    panels = [(unm[idx], "Unmodified hacking DQN (30k)", HACK, None),
              (full[idx], "Full-layer patch (k=128)", FULL, None)]
    if best is not None:
        panels.append((best["eval_traces"]["episodes"][idx], f"Best low-rank patch ({best['variant']}, k={best['k']})",
                       VAR_COLORS[best["variant"]], "low_rank_patching_best.gif"))
        panels.append((best["random_norm_matched_example_traces"]["episodes"][idx], f"Random norm-matched subspace, k={best['k']} (example)", RANDOM, None))
    else:
        for k in (8, 16):
            panels.append((d["sweep"]["pca"][str(k)]["eval_episodes"][idx], f"PCA patch, k={k} (no rank < 128 reached the dev criterion)", VAR_COLORS["pca"], None))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.2), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    gifs = []
    for ax, (ep, title, color, gif) in zip(axes, panels):
        tr = _replay(ep, title, f"lr_{title[:4]}")
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
            gifs.append(gif)
    fig.suptitle(f"Start state {tuple(unm[idx]['start_pos'])}: recorded episodes", fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES / "low_rank_patching_trajectory_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"gifs": gifs, "categories": [p[0]["category"] for p in panels]}


def main(make_gif: bool = False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating low-rank patching visualizations...")
    render_success_vs_rank(); print("  ✓ low_rank_patching_success_vs_rank.png")
    render_explained_variance(); print("  ✓ low_rank_patching_explained_variance.png")
    render_direction_impact(); print("  ✓ low_rank_patching_direction_impact.png")
    render_vs_random(); print("  ✓ low_rank_patching_vs_random.png")
    info = render_trajectories(make_gif)
    print(f"  ✓ low_rank_patching_trajectory_comparison.png {info}")
    print("Done.")


if __name__ == "__main__":
    main(make_gif=script_args(__doc__, "--gif").gif)
