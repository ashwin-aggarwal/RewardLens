"""Generate Phase 5 weight-editing visualizations from recorded results.

Usage:
    python -m scripts.visualize_weight_editing [--gif]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv
from rewardlens.envs.trajectory import record_episode
from rewardlens.visualization.gridworld_viz import (
    BG_COLOR, TEXT_COLOR, GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR,
    _draw_grid, _draw_agent, _positions, render_trajectory_gif,
)
from rewardlens.cli import script_args

FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
RESULTS = Path("results/weight_editing")
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm-then-finish", "farm_only": "Farm-only", "other": "Other / failure"}
HACK, DONOR, PATCH, EDIT = EXPLOIT_PATH_COLOR, GOAL_PATH_COLOR, "#7C3AED", "#0EA5E9"
GROUP_STYLE = {
    "lowrank|act|bias1": ("Low-rank layer-2 edit, activation matching", "#0EA5E9", "o"),
    "lowrank|q|bias1": ("Low-rank layer-2 edit, Q matching", "#2563EB", "s"),
    "lowrank|act|bias0": ("Low-rank, activation matching, no bias edit", "#7DD3FC", "^"),
    "l2_full|act": ("Full layer-2 fine-tune, activation matching", "#F97316", "D"),
    "l2_full|q": ("Full layer-2 fine-tune, Q matching", "#EA580C", "D"),
    "head|q": ("Output-head-only fine-tune", "#A855F7", "P"),
    "qbias|q": ("Constant Q-value bias", "#EC4899", "X"),
    "full|q": ("Whole-network fine-tune, Q matching", "#64748B", "*"),
    "full|ce": ("Whole-network policy distillation (CE)", "#334155", "*"),
}


def _load():
    return json.load(open(RESULTS / "weight_editing_results.json"))


def _style(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.3)


def _group_rows(block, prefix, rank=None):
    out = []
    for r in block["runs"]:
        key = f"{r['method']}|{r['objective']}" + (f"|bias{int(r['edit_bias'])}" if r["method"] == "lowrank" else "")
        if key == prefix and (rank is None or r["rank"] == rank):
            out.append(r)
    return out


def render_rank_curve():
    d = _load()
    ranks = d["config"]["ranks"]
    blocks = [("Seed 123 (primary)", d["primary"])] + [(f"Seed {s} (transfer)", b) for s, b in d["transfer"].items()]
    fig, axes = plt.subplots(1, len(blocks), figsize=(6 * len(blocks), 5.2), dpi=150, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, (title, b) in zip(axes, blocks):
        for key in ("lowrank|act|bias1", "lowrank|q|bias1", "lowrank|act|bias0"):
            rows_by_r = [_group_rows(b, key, r) for r in ranks]
            if not any(rows_by_r):
                continue
            lab, col, mk = GROUP_STYLE[key]
            mean = [np.mean([x["behavior_all62"]["direct_goal_rate"] for x in rr]) for rr in rows_by_r]
            lo = [np.min([x["behavior_all62"]["direct_goal_rate"] for x in rr]) for rr in rows_by_r]
            hi = [np.max([x["behavior_all62"]["direct_goal_rate"] for x in rr]) for rr in rows_by_r]
            ax.plot(ranks, mean, "-", marker=mk, color=col, lw=2, ms=6, label=lab + " (mean of 3 seeds)")
            ax.fill_between(ranks, lo, hi, color=col, alpha=0.15)
        if "random_lowrank" in b:
            ax.plot(ranks, [b["random_lowrank"][str(r)]["mean_direct_goal_rate"] for r in ranks], "--", color="#94A3B8",
                    marker="s", ms=4, label="Random low-rank edit, norm-matched (mean of 10)")
        bl = b["baselines"]
        ax.axhline(bl["recipient"]["direct_goal_rate"], color=HACK, ls=":", lw=1.2, label="Unmodified hacking DQN")
        ax.axhline(bl["pca_patch"]["direct_goal_rate"], color=PATCH, ls="--", lw=1.2, label="Donor-dependent PCA patch")
        ax.axhline(bl["donor"]["direct_goal_rate"], color=DONOR, ls=":", lw=1.2, label="Goal-following donor")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Rank of the layer-2 weight update", fontsize=9)
        _style(ax, f"{title}: donor {b['donor_step'] // 1000}k → recipient {b['recipient_step'] // 1000}k")
    axes[0].set_ylabel("Direct-goal rate (all 62 starts; band = seed min–max)", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Direct-goal completion of the edited standalone DQN vs weight-update rank (no donor, no hooks at inference)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "weight_editing_rank_curve.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _comparison_entries(b, best_rank):
    entries = []
    for r in sorted({1, best_rank} if best_rank else {1}):
        entries.append((f"Low-rank r={r}\nact. match", _group_rows(b, "lowrank|act|bias1", r), GROUP_STYLE["lowrank|act|bias1"][1]))
        entries.append((f"Low-rank r={r}\nQ match", _group_rows(b, "lowrank|q|bias1", r), GROUP_STYLE["lowrank|q|bias1"][1]))
    for key, lab in (("l2_full|act", "Full layer 2\nact. match"), ("l2_full|q", "Full layer 2\nQ match"), ("head|q", "Output head\nonly"),
                     ("qbias|q", "Constant\nQ bias"), ("full|q", "Whole network\nQ match"), ("full|ce", "Whole network\npolicy distill")):
        rows = _group_rows(b, key)
        if rows:
            entries.append((lab, rows, GROUP_STYLE[key][1]))
    return entries


def render_vs_finetuning():
    d = _load()
    b = d["primary"]
    best_rank = (d["primary"].get("best_model") or {}).get("rank")
    entries = _comparison_entries(b, best_rank)
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    x = np.arange(len(entries))
    for ax, (metric, ylabel, getter) in zip(axes, (
            ("direct", "Direct-goal rate (all 62 starts)", lambda r: r["behavior_all62"]["direct_goal_rate"]),
            ("agree", "Action agreement with the patch target\n(held-out dev positions)", lambda r: r["matching"]["val_positions"]["action_agreement"]))):
        for i, (lab, rows, col) in enumerate(entries):
            vals = [getter(r) for r in rows]
            ax.bar(i, np.mean(vals), 0.6, color=col, alpha=0.85, zorder=2)
            ax.scatter([i] * len(vals), vals, color=TEXT_COLOR, s=12, zorder=3)
            if metric == "direct":
                ax.text(i, 1.03, f"{rows[0]['n_trainable']:,} params", ha="center", fontsize=7, color=TEXT_COLOR)
        bl = b["baselines"]
        if metric == "direct":
            ax.axhline(bl["recipient"]["direct_goal_rate"], color=HACK, ls=":", lw=1.2, label="Unmodified hacking DQN")
            ax.axhline(bl["donor"]["direct_goal_rate"], color=DONOR, ls=":", lw=1.2, label="Goal-following donor")
            ax.axhline(bl["pca_patch"]["direct_goal_rate"], color=PATCH, ls="--", lw=1.2, label="Donor-dependent PCA patch")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in entries], fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylabel, fontsize=9)
        _style(ax, ylabel.split("\n")[0])
    fig.suptitle("Low-rank editing vs conventional fine-tuning on the same examples, same budget (seed 123; dots = 3 training seeds)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "weight_editing_vs_finetuning.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _key(r):
    return f"{r['method']}|{r['objective']}" + (f"|bias{int(r['edit_bias'])}" if r["method"] == "lowrank" else "")


def render_params_vs_success():
    d = _load()
    b = d["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    seen = set()
    for r in b["runs"]:
        k = _key(r)
        lab, col, mk = GROUP_STYLE[k]
        y = r["behavior_all62"]["direct_goal_rate"]
        for ax, xv in ((axes[0], r["n_trainable"]), (axes[1], max(r["weights"]["total_delta_norm"], 1e-3))):
            ax.scatter(xv, y, color=col, marker=mk, s=45, alpha=0.8, edgecolor="white", linewidth=0.5,
                       label=lab if (k not in seen and ax is axes[0]) else None)
        seen.add(k)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable parameters (log scale)", fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Total weight-change norm ||ΔW||_F over all layers (log scale)", fontsize=9)
    for ax, t in zip(axes, ("Parameter count vs success", "Size of the weight change vs success")):
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("Direct-goal rate (all 62 starts)", fontsize=9)
        _style(ax, t)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("How much modification does each method need? (seed 123; each point = one training seed)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "weight_editing_params_vs_success.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_matching_vs_behavior():
    d = _load()
    b = d["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    seen = set()
    for r in b["runs"]:
        k = _key(r)
        lab, col, mk = GROUP_STYLE[k]
        y = r["behavior_all62"]["direct_goal_rate"]
        axes[0].scatter(max(r["matching"]["val_positions"]["act_relative_error"], 1e-4), y, color=col, marker=mk, s=45, alpha=0.8,
                        edgecolor="white", linewidth=0.5, label=lab if k not in seen else None)
        axes[1].scatter(r["matching"]["val_positions"]["action_agreement"], y, color=col, marker=mk, s=45, alpha=0.8,
                        edgecolor="white", linewidth=0.5)
        seen.add(k)
    floor = b["realizability_floor"]["val_positions"]["relative_floor"]
    axes[0].axvline(max(floor, 1e-4), color=TEXT_COLOR, ls="--", lw=1)
    axes[0].text(max(floor, 1e-4) * 1.15, 0.5, "ReLU realizability floor\n(negative target entries)", fontsize=7, color=TEXT_COLOR)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Layer-2 activation error vs patch target, relative (held-out dev positions, log)", fontsize=8.5)
    axes[1].set_xlabel("Greedy-action agreement with the patch target (held-out dev positions)", fontsize=8.5)
    for ax, t in zip(axes, ("Activation matching error vs behavior", "Action agreement vs behavior")):
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("Direct-goal rate (all 62 starts)", fontsize=9)
        _style(ax, t)
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Matching the patched activations is neither necessary nor sufficient for behavior (seed 123; each point = one run)",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "weight_editing_matching_vs_behavior.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _replay(ep, name, eid):
    cfg = RewardGridConfig(agent_start=tuple(ep["start_pos"]))
    return record_episode(RewardGridEnv(config=cfg), ep["actions"], name, eid, seed=0)


def render_trajectories(make_gif: bool):
    d = _load()
    b = d["primary"]
    best = b.get("best_model")
    bl = b["baselines"]
    unm = bl["recipient"]["episodes"]
    idx = next((i for i, e in enumerate(unm) if e["category"] == "farm_only" and tuple(e["start_pos"]) == (0, 4)),
               next(i for i, e in enumerate(unm) if e["category"] == "farm_only"))
    panels = [(unm[idx], "Hacking DQN (30k)", HACK, None),
              (bl["pca_patch"]["episodes"][idx], "Donor-dependent PCA patch\n(runtime hook + donor)", PATCH, None),
              (bl["donor"]["episodes"][idx], "Goal-following donor (20k)", DONOR, None)]
    if best:
        panels.append((best["behavior_all62_reloaded"]["episodes"][idx],
                       f"Weight-edited DQN, rank {best['rank']} ({best['objective']} matching)\nstandalone, reloaded from disk", EDIT,
                       "weight_editing_best.gif"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.4), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    gifs = []
    for ax, (ep, title, color, gif) in zip(axes, panels):
        tr = _replay(ep, title.split("\n")[0], f"we_{len(gifs)}")
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
                f"{CAT_LABELS[ep['category']]} | visits {ep['reward_tile_visits']} | steps {ep['episode_steps']}",
                ha="center", va="top", fontsize=8, color=CAT_COLORS[ep["category"]], fontfamily="monospace", fontweight="bold")
        if gif and make_gif:
            GIFS.mkdir(parents=True, exist_ok=True)
            render_trajectory_gif(tr, GIFS / gif, max_frames=40, frame_duration_ms=200, last_frame_duration_ms=2000)
            gifs.append(gif)
    fig.suptitle(f"Start state {tuple(unm[idx]['start_pos'])}: recorded episodes (seed 123)", fontsize=12, fontweight="bold",
                 color=TEXT_COLOR, y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES / "weight_editing_trajectory_comparison.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"gifs": gifs, "categories": [p[0]["category"] for p in panels]}


def main(make_gif: bool = False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating weight-editing visualizations...")
    render_rank_curve(); print("  ✓ weight_editing_rank_curve.png")
    render_vs_finetuning(); print("  ✓ weight_editing_vs_finetuning.png")
    render_params_vs_success(); print("  ✓ weight_editing_params_vs_success.png")
    render_matching_vs_behavior(); print("  ✓ weight_editing_matching_vs_behavior.png")
    print(f"  ✓ weight_editing_trajectory_comparison.png {render_trajectories(make_gif)}")
    print("Done.")


if __name__ == "__main__":
    main(make_gif=script_args(__doc__, "--gif").gif)
