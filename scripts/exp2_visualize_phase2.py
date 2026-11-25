#!/usr/bin/env python3
"""Experiment 2, Phase 2: figures and GIFs from recorded results.

Usage: python -m scripts.exp2_visualize_phase2 [--gif]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rewardlens.envs.reward_grid import RewardGridEnv
from rewardlens.envs.trajectory import record_episode
from rewardlens.exp2.evaluation import categories, rollout
from rewardlens.exp2.patching import make_policy
from rewardlens.visualization.gridworld_viz import BG_COLOR, TEXT_COLOR, render_trajectory_gif

FIGURES, GIFS = Path("assets/figures"), Path("assets/gifs")
RESULTS = Path("results/exp2/phase2")
GOOD, HACK, PRE = "#16A34A", "#EF4444", "#64748B"
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm then finish", "farm_only": "Farm only",
              "other": "Other / failure"}
CATS = tuple(CAT_COLORS)
SEED_MARKERS = {"42": "o", "123": "s", "456": "^"}


def _style(ax, title=""):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=TEXT_COLOR)


def render_cka(rep):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, layer in zip(axes, ("layer1", "layer2")):
        cb = rep["cross_seed"][layer]
        m = np.array(cb["cka_matrix"])
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        labels = [l.replace("pretrain", "pre").replace("hacking", "hack").replace("_", " ") for l in cb["cka_labels"]]
        ax.set_xticks(range(9)); ax.set_yticks(range(9))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(labels, fontsize=8)
        for i in range(9):
            for j in range(9):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] < 0.6 else "black")
        for b in (2.5, 5.5):
            ax.axhline(b, color="white", lw=1.5); ax.axvline(b, color="white", lw=1.5)
        ax.set_title(f"Linear CKA, {layer.replace('layer', 'layer ')}", fontsize=11, fontweight="bold", color=TEXT_COLOR)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(f"Representation similarity on {rep['n_states']['dev']:,} identical development states "
                 "(pretrained, good and hacking DQNs, 3 seeds)", fontsize=10, color=TEXT_COLOR, y=0.98)
    fig.savefig(FIGURES / "exp2_phase2_cka.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_difference(rep):
    sc = np.load(RESULTS / "pca_scatter.npz")
    seeds = list(rep["within_seed"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    ax = axes[0]
    _style(ax, "Distance from the shared pretrained network")
    w = 0.38
    for i, s in enumerate(seeds):
        for j, layer in enumerate(("layer1", "layer2")):
            rc = rep["within_seed"][s]["layers"][layer]["relative_change_from_pretrain"]
            x = i * 2.4 + j
            ax.bar(x - w / 2, rc["good"], w, color=GOOD, label="Good" if i == j == 0 else None)
            ax.bar(x + w / 2, rc["hacking"], w, color=HACK, label="Hacking" if i == j == 0 else None)
            ax.text(x + w / 2, rc["hacking"] + 0.04, f"{rc['hacking']:.2f}", ha="center", fontsize=7.5)
            ax.text(x - w / 2, rc["good"] + 0.04, f"{rc['good']:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks([i * 2.4 + j for i in range(3) for j in range(2)])
    ax.set_xticklabels([f"L{j + 1}\nseed {s}" for s in seeds for j in range(2)], fontsize=8)
    ax.set_ylabel("Mean activation change / pretrained norm")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    _style(ax, "(Hacking − good), state-dependent part")
    for s in seeds:
        for layer, ls in (("layer2", "-"), ("layer1", "--")):
            d = rep["within_seed"][s]["layers"][layer]["difference_pca"]["hacking_minus_good"]
            ax.plot(range(1, 17), np.cumsum(d["explained_variance_ratio"])[:16], ls=ls, marker=SEED_MARKERS[s], ms=3.5,
                    color=HACK if layer == "layer2" else "#F59E0B", alpha=0.85,
                    label=f"{layer.replace('layer', 'Layer ')}, seed {s}")
    ax.axhline(0.9, color="#94A3B8", lw=0.8, ls=":")
    ax.set_xlabel("Principal components of the centered difference")
    ax.set_ylabel("Cumulative variance explained (development states)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    share = [rep["within_seed"][s]["layers"]["layer2"]["difference_pca"]["hacking_minus_good"]["constant_offset_share"] for s in seeds]
    ax.text(0.97, 0.42, f"Before centering, {min(share):.0%}–{max(share):.0%} of the\nlayer-2 difference is one\nconstant offset",
            transform=ax.transAxes, fontsize=8, color=TEXT_COLOR, ha="right")

    ax = axes[2]
    s = seeds[0]
    _style(ax, f"Layer 2, seed {s}: same states, three networks")
    for st, color, name in (("hacking", HACK, "Hacking"), ("good", GOOD, "Good"), ("pretrain", PRE, "Pretrained")):
        pts = sc[f"seed{s}_{st}"]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=color, alpha=0.45, label=name, linewidths=0)
    ax.set_xlabel("PC 1 of pooled good + hacking activations"); ax.set_ylabel("PC 2")
    ax.legend(fontsize=8, frameon=False, markerscale=2.5)
    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_phase2_difference_pca.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_cross_seed(rep):
    keys = (("r2_from_other_seed_difference", "Other seed's\n(hacking − good)\ntop-8 PCs", HACK),
            ("r2_from_other_seed_hacking_activations", "Other seed's\nhacking activations\ntop-8 PCs", "#F59E0B"),
            ("r2_from_other_seed_good_activations", "Other seed's\ngood activations\ntop-8 PCs", GOOD),
            ("r2_from_other_seed_difference_shuffled_states", "Other seed's difference,\nstates shuffled", PRE))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), dpi=150, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, layer in zip(axes, ("layer1", "layer2")):
        _style(ax, f"{layer.replace('layer', 'Layer ')}: predicting one seed's (hacking − good) top-8 PCs")
        pairs = rep["cross_seed"][layer]["pairs"]
        for i, (k, name, color) in enumerate(keys):
            vals = [p[k] for p in pairs.values()]
            ax.bar(i, np.mean(vals), color=color, alpha=0.75, width=0.65)
            ax.scatter([i] * len(vals), vals, color=TEXT_COLOR, s=14, zorder=3)
            ax.text(i, max(np.mean(vals), 0) + 0.04, f"{np.mean(vals):.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([n for _, n, _ in keys], fontsize=8)
        ax.set_ylim(-0.05, 1.1)
    axes[0].set_ylabel("Held-out R² (fit on training-layout states,\nscored on development states; dots = 6 seed pairs)")
    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_phase2_cross_seed.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _cat_values(block):
    return [block[c]["mean"] if isinstance(block[c], dict) else block[c] for c in CATS]


def render_patching(pat):
    seeds = list(pat["per_seed"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, s in zip(axes, seeds):
        b = pat["per_seed"][s]
        g2, g1 = b["good_to_hacking"]["layer2"], b["good_to_hacking"]["layer1"]
        bars = [("Hacking\nalone", b["unmodified"]["hacking"]), ("Constant\nQ bias", g2["q_bias"]),
                ("Random-\nstate\ndonor", g2["shuffled"]), ("Wrong-\nposition\ndonor", g2["wrong_position"]),
                ("Matched\npatch,\nlayer 2", g2["matched"]), ("Matched\npatch,\nlayer 1", g1["matched"]),
                ("Good\nalone", b["unmodified"]["good"])]
        _style(ax, f"Seed {s}")
        for x, (name, block) in enumerate(bars):
            bottom = 0.0
            for c, v in zip(CATS, _cat_values(block)):
                ax.bar(x, v, bottom=bottom, color=CAT_COLORS[c], width=0.8, label=CAT_LABELS[c] if x == 0 and s == seeds[0] else None)
                if v >= 0.08:
                    ax.text(x, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
                bottom += v
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([n for n, _ in bars], fontsize=7)
        ax.axvspan(3.5, 5.5, color="#FEF3C7", alpha=0.5, zorder=0)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(f"Fraction of {pat['n_dev_layouts']:,} development layouts")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.suptitle("Full-layer activation transplant into the hacking DQN (shaded = matched-state patches; controls use layer 2, "
                 f"mean of {pat['n_control_draws']} draws)", fontsize=10, color=TEXT_COLOR, y=1.0)
    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_phase2_patching.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_gifs():
    """Seed 42 on one development layout: unmodified hacking DQN and the layer-2 transplant.

    The layout is the first development layout with start-goal distance >= 6 and
    start-tile distance >= 3 on which the unmodified hacking agent farms. The
    patched outcome is whatever happens there; it is not selected.
    """
    from scripts.exp2_run_phase2 import GAMMA, load_setup
    setup = load_setup()
    params, dev = setup["params"], setup["layouts"]["dev"]
    trip = setup["nets"][42]
    pols = {"hacking": make_policy(trip["hacking"]), "patched": make_policy(trip["hacking"], trip["good"], 2, "matched")}
    res = {k: rollout(p, dev, params, GAMMA, record_actions=True) for k, p in pols.items()}
    cat = {k: categories(r, params.exploit_visit_threshold) for k, r in res.items()}
    idx = next(i for i, l in enumerate(dev) if l.manhattan_to_goal() >= 6
               and abs(l.start[0] - l.reward[0]) + abs(l.start[1] - l.reward[1]) >= 3 and cat["hacking"][i] == "farm_only")
    info = {"seed": 42, "dev_layout_index": idx, "layout": dev[idx].as_tuple()}
    GIFS.mkdir(parents=True, exist_ok=True)
    for key, name, fname in (("hacking", "Hacking DQN", "exp2_phase2_hacking.gif"),
                             ("patched", "Hacking DQN + good layer-2 transplant", "exp2_phase2_patched.gif")):
        actions = [int(a) for a in res[key]["actions"][idx] if a >= 0]
        traj = record_episode(RewardGridEnv(config=params.grid_config(dev[idx], "flawed")), actions, name, f"exp2p2_{key}", seed=0)
        render_trajectory_gif(traj, GIFS / fname, max_frames=40, frame_duration_ms=200, last_frame_duration_ms=2000)
        info[key] = {"category": str(cat[key][idx]), "reward_tile_visits": int(res[key]["reward_tile_visits"][idx]),
                     "steps": int(res[key]["steps"][idx])}
    json.dump(info, open(RESULTS / "gif_episode.json", "w"), indent=2)
    return info


def main(make_gif: bool = False):
    rep = json.load(open(RESULTS / "representation_analysis.json"))
    pat = json.load(open(RESULTS / "patching_results.json"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_cka(rep); render_difference(rep); render_cross_seed(rep); render_patching(pat)
    print("figures written")
    if make_gif:
        print(render_gifs())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    main(ap.parse_args().gif)
