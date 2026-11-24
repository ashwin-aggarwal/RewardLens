#!/usr/bin/env python3
"""Experiment 2, Phase 3: figures and the training-progression GIF, from recorded results.

Usage: python -m scripts.exp2_visualize_phase3 [--gif]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rewardlens.visualization.gridworld_viz import BG_COLOR, TEXT_COLOR, _draw_agent, _draw_grid

FIGURES, GIFS = Path("assets/figures"), Path("assets/gifs")
RESULTS = Path("results/exp2/phase3")
GOOD, HACK, GREY, WINDOW = "#16A34A", "#EF4444", "#94A3B8", "#FDE68A"
SEEDS = ("42", "123", "456")
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_VEC = ((0, -1), (1, 0), (0, 1), (-1, 0))
LABEL_COLORS = {"agree": "#CBD5E1", "equivalent_route": "#86EFAC", "tile_over_goal": "#EF4444", "other_disagreement": "#FBBF24"}
LABEL_NAMES = {"agree": "Same action", "equivalent_route": "Different, both toward goal",
               "tile_over_goal": "Tile instead of goal", "other_disagreement": "Other difference"}
SAMPLING_NAMES = {"uniform": "Uniform\nstates", "rollout_goal_following": "Visited by\ngoal-follower", "rollout_hacking": "Visited by\nhacker"}


def _style(ax, title=""):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=TEXT_COLOR)


def _mark(ax, tr, label=False):
    """Shade the behavioral transition: onset to completion, with the midpoint as a dotted line."""
    ax.axvspan(tr["onset"] / 1000, tr["completion"] / 1000, color=WINDOW, alpha=0.45, lw=0,
               label="Behavioral transition" if label else None)
    ax.axvline(tr["midpoint"] / 1000, color="#B45309", lw=1, ls=":")


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return name


def render_behavior(tl):
    k = np.array(tl["steps"]) / 1000
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, seed in zip(axes, SEEDS):
        b = tl["per_seed"][seed]
        tr, hk, gd = b["summary"]["transition"]["dev"], b["series"]["hacking"], b["series"]["good"]
        _mark(ax, tr, label=True)
        ax.plot(k, hk["dev_direct_goal"], color=GOOD, lw=2.2, label="Direct goal")
        ax.plot(k, hk["dev_farming"], color=HACK, lw=2.2, label="Farming")
        ax.plot(k, hk["dev_other"], color=GREY, lw=1.5, label="Other (wandering)")
        ax.plot(k, gd["dev_farming"], color=GOOD, lw=1.2, ls="--", label="Farming, good branch")
        _style(ax, f"Seed {seed}: onset {tr['onset'] // 1000}k, complete {tr['completion'] // 1000}k")
        ax.set_xlabel("Training steps after the fork (thousands)", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Share of 1,500 development layouts", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="center right", frameon=False)
    fig.suptitle("When the flawed-reward branch stops going to the goal", fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=1.03)
    return _save(fig, "exp2_phase3_behavior_timeline.png")


def render_activation(tl):
    k = np.array(tl["steps"]) / 1000
    rows = (("rel_change_prev", "Activation change\nper 10k steps"), ("cka_start", "Similarity to fork start\n(linear CKA)"),
            ("mean_norm", "Mean layer-2\nactivation norm"))
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.2), sharex=True, sharey="row")
    fig.patch.set_facecolor(BG_COLOR)
    for j, seed in enumerate(SEEDS):
        b = tl["per_seed"][seed]
        tr, hk, gd = b["summary"]["transition"]["dev"], b["series"]["hacking"], b["series"]["good"]
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            _mark(ax, tr, label=(i == 0))
            y = lambda s, layer: np.array([np.nan if v is None else v for v in s[f"{layer}_{key}"]])
            ax.plot(k, y(hk, "layer2"), color=HACK, lw=2.2, label="Hacking branch, layer 2")
            ax.plot(k, y(gd, "layer2"), color=GOOD, lw=2.2, label="Good branch, layer 2")
            if key != "mean_norm":
                ax.plot(k, y(hk, "layer1"), color=HACK, lw=1.2, ls="--", label="Hacking branch, layer 1")
                ax.plot(k, y(gd, "layer1"), color=GOOD, lw=1.2, ls="--", label="Good branch, layer 1")
            _style(ax, f"Seed {seed}" if i == 0 else "")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if i == 2:
                ax.set_xlabel("Training steps after the fork (thousands)", fontsize=9)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Hidden-layer change against the behavioral transition (same 6,000 development states at every checkpoint)",
                 fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=0.995)
    fig.tight_layout()
    return _save(fig, "exp2_phase3_activation_timeline.png")


def render_heatmap(tl):
    arr = np.load(RESULTS / "mean_activations.npz")
    k = arr["steps"] / 1000
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), sharex=True)
    fig.patch.set_facecolor(BG_COLOR)
    im = None
    for j, seed in enumerate(SEEDS):
        tr = tl["per_seed"][seed]["summary"]["transition"]["dev"]
        h = arr[f"mean_activation_seed{seed}_hacking_layer2"]
        g = arr[f"mean_activation_seed{seed}_good_layer2"]
        dh, dg = (h - h[0]).T, (g - g[0]).T
        # Units ordered by where they end up in the hacking branch; both rows use that order.
        order = np.argsort(dh[:, -1])
        for i, (d, name) in enumerate(((dh, "Hacking branch"), (dg, "Good branch"))):
            ax = axes[i, j]
            im = ax.imshow(d[order], aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                           extent=[k[0] - 5, k[-1] + 5, 128, 0], interpolation="nearest")
            for key, ls in (("onset", "-"), ("completion", "-"), ("midpoint", ":")):
                ax.axvline(tr[key] / 1000, color="black", lw=1, ls=ls)
            ax.set_title(f"Seed {seed}: {name.lower()}", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)
            ax.tick_params(labelsize=8.5)
            if j == 0:
                ax.set_ylabel("128 layer-2 units\n(sorted by final change)", fontsize=9)
            if i == 1:
                ax.set_xlabel("Training steps after the fork (thousands)", fontsize=9)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("Mean activation minus its value at the fork (clipped at ±4)", fontsize=9)
    fig.suptitle("Each layer-2 unit over training. Black lines: behavioral onset, midpoint (dotted) and completion",
                 fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=0.97)
    return _save(fig, "exp2_phase3_neuron_heatmap.png")


def render_pca(tl):
    steps = tl["steps"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    fig.patch.set_facecolor(BG_COLOR)
    sc = None
    for ax, seed in zip(axes, SEEDS):
        b = tl["per_seed"][seed]
        tr, p = b["summary"]["transition"]["dev"], b["trajectory_pca"]["layer2"]
        h, g = np.array(p["hacking"]), np.array(p["good"])
        ax.plot(h[:, 0], h[:, 1], color="#475569", lw=1, zorder=1)
        sc = ax.scatter(h[:, 0], h[:, 1], c=np.array(steps) / 1000, cmap="viridis", s=34, zorder=2, edgecolor="white", lw=0.4)
        ax.scatter(g[:, 0], g[:, 1], color=GOOD, s=22, marker="s", zorder=3, label="Good branch, all 41 checkpoints")
        for key, marker, name in (("onset", "^", "Onset"), ("completion", "v", "Completion")):
            i = steps.index(tr[key])
            ax.scatter(h[i, 0], h[i, 1], s=170, facecolor="none", edgecolor=HACK, lw=2, marker=marker, zorder=4, label=name)
        r = p["explained_variance_ratio"]
        _style(ax, f"Seed {seed}")
        ax.grid(axis="x", color="#E2E8F0", lw=0.8)
        ax.set_xlabel(f"PC 1 ({r[0]:.0%} of variance)", fontsize=9)
        ax.set_ylabel(f"PC 2 ({r[1]:.0%})", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower center", markerscale=0.55)
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("Hacking branch: training steps (thousands)", fontsize=9)
    fig.suptitle("Layer-2 trajectory through training (PCA of each checkpoint's activations on 2,000 fixed states)",
                 fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=1.02)
    return _save(fig, "exp2_phase3_pca_trajectory.png")


def render_decision_example(dec):
    e = dec["example"]
    b = dec["per_seed"][str(e["seed"])]
    pre_k, post_k = b["goal_following_checkpoint_step"] // 1000, b["hacking_checkpoint_step"] // 1000
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.15, 1.5]})
    fig.patch.set_facecolor(BG_COLOR)

    ax = axes[0]
    lay = e["layout"]
    _draw_grid(ax, {"width": 8, "height": 8, "goal_pos": lay[2:4], "reward_pos": lay[4:6]})
    x, y = e["pos"]
    _draw_agent(ax, x, y)
    for a, color, name in ((e["action_pre"], GOOD, f"{pre_k}k checkpoint"), (e["action_post"], HACK, f"{post_k}k checkpoint")):
        dx, dy = ACTION_VEC[a]
        ax.annotate("", xy=(x + 0.95 * dx, y + 0.95 * dy), xytext=(x + 0.3 * dx, y + 0.3 * dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=3), zorder=12)
        ax.plot([], [], color=color, lw=3, label=f"{name}: {ACTION_NAMES[a]}")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)
    ax.set_title(f"Same state, step {e['elapsed_steps']} of 100\n★ goal, $ reward tile", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    ax = axes[1]
    w = 0.38
    xs = np.arange(4)
    for off, q, color, name in ((-w / 2, e["q_pre"], GOOD, f"{pre_k}k (goal-following)"), (w / 2, e["q_post"], HACK, f"{post_k}k (hacking)")):
        adv = np.array(q) - np.mean(q)
        bars = ax.bar(xs + off, adv, w, color=color, label=f"{name}, mean Q {np.mean(q):.1f}")
        bars[int(np.argmax(q))].set_edgecolor("black")
        bars[int(np.argmax(q))].set_linewidth(2)
    ax.set_xticks(xs)
    ax.set_xticklabels(ACTION_NAMES)
    ax.axhline(0, color="#475569", lw=0.8)
    _style(ax, "Q-value of each action, minus the mean\n(black outline: chosen action)")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    d, g = np.array(e["layer2_change"]), np.array(e["layer2_change_good_branch"])
    order = np.argsort(d)
    ax.bar(np.arange(128), d[order], color=HACK, width=1.0, label="Hacking branch")
    ax.bar(np.arange(128), g[order], color=GOOD, width=1.0, label="Good branch, same two steps")
    ax.axhline(0, color="#475569", lw=0.8)
    _style(ax, f"Layer-2 change at this state, {pre_k}k to {post_k}k\n(128 units, sorted)")
    ax.set_xlabel("Layer-2 unit", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"One decision that flipped (seed {e['seed']}, chosen by a fixed rule)", fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=1.04)
    return _save(fig, "exp2_phase3_decision_divergence.png")


def render_decision_stats(dec):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})
    fig.patch.set_facecolor(BG_COLOR)

    ax = axes[0]
    pos, ticks = 0, []
    for name in SAMPLING_NAMES:
        for seed in SEEDS:
            frac = dec["per_seed"][seed]["sampling"][name]["label_fraction"]
            bottom = 0.0
            for label, color in LABEL_COLORS.items():
                ax.bar(pos, frac[label], 0.8, bottom=bottom, color=color,
                       label=LABEL_NAMES[label] if pos == 0 else None)
                bottom += frac[label]
            ticks.append((pos, seed))
            pos += 1
        pos += 0.6
    ax.set_xticks([p for p, _ in ticks])
    ax.set_xticklabels([s for _, s in ticks], fontsize=7.5)
    for i, name in enumerate(SAMPLING_NAMES.values()):
        ax.text(1 + i * 3.6, -0.17, name, ha="center", va="top", fontsize=8.5, transform=ax.get_xaxis_transform())
    _style(ax, "How the two checkpoints' actions compare")
    ax.set_ylabel("Share of states", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1, columnspacing=0.8)

    ax = axes[1]
    w = 0.36
    for i, seed in enumerate(SEEDS):
        lb = dec["per_seed"][seed]["sampling"]["uniform"]["layers"]["layer2"]["activation_change_by_label"]
        gb = dec["per_seed"][seed]["sampling"]["uniform"]["layers"]["layer2"]["good_branch_change_by_label"]
        ax.bar(i - w / 2, lb["agree"], w, color=LABEL_COLORS["agree"], label="Same action" if i == 0 else None)
        ax.bar(i + w / 2, lb["tile_over_goal"], w, color=LABEL_COLORS["tile_over_goal"], label="Tile instead of goal" if i == 0 else None)
        ax.plot([i - w, i + w], [gb["agree"]] * 2, color=GOOD, lw=2.5, label="Good branch, same steps" if i == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Seed {s}" for s in SEEDS])
    _style(ax, "Layer-2 change is as large where\nthe action stayed the same")
    ax.set_ylabel("Change relative to activation norm", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))

    ax = axes[2]
    rows = (("State features only", lambda r: r["predict_tile_over_goal_auroc"]["state_features"]),
            ("+ earlier checkpoint's Q-values", lambda r: r["predict_tile_over_goal_auroc"]["plus_pre_checkpoint_q_values"]),
            ("+ size of activation change", lambda r: r["predict_tile_over_goal_auroc"]["plus_activation_change_magnitude"]),
            ("Size of activation change alone", lambda r: r["predict_tile_over_goal_auroc"]["activation_change_magnitude_alone"]),
            ("Layer-2 change direction", lambda r: r["direction"]["auroc"]),
            ("Tile distance alone", lambda r: r["direction"]["auroc_tile_distance_only"]))
    markers = {"42": "o", "123": "s", "456": "^"}
    for i, (name, get) in enumerate(rows):
        for seed in SEEDS:
            ax.scatter(get(dec["per_seed"][seed]["sampling"]["uniform"]), len(rows) - 1 - i, marker=markers[seed], s=55,
                       color="#1E293B", alpha=0.8, label=f"Seed {seed}" if i == 0 else None)
    ax.axvline(0.5, color=GREY, lw=1, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows][::-1], fontsize=8.5)
    ax.set_xlim(0.45, 1.01)
    _style(ax, "Predicting which decisions flip\n(held-out AUROC, uniform states)")
    ax.grid(axis="x", color="#E2E8F0", lw=0.8)
    ax.set_xlabel("AUROC (0.5 = chance)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="center")
    fig.tight_layout()
    return _save(fig, "exp2_phase3_decision_stats.png")


def render_cross_seed(tl, dec):
    k = np.array(tl["steps"]) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(BG_COLOR)
    ax = axes[0]
    cs = tl["cross_seed_cka_same_step"]["layer2"]
    for i, pair in enumerate(cs["hacking"]):
        ax.plot(k, cs["hacking"][pair], color=HACK, lw=2, alpha=0.55 + 0.2 * i, ls=("-", "--", ":")[i], label=f"Hacking branches, seeds {pair}")
        ax.plot(k, cs["good"][pair], color=GOOD, lw=1.6, ls=("-", "--", ":")[i], label=f"Good branches, seeds {pair}")
    lo = min(tl["per_seed"][s]["summary"]["transition"]["dev"]["onset"] for s in SEEDS) / 1000
    hi = max(tl["per_seed"][s]["summary"]["transition"]["dev"]["completion"] for s in SEEDS) / 1000
    ax.axvspan(lo, hi, color=WINDOW, alpha=0.45, lw=0, label="Any seed in transition")
    _style(ax, "Layer-2 similarity between seeds at the same training step")
    ax.set_xlabel("Training steps after the fork (thousands)", fontsize=9)
    ax.set_ylabel("Linear CKA", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")

    ax = axes[1]
    pairs = list(dec["cross_seed_uniform"])
    w = 0.36
    for i, p in enumerate(pairs):
        c = dec["cross_seed_uniform"][p]
        ax.bar(i - w / 2, c["tile_over_goal_both"], w, color=HACK, label="Observed" if i == 0 else None)
        ax.bar(i + w / 2, c["expected_if_independent"], w, color=GREY, label="If the seeds were unrelated" if i == 0 else None)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"Seeds {p}" for p in pairs], fontsize=8.5)
    _style(ax, "States where both seeds flip\nfrom goal to tile")
    ax.set_ylabel("Share of 6,000 identical states", fontsize=9)
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, "exp2_phase3_cross_seed.png")


def render_gif(tl):
    """Seed 42's hacking branch on one development layout at five checkpoints.

    The layout is development layout 0, the one used for the Phase 2 GIFs. The
    checkpoints are the fork start, the last one before onset, the midpoint,
    completion and the final one. Each is whatever the greedy policy does there.
    """
    from rewardlens.exp2.emergence import rollout_states
    from scripts.exp2_run_phase3 import _net, load_setup
    setup = load_setup()
    params, layout = setup["params"], setup["layouts"]["dev"][0]
    seed, max_show = 42, 26
    b = tl["per_seed"][str(seed)]
    tr = b["summary"]["transition"]["dev"]
    steps = [0, tr["last_before_onset"], tr["midpoint"], tr["completion"], tl["steps"][-1]]
    config = {"width": params.width, "height": params.height, "goal_pos": layout.goal, "reward_pos": layout.reward}
    frames, durations, info = [], [], {"seed": seed, "dev_layout_index": 0, "layout": list(layout.as_tuple()), "checkpoints": []}
    for step in steps:
        st = rollout_states(_net(setup, seed, "hacking", step), [layout], params)
        pos = [tuple(p) for p in st["pos"]]
        n = len(pos)
        farming = b["series"]["hacking"]["dev_farming"][tl["steps"].index(step)]
        visits = sum(1 for i in range(1, n) if pos[i] == tuple(layout.reward) and pos[i - 1] != tuple(layout.reward))
        info["checkpoints"].append({"step": step, "episode_states": n, "tile_entries_seen": visits,
                                    "dev_farming_rate": farming, "reached_goal": bool(n < params.max_steps)})
        shown = pos[:max_show]
        for i, (x, y) in enumerate(shown):
            fig, ax = plt.subplots(figsize=(4.6, 5.3), dpi=80)
            fig.patch.set_facecolor(BG_COLOR)
            _draw_grid(ax, config, title=f"Checkpoint {step // 1000}k of 400k\nfarming in {farming:.0%} of development layouts")
            path = shown[:i + 1]
            ax.plot([p[0] for p in path], [p[1] for p in path], color=HACK if farming >= 0.5 else GOOD, lw=2.5, alpha=0.6, zorder=5)
            _draw_agent(ax, x, y)
            note = f"episode step {i}"
            if i == len(shown) - 1:
                note = "reached the goal next step" if n < params.max_steps and n <= max_show else f"continues to step {n}" if n > max_show else note
            ax.text(0.5, -0.04, note, transform=ax.transAxes, ha="center", va="top", fontsize=10, color=TEXT_COLOR)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
            plt.close(fig)
            frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
            durations.append(1600 if i == len(shown) - 1 else 140)
    GIFS.mkdir(parents=True, exist_ok=True)
    frames[0].save(GIFS / "exp2_phase3_training_progression.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
    json.dump(info, open(RESULTS / "gif_episode.json", "w"), indent=2)
    return info


def main(make_gif: bool = False):
    tl = json.load(open(RESULTS / "timeline.json"))
    dec = json.load(open(RESULTS / "decision_analysis.json"))
    for fn, args in ((render_behavior, (tl,)), (render_activation, (tl,)), (render_heatmap, (tl,)), (render_pca, (tl,)),
                     (render_decision_example, (dec,)), (render_decision_stats, (dec,)), (render_cross_seed, (tl, dec))):
        print("wrote", fn(*args))
    if make_gif:
        print(render_gif(tl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    main(ap.parse_args().gif)
