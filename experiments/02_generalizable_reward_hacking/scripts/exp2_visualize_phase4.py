#!/usr/bin/env python3
"""Experiment 2, Phase 4: figures and a comparison GIF, from recorded results.

Usage: python -m scripts.exp2_visualize_phase4 [--gif]
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
RESULTS = Path("results/exp2/phase4")
GOOD, HACK, GREY, BLUE, AMBER = "#16A34A", "#EF4444", "#94A3B8", "#2563EB", "#F59E0B"
SEEDS = ("42", "123", "456")
MARK = {"42": "o", "123": "s", "456": "^"}
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm then finish", "farm_only": "Farm only", "other": "Other / failure"}
PAIR_LABEL = {"minus_10k": "−10k", "minus_20k": "−20k", "minus_50k": "−50k", "pre_onset": "pre-onset",
              "fork_start": "fork start", "pre_onset_to_final": "pre-onset\n→ 400k", "good_same_step": "good,\nsame step"}


def _style(ax, title=""):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=TEXT_COLOR)


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return name


def _pairs(res, seed, kind=None):
    return [p for p in res["per_seed"][seed]["pairs"] if kind is None or p["kind"] == kind]


def render_gap(res):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, seed in zip(axes, SEEDS):
        ps = _pairs(res, seed, "main")
        x = np.array([p["gap"] for p in ps]) / 1000
        l2 = [p["layers"]["layer2"] for p in ps]
        lo = [min(l["shuffled"]["direct_goal"]["mean"], l["wrong_position"]["direct_goal"]["mean"]) for l in l2]
        hi = [max(l["shuffled"]["direct_goal"]["mean"], l["wrong_position"]["direct_goal"]["mean"]) for l in l2]
        ax.fill_between(x, lo, hi, color=GREY, alpha=0.35, lw=0, label="Wrong-state donors (shuffled, wrong position)")
        ax.plot(x, [p["donor"]["direct_goal"] for p in ps], color=GOOD, ls="--", lw=1.6, marker="o", ms=4, label="Donor checkpoint alone")
        ax.plot(x, [l["matched"]["direct_goal"] for l in l2], color=BLUE, lw=2.4, marker="o", label="Layer-2 patch, matched state")
        ax.plot(x, [p["layers"]["layer1"]["matched"]["direct_goal"] for p in ps], color=BLUE, lw=1.3, ls=":", marker="s", ms=4,
                label="Layer-1 patch, matched state")
        ax.axhline(ps[0]["recipient"]["direct_goal"], color=HACK, lw=1.4, label="Recipient alone (hacking)")
        ax.axhline(res["random_policy_baseline"]["direct_goal"]["mean"], color="#475569", lw=1.2, ls="-.",
                   label="Uniformly random policy (efficiency 0.23)")
        ref = [p for p in _pairs(res, seed, "reference") if p["name"] == "pre_onset_to_final"][0]
        ax.scatter([ref["gap"] / 1000], [ref["layers"]["layer2"]["matched"]["direct_goal"]], marker="*", s=140, color=BLUE,
                   edgecolor="black", lw=0.6, zorder=5, label="Same pre-onset donor → 400k recipient")
        _style(ax, f"Seed {seed}: recipient = {ps[0]['recipient_step'] // 1000}k (completion)")
        ax.set_xlabel("Steps between donor and recipient (thousands)", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Direct-goal rate, 1,500 development layouts", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.suptitle("Patching earlier checkpoints of the same hacking run into the completion checkpoint",
                 fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=1.03)
    return _save(fig, "exp2_phase4_gap.png")


def render_matched_vs_shuffled(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 2.3]})
    fig.patch.set_facecolor(BG_COLOR)
    ax = axes[0]
    for seed in SEEDS:
        for p in _pairs(res, seed):
            for layer, color in (("layer2", BLUE), ("layer1", AMBER)):
                l = p["layers"][layer]
                best = max(l["shuffled"]["direct_goal"]["mean"], l["wrong_position"]["direct_goal"]["mean"])
                ax.scatter(best, l["matched"]["direct_goal"], marker=MARK[seed], s=45, color=color, alpha=0.85, edgecolor="white", lw=0.4)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    ax.plot([0, 0.9], [0.1, 1.0], color=GREY, lw=0.8, ls=":")
    for layer, color in (("Layer 2", BLUE), ("Layer 1", AMBER)):
        ax.scatter([], [], color=color, label=layer)
    for seed in SEEDS:
        ax.scatter([], [], marker=MARK[seed], color="#334155", label=f"Seed {seed}")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(ax, "Matched donor vs better wrong-state donor\n(every pair; dotted line: +0.10)")
    ax.grid(axis="x", color="#E2E8F0", lw=0.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Direct goal, better of shuffled / wrong-position donor", fontsize=9)
    ax.set_ylabel("Direct goal, matched-state donor", fontsize=9)

    ax = axes[1]
    names = [p["name"] for p in _pairs(res, SEEDS[0])]
    x, ticks = 0.0, []
    for name in names:
        for seed in SEEDS:
            p = [q for q in _pairs(res, seed) if q["name"] == name][0]
            for off, block, hatch in ((-0.2, p["layers"]["layer2"]["matched"], None),
                                      (0.2, {k: v["mean"] for k, v in p["layers"]["layer2"]["shuffled"].items() if k != "n_draws"}, "///")):
                bottom = 0.0
                for c in CAT_COLORS:
                    ax.bar(x + off, block[c], 0.38, bottom=bottom, color=CAT_COLORS[c], hatch=hatch, edgecolor="white", lw=0.3,
                           label=CAT_LABELS[c] if (x == 0 and off < 0) else None)
                    bottom += block[c]
            ticks.append((x, seed))
            x += 1
        ax.text(x - 2, -0.2, PAIR_LABEL[name], ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
        x += 0.6
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([s for _, s in ticks], fontsize=6.5)
    ax.bar([0], [0], color="white", edgecolor="#475569", hatch="///", label="Hatched: shuffled donor (mean of 5)")
    _style(ax, "Layer-2 outcomes by pair and seed (plain: matched donor; hatched: shuffled donor)")
    ax.set_ylabel("Share of development episodes", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5)
    fig.tight_layout()
    return _save(fig, "exp2_phase4_matched_vs_shuffled.png")


def render_compatibility(res):
    rows = res["relations"]["rows"]
    specs = (("cka_layer2", "Layer-2 CKA, donor vs recipient"), ("norm_ratio_layer2", "Recipient / donor layer-2 norm"),
             ("output_head_drift", "Output-layer weight drift"), ("donor_direct_goal", "Donor's own direct-goal rate"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    rho = res["relations"]["spearman_pooled"]["matched_minus_best_control"]
    for ax, (key, label) in zip(axes, specs):
        for r in rows:
            ax.scatter(r[key], r["matched_minus_best_control"], marker=MARK[r["seed"]], s=55, c=[r["donor_direct_goal"]],
                       cmap="Greens", vmin=-0.2, vmax=1, edgecolor="#334155", lw=0.6)
        ax.axhline(0, color=GREY, lw=1, ls="--")
        ax.axhline(0.1, color=GREY, lw=0.8, ls=":")
        r_ = rho[key]
        _style(ax, f"{label}\nSpearman ρ = {r_:+.2f}" if r_ is not None else label)
        ax.grid(axis="x", color="#E2E8F0", lw=0.8)
        ax.set_xlabel(label, fontsize=9)
    axes[0].set_ylabel("Layer-2 matched direct goal\nminus better wrong-state donor", fontsize=9)
    for seed in SEEDS:
        axes[3].scatter([], [], marker=MARK[seed], color="#334155", label=f"Seed {seed}")
    axes[3].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("State-specific benefit of the patch against donor-recipient compatibility "
                 "(15 main pairs; fill = donor's direct-goal rate; correlational)", fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05)
    return _save(fig, "exp2_phase4_compatibility.png")


def render_actions(res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.patch.set_facecolor(BG_COLOR)
    names = [p["name"] for p in _pairs(res, SEEDS[0])]
    w = 0.26
    for i, seed in enumerate(SEEDS):
        ps = {p["name"]: p for p in _pairs(res, seed)}
        xs = np.arange(len(names)) + (i - 1) * w
        flip = [ps[n]["layers"]["layer2"]["actions_uniform"]["at_flip_states"]["restored_to_donor_action"] or 0 for n in names]
        agree = [ps[n]["layers"]["layer2"]["actions_uniform"]["at_agree_states"]["action_changed"] or 0 for n in names]
        axes[0].bar(xs, flip, w, color=BLUE, alpha=0.45 + 0.25 * i, label=f"Seed {seed}")
        axes[1].bar(xs, agree, w, color=HACK, alpha=0.45 + 0.25 * i, label=f"Seed {seed}")
        dg_patch = [ps[n]["layers"]["layer2"]["matched"]["direct_goal"] for n in names]
        dg_swap = [ps[n]["head_swap_layer2"]["direct_goal"] for n in names]
        axes[2].scatter(dg_patch, dg_swap, marker=MARK[seed], s=50, color="#334155", label=f"Seed {seed}")
    for ax, title, ylabel in ((axes[0], "Where donor goes to the goal and recipient\nto the tile: patched action = donor's",
                               "Share of those uniform states"),
                              (axes[1], "Where donor and recipient agree:\npatch changes the action anyway", "Share of those uniform states")):
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([PAIR_LABEL[n] for n in names], fontsize=7.5)
        _style(ax, title)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7.5, frameon=False)
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    _style(ax, "Features vs readout (layer 2)")
    ax.grid(axis="x", color="#E2E8F0", lw=0.8)
    ax.set_xlabel("Donor features → recipient head (the patch)", fontsize=9)
    ax.set_ylabel("Recipient features → donor head", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("What the layer-2 patch does to individual decisions (6,000 uniform development states) and where the behavior lives",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.03)
    fig.tight_layout()
    return _save(fig, "exp2_phase4_actions.png")


def render_gif(res):
    """Seed 42, development layout 0: recipient, donor and the layer-2 patch, side by side.

    Pair: the pre-onset donor into the completion recipient (fixed in advance as the most
    goal-following donor of the main pairs). Layout: the Phase 2 rule, the first development
    layout with start-goal distance >= 6 and start-tile distance >= 3 on which the unmodified
    recipient farms. The patched outcome is whatever happens there.
    """
    from rewardlens.exp2.emergence import rollout_states
    from rewardlens.exp2.patching import make_policy
    from scripts.exp2_run_phase4 import load_setup, net
    from rewardlens.exp2.evaluation import rollout
    from rewardlens.exp2.evaluation import categories
    setup = load_setup()
    params, dev = setup["params"], setup["layouts"]["dev"]
    seed = 42
    pair = [p for p in res["per_seed"][str(seed)]["pairs"] if p["name"] == "pre_onset"][0]
    donor, rec = net(setup, seed, "hacking", pair["donor_step"]), net(setup, seed, "hacking", pair["recipient_step"])
    cat = categories(rollout(make_policy(rec), dev, params), params.exploit_visit_threshold)
    idx = next(i for i, l in enumerate(dev) if l.manhattan_to_goal() >= 6
               and abs(l.start[0] - l.reward[0]) + abs(l.start[1] - l.reward[1]) >= 3 and cat[i] == "farm_only")
    layout = dev[idx]
    policies = {"Recipient ({}k, hacking)".format(pair["recipient_step"] // 1000): make_policy(rec),
                "Donor ({}k, pre-onset)".format(pair["donor_step"] // 1000): make_policy(donor),
                "Recipient + donor's layer 2": make_policy(rec, donor, 2, "matched")}
    paths, info = {}, {"seed": seed, "dev_layout_index": idx, "layout": list(layout.as_tuple()), "pair": pair["name"], "episodes": {}}
    for name, pol in policies.items():
        acts = rollout(pol, [layout], params, record_actions=True)["actions"][0]
        pos, out = list(layout.start), [tuple(layout.start)]
        for a in acts[acts >= 0]:
            dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[a]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < params.width and 0 <= ny < params.height:
                pos = [nx, ny]
            out.append(tuple(pos))
        paths[name] = out
        entries = sum(1 for i in range(1, len(out)) if out[i] == tuple(layout.reward) and out[i - 1] != tuple(layout.reward))
        info["episodes"][name] = {"steps": len(out) - 1, "reached_goal": out[-1] == tuple(layout.goal), "tile_entries": entries}
    config = {"width": params.width, "height": params.height, "goal_pos": layout.goal, "reward_pos": layout.reward}
    n_frames = min(40, max(len(p) for p in paths.values()))
    frames, durations = [], []
    for t in range(n_frames):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), dpi=72)
        fig.patch.set_facecolor(BG_COLOR)
        for ax, (name, path) in zip(axes, paths.items()):
            i = min(t, len(path) - 1)
            ep = info["episodes"][name]
            if ep["reached_goal"] and i == len(path) - 1:
                status = f"goal reached in {ep['steps']} steps"
            elif t == n_frames - 1:
                status = f"{ep['steps']} steps, no goal, {ep['tile_entries']} tile entries"
            else:
                status = f"step {i}"
            _draw_grid(ax, config, title=f"{name}\n{status}")
            ax.title.set_fontsize(9.5)
            seg = path[:i + 1]
            ax.plot([p[0] for p in seg], [p[1] for p in seg], color=BLUE, lw=2.2, alpha=0.55, zorder=5)
            _draw_agent(ax, *path[i])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
        durations.append(2000 if t == n_frames - 1 else 180)
    GIFS.mkdir(parents=True, exist_ok=True)
    frames[0].save(GIFS / "exp2_phase4_patch_comparison.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
    info["frames_shown"] = n_frames
    json.dump(info, open(RESULTS / "gif_episode.json", "w"), indent=2)
    return info


def main(make_gif: bool = False):
    res = json.load(open(RESULTS / "patching_results.json"))
    for fn in (render_gap, render_matched_vs_shuffled, render_compatibility, render_actions):
        print("wrote", fn(res))
    if make_gif:
        print(render_gif(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    main(ap.parse_args().gif)
