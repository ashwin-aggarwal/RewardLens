#!/usr/bin/env python3
"""Experiment 2, Phase 1: figures and GIFs from recorded results.

Usage: python -m scripts.exp2_visualize [--gif]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.envs.reward_grid import RewardGridEnv
from rewardlens.envs.trajectory import record_episode
from rewardlens.exp2.evaluation import categories, greedy_policy, rollout
from rewardlens.exp2.layouts import load_manifest
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.visualization.gridworld_viz import BG_COLOR, TEXT_COLOR, render_trajectory_gif
from scripts.exp2_train import load_configs, seed_dir

FIGURES, GIFS = Path("assets/figures"), Path("assets/gifs")
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm then finish", "farm_only": "Farm only",
              "other": "Other / failure"}
CATS = ("direct_goal", "farm_then_finish", "farm_only", "other")
GOOD, HACK, PRE = "#16A34A", "#EF4444", "#64748B"


def _style(ax, title=""):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT_COLOR)


def _curve(tc, seed, stage):
    rows = list(csv.DictReader(open(seed_dir(tc, seed) / stage / "eval_metrics.csv")))
    return (np.array([int(r["timestep"]) for r in rows]),
            {k: np.array([float(r[f"dev_{k}"]) for r in rows]) for k in ("direct_goal", "farming", "other")})


def render_training_curves(tc):
    n_pre = tc["pretrain"]["total_timesteps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, key, title in zip(axes, ("direct_goal", "farming"),
                              ("Direct-goal rate (development layouts)",
                               "Farming rate: farm-only + farm-then-finish (development layouts)")):
        _style(ax, title)
        for i, seed in enumerate(tc["seeds"]):
            x, y = _curve(tc, seed, "pretrain")
            ax.plot(x / 1000, y[key], color=PRE, lw=1.6, alpha=0.75, label="Shared pretraining (corrected reward)" if i == 0 else None)
            for stage, color, name in (("good", GOOD, "Good DQN (corrected reward)"), ("hacking", HACK, "Hacking DQN (flawed reward)")):
                x, y = _curve(tc, seed, stage)
                ax.plot((x + n_pre) / 1000, y[key], color=color, lw=1.8, alpha=0.8, label=name if i == 0 else None)
        ax.axvline(n_pre / 1000, color=TEXT_COLOR, lw=1, ls="--")
        ax.text(n_pre / 1000 + 3, 0.5, "fork", fontsize=9, color=TEXT_COLOR)
        ax.axhline(0.8, color="#94A3B8", lw=0.8, ls=":")
        ax.set_xlabel("Environment steps (thousands)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Fraction of 1,500 development layouts")
    axes[0].legend(fontsize=8, loc="center left", frameon=False)
    fig.suptitle(f"Experiment 2: one line per training seed ({', '.join(map(str, tc['seeds']))}); dotted line = 0.80 target",
                 fontsize=10, color=TEXT_COLOR, y=1.0)
    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_training_curves.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_behavior_by_seed(ev, split):
    seeds = [str(s) for s in ev["seeds"]]
    stages = (("pretrain", "Pretrained"), ("good", "Good"), ("hacking", "Hacking"))
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    _style(ax, f"Behavior on the {ev['n_layouts'][split]:,} {split} layouts, same layouts for every agent")
    xs, labels = [], []
    for si, seed in enumerate(seeds):
        for gi, (stage, name) in enumerate(stages):
            x = si * 4 + gi
            bottom = 0.0
            for c in CATS:
                v = ev["per_seed"][seed][stage][split][c]
                ax.bar(x, v, bottom=bottom, color=CAT_COLORS[c], width=0.8,
                       label=CAT_LABELS[c] if si == 0 and gi == 0 else None)
                if v >= 0.06:
                    ax.text(x, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
                bottom += v
            xs.append(x)
            labels.append(f"{name}\nseed {seed}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of layouts")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_behavior_by_seed.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_generalization(ev):
    splits = [s for s in ("train_sample", "dev", "test") if s in ev["splits_evaluated"]]
    groups = [(s, None) for s in splits]
    names = {"train_sample": "Training layouts\n(1,500 sample)", "dev": "Development\n(unseen layouts)",
             "test": "Final test\n(unseen layouts)"}
    if "test" in splits:
        groups.append(("test", "unseen_goal_tile_pair"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, stage, key, color, title in ((axes[0], "good", "direct_goal", GOOD, "Good DQN: direct-goal rate"),
                                         (axes[1], "hacking", "farming", HACK, "Hacking DQN: farming rate")):
        _style(ax, title)
        labels = []
        for i, (split, sub) in enumerate(groups):
            agg = ev["aggregate"][stage][split]
            a = agg[sub][key] if sub else agg[key]
            ax.bar(i, a["mean"], color=color, alpha=0.75 if not sub else 0.45, width=0.65)
            ax.scatter([i] * len(a["per_seed"]), a["per_seed"], color=TEXT_COLOR, s=18, zorder=3)
            ax.text(i, a["mean"] + 0.03, f"{a['mean']:.3f}", ha="center", fontsize=9, color=TEXT_COLOR)
            n = ev["n_unseen_goal_tile_pair"][split] if sub else ev["n_layouts"][split]
            labels.append((names[split] if not sub else "Test, goal/tile pair\nnever seen in training") + f"\nn = {n:,}")
        ax.axhline(0.8, color="#94A3B8", lw=0.8, ls=":")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("Rate (bar = mean of 3 seeds, dots = seeds)")
    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_generalization.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_scripted_returns(tc):
    d = json.load(open(Path(tc["results_dir"]) / "scripted_returns.json"))
    pol = [("goal", "Walk to goal"), ("farm", "Farm until time runs out"), ("farm_then_finish", "Farm, then finish")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    for ax, key, title in ((axes[0], "discounted_flawed", "Flawed reward (tile pays +2)"),
                           (axes[1], "discounted_corrected", "Corrected reward (tile pays 0)")):
        _style(ax, title)
        for i, (p, name) in enumerate(pol):
            s = d["policies"][p][key]
            ax.bar(i, s["mean"], color=[GOOD, HACK, "#F59E0B"][i], alpha=0.8, width=0.6)
            ax.errorbar(i, s["mean"], yerr=[[max(0.0, s["mean"] - s["min"])], [max(0.0, s["max"] - s["mean"])]], color=TEXT_COLOR, capsize=4, lw=1)
            ax.text(i, s["max"] + 1.5, f"{s['mean']:.1f}", ha="center", fontsize=9, color=TEXT_COLOR)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n for _, n in pol], fontsize=8)
        ax.axhline(0, color=TEXT_COLOR, lw=0.6)
    axes[0].set_ylabel(f"Discounted return, γ = {d['gamma']}\n(bar = mean, whiskers = min/max)")
    fig.suptitle(f"Scripted policies on {d['n_layouts']:,} training + development layouts", fontsize=10, color=TEXT_COLOR, y=1.0)
    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_scripted_returns.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_gifs(ec, tc, ev):
    """Good and hacking DQN (first seed) on one test layout.

    Selection rule: the first test layout, in manifest order, with start-goal
    distance >= 6 and start-tile distance >= 3 on which each agent shows the
    behavior it shows on most layouts. The choice is illustrative; rates are in
    the tables.
    """
    params = ProceduralEnvParams(**ec["env"])
    splits, _ = load_manifest(ec["layouts"]["manifest_path"])
    seed = tc["seeds"][0]
    nets = {s: load_q_network_for_eval(seed_dir(tc, seed) / s / "checkpoints" / "checkpoint_final.pt")[0] for s in ("good", "hacking")}
    res = {s: rollout(greedy_policy(n), splits["test"], params, tc["dqn"]["gamma"], record_actions=True) for s, n in nets.items()}
    cat = {s: categories(r, params.exploit_visit_threshold) for s, r in res.items()}
    idx = next(i for i, l in enumerate(splits["test"])
               if l.manhattan_to_goal() >= 6 and abs(l.start[0] - l.reward[0]) + abs(l.start[1] - l.reward[1]) >= 3
               and cat["good"][i] == "direct_goal" and cat["hacking"][i] in ("farm_only", "farm_then_finish"))
    layout = splits["test"][idx]
    info = {"seed": seed, "test_layout_index": idx, "layout": layout.as_tuple()}
    GIFS.mkdir(parents=True, exist_ok=True)
    for stage, name, fname in (("good", "Good DQN (goal-following)", "exp2_good_dqn.gif"), ("hacking", "Hacking DQN", "exp2_hacking_dqn.gif")):
        actions = [int(a) for a in res[stage]["actions"][idx] if a >= 0]
        traj = record_episode(RewardGridEnv(config=params.grid_config(layout, "flawed")), actions, name, f"exp2_{stage}", seed=0)
        assert traj.final_reward_tile_visits == res[stage]["reward_tile_visits"][idx]
        render_trajectory_gif(traj, GIFS / fname, max_frames=40, frame_duration_ms=200, last_frame_duration_ms=2000)
        info[stage] = {"category": str(cat[stage][idx]), "reward_tile_visits": int(res[stage]["reward_tile_visits"][idx]),
                       "steps": int(res[stage]["steps"][idx])}
    json.dump(info, open(Path(tc["results_dir"]) / "gif_episode.json", "w"), indent=2)
    return info


def main(make_gif: bool = False):
    ec, tc = load_configs()
    ev = json.load(open(Path(tc["results_dir"]) / "evaluation.json"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_scripted_returns(tc)
    render_training_curves(tc)
    render_behavior_by_seed(ev, "test" if "test" in ev["splits_evaluated"] else "dev")
    render_generalization(ev)
    print("figures written")
    if make_gif:
        if "test" not in ev["splits_evaluated"]:
            raise SystemExit("GIFs use a test layout; run the final test evaluation first.")
        print(render_gifs(ec, tc, ev))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    main(ap.parse_args().gif)
