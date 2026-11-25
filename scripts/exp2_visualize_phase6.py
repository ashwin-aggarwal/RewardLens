"""Experiment 2, Phase 6 figures and GIF.

Usage: python -m scripts.exp2_visualize_phase6 [--gif]
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
from matplotlib.lines import Line2D
from PIL import Image

from rewardlens.visualization.gridworld_viz import BG_COLOR, TEXT_COLOR, _draw_agent, _draw_grid

RESULTS = Path("results/exp2/phase6")
FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
SEEDS = ("42", "123", "456")
SEED_COLORS = {"42": "#2563EB", "123": "#EA580C", "456": "#16A34A"}
MARK = {"42": "o", "123": "s", "456": "^"}
METHOD_COLORS = {"coordinated": "#111827", "layer2_only": "#7C3AED", "head_only": "#EA580C", "layer1_only": "#0EA5E9",
                 "full_network": "#DC2626"}
METHOD_LABELS = {"coordinated": "All three layers, low rank", "layer2_only": "Layer 2 only", "head_only": "Output head only",
                 "layer1_only": "Layer 1 only", "full_network": "Full network, ordinary fine-tuning"}
LR_LABELS = {"primary": "learning rate 0.00025 (Phase 5 setting)", "secondary": "learning rate 0.001 (sensitivity check)"}
BLUE = "#2563EB"
FAST = "#0F766E"


def _summary():
    return json.load(open(RESULTS / "summary.json"))


def _runs():
    return json.load(open(RESULTS / "runs.json"))["runs"]


def _group(summary, method, rank, lr):
    return next(g for g in summary["groups"] if g["method"] == method and g["rank"] == rank and g["lr_name"] == lr)


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


def render_repair_vs_rank():
    """Direct goal and path efficiency of coordinated low-rank edits by rank, every run shown."""
    s = _summary()
    crit = s["repair_criterion"]
    ranks = sorted(g["rank"] for g in s["groups"] if g["method"] == "coordinated" and g["lr_name"] == "primary")
    cols = [(r, "coordinated") for r in ranks] + [(None, "full_network")]
    labels = [("biases\nonly" if r == 0 else f"rank {r}") + f"\n{_group(s, 'coordinated', r, 'primary')['trainable_parameters']:,}" for r in ranks] + ["full\n18,052"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for c, lr in enumerate(("primary", "secondary")):
        for r, (key, ylabel, line) in enumerate((("direct_goal", "Direct goal", crit["min_direct_goal"]),
                                                 ("path_efficiency", "Path efficiency", crit["min_path_efficiency"]))):
            ax = axes[r, c]
            for x, (rank, method) in enumerate(cols):
                g = _group(s, method, rank, lr)
                for j, seed in enumerate(SEEDS):
                    vals = g["per_seed"][seed][key]
                    xs = x + (j - 1) * 0.2 + np.linspace(-0.05, 0.05, len(vals))
                    ax.scatter(xs, vals, color=SEED_COLORS[seed], marker=MARK[seed], s=42, edgecolor="white", lw=0.6, zorder=4)
                if r == 0:
                    ax.text(x, 1.07, f"{g['n_repaired']}/{g['n_runs']}", ha="center", fontsize=9, fontweight="bold",
                            color="#16A34A" if g["reliably_successful"] else "#B91C1C")
            ax.axhline(line, color="#16A34A", ls="--", lw=1.2)
            ax.axvline(len(cols) - 1.5, color="#CBD5E1", lw=1)
            ax.set_ylim(0, 1.16)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(labels, fontsize=8.5)
            _style(ax, f"{ylabel}, {LR_LABELS[lr]}")
            if c == 0:
                ax.set_ylabel(f"{ylabel} (750 reporting layouts)", fontsize=9)
        axes[1, c].set_xlabel("Rank of the update in every layer, and trainable parameters", fontsize=9)
    handles = [Line2D([], [], marker=MARK[x], ls="", color=SEED_COLORS[x], ms=8, label=f"DQN seed {x} (3 optimization seeds)") for x in SEEDS]
    handles += [Line2D([], [], color="#16A34A", ls="--", label="Repair criterion (0.9)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Coordinated low-rank edits: repair by rank. Numbers above each column: runs meeting the full repair criterion",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.0)
    fig.tight_layout()
    return _save(fig, "exp2_phase6_repair_vs_rank.png")


def _bar(ax, x, g_or_vals, color, key=None, hatch=None):
    if isinstance(g_or_vals, dict):
        st = g_or_vals[key]
        mean, lo, hi = st["mean"], st["min"], st["max"]
    else:
        mean, lo, hi = float(np.mean(g_or_vals)), float(np.min(g_or_vals)), float(np.max(g_or_vals))
    ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=color, ms=7, capsize=3, lw=1.4, mec="white", mew=0.7, zorder=4,
                mfc="white" if hatch else color)
    if hatch:
        ax.scatter([x], [mean], marker="o", s=45, facecolor="white", edgecolor=color, lw=1.6, zorder=5)


def render_parameter_efficiency():
    """Trainable parameters against repair for every method (mean and range over runs)."""
    s = _summary()
    crit = s["repair_criterion"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), sharex=True)
    fig.patch.set_facecolor("white")
    p5 = s["baselines"]["per_seed"]
    for ax, (key, p5key, ylabel, line) in zip(axes, (("direct_goal", "direct_goal", "Direct goal", crit["min_direct_goal"]),
                                                    ("path_efficiency", "path_efficiency_direct", "Path efficiency", crit["min_path_efficiency"]))):
        for method in ("coordinated", "layer2_only", "head_only"):
            gs = sorted((g for g in s["groups"] if g["method"] == method and g["lr_name"] == "primary"), key=lambda g: g["trainable_parameters"])
            ax.plot([g["trainable_parameters"] for g in gs], [g[key]["mean"] for g in gs], color=METHOD_COLORS[method], lw=1.6, alpha=0.6, zorder=3)
            for g in gs:
                _bar(ax, g["trainable_parameters"], g, METHOD_COLORS[method], key)
        gs = sorted((g for g in s["groups"] if g["method"] == "coordinated" and g["lr_name"] == "secondary"), key=lambda g: g["trainable_parameters"])
        ax.plot([g["trainable_parameters"] for g in gs], [g[key]["mean"] for g in gs], color=FAST, lw=1.6, ls="--", alpha=0.7, zorder=3)
        for g in gs:
            _bar(ax, g["trainable_parameters"], g, FAST, key)
        g = _group(s, "full_network", None, "primary")
        _bar(ax, g["trainable_parameters"], g, METHOD_COLORS["full_network"], key)
        ax.set_xlim(100, 30000)
        for cond, method in (("head_only", "head_only"), ("layer1_only", "layer1_only"), ("layer2_only", "layer2_only")):
            vals = [p5[seed]["distillation"][cond]["dev_report"][p5key] for seed in SEEDS]
            _bar(ax, p5[SEEDS[0]]["distillation"][cond]["trainable_parameters"], vals, METHOD_COLORS[method], hatch=True)
        ax.axhline(line, color="#16A34A", ls="--", lw=1.2)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        _style(ax, ylabel)
        ax.set_xlabel("Trainable parameters (log scale)", fontsize=9)
        ax.set_ylabel(f"{ylabel} (750 reporting layouts)", fontsize=9)
    handles = [Line2D([], [], marker="o", color=METHOD_COLORS[m], ms=8, lw=1.6, label=f"{METHOD_LABELS[m]}" + ("" if m == "coordinated" else ", low rank"))
               for m in ("coordinated", "layer2_only", "head_only")]
    handles.insert(1, Line2D([], [], marker="o", color=FAST, ms=8, lw=1.6, ls="--", label="All three layers, low rank, learning rate 0.001"))
    handles += [Line2D([], [], marker="o", ls="", color=METHOD_COLORS["full_network"], ms=8, label=METHOD_LABELS["full_network"]),
                Line2D([], [], marker="o", ls="", mfc="white", mec="#64748B", mew=1.6, ms=8, label="Single layer, ordinary fine-tuning (Phase 5; colors as above, blue = layer 1)"),
                Line2D([], [], color="#16A34A", ls="--", label="Repair criterion (0.9)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Parameter efficiency: mean and range over 9 runs (3 for Phase 5 single layers). Learning rate 0.00025 unless marked",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.0)
    fig.tight_layout()
    return _save(fig, "exp2_phase6_parameter_efficiency.png")


def render_single_vs_coordinated():
    """At equal rank: one layer edited against all three edited together."""
    s = _summary()
    crit = s["repair_criterion"]
    ranks = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True, gridspec_kw={"width_ratios": [4, 2.2]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    width = 0.25
    for i, method in enumerate(("head_only", "layer2_only", "coordinated")):
        for x, rank in enumerate(ranks):
            try:
                g = _group(s, method, rank, "primary")
            except StopIteration:
                g = _group(s, method, 4, "primary") if method == "head_only" else None   # the 4 x 128 head has rank 4 at most
            if g is None:
                continue
            xpos = x + (i - 1) * width
            ax.bar(xpos, g["direct_goal"]["mean"], width * 0.92, color=METHOD_COLORS[method], alpha=0.85 if not (method == "head_only" and rank == 8) else 0.4,
                   label=METHOD_LABELS[method] if x == 0 else None)
            vals = [v for seed in SEEDS for v in g["per_seed"][seed]["direct_goal"]]
            ax.scatter(xpos + np.linspace(-0.07, 0.07, len(vals)), vals, color="white", edgecolor="#111827", s=13, lw=0.7, zorder=4)
    ax.axhline(crit["min_direct_goal"], color="#16A34A", ls="--", lw=1.2, label="Repair criterion (0.9)")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"rank {r}" for r in ranks])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Direct goal (750 reporting layouts)", fontsize=9)
    _style(ax, "Low-rank edits at equal rank (bars: mean of 9 runs; dots: runs)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax = axes[1]
    p5 = s["baselines"]["per_seed"]
    for x, cond in enumerate(("head_only", "layer1_only", "layer2_only", "full_network")):
        vals = [p5[seed]["distillation"][cond]["dev_report"]["direct_goal"] for seed in SEEDS]
        ax.bar(x, np.mean(vals), 0.62, color=METHOD_COLORS[cond], alpha=0.85)
        ax.scatter(x + np.linspace(-0.1, 0.1, 3), vals, color="white", edgecolor="#111827", s=13, lw=0.7, zorder=4)
    ax.axhline(crit["min_direct_goal"], color="#16A34A", ls="--", lw=1.2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["head", "layer 1", "layer 2", "all three"], fontsize=8.5)
    _style(ax, "Ordinary fine-tuning, no rank limit (Phase 5)")
    fig.suptitle("Editing one layer against editing all three together (learning rate 0.00025 throughout)",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.0)
    fig.tight_layout()
    return _save(fig, "exp2_phase6_single_vs_coordinated.png")


def render_norm_vs_behavior():
    """Size of the weight change against direct goal, for learned edits and for random edits of the same size."""
    s, runs = _summary(), _runs()
    hack = np.mean([s["baselines"]["per_seed"][seed]["hacking"]["direct_goal"] for seed in SEEDS])
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    fig.patch.set_facecolor("white")
    for r in runs:
        if r["lr_name"] != "primary":
            continue
        sel = r["selected"]
        x = sel["update"]["network"]["relative_norm"]
        ax.scatter(x, sel["dev"]["direct_goal"], color=METHOD_COLORS[r["method"]], marker=MARK[str(r["dqn_seed"])], s=38,
                   edgecolor="white", lw=0.5, alpha=0.9, zorder=4)
        ctrl = sel.get("random_low_rank") or sel["random_full_rank"]
        ax.scatter(x, ctrl["direct_goal"]["mean"], color="#94A3B8", marker="x", s=22, lw=1.1, zorder=3)
    ax.axhline(s["repair_criterion"]["min_direct_goal"], color="#16A34A", ls="--", lw=1.2)
    ax.axhline(hack, color="#B91C1C", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Size of the weight change: ‖Δθ‖ / ‖θ‖ over the whole network (log scale)", fontsize=9)
    ax.set_ylabel("Direct goal (1,500 development layouts)", fontsize=9)
    _style(ax)
    handles = [Line2D([], [], marker="o", ls="", color=METHOD_COLORS[m], ms=7, label=METHOD_LABELS[m] + ("" if m in ("coordinated", "full_network") else ", low rank"))
               for m in ("coordinated", "layer2_only", "head_only", "full_network")]
    handles += [Line2D([], [], marker="x", ls="", color="#94A3B8", ms=6, mew=1.2, label="Random edit of the same size and rank (mean of 5)"),
                Line2D([], [], color="#16A34A", ls="--", label="Repair criterion"), Line2D([], [], color="#B91C1C", ls=":", label="Unedited hacking network")]
    ax.legend(handles=handles, fontsize=8.3, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Size of the edit does not determine behavior: learned and random edits of equal size (learning rate 0.00025)",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.tight_layout()
    return _save(fig, "exp2_phase6_norm_vs_behavior.png")


def render_gif():
    """Seed 42, first development layout (Phase 2 rule) where the hacking network farms: hacking, low-rank repaired, good."""
    import torch

    from rewardlens.exp2.evaluation import categories, greedy_policy, rollout
    from rewardlens.exp2.low_rank import merged_network
    from scripts.exp2_run_phase5 import load_setup, net_at
    s, runs = _summary(), _runs()
    setup = load_setup()
    params, dev, seed = setup["params"], setup["layouts"]["dev"], 42
    # The edit shown is fixed by rule: the smallest reliably successful rank (primary rate first), optimization seed 0.
    lr = next((l for l in ("primary", "secondary") if s["minimum_reliable_rank"].get(l) is not None), "primary")
    rank = s["minimum_reliable_rank"].get(lr) or 8
    run = next(r for r in runs if r["method"] == "coordinated" and r["rank"] == rank and r["lr_name"] == lr and r["opt_seed"] == 0 and r["dqn_seed"] == seed)
    hack = net_at(setup, seed, ("hacking", "final"))
    repaired = merged_network(hack.state_dict(), torch.load(run["edit_file"], weights_only=False)["factors"])
    cat = categories(rollout(greedy_policy(hack), dev, params), params.exploit_visit_threshold)
    idx = next(i for i, l in enumerate(dev) if l.manhattan_to_goal() >= 6
               and abs(l.start[0] - l.reward[0]) + abs(l.start[1] - l.reward[1]) >= 3 and cat[i] == "farm_only")
    layout = dev[idx]
    nets = {"Hacking network": hack, f"Same network after a merged\nrank-{rank} edit of all three layers": repaired,
            "Goal-following network": net_at(setup, seed, ("good", "final"))}
    paths, info = {}, {"seed": seed, "dev_layout_index": idx, "layout": list(layout.as_tuple()), "edit_run": run["run_id"], "rank": rank,
                       "learning_rate": lr, "episodes": {}}
    for name, net in nets.items():
        acts = rollout(greedy_policy(net), [layout], params, record_actions=True)["actions"][0]
        pos, out = list(layout.start), [tuple(layout.start)]
        for a in acts[acts >= 0]:
            dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[a]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < params.width and 0 <= ny < params.height:
                pos = [nx, ny]
            out.append(tuple(pos))
        paths[name] = out
        entries = sum(1 for i in range(1, len(out)) if out[i] == tuple(layout.reward) and out[i - 1] != tuple(layout.reward))
        info["episodes"][name.replace("\n", " ")] = {"steps": len(out) - 1, "reached_goal": out[-1] == tuple(layout.goal), "tile_entries": entries}
    config = {"width": params.width, "height": params.height, "goal_pos": layout.goal, "reward_pos": layout.reward}
    n_frames = min(40, max(len(p) for p in paths.values()))
    frames, durations = [], []
    for t in range(n_frames):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.7), dpi=72)
        fig.patch.set_facecolor(BG_COLOR)
        for ax, (name, path) in zip(axes, paths.items()):
            i = min(t, len(path) - 1)
            ep = info["episodes"][name.replace("\n", " ")]
            if ep["reached_goal"] and i == len(path) - 1:
                status = f"goal in {ep['steps']} steps"
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
        durations.append(2500 if t == n_frames - 1 else 180)
    GIFS.mkdir(parents=True, exist_ok=True)
    frames[0].save(GIFS / "exp2_phase6_low_rank_repair.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
    info["frames_shown"] = n_frames
    json.dump(info, open(RESULTS / "gif_episode.json", "w"), indent=2)
    return info


def main(make_gif: bool = False):
    for fn in (render_repair_vs_rank, render_parameter_efficiency, render_single_vs_coordinated, render_norm_vs_behavior):
        print("wrote", fn())
    if make_gif:
        print(render_gif())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    main(ap.parse_args().gif)
