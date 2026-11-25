"""Experiment 2, Phase 5 figures and GIF.

Usage: python -m scripts.exp2_visualize_phase5 [--gif]
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

RESULTS = Path("results/exp2/phase5")
FIGURES = Path("assets/figures")
GIFS = Path("assets/gifs")
SEEDS = ("42", "123", "456")
MARK = {"42": "o", "123": "s", "456": "^"}
BLUE = "#2563EB"
CAT_COLORS = {"direct_goal": "#16A34A", "farm_then_finish": "#F59E0B", "farm_only": "#EF4444", "other": "#94A3B8"}
CAT_LABELS = {"direct_goal": "Direct goal", "farm_then_finish": "Farm then finish", "farm_only": "Farm only", "other": "Other / failure"}
LAYER_COLORS = {"fc1": "#0EA5E9", "fc2": "#7C3AED", "fc3": "#EA580C"}
LAYER_LABELS = {"fc1": "Layer 1", "fc2": "Layer 2", "fc3": "Output head"}
COND_COLORS = {"head_only": "#EA580C", "layer2_only": "#7C3AED", "layer1_only": "#0EA5E9", "full_network": "#111827"}
COND_LABELS = {"head_only": "Output head only (516)", "layer2_only": "Layer 2 only (16,512)",
               "layer1_only": "Layer 1 only (1,024)", "full_network": "Full network (18,052)"}
SWAP_LABELS = {"head": "Output head", "layer2": "Layer 2", "layer1": "Layer 1", "layer2_head": "Layer 2 + head",
               "layer1_layer2": "Layer 1 + layer 2", "layer1_head": "Layer 1 + head"}


def _load(name):
    return json.load(open(RESULTS / name))


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


def render_weight_drift():
    """Per-layer weight change from the fork start, both branches, against the farming rate."""
    wt = _load("weight_trajectory.json")
    timeline = json.load(open("results/exp2/phase3/timeline.json"))
    x = np.array(wt["steps"]) / 1000
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, seed in zip(axes, SEEDS):
        b = wt["per_seed"][seed]
        tr = b["transition"]
        ax.axvspan(tr["onset"] / 1000, tr["completion"] / 1000, color="#FEE2E2", alpha=0.7, lw=0, zorder=0)
        farming = timeline["per_seed"][seed]["series"]["hacking"]["dev_farming"]
        ax2 = ax.twinx()
        ax2.fill_between(x, 0, farming, color="#94A3B8", alpha=0.22, lw=0, zorder=0)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=8, colors="#64748B")
        for s in ("top",):
            ax2.spines[s].set_visible(False)
        if seed == SEEDS[-1]:
            ax2.set_ylabel("Farming rate (grey area)", fontsize=9, color="#64748B")
        else:
            ax2.set_yticklabels([])
        for layer in wt["layers"]:
            ax.plot(x, b["series"]["hacking"][layer]["relative_change_from_start"], color=LAYER_COLORS[layer], lw=2.2,
                    label=f"{LAYER_LABELS[layer]}, hacking branch")
            ax.plot(x, b["series"]["good"][layer]["relative_change_from_start"], color=LAYER_COLORS[layer], lw=1.5, ls="--",
                    label=f"{LAYER_LABELS[layer]}, good branch")
        _style(ax, f"Seed {seed}")
        ax.set_xlabel("Training steps after the fork (thousands)", fontsize=9)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    axes[0].set_ylabel("Weight change from fork start\n(relative to starting norm)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("All three layers of the hacking branch change, mostly during the behavioral transition (red band: onset to completion)",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.0)
    fig.tight_layout()
    return _save(fig, "exp2_phase5_weight_drift.png")


def render_weight_swaps():
    """Behavior after copying goal-following weights into the final hacking network, layer subset by layer subset."""
    ws, base = _load("weight_swaps.json"), _load("baselines.json")
    pair = "pretrain_to_final"
    random_dg = base["references"]["dev"]["random_policy"]["direct_goal"]["mean"]
    order = ["head", "layer2", "layer1", "layer2_head", "layer1_layer2", "layer1_head"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, seed in zip(axes, SEEDS):
        block = ws["per_seed"][seed]["pairs"][pair]
        rows = [("Hacking network (no edit)", block["recipient"]["behavior"], None)]
        rows += [(SWAP_LABELS[n], block["swaps"][n]["behavior"], block["swaps"][n]["random_control"]) for n in order if n in block["swaps"]]
        rows += [("Pretrained network (all three)", block["donor"]["behavior"], None)]
        y = np.arange(len(rows))[::-1]
        for yi, (name, b, ctrl) in zip(y, rows):
            left = 0.0
            for cat in CAT_COLORS:
                ax.barh(yi, b[cat], left=left, color=CAT_COLORS[cat], height=0.68, label=CAT_LABELS[cat] if yi == y[0] else None)
                left += b[cat]
            eff = b["path_efficiency_direct"]
            ax.text(1.02, yi, "–" if eff is None else f"{eff:.2f}", va="center", fontsize=8.5, color=TEXT_COLOR)
            if ctrl is not None:
                ax.plot(ctrl["direct_goal"]["mean"], yi, marker="D", ms=6, color="white", mec="black", mew=1.3, zorder=5,
                        label="Random edit of the same size: direct goal" if yi == y[1] else None)
        ax.axvline(random_dg, color="#111827", ls=":", lw=1.3, label="Random policy: direct goal")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.text(1.02, y[0] + 0.62, "path\neff.", fontsize=8, color="#64748B", va="bottom")
        _style(ax, f"Seed {seed}")
        ax.grid(False)
        ax.set_xlabel("Share of 1,500 development layouts", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Copying the pretrained network's weights into the hacking network, one layer subset at a time: no subset restores navigation",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.0)
    fig.tight_layout()
    return _save(fig, "exp2_phase5_weight_swaps.png")


def render_finetune_curves():
    """Repair over training: RL fine-tuning (direct goal and farming) and supervised distillation."""
    ft, ds, base = _load("finetune.json"), _load("distill.json"), _load("baselines.json")
    random_dg = base["references"]["dev"]["random_policy"]["direct_goal"]["mean"]
    crit = ft["repair_criterion"]["min_direct_goal"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.2), sharey=True)
    fig.patch.set_facecolor("white")
    rows = [(ft, "direct_goal", "RL fine-tuning: direct goal", 1000, "Environment steps (thousands)"),
            (ft, "farming", "RL fine-tuning: farming", 1000, "Environment steps (thousands)"),
            (ds, "direct_goal", "Supervised distillation: direct goal", 1000, "Gradient updates (thousands)")]
    for r, (res, key, label, div, xlabel) in enumerate(rows):
        for c, seed in enumerate(SEEDS):
            ax = axes[r, c]
            for cond in COND_COLORS:
                curve = res["per_seed"][seed][cond]["curve"]
                ax.plot([e["step"] / div for e in curve], [e[key] for e in curve], color=COND_COLORS[cond], lw=2,
                        marker="o", ms=2.5, label=COND_LABELS[cond])
            if key == "direct_goal":
                ax.axhline(crit, color="#16A34A", ls="--", lw=1.2, label="Repair criterion (0.9)")
                ax.axhline(random_dg, color="#111827", ls=":", lw=1.2, label="Random policy")
            _style(ax, f"{label}, seed {seed}")
            ax.set_ylim(-0.03, 1.05)
            ax.set_xlabel(xlabel, fontsize=8.5)
        axes[r, 0].set_ylabel("Share of development layouts", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Fine-tuning the hacking network toward the corrected task: which layers are allowed to change (trainable parameters in brackets)",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.0)
    fig.tight_layout()
    return _save(fig, "exp2_phase5_finetune_curves.png")


def render_params_vs_repair():
    """Trainable parameters against repaired performance on the 750 reporting layouts."""
    ft, ds = _load("finetune.json"), _load("distill.json")
    crit = ft["repair_criterion"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    rng = {"42": 0.88, "123": 1.0, "456": 1.13}
    for c, (res, title) in enumerate(((ft, "RL fine-tuning, 200k steps"), (ds, "Supervised distillation, 50k updates"))):
        for r, (key, label, line) in enumerate((("direct_goal", "Direct goal", crit["min_direct_goal"]),
                                                ("path_efficiency_direct", "Path efficiency of direct-goal episodes", crit["min_path_efficiency"]))):
            ax = axes[r, c]
            for seed in SEEDS:
                for cond in COND_COLORS:
                    run = res["per_seed"][seed][cond]
                    v = run["selected"]["dev_report"][key]
                    ax.scatter(run["n_trainable_parameters"] * rng[seed], v, marker=MARK[seed], s=75, color=COND_COLORS[cond],
                               edgecolor="white", lw=0.8, zorder=4)
                if res is ds:
                    v = ds["per_seed"][seed]["head_closed_form"]["dev"][key]
                    ax.scatter(516 * rng[seed], v, marker=MARK[seed], s=75, facecolor="none", edgecolor=COND_COLORS["head_only"], lw=1.6, zorder=4)
            ax.axhline(line, color="#16A34A", ls="--", lw=1.2)
            ax.set_xscale("log")
            ax.set_xlim(300, 32000)
            ax.set_ylim(0, 1.05)
            _style(ax, f"{title}: {label.lower()}" if r == 0 else label)
            ax.set_xticks([516, 1024, 16512])
            ax.set_xticklabels(["516\nhead", "1,024\nlayer 1", "16,512 / 18,052\nlayer 2 / full"], fontsize=8)
            ax.minorticks_off()
        axes[1, c].set_xlabel("Trainable parameters", fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COND_COLORS[c], ms=8, label=COND_LABELS[c].split(" (")[0]) for c in COND_COLORS]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=COND_COLORS["head_only"], mew=1.6, ms=8, label="Head, least-squares fit"),
                Line2D([], [], color="#16A34A", ls="--", label="Repair criterion (0.9)")]
    handles += [Line2D([], [], marker=MARK[s], ls="", color="#64748B", ms=8, label=f"Seed {s}") for s in SEEDS]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("More trainable parameters do not explain repair: layer 2 alone (16,512) fails where the full network (18,052) succeeds",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.0)
    fig.tight_layout()
    return _save(fig, "exp2_phase5_params_vs_repair.png")


def _episodes():
    """Seed 42 on the first development layout (Phase 2 rule) where the final hacking network farms."""
    import torch

    from rewardlens.agents.dqn import QNetwork
    from rewardlens.exp2.evaluation import categories, greedy_policy, rollout
    from scripts.exp2_run_phase5 import load_setup, net_at
    setup = load_setup()
    params, dev, seed = setup["params"], setup["layouts"]["dev"], 42

    def saved(folder, cond):
        net = QNetwork()
        net.load_state_dict(torch.load(RESULTS / folder / f"seed{seed}_{cond}.pt", weights_only=False)["q_net_state_dict"])
        net.eval()
        return net

    hack = net_at(setup, seed, ("hacking", "final"))
    cat = categories(rollout(greedy_policy(hack), dev, params), params.exploit_visit_threshold)
    idx = next(i for i, l in enumerate(dev) if l.manhattan_to_goal() >= 6
               and abs(l.start[0] - l.reward[0]) + abs(l.start[1] - l.reward[1]) >= 3 and cat[i] == "farm_only")
    layout = dev[idx]
    nets = {"Hacking network": hack, "RL fine-tuned,\nfull network": saved("finetuned", "full_network"),
            "Distilled,\noutput head only": saved("distilled", "head_only"),
            "Distilled,\nfull network": saved("distilled", "full_network"), "Good network": net_at(setup, seed, ("good", "final"))}
    paths, info = {}, {"seed": seed, "dev_layout_index": idx, "layout": list(layout.as_tuple()), "episodes": {}}
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
    return paths, info, config


def _outcome(ep):
    if ep["reached_goal"]:
        return f"goal in {ep['steps']} steps"
    return f"no goal, {ep['tile_entries']} tile entries"


def _draw_panel(ax, config, name, path, i, status):
    _draw_grid(ax, config, title=f"{name}\n{status}")
    ax.title.set_fontsize(9.5)
    seg = path[:i + 1]
    ax.plot([p[0] for p in seg], [p[1] for p in seg], color=BLUE, lw=2.2, alpha=0.55, zorder=5)
    _draw_agent(ax, *path[i])


def render_trajectories(make_gif: bool = False):
    paths, info, config = _episodes()
    fig, axes = plt.subplots(1, len(paths), figsize=(3.7 * len(paths), 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    for ax, (name, path) in zip(axes, paths.items()):
        _draw_panel(ax, config, name, path, len(path) - 1, _outcome(info["episodes"][name.replace("\n", " ")]))
    fig.suptitle("One development layout, seed 42: the hacking network and four attempts at goal-following",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR, y=1.02)
    _save(fig, "exp2_phase5_trajectories.png")
    if make_gif:
        n_frames = min(70, max(len(p) for p in paths.values()))
        frames, durations = [], []
        for t in range(n_frames):
            fig, axes = plt.subplots(1, len(paths), figsize=(3.3 * len(paths), 4.4), dpi=72)
            fig.patch.set_facecolor(BG_COLOR)
            for ax, (name, path) in zip(axes, paths.items()):
                i = min(t, len(path) - 1)
                ep = info["episodes"][name.replace("\n", " ")]
                done = (ep["reached_goal"] and i == len(path) - 1) or t == n_frames - 1
                _draw_panel(ax, config, name, path, i, _outcome(ep) if done else f"step {i}")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
            plt.close(fig)
            frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
            durations.append(2500 if t == n_frames - 1 else 140)
        GIFS.mkdir(parents=True, exist_ok=True)
        frames[0].save(GIFS / "exp2_phase5_repair_comparison.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
        info["gif_frames_shown"] = n_frames
    json.dump(info, open(RESULTS / "trajectory_episode.json", "w"), indent=2)
    return info


def main(make_gif: bool = False):
    for fn in (render_weight_drift, render_weight_swaps, render_finetune_curves, render_params_vs_repair):
        print("wrote", fn())
    print(render_trajectories(make_gif))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    main(ap.parse_args().gif)
