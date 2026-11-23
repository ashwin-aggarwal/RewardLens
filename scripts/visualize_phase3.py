"""Generate Phase 3 visualizations from collected activation and probing data.

Usage:
    python -m scripts.visualize_phase3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from rewardlens.analysis.activation_extraction import (
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
)
from rewardlens.analysis.representation_analysis import fit_pca_projection
from rewardlens.visualization.gridworld_viz import (
    BG_COLOR, TEXT_COLOR, GOAL_PATH_COLOR, EXPLOIT_PATH_COLOR,
)

FIGURES = Path("assets/figures")
RESULTS = Path("results/phase3")
FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")

FLAWED_COLOR = EXPLOIT_PATH_COLOR   # red
CORRECTED_COLOR = GOAL_PATH_COLOR   # green
NEUTRAL_COLOR = "#6366F1"           # indigo
TRANSITION_CMAP = "viridis"


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(BG_COLOR)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)


# ---------------------------------------------------------------------------
# 1. Activation Overview Diagram
# ---------------------------------------------------------------------------

def render_activation_overview() -> None:
    """Diagram of DQN architecture showing where activations are captured."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2, 6)
    ax.axis("off")

    layers = [
        (0.5, "Input\n(7 features)", 7, "#94A3B8", False),
        (3.0, "Hidden 1\n(128 units)", 10, "#F59E0B", True),
        (5.5, "Hidden 2\n(128 units)", 10, "#F59E0B", True),
        (8.0, "Q-values\n(4 actions)", 4, "#3B82F6", False),
    ]

    for x, label, n_dots, color, captured in layers:
        spacing = min(0.4, 4.0 / n_dots)
        y_start = 2.5 - (n_dots - 1) * spacing / 2
        for i in range(n_dots):
            y = y_start + i * spacing
            circle = plt.Circle((x, y), 0.12, color=color, alpha=0.8)
            ax.add_patch(circle)
            if captured:
                ring = plt.Circle((x, y), 0.17, fill=False,
                                  edgecolor="#EF4444", linewidth=2, alpha=0.6)
                ax.add_patch(ring)
        ax.text(x, -1.0, label, ha="center", va="top", fontsize=9,
                color=TEXT_COLOR, fontweight="bold")
        if captured:
            ax.text(x, -1.8, "⬆ captured", ha="center", va="top", fontsize=7,
                    color="#EF4444", fontstyle="italic")

    # Arrows between layers
    for x1, x2 in [(0.7, 2.8), (3.2, 5.3), (5.7, 7.8)]:
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#64748B", lw=2))

    # ReLU labels
    for x in [2.0, 4.5]:
        ax.text(x, 3.2, "ReLU", ha="center", fontsize=7, color="#64748B",
                fontstyle="italic")

    # Legend
    ax.text(5.0, 5.5,
            "Red rings mark where post-ReLU activations are extracted for probing",
            ha="center", va="center", fontsize=8, color="#EF4444",
            fontstyle="italic")

    ax.set_title("DQN Architecture: Activation Capture Points",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=15)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase3_activation_overview.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Activation Heatmaps
# ---------------------------------------------------------------------------

def render_activation_heatmaps(seed: int = 42) -> None:
    """Heatmaps of hidden-layer activations at representative states."""
    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)

    flawed_ckpt = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    corrected_ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"

    flawed = extract_activations(flawed_ckpt, observations, seed, "flawed", 100000)
    corrected = extract_activations(corrected_ckpt, observations, seed, "corrected", 100000)

    # Pick 6 representative states: near goal, near reward, center, corner, far from both
    state_indices = []
    ax, ay = metadata["agent_x"], metadata["agent_y"]
    dg, dr = metadata["dist_to_goal"], metadata["dist_to_reward"]
    t = metadata["time_remaining"]

    target_states = [
        ("Near goal (t=0.5)", lambda i: dg[i] <= 2 and abs(t[i] - 0.5) < 0.01),
        ("Near reward (t=0.5)", lambda i: dr[i] <= 1 and abs(t[i] - 0.5) < 0.01),
        ("Center (t=0.75)", lambda i: ax[i] == 4 and ay[i] == 4 and abs(t[i] - 0.75) < 0.01),
        ("Start pos (t=1.0)", lambda i: ax[i] == 1 and ay[i] == 6 and abs(t[i] - 1.0) < 0.01),
        ("Corner (t=0.25)", lambda i: ax[i] == 0 and ay[i] == 0 and abs(t[i] - 0.25) < 0.01),
        ("Far from both (t=0.5)", lambda i: dg[i] >= 8 and dr[i] >= 5 and abs(t[i] - 0.5) < 0.01),
    ]

    selected = []
    for label, cond in target_states:
        for i in range(len(observations)):
            if cond(i):
                selected.append((i, label))
                break
        else:
            # Fallback: just pick a state
            for i in range(len(observations)):
                if i not in [s[0] for s in selected]:
                    selected.append((i, label + " (approx)"))
                    break

    fig, axes = plt.subplots(len(selected), 4, figsize=(14, 2.2 * len(selected)), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    from rewardlens.envs.reward_grid import ACTION_NAMES
    for row, (idx, label) in enumerate(selected):
        # Flawed hidden2
        ax = axes[row, 0]
        ax.imshow(flawed.hidden2[idx:idx+1, :64], aspect="auto", cmap="YlOrRd",
                  interpolation="nearest")
        if row == 0:
            ax.set_title("Flawed DQN\nHidden 2 (units 0–63)", fontsize=8,
                         fontweight="bold", color=TEXT_COLOR)
        ax.set_ylabel(label, fontsize=7, color=TEXT_COLOR, rotation=0,
                      labelpad=100, va="center")
        ax.set_yticks([])
        ax.set_xticks([])

        # Corrected hidden2
        ax = axes[row, 1]
        ax.imshow(corrected.hidden2[idx:idx+1, :64], aspect="auto", cmap="YlOrRd",
                  interpolation="nearest")
        if row == 0:
            ax.set_title("Corrected DQN\nHidden 2 (units 0–63)", fontsize=8,
                         fontweight="bold", color=TEXT_COLOR)
        ax.set_yticks([])
        ax.set_xticks([])

        # Flawed Q-values
        ax = axes[row, 2]
        ax.set_facecolor(BG_COLOR)
        q_f = flawed.q_values[idx]
        bars = ax.bar(range(4), q_f, color=[FLAWED_COLOR if i == q_f.argmax() else "#94A3B8"
                                             for i in range(4)])
        if row == 0:
            ax.set_title("Flawed Q-values", fontsize=8, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(range(4))
        ax.set_xticklabels([ACTION_NAMES[i] for i in range(4)], fontsize=6)
        ax.tick_params(labelsize=6, colors=TEXT_COLOR)

        # Corrected Q-values
        ax = axes[row, 3]
        ax.set_facecolor(BG_COLOR)
        q_c = corrected.q_values[idx]
        bars = ax.bar(range(4), q_c, color=[CORRECTED_COLOR if i == q_c.argmax() else "#94A3B8"
                                             for i in range(4)])
        if row == 0:
            ax.set_title("Corrected Q-values", fontsize=8, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(range(4))
        ax.set_xticklabels([ACTION_NAMES[i] for i in range(4)], fontsize=6)
        ax.tick_params(labelsize=6, colors=TEXT_COLOR)

    fig.suptitle(
        f"Hidden Activations at Matched States (seed {seed})\n"
        "Same observations, independently trained networks — neuron indices are NOT aligned",
        fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIGURES / "phase3_activation_heatmaps.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Probe Performance Comparison
# ---------------------------------------------------------------------------

def render_probe_comparison() -> None:
    """Bar chart comparing probe AUROC across feature sets and seeds."""
    with open(RESULTS / "probe_results.json") as f:
        data = json.load(f)

    feature_order = [
        "position_time", "observation_raw", "distances",
        "combined_observable", "q_values",
        "hidden1", "hidden2", "hidden1+hidden2",
    ]
    display_names = {
        "position_time": "Position +\nTime",
        "observation_raw": "Raw\nObservation",
        "distances": "Distances\n(goal/reward)",
        "combined_observable": "All\nObservable",
        "q_values": "Q-values",
        "hidden1": "Hidden\nLayer 1",
        "hidden2": "Hidden\nLayer 2",
        "hidden1+hidden2": "Hidden\n1 + 2",
    }
    colors = {
        "position_time": "#94A3B8", "observation_raw": "#94A3B8",
        "distances": "#94A3B8", "combined_observable": "#64748B",
        "q_values": "#3B82F6",
        "hidden1": "#F59E0B", "hidden2": "#F59E0B", "hidden1+hidden2": "#EF4444",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    for metric_idx, (metric, title) in enumerate([("auroc", "AUROC"), ("balanced_accuracy", "Balanced Accuracy")]):
        ax = axes[metric_idx]
        _style_ax(ax, title)

        x = np.arange(len(feature_order))
        width = 0.25
        seeds = ["42", "123", "456"]
        offsets = [-width, 0, width]

        for s_idx, seed_str in enumerate(seeds):
            if not data[seed_str]["viable"]:
                continue
            vals = []
            for feat in feature_order:
                if feat in data[seed_str]["probes"]:
                    v = data[seed_str]["probes"][feat][metric]
                    vals.append(v if v is not None else 0)
                else:
                    vals.append(0)
            bar_colors = [colors.get(f, "#94A3B8") for f in feature_order]
            alpha = [0.5, 0.75, 1.0][s_idx]
            bars = ax.bar(x + offsets[s_idx], vals, width, alpha=alpha,
                         color=bar_colors, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([display_names.get(f, f) for f in feature_order],
                           fontsize=7, ha="center")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="#CBD5E1", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        if metric_idx == 0:
            ax.set_ylabel(title, fontsize=9, color=TEXT_COLOR)

        # Seed legend
        for s_idx, (seed_str, alpha) in enumerate(zip(seeds, [0.5, 0.75, 1.0])):
            ax.bar([], [], color="#94A3B8", alpha=alpha, label=f"Seed {seed_str}")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Linear Probe Performance: Hidden Activations vs Observable Baselines\n"
        "Target: states where flawed-reward DQN distinctively farms reward "
        "(moves toward reward, not toward goal, corrected DQN disagrees)",
        fontsize=10, fontweight="bold", color=TEXT_COLOR, y=1.06,
    )

    # Annotation
    fig.text(0.5, -0.04,
             "Observable baselines (gray) show what can be predicted from raw state alone. "
             "Hidden activations (orange/red) perform comparably or better,\n"
             "but activations are deterministic functions of the input — improved "
             "probe accuracy shows the network has learned useful features, not new information.",
             ha="center", fontsize=7, color="#64748B", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(FIGURES / "phase3_probe_comparison.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Representation Projection (PCA)
# ---------------------------------------------------------------------------

def render_representation_projection(seed: int = 42) -> None:
    """PCA of hidden-layer activations colored by behavioral class."""
    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)

    flawed_ckpt = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    corrected_ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"

    flawed = extract_activations(flawed_ckpt, observations, seed, "flawed", 100000)
    corrected = extract_activations(corrected_ckpt, observations, seed, "corrected", 100000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    for ax_idx, (rec, name, base_color) in enumerate([
        (flawed, "Flawed-Reward DQN", FLAWED_COLOR),
        (corrected, "Corrected-Reward DQN", CORRECTED_COLOR),
    ]):
        ax = axes[ax_idx]
        _style_ax(ax, f"{name} — Hidden Layer 2 (PCA)")

        projected, pca = fit_pca_projection(rec.hidden2, n_components=2, seed=42)

        # Color by distance to reward tile (continuous)
        c = metadata["dist_to_reward"]
        sc = ax.scatter(projected[:, 0], projected[:, 1],
                       c=c, cmap="coolwarm", s=12, alpha=0.6, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Distance to reward tile", shrink=0.8)

        # Mark action disagreements
        agree = flawed.greedy_actions == corrected.greedy_actions
        disagree_mask = ~agree
        ax.scatter(projected[disagree_mask, 0], projected[disagree_mask, 1],
                  facecolors="none", edgecolors="black", s=30, linewidths=0.8,
                  alpha=0.4, label="Policy disagreement")

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Hidden Representations in PCA Space (seed {seed})\n"
        "Each point is one of 320 matched observations — same states, different networks.\n"
        "PCA fitted independently per model; axes are NOT aligned across panels.",
        fontsize=10, fontweight="bold", color=TEXT_COLOR, y=1.08,
    )
    plt.tight_layout()
    fig.savefig(FIGURES / "phase3_representation_projection.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Training-Transition Evolution
# ---------------------------------------------------------------------------

def render_activation_evolution() -> None:
    """Multi-panel figure showing representation + policy changes across training."""
    with open(RESULTS / "transition_analysis.json") as f:
        data = json.load(f)

    fig = plt.figure(figsize=(14, 12), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.35)

    seed_colors = {"42": "#3B82F6", "123": "#EF4444", "456": "#22C55E"}

    # Panel 1: Policy behavior (toward-goal vs toward-reward fractions)
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax, "Policy Direction: Goal vs Reward-Seeking")
    for seed_str, color in seed_colors.items():
        s = data[seed_str]
        steps = [pb["step"] for pb in s["policy_behavior"]]
        goal_frac = [pb["toward_goal"] for pb in s["policy_behavior"]]
        reward_frac = [pb["toward_reward"] for pb in s["policy_behavior"]]
        ax.plot(steps, goal_frac, "-o", color=color, markersize=3,
                label=f"Seed {seed_str} → goal", alpha=0.8)
        ax.plot(steps, reward_frac, "--s", color=color, markersize=3,
                label=f"Seed {seed_str} → reward", alpha=0.5)
    ax.set_xlabel("Training Step", fontsize=8)
    ax.set_ylabel("Fraction of Greedy Actions", fontsize=8)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: CKA vs first checkpoint
    ax = fig.add_subplot(gs[0, 1])
    _style_ax(ax, "CKA vs First Checkpoint (Representation Stability)")
    for seed_str, color in seed_colors.items():
        evo = data[seed_str]["evolution"]
        ax.plot(evo["steps"], evo["cka_h1_vs_first"], "-o", color=color,
                markersize=3, label=f"Seed {seed_str} H1", alpha=0.8)
        ax.plot(evo["steps"], evo["cka_h2_vs_first"], "--s", color=color,
                markersize=3, label=f"Seed {seed_str} H2", alpha=0.5)
    ax.set_xlabel("Training Step", fontsize=8)
    ax.set_ylabel("Linear CKA", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Layer 2 sparsity evolution
    ax = fig.add_subplot(gs[1, 0])
    _style_ax(ax, "Hidden Layer 2: Sparsity Over Training")
    for seed_str, color in seed_colors.items():
        evo = data[seed_str]["evolution"]
        ax.plot(evo["steps"], evo["sparsity_h2"], "-o", color=color,
                markersize=3, label=f"Seed {seed_str}")
    ax.set_xlabel("Training Step", fontsize=8)
    ax.set_ylabel("Sparsity (fraction of zero activations)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: Within-run probe AUROC
    ax = fig.add_subplot(gs[1, 1])
    _style_ax(ax, "Within-Run Probe AUROC (H2 → Farming Label)")
    for seed_str, color in seed_colors.items():
        probes = data[seed_str]["within_run_probes"]
        steps_v, aurocs = [], []
        for step_str, p in probes.items():
            if p["viable"] and p["auroc"] is not None:
                steps_v.append(int(step_str))
                aurocs.append(p["auroc"])
        if steps_v:
            ax.plot(steps_v, aurocs, "-o", color=color, markersize=4,
                    label=f"Seed {seed_str}")
    ax.set_xlabel("Training Step", fontsize=8)
    ax.set_ylabel("AUROC", fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.5, color="#CBD5E1", linestyle="--", linewidth=1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 5: CKA vs corrected-reward final checkpoint
    ax = fig.add_subplot(gs[2, 0])
    _style_ax(ax, "CKA vs Corrected-Reward Final (Cross-Condition)")
    for seed_str, color in seed_colors.items():
        cross = data[seed_str]["cross_condition_cka"]
        steps = [c["step"] for c in cross]
        cka_h2 = [c["cka_h2_vs_corrected"] for c in cross]
        ax.plot(steps, cka_h2, "-o", color=color, markersize=3,
                label=f"Seed {seed_str} H2")
    ax.set_xlabel("Training Step", fontsize=8)
    ax.set_ylabel("Linear CKA (H2 vs corrected final)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, -0.2,
            "Independently trained networks — CKA measures representational geometry similarity,\n"
            "not aligned semantic content. Changes here may reflect training dynamics, not meaning.",
            transform=ax.transAxes, ha="center", fontsize=7, color="#64748B", fontstyle="italic")

    # Panel 6: L2 norm evolution
    ax = fig.add_subplot(gs[2, 1])
    _style_ax(ax, "Hidden Layer 2: Mean Activation Magnitude")
    for seed_str, color in seed_colors.items():
        evo = data[seed_str]["evolution"]
        ax.plot(evo["steps"], evo["l2_norm_h2"], "-o", color=color,
                markersize=3, label=f"Seed {seed_str}")
    ax.set_xlabel("Training Step", fontsize=8)
    ax.set_ylabel("Mean L2 Norm", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Training-Transition Analysis: How Representations Change\n"
        "as the DQN Shifts from Goal-Seeking to Reward Farming",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    fig.savefig(FIGURES / "phase3_activation_evolution.png",
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating Phase 3 visualizations...")

    print("  1. Activation overview diagram...")
    render_activation_overview()
    print(f"     ✓ {FIGURES / 'phase3_activation_overview.png'}")

    print("  2. Activation heatmaps...")
    render_activation_heatmaps(seed=42)
    print(f"     ✓ {FIGURES / 'phase3_activation_heatmaps.png'}")

    print("  3. Probe performance comparison...")
    render_probe_comparison()
    print(f"     ✓ {FIGURES / 'phase3_probe_comparison.png'}")

    print("  4. Representation projection (PCA)...")
    render_representation_projection(seed=42)
    print(f"     ✓ {FIGURES / 'phase3_representation_projection.png'}")

    print("  5. Training-transition evolution...")
    render_activation_evolution()
    print(f"     ✓ {FIGURES / 'phase3_activation_evolution.png'}")

    print("Done.")


if __name__ == "__main__":
    main()
