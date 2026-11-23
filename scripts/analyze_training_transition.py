"""Experiment C: Analyze representational changes during the goal-to-exploit transition.

For each flawed-reward training seed, load checkpoints sequentially, extract
activations on matched states, and track how representations and policy change.

Usage:
    python -m scripts.analyze_training_transition
    python -m scripts.analyze_training_transition --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rewardlens.analysis.activation_extraction import (
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
    verify_extraction_preserves_predictions,
)
from rewardlens.analysis.representation_analysis import (
    activation_statistics,
    extract_checkpoint_sequence,
    track_representation_evolution,
    centered_kernel_alignment,
    fit_pca_projection,
    project_with_fitted_pca,
)
from rewardlens.analysis.probing import (
    fit_probe,
    spatial_train_test_split,
    check_label_quality,
)
from rewardlens.envs.reward_grid import ACTION_DELTAS


FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")
OUTPUT_DIR = Path("results/phase3")


def classify_greedy_actions(
    greedy_actions: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> dict[str, float]:
    """Classify what fraction of greedy actions move toward goal vs reward tile."""
    n = len(greedy_actions)
    toward_goal = 0
    toward_reward = 0
    toward_both = 0
    toward_neither = 0

    ax, ay = metadata["agent_x"], metadata["agent_y"]
    gx, gy = metadata["goal_x"], metadata["goal_y"]
    rx, ry = metadata["reward_x"], metadata["reward_y"]

    for i in range(n):
        dx, dy = ACTION_DELTAS[int(greedy_actions[i])]
        nx, ny = max(0, min(7, ax[i] + dx)), max(0, min(7, ay[i] + dy))

        closer_goal = (abs(nx - gx[i]) + abs(ny - gy[i])) < (abs(ax[i] - gx[i]) + abs(ay[i] - gy[i]))
        closer_reward = (abs(nx - rx[i]) + abs(ny - ry[i])) < (abs(ax[i] - rx[i]) + abs(ay[i] - ry[i]))

        if closer_goal and closer_reward:
            toward_both += 1
        elif closer_goal:
            toward_goal += 1
        elif closer_reward:
            toward_reward += 1
        else:
            toward_neither += 1

    return {
        "toward_goal": toward_goal / n,
        "toward_reward": toward_reward / n,
        "toward_both": toward_both / n,
        "toward_neither": toward_neither / n,
    }


def analyze_seed(seed: int, observations: np.ndarray, metadata: dict) -> dict:
    """Full transition analysis for one training seed."""
    ckpt_dir = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints"
    if not ckpt_dir.exists():
        return {"seed": seed, "error": f"Checkpoint dir not found: {ckpt_dir}"}

    print(f"  Loading checkpoint sequence...")
    records = extract_checkpoint_sequence(ckpt_dir, observations, seed, "flawed")
    print(f"  {len(records)} checkpoints loaded")

    # Verify one extraction
    verify_extraction_preserves_predictions(
        records[0].checkpoint_path, observations,
    )

    # Track representation evolution
    print(f"  Computing representation evolution...")
    evolution = track_representation_evolution(records)

    # Policy behavior at each checkpoint
    print(f"  Classifying policy behavior per checkpoint...")
    policy_behavior = []
    for rec in records:
        behavior = classify_greedy_actions(rec.greedy_actions, metadata)
        behavior["step"] = rec.step
        policy_behavior.append(behavior)

    # Activation statistics per checkpoint
    all_stats = []
    for rec in records:
        stats = activation_statistics(rec)
        stats["step"] = rec.step
        all_stats.append(stats)

    # CKA between consecutive checkpoints
    consecutive_cka = []
    for i in range(1, len(records)):
        cka_h1 = centered_kernel_alignment(records[i-1].hidden1, records[i].hidden1)
        cka_h2 = centered_kernel_alignment(records[i-1].hidden2, records[i].hidden2)
        consecutive_cka.append({
            "step_from": records[i-1].step,
            "step_to": records[i].step,
            "cka_h1": cka_h1,
            "cka_h2": cka_h2,
        })

    # Also load the corrected-reward final checkpoint for reference
    corrected_ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    corrected_rec = extract_activations(corrected_ckpt, observations, seed, "corrected", 100000)

    # CKA between each flawed checkpoint and the corrected final
    cross_cka = []
    for rec in records:
        cka_h1 = centered_kernel_alignment(rec.hidden1, corrected_rec.hidden1)
        cka_h2 = centered_kernel_alignment(rec.hidden2, corrected_rec.hidden2)
        cross_cka.append({
            "step": rec.step,
            "cka_h1_vs_corrected": cka_h1,
            "cka_h2_vs_corrected": cka_h2,
        })

    # PCA on the full checkpoint sequence (layer 2) for visualization
    print(f"  Computing PCA projections...")
    all_h2 = np.concatenate([r.hidden2 for r in records], axis=0)
    projected_all, pca = fit_pca_projection(all_h2, n_components=2, seed=42)

    pca_per_checkpoint = []
    offset = 0
    for rec in records:
        n = len(rec.hidden2)
        proj = projected_all[offset:offset + n]
        offset += n
        pca_per_checkpoint.append({
            "step": rec.step,
            "centroid": proj.mean(axis=0).tolist(),
            "std": proj.std(axis=0).tolist(),
        })

    # Within-run probing: use the transition to define labels
    # Label = 1 for states where the checkpoint's greedy action moves toward
    # reward and away from goal (same logic as Experiment B but within-run)
    print(f"  Running within-run probes at key checkpoints...")
    probe_results = {}
    for rec in records:
        step = rec.step
        # Compute farming-like labels for this checkpoint
        labels = np.zeros(len(observations), dtype=np.int64)
        ax, ay = metadata["agent_x"], metadata["agent_y"]
        gx, gy = metadata["goal_x"], metadata["goal_y"]
        rx, ry = metadata["reward_x"], metadata["reward_y"]

        for i in range(len(observations)):
            action = int(rec.greedy_actions[i])
            dx, dy = ACTION_DELTAS[action]
            nx = max(0, min(7, ax[i] + dx))
            ny = max(0, min(7, ay[i] + dy))
            closer_reward = (abs(nx - rx[i]) + abs(ny - ry[i])) < (abs(ax[i] - rx[i]) + abs(ay[i] - ry[i]))
            closer_goal = (abs(nx - gx[i]) + abs(ny - gy[i])) < (abs(ax[i] - gx[i]) + abs(ay[i] - gy[i]))
            if closer_reward and not closer_goal:
                labels[i] = 1

        info = check_label_quality(labels)
        if not info["viable"]:
            probe_results[step] = {"viable": False, "label_info": info}
            continue

        X_tr, y_tr, X_te, y_te = spatial_train_test_split(
            rec.hidden2, labels, seed=42, observations=observations,
        )
        tr_info = check_label_quality(y_tr)
        te_info = check_label_quality(y_te)
        if not tr_info["viable"] or not te_info["viable"]:
            probe_results[step] = {"viable": False, "label_info": info}
            continue

        result = fit_probe(X_tr, y_tr, X_te, y_te, name=f"h2_step{step}")
        probe_results[step] = {
            "viable": True,
            "auroc": result.auroc,
            "balanced_accuracy": result.balanced_accuracy,
            "label_positive_fraction": info["positive_fraction"],
        }

    # Save checkpoint-level activations for visualization
    np.savez_compressed(
        OUTPUT_DIR / f"transition_pca_seed{seed}.npz",
        pca_projections=projected_all,
        explained_variance_ratio=pca.explained_variance_ratio_,
        checkpoint_sizes=[len(r.hidden2) for r in records],
        checkpoint_steps=[r.step for r in records],
    )

    return {
        "seed": seed,
        "n_checkpoints": len(records),
        "checkpoint_steps": [r.step for r in records],
        "evolution": evolution,
        "policy_behavior": policy_behavior,
        "activation_stats": all_stats,
        "consecutive_cka": consecutive_cka,
        "cross_condition_cka": cross_cka,
        "pca_per_checkpoint": pca_per_checkpoint,
        "within_run_probes": {str(k): v for k, v in probe_results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)

    seeds = [args.seed] if args.seed else [42, 123, 456]
    all_results = {}

    for seed in seeds:
        print(f"\n=== Transition analysis: seed {seed} ===")
        result = analyze_seed(seed, observations, metadata)
        all_results[str(seed)] = result

    with open(OUTPUT_DIR / "transition_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'transition_analysis.json'}")


if __name__ == "__main__":
    main()
