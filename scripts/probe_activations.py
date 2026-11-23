"""Experiment B: Linear probing of hidden activations for reward-farming behavior.

Behavioral target definition:
  For each observation fed to the flawed-reward DQN, the label is 1 (reward-farming)
  if the greedy action moves the agent toward the reward tile AND away from the goal
  (or stays equidistant to the goal), AND the corrected-reward DQN disagrees (selects
  a different action at the same state). This isolates states where the flawed-reward
  network makes a distinctively exploitative choice — not just any action difference.

  States where both models agree, or where the flawed-reward agent moves toward the
  goal, are labeled 0 (non-farming).

  This avoids the tautological classifier problem: we do NOT use model identity as
  the label. The label depends on what the flawed-reward model *does* at each state,
  evaluated against spatial criteria the probe must learn from activations alone.

Usage:
    python -m scripts.probe_activations
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rewardlens.analysis.activation_extraction import (
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
)
from rewardlens.analysis.probing import (
    fit_probe,
    build_observable_baselines,
    spatial_train_test_split,
    check_label_quality,
)
from rewardlens.envs.reward_grid import ACTION_DELTAS


FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")
OUTPUT_DIR = Path("results/phase3")


def compute_farming_labels(
    flawed_actions: np.ndarray,
    corrected_actions: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> np.ndarray:
    """Label states where the flawed-reward DQN makes a distinctively farming choice.

    A state is labeled 1 (farming) when ALL of:
      1. The two models disagree on greedy action.
      2. The flawed-reward agent's action moves closer to the reward tile.
      3. The flawed-reward agent's action does NOT move closer to the goal.

    This separates "actively farms reward" from "happens to pick a different action"
    and from "moves toward both reward and goal simultaneously."
    """
    n = len(flawed_actions)
    labels = np.zeros(n, dtype=np.int64)

    ax = metadata["agent_x"]
    ay = metadata["agent_y"]
    gx = metadata["goal_x"]
    gy = metadata["goal_y"]
    rx = metadata["reward_x"]
    ry = metadata["reward_y"]

    for i in range(n):
        if flawed_actions[i] == corrected_actions[i]:
            continue

        action = int(flawed_actions[i])
        dx, dy = ACTION_DELTAS[action]
        nx, ny = ax[i] + dx, ay[i] + dy

        # Clamp to grid
        nx = max(0, min(7, nx))
        ny = max(0, min(7, ny))

        old_dist_reward = abs(ax[i] - rx[i]) + abs(ay[i] - ry[i])
        new_dist_reward = abs(nx - rx[i]) + abs(ny - ry[i])
        old_dist_goal = abs(ax[i] - gx[i]) + abs(ay[i] - gy[i])
        new_dist_goal = abs(nx - gx[i]) + abs(ny - gy[i])

        moves_toward_reward = new_dist_reward < old_dist_reward
        moves_toward_goal = new_dist_goal < old_dist_goal

        if moves_toward_reward and not moves_toward_goal:
            labels[i] = 1

    return labels


def run_probes(
    seed: int = 42,
    probe_seed: int = 42,
) -> dict:
    """Run the full probing experiment for one training seed."""
    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)

    flawed_ckpt = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    corrected_ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"

    flawed_rec = extract_activations(flawed_ckpt, observations, seed, "flawed", 100000)
    corrected_rec = extract_activations(corrected_ckpt, observations, seed, "corrected", 100000)

    labels = compute_farming_labels(
        flawed_rec.greedy_actions, corrected_rec.greedy_actions, metadata,
    )

    label_info = check_label_quality(labels)
    print(f"  Label quality (seed {seed}): {label_info}")

    if not label_info["viable"]:
        print(f"  WARNING: Labels not viable for probing (seed {seed})")
        return {"seed": seed, "label_info": label_info, "viable": False, "probes": {}}

    # Spatial train/test split on the flawed-reward observations
    obs_train, y_train, obs_test, y_test = spatial_train_test_split(
        observations, labels, seed=probe_seed,
    )

    # We also need split indices to slice activations consistently
    ax = (observations[:, 0] * 8).round().astype(int)
    ay = (observations[:, 1] * 8).round().astype(int)
    rng = np.random.RandomState(probe_seed)
    positions = sorted(set(zip(ax.tolist(), ay.tolist())))
    rng.shuffle(positions)  # type: ignore[arg-type]
    n_test = max(1, int(len(positions) * 0.25))
    test_positions = set(positions[:n_test])
    test_mask = np.array([
        (int(ax[i]), int(ay[i])) in test_positions for i in range(len(ax))
    ])
    train_mask = ~test_mask

    # Feature sets to probe
    baselines = build_observable_baselines(metadata)
    feature_sets = {
        "hidden1": flawed_rec.hidden1,
        "hidden2": flawed_rec.hidden2,
        "hidden1+hidden2": np.concatenate([flawed_rec.hidden1, flawed_rec.hidden2], axis=1),
        "q_values": flawed_rec.q_values,
        "position_time": baselines["position_time"],
        "distances": baselines["distances"],
        "combined_observable": baselines["combined_observable"],
        "observation_raw": observations,
    }

    results = {}
    for name, features in feature_sets.items():
        X_tr = features[train_mask]
        X_te = features[test_mask]
        y_tr = labels[train_mask]
        y_te = labels[test_mask]

        # Check both splits have both classes
        train_info = check_label_quality(y_tr)
        test_info = check_label_quality(y_te)
        if not train_info["viable"] or not test_info["viable"]:
            print(f"    {name}: SKIP — insufficient class variety in split")
            continue

        result = fit_probe(X_tr, y_tr, X_te, y_te, name=name, seed=probe_seed)
        results[name] = {
            "auroc": result.auroc,
            "balanced_accuracy": result.balanced_accuracy,
            "confusion_matrix": result.confusion.tolist(),
            "class_counts": result.class_counts,
            "n_train": result.n_train,
            "n_test": result.n_test,
            "train_prevalence": result.train_prevalence,
            "test_prevalence": result.test_prevalence,
        }
        auroc_str = f"{result.auroc:.3f}" if result.auroc else "N/A"
        print(f"    {name:25s}  AUROC={auroc_str}  BalAcc={result.balanced_accuracy:.3f}")

    return {
        "seed": seed,
        "label_info": label_info,
        "viable": True,
        "n_disagreements": int((flawed_rec.greedy_actions != corrected_rec.greedy_actions).sum()),
        "n_total": len(observations),
        "probes": results,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for seed in [42, 123, 456]:
        print(f"\n=== Probing seed {seed} ===")
        result = run_probes(seed=seed)
        all_results[str(seed)] = result

    with open(OUTPUT_DIR / "probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'probe_results.json'}")

    # Summary table
    print("\n=== Probe Summary ===")
    print(f"{'Seed':>6} {'Feature Set':>25} {'AUROC':>8} {'BalAcc':>8}")
    print("-" * 55)
    for seed_str, res in all_results.items():
        if not res["viable"]:
            print(f"{seed_str:>6} Labels not viable")
            continue
        for name, probe in res["probes"].items():
            auroc = f"{probe['auroc']:.3f}" if probe["auroc"] else "N/A"
            print(f"{seed_str:>6} {name:>25} {auroc:>8} {probe['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
