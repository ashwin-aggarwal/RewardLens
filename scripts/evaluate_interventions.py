"""Phase 4 Experiment 4: interventions before vs after the goal-to-exploit transition.

Uses seed 123's checkpoints at 20k steps (goal-preferring) and 30k steps
(farming), confirmed from behavior, not assumed from the documented timeline.
The farming direction is fitted at the 30k checkpoint in its own activation
space using the within-run behavioral label (greedy action moves toward the
reward tile and not toward the goal). No within-checkpoint direction is
fittable at 20k because that checkpoint produces no farming-labeled actions
(single-class); applying the 30k direction to the 20k checkpoint is therefore
a cross-checkpoint transfer within one training run, flagged as a caveat.

Usage:
    python -m scripts.evaluate_interventions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_extraction import (
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
)
from rewardlens.analysis.activation_intervention import (
    evaluate_on_start_states,
    fit_probe_direction,
    partition_start_positions,
    random_unit_directions,
)
from rewardlens.envs.reward_grid import ACTION_DELTAS, RewardGridConfig

FLAWED_DIR = Path("results/phase2")
OUTPUT_DIR = Path("results/phase4")
CONFIG_PATH = Path("configs/phase4_interventions.json")


def within_run_farming_labels(greedy_actions: np.ndarray, metadata: dict) -> np.ndarray:
    """Label = 1 when the greedy action moves toward the reward tile and not toward the goal."""
    n = len(greedy_actions)
    labels = np.zeros(n, dtype=np.int64)
    ax, ay = metadata["agent_x"], metadata["agent_y"]
    gx, gy = metadata["goal_x"], metadata["goal_y"]
    rx, ry = metadata["reward_x"], metadata["reward_y"]
    for i in range(n):
        dx, dy = ACTION_DELTAS[int(greedy_actions[i])]
        nx, ny = max(0, min(7, ax[i] + dx)), max(0, min(7, ay[i] + dy))
        closer_reward = (abs(nx - rx[i]) + abs(ny - ry[i])) < (abs(ax[i] - rx[i]) + abs(ay[i] - ry[i]))
        closer_goal = (abs(nx - gx[i]) + abs(ny - gy[i])) < (abs(ax[i] - gx[i]) + abs(ay[i] - gy[i]))
        if closer_reward and not closer_goal:
            labels[i] = 1
    return labels


def _strip(result: dict) -> dict:
    out = dict(result)
    out["episodes"] = [{k: v for k, v in e.items() if k != "actions"} for e in result["episodes"]]
    return out


def evaluate_checkpoint(net, parts, direction, rel: float, typical_norm: float,
                        randoms, label: str) -> dict:
    base_cfg = RewardGridConfig()
    alpha = rel * typical_norm
    out = {"relative_strength": rel, "alpha": alpha, "typical_norm": typical_norm}
    out["baseline"] = _strip(evaluate_on_start_states(net, base_cfg, parts["test"], None, 0.0, "none"))
    out["probe_subtract"] = _strip(evaluate_on_start_states(
        net, base_cfg, parts["test"], direction, alpha, "subtract"))
    out["probe_add"] = _strip(evaluate_on_start_states(
        net, base_cfg, parts["test"], direction, alpha, "add"))
    rand = []
    for ri in range(len(randoms)):
        r = evaluate_on_start_states(net, base_cfg, parts["test"], randoms[ri], alpha, "subtract")
        rand.append({k: r[k] for k in ("goal_rate", "hacking_rate", "mean_proxy_reward",
                                        "mean_episode_steps", "mean_no_move_steps")})
    out["random_subtract"] = rand

    # Targeted-subtraction strength sweep, scaled to this checkpoint's own typical norm
    out["subtract_sweep"] = {}
    for rel_s in (0.05, 0.1, 0.25, 0.5, 1.0):
        s = evaluate_on_start_states(
            net, base_cfg, parts["test"], direction, rel_s * typical_norm, "subtract")
        out["subtract_sweep"][str(rel_s)] = {
            k: s[k] for k in ("goal_rate", "hacking_rate", "mean_proxy_reward",
                              "mean_episode_steps", "mean_no_move_steps")}

    b, s = out["baseline"], out["probe_subtract"]
    rg = np.mean([r["goal_rate"] for r in rand])
    rh = np.mean([r["hacking_rate"] for r in rand])
    print(f"  [{label}] baseline: goal={b['goal_rate']:.2f} hack={b['hacking_rate']:.2f}")
    print(f"  [{label}] subtract: goal={s['goal_rate']:.2f} hack={s['hacking_rate']:.2f}"
          f" | add: goal={out['probe_add']['goal_rate']:.2f} hack={out['probe_add']['hacking_rate']:.2f}"
          f" | random: goal={rg:.2f} hack={rh:.2f}")
    sweep_str = " ".join(
        f"{k}:g={v['goal_rate']:.2f},h={v['hacking_rate']:.2f}"
        for k, v in out["subtract_sweep"].items())
    print(f"  [{label}] subtract sweep: {sweep_str}")
    return out


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    tc = cfg["transition_checkpoints"]
    seed = tc["seed"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = partition_start_positions(
        RewardGridConfig(), n_dev=cfg["n_dev_positions"],
        n_test=cfg["n_test_positions"], seed=cfg["partition_seed"])

    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)
    randoms = random_unit_directions(128, cfg["n_random_directions"], cfg["random_direction_seed"])

    before_ckpt = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints" / f"checkpoint_{tc['before_step']}.pt"
    after_ckpt = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints" / f"checkpoint_{tc['after_step']}.pt"
    for p in (before_ckpt, after_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Required checkpoint missing: {p}")

    net_before, _ = load_q_network_for_eval(before_ckpt)
    net_after, _ = load_q_network_for_eval(after_ckpt)

    # Behavioral confirmation + within-run labels at each checkpoint
    rec_before = extract_activations(before_ckpt, observations, seed, "flawed", tc["before_step"])
    rec_after = extract_activations(after_ckpt, observations, seed, "flawed", tc["after_step"])
    labels_before = within_run_farming_labels(rec_before.greedy_actions, metadata)
    labels_after = within_run_farming_labels(rec_after.greedy_actions, metadata)
    print(f"Within-run farming label prevalence: before(20k)={labels_before.mean():.3f}, "
          f"after(30k)={labels_after.mean():.3f}")

    before_direction_fittable = len(np.unique(labels_before)) >= 2 and min(
        np.bincount(labels_before)) >= 5

    # Fit direction at the AFTER checkpoint, in its own space, train positions only
    from scripts.run_interventions import _position_mask
    train_mask = _position_mask(observations, parts["train"])
    direction = fit_probe_direction(
        rec_after.hidden2[train_mask], labels_after[train_mask],
        str(after_ckpt), C=cfg["probe_C"], seed=cfg["probe_seed"])
    typical_norm_after = float(np.linalg.norm(rec_after.hidden2[train_mask], axis=1).mean())
    typical_norm_before = float(np.linalg.norm(rec_before.hidden2[train_mask], axis=1).mean())

    # Use the strength selected for seed 123's final checkpoint if available,
    # else a mid-range default; keep relative scaling per-checkpoint.
    final_results_path = OUTPUT_DIR / "intervention_results.json"
    rel = 0.1
    if final_results_path.exists():
        with open(final_results_path) as f:
            prior = json.load(f)
        rel = prior["flawed"][str(seed)]["selected_relative_strength"]
    print(f"Using relative strength {rel} (selected on seed {seed} final dev states)")

    results = {
        "seed": seed,
        "relative_strength": rel,
        "direction_fitted_at": str(after_ckpt),
        "before_direction_fittable": bool(before_direction_fittable),
        "label_prevalence": {
            "before": float(labels_before.mean()),
            "after": float(labels_after.mean()),
        },
        "typical_norms": {"before": typical_norm_before, "after": typical_norm_after},
        "caveat": (
            "The direction is fitted at the 30k checkpoint. The 20k checkpoint has no "
            "farming-labeled actions, so no within-checkpoint direction exists; applying "
            "the 30k direction to the 20k network is a cross-checkpoint transfer within "
            "one training run, and the weights differ between checkpoints."
        ),
    }

    print(f"\n=== Checkpoint {tc['before_step']} (goal-preferring) ===")
    results["before"] = evaluate_checkpoint(
        net_before, parts, direction.vector, rel, typical_norm_before, randoms, "20k")

    print(f"\n=== Checkpoint {tc['after_step']} (farming) ===")
    results["after"] = evaluate_checkpoint(
        net_after, parts, direction.vector, rel, typical_norm_after, randoms, "30k")

    out_path = OUTPUT_DIR / "transition_interventions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
