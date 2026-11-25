"""Phase 4 Experiments 1-3: intervention strength sweeps, random controls, cross-seed.

For each flawed-reward seed, fits a probe direction and a contrastive direction
in that network's own layer-2 activation space (train positions only), selects
the intervention strength on dev start states, and reports final behavior on
held-out test start states against the unmodified policy and 20 norm-matched
random-direction controls. Also measures corrected-reward network fragility
under the same random perturbations.

Usage:
    python -m scripts.run_interventions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_extraction import (
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
)
from rewardlens.analysis.activation_intervention import (
    evaluate_on_start_states,
    fit_contrastive_direction,
    fit_probe_direction,
    partition_start_positions,
    random_unit_directions,
)
from rewardlens.envs.reward_grid import RewardGridConfig
from rewardlens.cli import script_args
from scripts.probe_activations import compute_farming_labels

FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")
OUTPUT_DIR = Path("results/phase4")
CONFIG_PATH = Path("experiments/01_fixed_gridworld_repair/configs/phase4_interventions.json")


def _strip_actions(result: dict) -> dict:
    out = dict(result)
    out["episodes"] = [
        {k: v for k, v in e.items() if k != "actions"} for e in result["episodes"]
    ]
    return out


def _position_mask(observations: np.ndarray, positions: list, width: int = 8, height: int = 8) -> np.ndarray:
    ax = (observations[:, 0] * width).round().astype(int)
    ay = (observations[:, 1] * height).round().astype(int)
    pos_set = {tuple(p) for p in positions}
    return np.array([(int(ax[i]), int(ay[i])) in pos_set for i in range(len(ax))])


def q_value_effects(net, observations: np.ndarray, direction: np.ndarray, alpha: float, mode: str) -> dict:
    """Greedy-action flip rate and mean |dQ| on a fixed observation set."""
    from rewardlens.analysis.activation_intervention import layer2_intervention

    obs_t = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        q_base = net(obs_t).numpy()
    with layer2_intervention(net, direction, alpha, mode), torch.no_grad():
        q_edit = net(obs_t).numpy()
    flips = (q_base.argmax(1) != q_edit.argmax(1)).mean()
    return {
        "greedy_flip_rate": float(flips),
        "mean_abs_q_change": float(np.abs(q_edit - q_base).mean()),
    }


def analyze_flawed_seed(seed: int, cfg: dict, observations, metadata, parts) -> dict:
    print(f"\n=== Flawed-reward seed {seed} ===")
    ckpt = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    corrected_ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    net, _ = load_q_network_for_eval(ckpt)
    base_env_cfg = RewardGridConfig()

    # --- Direction fitting on TRAIN positions only ---
    flawed_rec = extract_activations(ckpt, observations, seed, "flawed", 100000)
    corrected_rec = extract_activations(corrected_ckpt, observations, seed, "corrected", 100000)
    labels = compute_farming_labels(flawed_rec.greedy_actions, corrected_rec.greedy_actions, metadata)

    train_mask = _position_mask(observations, parts["train"])
    h_train = flawed_rec.hidden2[train_mask]
    y_train = labels[train_mask]
    print(f"  Direction-fit states: {train_mask.sum()} (pos frac {y_train.mean():.3f})")

    probe_dir = fit_probe_direction(h_train, y_train, str(ckpt), C=cfg["probe_C"], seed=cfg["probe_seed"])
    contrastive_dir = fit_contrastive_direction(h_train, y_train, str(ckpt))
    typical_norm = float(np.linalg.norm(h_train, axis=1).mean())
    print(f"  Typical layer-2 norm (train states): {typical_norm:.2f}")

    rels = cfg["relative_strengths"]
    randoms = random_unit_directions(128, cfg["n_random_directions"], cfg["random_direction_seed"])

    # --- Dev sweep: targeted modes + random controls at every strength ---
    dev_sweep = {"probe_subtract": {}, "probe_add": {}, "random_subtract": {}}
    for rel in rels:
        alpha = rel * typical_norm
        dev_sweep["probe_subtract"][str(rel)] = _strip_actions(evaluate_on_start_states(
            net, base_env_cfg, parts["dev"], probe_dir.vector, alpha, "subtract"))
        dev_sweep["probe_add"][str(rel)] = _strip_actions(evaluate_on_start_states(
            net, base_env_cfg, parts["dev"], probe_dir.vector, alpha, "add"))
        rand_out = []
        for ri in range(len(randoms)):
            r = evaluate_on_start_states(net, base_env_cfg, parts["dev"], randoms[ri], alpha, "subtract")
            rand_out.append({k: r[k] for k in
                             ("goal_rate", "hacking_rate", "mean_proxy_reward",
                              "mean_episode_steps", "mean_no_move_steps", "mean_applied_norm")})
        dev_sweep["random_subtract"][str(rel)] = rand_out
        ps = dev_sweep["probe_subtract"][str(rel)]
        print(f"  dev rel={rel:<5} subtract: goal={ps['goal_rate']:.2f} hack={ps['hacking_rate']:.2f}"
              f" | add: goal={dev_sweep['probe_add'][str(rel)]['goal_rate']:.2f}"
              f" hack={dev_sweep['probe_add'][str(rel)]['hacking_rate']:.2f}")

    # Projection removal on dev (no alpha)
    dev_project = _strip_actions(evaluate_on_start_states(
        net, base_env_cfg, parts["dev"], probe_dir.vector, 0.0, "project_out"))
    print(f"  dev project_out: goal={dev_project['goal_rate']:.2f} hack={dev_project['hacking_rate']:.2f}")

    # --- Strength selection on dev: max goal rate under probe subtraction;
    #     ties broken by lower hacking rate, then smaller strength. ---
    def key(rel):
        s = dev_sweep["probe_subtract"][str(rel)]
        return (-s["goal_rate"], s["hacking_rate"], rel)
    selected_rel = min([r for r in rels if r > 0], key=key)
    selected_alpha = selected_rel * typical_norm
    print(f"  Selected strength (dev): rel={selected_rel} (alpha={selected_alpha:.2f})")

    # --- Held-out TEST evaluation ---
    test = {}
    test["baseline"] = _strip_actions(evaluate_on_start_states(
        net, base_env_cfg, parts["test"], None, 0.0, "none"))
    test["probe_subtract"] = _strip_actions(evaluate_on_start_states(
        net, base_env_cfg, parts["test"], probe_dir.vector, selected_alpha, "subtract"))
    test["probe_add"] = _strip_actions(evaluate_on_start_states(
        net, base_env_cfg, parts["test"], probe_dir.vector, selected_alpha, "add"))
    test["probe_project_out"] = _strip_actions(evaluate_on_start_states(
        net, base_env_cfg, parts["test"], probe_dir.vector, 0.0, "project_out"))
    test["contrastive_subtract"] = _strip_actions(evaluate_on_start_states(
        net, base_env_cfg, parts["test"], contrastive_dir.vector, selected_alpha, "subtract"))
    rand_test = []
    for ri in range(len(randoms)):
        r = evaluate_on_start_states(net, base_env_cfg, parts["test"], randoms[ri], selected_alpha, "subtract")
        rand_test.append({k: r[k] for k in
                          ("goal_rate", "hacking_rate", "mean_proxy_reward",
                           "mean_episode_steps", "mean_no_move_steps",
                           "mean_unique_positions", "mean_applied_norm")})
    test["random_subtract"] = rand_test

    for name in ("baseline", "probe_subtract", "probe_add", "probe_project_out", "contrastive_subtract"):
        t = test[name]
        print(f"  TEST {name:22s} goal={t['goal_rate']:.2f} hack={t['hacking_rate']:.2f}"
              f" reward={t['mean_proxy_reward']:7.2f} steps={t['mean_episode_steps']:.1f}")
    rg = [r["goal_rate"] for r in rand_test]
    rh = [r["hacking_rate"] for r in rand_test]
    print(f"  TEST random (n=20)       goal={np.mean(rg):.2f}±{np.std(rg):.2f}"
          f" hack={np.mean(rh):.2f}±{np.std(rh):.2f}")

    # --- Q-value effects at selected strength (matched states, test positions) ---
    test_mask = _position_mask(observations, parts["test"])
    q_fx = {
        "probe_subtract": q_value_effects(net, observations[test_mask], probe_dir.vector, selected_alpha, "subtract"),
        "probe_add": q_value_effects(net, observations[test_mask], probe_dir.vector, selected_alpha, "add"),
    }

    return {
        "seed": seed,
        "checkpoint": str(ckpt),
        "typical_layer2_norm": typical_norm,
        "direction_provenance": {
            "probe": {
                "source": probe_dir.source, "checkpoint": probe_dir.checkpoint,
                "sign_verified": probe_dir.sign_verified,
                "fit_n_states": probe_dir.fit_n_states,
                "fit_positive_fraction": probe_dir.fit_positive_fraction,
            },
            "contrastive_cosine_to_probe": float(contrastive_dir.vector @ probe_dir.vector),
        },
        "selected_relative_strength": selected_rel,
        "selected_alpha": selected_alpha,
        "dev_sweep": dev_sweep,
        "dev_project_out": dev_project,
        "test": test,
        "q_value_effects": q_fx,
    }


def analyze_corrected_control(seed: int, cfg: dict, parts) -> dict:
    """Fragility control: the corrected-reward net under the same random perturbations.

    A farming direction is not identifiable from this network's own behavior
    (it never farms, so the behavioral label has a single class). We therefore
    apply only norm-matched random perturbations, which measure generic policy
    fragility, and explicitly do not transfer directions from other networks.
    """
    print(f"\n=== Corrected-reward control (seed {seed}) ===")
    ckpt = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    net, loaded = load_q_network_for_eval(ckpt)
    env_cfg = RewardGridConfig(reward_tile_bonus=0.0)

    observations = build_matched_state_dataset()
    rec = extract_activations(ckpt, observations, seed, "corrected", 100000)
    train_mask = _position_mask(observations, parts["train"])
    typical_norm = float(np.linalg.norm(rec.hidden2[train_mask], axis=1).mean())

    randoms = random_unit_directions(128, cfg["n_random_directions"], cfg["random_direction_seed"])
    out = {"seed": seed, "checkpoint": str(ckpt), "typical_layer2_norm": typical_norm,
           "farming_direction_identifiable": False, "random_subtract": {}}
    out["baseline"] = _strip_actions(evaluate_on_start_states(
        net, env_cfg, parts["test"], None, 0.0, "none"))
    print(f"  baseline: goal={out['baseline']['goal_rate']:.2f}")
    for rel in cfg["relative_strengths"]:
        if rel == 0.0:
            continue
        alpha = rel * typical_norm
        rand_out = []
        for ri in range(len(randoms)):
            r = evaluate_on_start_states(net, env_cfg, parts["test"], randoms[ri], alpha, "subtract")
            rand_out.append({k: r[k] for k in ("goal_rate", "mean_episode_steps", "mean_no_move_steps")})
        out["random_subtract"][str(rel)] = rand_out
        print(f"  rel={rel:<5} random goal={np.mean([r['goal_rate'] for r in rand_out]):.2f}"
              f"±{np.std([r['goal_rate'] for r in rand_out]):.2f}")
    return out


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_env_cfg = RewardGridConfig()
    parts = partition_start_positions(
        base_env_cfg, n_dev=cfg["n_dev_positions"],
        n_test=cfg["n_test_positions"], seed=cfg["partition_seed"])
    print(f"Partition: {len(parts['train'])} train / {len(parts['dev'])} dev / {len(parts['test'])} test positions")

    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)

    results = {"config": cfg,
               "partition": {k: [list(p) for p in v] for k, v in parts.items()},
               "flawed": {}, "corrected_control": None}
    for seed in cfg["flawed_seeds"]:
        results["flawed"][str(seed)] = analyze_flawed_seed(seed, cfg, observations, metadata, parts)
    results["corrected_control"] = analyze_corrected_control(cfg["corrected_control_seed"], cfg, parts)

    out_path = OUTPUT_DIR / "intervention_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    script_args(__doc__)
    main()
