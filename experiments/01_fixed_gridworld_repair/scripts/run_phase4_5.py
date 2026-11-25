"""Phase 4.5: goal-directed activation patching.

Experiment A (same run, seed 123): donor = goal-following 20k checkpoint,
recipients = farming 30k and 100k checkpoints.
Experiment B (cross model, seed 123): donor = corrected-reward final DQN,
recipient = flawed-reward final DQN, with identity / ridge-all /
ridge-agreement alignment mappings.

A fresh prespecified start-state partition (partition_seed=11, disjoint from
the Phase 4 split's purpose) is used: fit positions for alignment mappings,
dev positions for strength selection, test positions for final evaluation.

Usage:
    python -m scripts.run_phase4_5
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
from rewardlens.analysis.activation_intervention import partition_start_positions
from rewardlens.analysis.activation_patching import (
    activation_distribution_comparison,
    evaluate_patching,
    fit_linear_mapping,
    identity_mapping,
    make_matched_donor,
    make_shuffled_donor,
    q_change_analysis,
)
from rewardlens.envs.reward_grid import RewardGridConfig
from scripts.run_interventions import _position_mask

FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")
OUTPUT_DIR = Path("results/phase4_5")
CONFIG_PATH = Path("experiments/01_fixed_gridworld_repair/configs/phase4_5.json")


def _summ(r: dict) -> dict:
    return {k: r[k] for k in ("goal_rate", "hacking_rate", "mean_reward_tile_visits",
                              "mean_proxy_reward", "mean_episode_steps", "mean_no_move_steps",
                              "mean_unique_positions", "mean_applied_norm",
                              "mean_negative_fraction")}


def _ckpt(step: int) -> Path:
    name = "checkpoint_final.pt" if step == 100000 else f"checkpoint_{step}.pt"
    return FLAWED_DIR / "baseline_seed123" / "checkpoints" / name


def run_patching_block(
    label: str, recipient, donor, mapping, observations, parts, cfg, pool_obs,
) -> dict:
    """Dev sweep, strength selection, held-out evaluation with all controls."""
    env_cfg = RewardGridConfig()
    alphas = cfg["alphas"]
    donor_fn = make_matched_donor(donor, mapping)
    print(f"\n--- {label} ---")

    # Dev sweep (matched patching + random + shuffled at each alpha)
    dev = {}
    for a in alphas:
        m = _summ(evaluate_patching(recipient, env_cfg, parts["dev"], a, donor_fn, "matched"))
        rnd = [_summ(evaluate_patching(recipient, env_cfg, parts["dev"], a, donor_fn, "random",
                                       random_seed=cfg["control_seed"] + i))
               for i in range(cfg["n_controls"])] if a > 0 else []
        shf = [_summ(evaluate_patching(recipient, env_cfg, parts["dev"], a,
                                       make_shuffled_donor(donor, pool_obs, cfg["control_seed"] + i, mapping),
                                       "matched"))
               for i in range(cfg["n_controls"])] if a > 0 else []
        dev[str(a)] = {"matched": m, "random": rnd, "shuffled": shf}
        rg = np.mean([r["goal_rate"] for r in rnd]) if rnd else m["goal_rate"]
        sg = np.mean([r["goal_rate"] for r in shf]) if shf else m["goal_rate"]
        print(f"  dev a={a:<4} matched goal={m['goal_rate']:.2f} hack={m['hacking_rate']:.2f}"
              f" | random goal={rg:.2f} | shuffled goal={sg:.2f} | norm={m['mean_applied_norm']:.1f}")

    # Selection on dev: max goal rate, tie -> lower hacking, tie -> smaller alpha
    sel = min([a for a in alphas if a > 0],
              key=lambda a: (-dev[str(a)]["matched"]["goal_rate"],
                             dev[str(a)]["matched"]["hacking_rate"], a))
    print(f"  selected alpha (dev) = {sel}")

    # Constant Q-bias control: mean layer-2 delta over DEV matched states
    dev_mask = _position_mask(observations, parts["dev"])
    obs_dev = torch.as_tensor(observations[dev_mask], dtype=torch.float32)
    with torch.no_grad():
        h_rec = recipient.relu2(recipient.fc2(recipient.relu1(recipient.fc1(obs_dev))))
        h_don = donor_fn(obs_dev)
    constant_delta = (h_don - h_rec).mean(0).numpy()

    # Held-out test evaluation
    test = {
        "baseline": _summ(evaluate_patching(recipient, env_cfg, parts["test"], 0.0)),
        "matched": _summ(evaluate_patching(recipient, env_cfg, parts["test"], sel, donor_fn, "matched")),
        "constant_qbias": _summ(evaluate_patching(recipient, env_cfg, parts["test"], sel, None,
                                                  "constant", constant_delta=constant_delta)),
        "random": [_summ(evaluate_patching(recipient, env_cfg, parts["test"], sel, donor_fn, "random",
                                           random_seed=cfg["control_seed"] + i))
                   for i in range(cfg["n_controls"])],
        "shuffled": [_summ(evaluate_patching(recipient, env_cfg, parts["test"], sel,
                                             make_shuffled_donor(donor, pool_obs, cfg["control_seed"] + i, mapping),
                                             "matched"))
                     for i in range(cfg["n_controls"])],
        "donor_independent": _summ(evaluate_patching(donor, env_cfg, parts["test"], 0.0)),
        "matched_sweep": {str(a): _summ(evaluate_patching(recipient, env_cfg, parts["test"], a, donor_fn, "matched"))
                          for a in alphas},
    }
    for k in ("baseline", "matched", "constant_qbias", "donor_independent"):
        t = test[k]
        print(f"  TEST {k:18s} goal={t['goal_rate']:.2f} hack={t['hacking_rate']:.2f}"
              f" reward={t['mean_proxy_reward']:7.2f} steps={t['mean_episode_steps']:.1f}")
    for k in ("random", "shuffled"):
        g = [r["goal_rate"] for r in test[k]]
        h = [r["hacking_rate"] for r in test[k]]
        print(f"  TEST {k:18s} goal={np.mean(g):.2f}±{np.std(g):.2f} hack={np.mean(h):.2f}±{np.std(h):.2f}")

    # Scientific checks on matched test states
    test_mask = _position_mask(observations, parts["test"])
    obs_test = observations[test_mask]
    with torch.no_grad():
        ot = torch.as_tensor(obs_test, dtype=torch.float32)
        h_rec_t = recipient.relu2(recipient.fc2(recipient.relu1(recipient.fc1(ot)))).numpy()
        h_don_t = donor_fn(ot).numpy()
    checks = {
        "distributions": activation_distribution_comparison(h_don_t, h_rec_t),
        "q_change_matched": q_change_analysis(recipient, obs_test, sel, donor_fn, "matched"),
        "q_change_constant": q_change_analysis(recipient, obs_test, sel, None, "constant",
                                               constant_delta=constant_delta),
    }
    qc = checks["q_change_matched"]
    print(f"  checks: delta_norm={checks['distributions']['mean_delta_norm']:.1f}"
          f" cos(donor,rec)={checks['distributions']['mean_cosine_donor_recipient']:.2f}"
          f" flip={qc['greedy_flip_rate']:.2f} const_frac_dQ={qc['constant_fraction_of_dq_energy']:.2f}")

    return {"label": label, "selected_alpha": sel, "dev_sweep": dev, "test": test,
            "checks": checks, "constant_delta_norm": float(np.linalg.norm(constant_delta)),
            "mapping": {"name": mapping.name if mapping else "none",
                        "fit_r2": mapping.fit_r2 if mapping else None,
                        "fit_n": mapping.fit_n if mapping else None}}


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    parts = partition_start_positions(RewardGridConfig(), n_dev=cfg["n_dev_positions"],
                                      n_test=cfg["n_test_positions"], seed=cfg["partition_seed"])
    print(f"Partition (seed {cfg['partition_seed']}): {len(parts['train'])} fit / "
          f"{len(parts['dev'])} dev / {len(parts['test'])} test positions")
    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)
    fit_mask = _position_mask(observations, parts["train"])
    pool_obs = observations[fit_mask]  # shuffled-donor pool: fit states only

    results = {"config": cfg, "partition": {k: [list(p) for p in v] for k, v in parts.items()},
               "same_run": {}, "cross_model": {}}

    # ---------------- Experiment A: same-run checkpoint patching ----------------
    donor_step = cfg["same_run"]["donor_step"]
    donor, _ = load_q_network_for_eval(_ckpt(donor_step))
    env_cfg = RewardGridConfig()
    donor_beh = _summ(evaluate_patching(donor, env_cfg, parts["test"], 0.0))
    print(f"\nDonor {donor_step} verified on test states: goal={donor_beh['goal_rate']:.2f} "
          f"hack={donor_beh['hacking_rate']:.2f}")
    for rstep in cfg["same_run"]["recipient_steps"]:
        recipient, _ = load_q_network_for_eval(_ckpt(rstep))
        rb = _summ(evaluate_patching(recipient, env_cfg, parts["test"], 0.0))
        print(f"Recipient {rstep} verified: goal={rb['goal_rate']:.2f} hack={rb['hacking_rate']:.2f}")
        results["same_run"][str(rstep)] = run_patching_block(
            f"same-run donor {donor_step} -> recipient {rstep}", recipient, donor, None,
            observations, parts, cfg, pool_obs)
        results["same_run"][str(rstep)]["donor_step"] = donor_step
        results["same_run"][str(rstep)]["recipient_step"] = rstep

    # ---------------- Experiment B: cross-model patching ----------------
    donor_c, _ = load_q_network_for_eval(CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt")
    recipient_f, _ = load_q_network_for_eval(_ckpt(100000))
    rec_d = extract_activations(CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt",
                                observations, seed, "corrected", 100000)
    rec_r = extract_activations(_ckpt(100000), observations, seed, "flawed", 100000)
    agree = rec_d.greedy_actions == rec_r.greedy_actions
    print(f"\nCross-model: policies agree on {agree.mean():.2%} of matched states; "
          f"fit states={fit_mask.sum()}, agreement fit states={(fit_mask & agree).sum()}")

    mappings = {
        "identity": identity_mapping(),
        "ridge_all": fit_linear_mapping(rec_d.hidden2[fit_mask], rec_r.hidden2[fit_mask],
                                        cfg["cross_model"]["ridge_alpha"], "ridge_all"),
        "ridge_agreement": fit_linear_mapping(rec_d.hidden2[fit_mask & agree], rec_r.hidden2[fit_mask & agree],
                                              cfg["cross_model"]["ridge_alpha"], "ridge_agreement"),
    }
    # Alignment audit: how close is M(h_donor) to h_recipient on held-out test states?
    test_mask = _position_mask(observations, parts["test"])
    audit = {}
    for name, M in mappings.items():
        mapped = rec_d.hidden2[test_mask] @ M.weight.T + M.bias
        resid = np.linalg.norm(mapped - rec_r.hidden2[test_mask], axis=1).mean()
        base = np.linalg.norm(rec_d.hidden2[test_mask] - rec_r.hidden2[test_mask], axis=1).mean()
        audit[name] = {"fit_r2": M.fit_r2, "fit_n": M.fit_n,
                       "test_mean_residual_norm_to_recipient": float(resid),
                       "unaligned_mean_delta_norm": float(base),
                       "disagreement_residual": float(np.linalg.norm(
                           (mapped - rec_r.hidden2[test_mask])[~agree[test_mask]], axis=1).mean())
                       if (~agree[test_mask]).any() else None}
        print(f"  mapping {name:16s} fit_r2={M.fit_r2 if not np.isnan(M.fit_r2) else float('nan'):.3f}"
              f" test residual->recipient={resid:.1f} (unaligned delta {base:.1f})")
    results["cross_model"]["alignment_audit"] = audit
    results["cross_model"]["policy_agreement_fraction"] = float(agree.mean())

    for name in cfg["cross_model"]["mappings"]:
        results["cross_model"][name] = run_patching_block(
            f"cross-model corrected -> flawed [{name}]", recipient_f, donor_c, mappings[name],
            observations, parts, cfg, pool_obs)

    out = OUTPUT_DIR / "patching_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
