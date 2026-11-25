"""Phase 4.8 (experimental extension): sparse causal activation patching.

Reproduces the Phase 4.7 full-layer patch (donor 20k, recipient 30k, seed
123, donor sees remaining time 0.2, alpha = 1) and asks how many of the 128
layer-2 units need to be replaced. Three selection methods (activation
difference, causal ablation, greedy forward selection) are run on the 16
dev positions; subsets of size k in {1,...,128} are evaluated on the 46
other positions against 20 random subsets of the same size, a constant
Q-bias control, and (for promising subsets) 20 shuffled-state donors.

Usage:
    python -m scripts.run_sparse_patching
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_extraction import build_matched_state_dataset
from rewardlens.analysis.activation_intervention import partition_start_positions
from rewardlens.analysis.behavior_classification import CATEGORIES
from rewardlens.analysis.sparse_patching import (
    DIM,
    ablation_importance,
    evaluate_mask,
    greedy_actions_under_mask,
    greedy_selection,
    make_mask,
    masked_delta,
    random_subsets,
    rank_by_activation_difference,
)
from rewardlens.analysis.steering import clamp_time_tensor, evaluate_steering, summarize
from rewardlens.analysis.time_dependence import observations_for_positions
from rewardlens.envs.reward_grid import RewardGridConfig
from rewardlens.cli import script_args
from scripts.run_interventions import _position_mask

FLAWED_DIR = Path("results/phase2")
OUTPUT_DIR = Path("results/sparse_patching")
CONFIG_PATH = Path("experiments/01_fixed_gridworld_repair/sparse_patching/configs/sparse_patching.json")


def _ckpt(step: int) -> Path:
    name = "checkpoint_final.pt" if step == 100000 else f"checkpoint_{step}.pt"
    return FLAWED_DIR / "baseline_seed123" / "checkpoints" / name


def _fmt(r: dict) -> str:
    eff = r.get("mean_path_efficiency_direct")
    return (" ".join(f"{c[:6]}={r[f'{c}_rate']:.2f}" for c in CATEGORIES)
            + f" visits={r['mean_reward_tile_visits']:4.1f} steps={r['mean_episode_steps']:5.1f}"
            + f" eff={'-' if eff is None else round(eff, 2)} norm={r['mean_applied_norm']:4.1f}")


def _agg(lst):
    eff = [r["mean_path_efficiency_direct"] for r in lst if r.get("mean_path_efficiency_direct") is not None]
    return {"n": len(lst),
            **{f"mean_{c}_rate": float(np.mean([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
            **{f"max_{c}_rate": float(np.max([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
            **{f"std_{c}_rate": float(np.std([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
            "mean_reward_tile_visits": float(np.mean([r["mean_reward_tile_visits"] for r in lst])),
            "mean_episode_steps": float(np.mean([r["mean_episode_steps"] for r in lst])),
            "mean_path_efficiency_direct": float(np.mean(eff)) if eff else None,
            "mean_applied_norm": float(np.mean([r["mean_applied_norm"] for r in lst])),
            "per_subset": lst}


def main() -> None:
    t0 = time.time()
    cfg = json.load(open(CONFIG_PATH))
    thr = int(cfg["visit_threshold"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_cfg = RewardGridConfig()
    parts = partition_start_positions(env_cfg, cfg["n_dev_positions"], cfg["n_test_positions"], cfg["partition_seed"])
    dev = parts["dev"]
    evalset = parts["test"] + parts["train"]
    all_positions = [(x, y) for x in range(env_cfg.width) for y in range(env_cfg.height)
                     if (x, y) not in (env_cfg.goal_pos, env_cfg.reward_pos)]
    assert not (set(dev) & set(evalset)) and len(dev) + len(evalset) == 62
    observations = build_matched_state_dataset()
    obs_dev = observations[_position_mask(observations, dev)]
    pool_obs = observations[_position_mask(observations, parts["train"])]
    obs_eval_grid, _, _ = observations_for_positions(evalset, cfg["q_time_fractions"], env_cfg)
    transform = clamp_time_tensor(cfg["donor_time"])
    donor, _ = load_q_network_for_eval(_ckpt(cfg["donor_step"]))
    recipient, _ = load_q_network_for_eval(_ckpt(cfg["recipient_step"]))
    ks = [int(k) for k in cfg["subset_sizes"]]
    full = np.ones(DIM, dtype=np.float32)
    empty = make_mask([])

    results = {"config": cfg, "dev_positions": [list(p) for p in dev], "eval_positions": [list(p) for p in evalset]}

    # ---- Baseline reproduction ----
    print("=== Baseline (all 62 starts) ===")
    base_all = {"unmodified": evaluate_mask(recipient, donor, all_positions, empty, transform, env_cfg, thr),
                "full_patch": evaluate_mask(recipient, donor, all_positions, full, transform, env_cfg, thr),
                "donor_alone": summarize(evaluate_steering(donor, env_cfg, all_positions, threshold=thr))}
    for k, r in base_all.items():
        print(f"  {k:12s} {_fmt(r)}")
    assert base_all["full_patch"]["direct_goal_rate"] == 1.0, "Phase 4.7 full patch did not reproduce 1.00"
    results["baseline_all_62"] = base_all
    results["baseline_dev"] = {"unmodified": evaluate_mask(recipient, donor, dev, empty, transform, env_cfg, thr),
                               "full_patch": evaluate_mask(recipient, donor, dev, full, transform, env_cfg, thr)}
    results["baseline_eval"] = {"unmodified": evaluate_mask(recipient, donor, evalset, empty, transform, env_cfg, thr, keep_traces=True),
                                "full_patch": evaluate_mask(recipient, donor, evalset, full, transform, env_cfg, thr, keep_traces=True)}
    full_actions_eval = greedy_actions_under_mask(recipient, donor, full, obs_eval_grid, transform)
    rec_actions_eval = greedy_actions_under_mask(recipient, donor, empty, obs_eval_grid, transform)

    # ---- Method A ----
    print("\n=== Method A: activation difference (dev states) ===")
    A = rank_by_activation_difference(recipient, donor, obs_dev, transform)
    print(f"  top 8 units: {A['order'][:8]}  scores: {[round(A['score'][u], 2) for u in A['order'][:8]]}")

    # ---- Method B ----
    print("\n=== Method B: single-unit ablation from the full patch (dev) ===")
    B = ablation_importance(recipient, donor, dev, obs_dev, transform, env_cfg, thr, cfg["ablation"]["group_size"])
    drops = [r["direct_goal_drop"] for r in B["rows"]]
    print(f"  units whose removal lowers dev direct-goal: {sum(d > 0 for d in drops)} / {DIM}; "
          f"max drop {max(drops):.2f}; top 8: {B['order'][:8]}")

    # ---- Method C ----
    print("\n=== Method C: greedy forward selection (dev) ===")
    n_top = cfg["greedy"]["pool_top_n"]
    candidates = sorted(set(A["order"][:n_top]) | set(B["order"][:n_top]))
    C = greedy_selection(recipient, donor, dev, transform, candidates, cfg["greedy"]["max_k"],
                         cfg["greedy"]["eval_budget"], env_cfg, thr)
    for h in C["history"]:
        print(f"  k={h['k']:2d} add unit {h['added']:3d} -> dev {_fmt(h['dev'])}")
    print(f"  evaluations used: {C['n_evaluations']} / {C['budget']}; candidates: {len(candidates)}")
    results["selection"] = {"A_activation_difference": A, "B_ablation": B, "C_greedy": C}

    # ---- Evaluate subsets per method and size ----
    print("\n=== Subsets by size (dev selection -> eval) ===")
    methods = {"A": A["order"], "B": B["order"], "C": C["order"]}
    per_method = {}
    for name, order in methods.items():
        per_method[name] = {}
        for k in ks:
            if k > len(order):
                continue
            units = [int(u) for u in order[:k]]
            m = make_mask(units)
            r_dev = evaluate_mask(recipient, donor, dev, m, transform, env_cfg, thr)
            r_eval = evaluate_mask(recipient, donor, evalset, m, transform, env_cfg, thr)
            acts = greedy_actions_under_mask(recipient, donor, m, obs_eval_grid, transform)
            cdelta = masked_delta(recipient, donor, m, obs_dev, transform).mean(0)
            r_const = summarize(evaluate_steering(recipient, env_cfg, evalset, None, lambda s: 1.0, 2, "constant",
                                                  constant_delta=cdelta, threshold=thr))
            per_method[name][str(k)] = {
                "units": units, "dev": r_dev, "eval": r_eval,
                "eval_greedy_flip_vs_recipient": float((acts != rec_actions_eval).mean()),
                "eval_greedy_agreement_with_full": float((acts == full_actions_eval).mean()),
                "constant_qbias_eval": r_const, "constant_delta_norm": float(np.linalg.norm(cdelta)),
            }
            print(f"  {name} k={k:3d} dev direct={r_dev['direct_goal_rate']:.2f} | eval {_fmt(r_eval)}"
                  f" | qbias direct={r_const['direct_goal_rate']:.2f}")
    results["subsets"] = per_method

    # ---- Random subsets per size (eval; also dev for reference) ----
    print("\n=== Random subsets (20 per size, eval) ===")
    rnd = {}
    for k in ks:
        subs = random_subsets(k, cfg["n_random_subsets"], cfg["random_seed"])
        lst = []
        for s in subs:
            r = evaluate_mask(recipient, donor, evalset, make_mask(s), transform, env_cfg, thr)
            r["units"] = [int(u) for u in s]
            lst.append(r)
        rnd[str(k)] = _agg(lst)
        rnd[str(k)]["dev"] = _agg([evaluate_mask(recipient, donor, dev, make_mask(s), transform, env_cfg, thr) for s in subs])
        print(f"  k={k:3d} eval direct mean={rnd[str(k)]['mean_direct_goal_rate']:.2f} max={rnd[str(k)]['max_direct_goal_rate']:.2f}"
              f" visits={rnd[str(k)]['mean_reward_tile_visits']:.1f} norm={rnd[str(k)]['mean_applied_norm']:.1f}")
    results["random_subsets"] = rnd

    # ---- Fine sweep for method A between the coarse sizes (prespecified in config) ----
    print("\n=== Method A fine sweep (eval) ===")
    fine = {}
    for k in cfg.get("fine_sizes_method_A", []):
        units = [int(u) for u in A["order"][:k]]
        m = make_mask(units)
        r_dev = evaluate_mask(recipient, donor, dev, m, transform, env_cfg, thr)
        r_eval = evaluate_mask(recipient, donor, evalset, m, transform, env_cfg, thr)
        subs = random_subsets(k, cfg["n_random_subsets"], cfg["random_seed"])
        rr = _agg([evaluate_mask(recipient, donor, evalset, make_mask(s), transform, env_cfg, thr) for s in subs])
        fine[str(k)] = {"units": units, "dev": r_dev, "eval": r_eval, "random": rr}
        print(f"  A k={k:3d} dev direct={r_dev['direct_goal_rate']:.2f} | eval {_fmt(r_eval)}"
              f" | random mean={rr['mean_direct_goal_rate']:.2f} max={rr['max_direct_goal_rate']:.2f}")
    results["fine_sweep_A"] = fine

    # ---- Shuffled-state donor controls for promising subsets ----
    print("\n=== Shuffled-state donor controls (promising subsets) ===")
    shuffled = {}
    thr_s = cfg["shuffled_control"]["min_eval_direct_goal_to_test"]
    for name in methods:
        for k, row in per_method[name].items():
            if int(k) < DIM and row["eval"]["direct_goal_rate"] >= thr_s:
                m = make_mask(row["units"])
                lst = [evaluate_mask(recipient, donor, evalset, m, transform, env_cfg, thr, pool_obs=pool_obs,
                                     seed=cfg["random_seed"] + i) for i in range(cfg["shuffled_control"]["n"])]
                shuffled[f"{name}:{k}"] = _agg(lst)
                print(f"  {name} k={k}: shuffled direct mean={shuffled[f'{name}:{k}']['mean_direct_goal_rate']:.2f}"
                      f" max={shuffled[f'{name}:{k}']['max_direct_goal_rate']:.2f} eff={shuffled[f'{name}:{k}']['mean_path_efficiency_direct']}")
    results["shuffled_controls"] = shuffled

    # ---- Knockout curves: full patch minus top-k (eval) ----
    print("\n=== Knockout from the full patch: restore top-k units to recipient values (eval) ===")
    knock = {}
    for name in ("A", "B"):
        knock[name] = {}
        for k in ks:
            m = full.copy()
            m[methods[name][:k]] = 0.0
            r = evaluate_mask(recipient, donor, evalset, m, transform, env_cfg, thr)
            knock[name][str(k)] = r
            print(f"  {name} remove top-{k:3d}: {_fmt(r)}")
    results["knockout"] = knock

    # ---- Best sparse subset, chosen on dev only ----
    cands = []
    for name in ("A", "B", "C"):
        for k, row in per_method[name].items():
            if row["dev"]["direct_goal_rate"] >= 0.95:
                cands.append((int(k), -(row["dev"]["mean_path_efficiency_direct"] or 0.0), "ABC".index(name), name))
    cands.sort()
    if cands:
        k_best, _, _, m_best = cands[0]
        row = per_method[m_best][str(k_best)]
        m = make_mask(row["units"])
        best = {"method": m_best, "k": k_best, "units": row["units"], "dev": row["dev"], "eval": row["eval"],
                "eval_traces": evaluate_mask(recipient, donor, evalset, m, transform, env_cfg, thr, keep_traces=True),
                "random_same_size_example_traces": evaluate_mask(
                    recipient, donor, evalset, make_mask(random_subsets(k_best, 1, cfg["random_seed"])[0]),
                    transform, env_cfg, thr, keep_traces=True),
                "all_62": evaluate_mask(recipient, donor, all_positions, m, transform, env_cfg, thr)}
        print(f"\nBest sparse subset (dev rule): method {m_best}, k={k_best}, units {row['units']}")
        print(f"  eval {_fmt(row['eval'])}\n  all62 {_fmt(best['all_62'])}")
    else:
        best = None
        print("\nNo sparse subset reached dev direct-goal >= 0.95")
    results["best_sparse"] = best

    out = OUTPUT_DIR / "sparse_patching_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    script_args(__doc__)
    main()
