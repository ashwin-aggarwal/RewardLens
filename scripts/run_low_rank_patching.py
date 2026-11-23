"""Low-rank activation patching (experimental extension).

Fits PCA (centered) and SVD (uncentered) bases to delta(s) = h_donor(s') -
h_rec(s) on dev observations, patches the recipient with the rank-k
projection of its own per-state delta, and evaluates on the 46 non-dev
positions against random subspaces (plain and norm-matched), shuffled-state
donors, the constant mean delta, the full patch and the donor alone.

Usage:
    python -m scripts.run_low_rank_patching
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
from rewardlens.analysis.low_rank_patching import (
    DIM,
    LowRankBasis,
    collect_deltas,
    evaluate_donor_fn,
    fit_basis,
    make_low_rank_donor,
    make_subspace_donor,
    random_orthonormal_bases,
    reconstruction_error,
)
from rewardlens.analysis.steering import clamp_time_tensor, evaluate_steering, summarize
from rewardlens.analysis.time_dependence import observations_for_positions
from rewardlens.envs.reward_grid import RewardGridConfig
from scripts.run_interventions import _position_mask

FLAWED_DIR = Path("results/phase2")
OUTPUT_DIR = Path("results/low_rank_patching")
CONFIG_PATH = Path("experiments/low_rank_patching/configs/low_rank_patching.json")


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
            "per_control": lst}


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
    pool_obs = observations[_position_mask(observations, parts["train"])]
    obs_dev, _, _ = observations_for_positions(dev, cfg["fit_time_fractions"], env_cfg)
    obs_eval, _, _ = observations_for_positions(evalset, cfg["fit_time_fractions"], env_cfg)
    transform = clamp_time_tensor(cfg["donor_time"])
    donor, _ = load_q_network_for_eval(_ckpt(cfg["donor_step"]))
    recipient, _ = load_q_network_for_eval(_ckpt(cfg["recipient_step"]))
    ks = [int(k) for k in cfg["ranks"]]
    results = {"config": cfg, "dev_positions": [list(p) for p in dev], "eval_positions": [list(p) for p in evalset]}

    # ---- Baselines ----
    print("=== Baselines ===")
    base = {}
    for label, positions, traces in (("eval", evalset, True), ("all_62", all_positions, False), ("dev", dev, False)):
        base[label] = {
            "unmodified": evaluate_donor_fn(recipient, positions, None, env_cfg, thr, keep_traces=traces, mode="none"),
            "donor_alone": summarize(evaluate_steering(donor, env_cfg, positions, threshold=thr)),
        }
    results["baselines"] = base
    print(f"  unmodified all62 {_fmt(base['all_62']['unmodified'])}\n  donor alone all62 {_fmt(base['all_62']['donor_alone'])}")

    # ---- Fit bases on dev deltas ----
    deltas_dev = collect_deltas(recipient, donor, obs_dev, transform)
    deltas_eval = collect_deltas(recipient, donor, obs_eval, transform)
    bases = {v: fit_basis(deltas_dev, v) for v in cfg["variants"]}
    results["fit"] = {}
    for v, b in bases.items():
        cum = np.cumsum(b.explained_variance_ratio)
        results["fit"][v] = {
            "n_fit": b.n_fit, "explained_variance_ratio": b.explained_variance_ratio.tolist(),
            "cumulative_explained_variance": cum.tolist(),
            "mean_delta_norm": float(np.linalg.norm(deltas_dev, axis=1).mean()),
            "mean_norm": None if b.mean is None else float(np.linalg.norm(b.mean)),
            "reconstruction": {str(k): {"dev": reconstruction_error(b, deltas_dev, k),
                                        "eval": reconstruction_error(b, deltas_eval, k)} for k in ks},
        }
        print(f"  {v}: n_fit={b.n_fit}, cumulative explained variance at k=1/2/4/8/16/32/64: "
              + "/".join(f"{cum[k - 1]:.2f}" for k in (1, 2, 4, 8, 16, 32, 64)))
    np.savez(OUTPUT_DIR / "bases.npz", **{f"{v}_components": b.components for v, b in bases.items()},
             **{f"{v}_mean": (b.mean if b.mean is not None else np.zeros(DIM, dtype=np.float32)) for v, b in bases.items()})

    # ---- Rank sweep ----
    print("\n=== Rank sweep (dev selection -> eval) ===")
    sweep = {}
    for v, b in bases.items():
        sweep[v] = {}
        for k in ks:
            fn = make_low_rank_donor(recipient, donor, b, k, transform)
            r_dev = evaluate_donor_fn(recipient, dev, fn, env_cfg, thr)
            r_eval = evaluate_donor_fn(recipient, evalset, fn, env_cfg, thr, keep_traces=(k in (1, 2, 4, 8, 16, 128)))
            r_eval_s = {kk: vv for kk, vv in r_eval.items() if kk != "episodes"}
            sweep[v][str(k)] = {"dev": r_dev, "eval": r_eval_s,
                                "eval_episodes": r_eval.get("episodes"),
                                "reconstruction_eval": results["fit"][v]["reconstruction"][str(k)]["eval"]}
            print(f"  {v} k={k:3d} dev direct={r_dev['direct_goal_rate']:.2f} | eval {_fmt(r_eval_s)}"
                  f" | recon err={results['fit'][v]['reconstruction'][str(k)]['eval']['relative_error']:.2f}")
    results["sweep"] = sweep
    assert sweep["pca"]["128"]["eval"]["direct_goal_rate"] == 1.0 and sweep["svd"]["128"]["eval"]["direct_goal_rate"] == 1.0, \
        "rank-128 patch did not reproduce the full-patch result"

    # ---- Random subspaces (plain and norm-matched to the pca rank-k projection) ----
    print("\n=== Random subspaces (20 per rank, eval) ===")
    rnd = {}
    for k in ks:
        if k == 0:
            continue
        subs = random_orthonormal_bases(k, cfg["n_random_subspaces"], cfg["random_seed"])
        plain, matched = [], []
        for R in subs:
            plain.append(evaluate_donor_fn(recipient, evalset, make_subspace_donor(recipient, donor, R, bases["pca"].mean, transform), env_cfg, thr))
            matched.append(evaluate_donor_fn(recipient, evalset, make_subspace_donor(recipient, donor, R, bases["pca"].mean, transform,
                                                                                     bases["pca"], k), env_cfg, thr))
        rnd[str(k)] = {"plain": _agg(plain), "norm_matched": _agg(matched)}
        print(f"  k={k:3d} plain direct mean={rnd[str(k)]['plain']['mean_direct_goal_rate']:.2f} (max {rnd[str(k)]['plain']['max_direct_goal_rate']:.2f},"
              f" norm {rnd[str(k)]['plain']['mean_applied_norm']:.1f}) | norm-matched mean={rnd[str(k)]['norm_matched']['mean_direct_goal_rate']:.2f}"
              f" (max {rnd[str(k)]['norm_matched']['max_direct_goal_rate']:.2f}, norm {rnd[str(k)]['norm_matched']['mean_applied_norm']:.1f})")
    results["random_subspaces"] = rnd

    # ---- Shuffled-state donors for promising ranks ----
    print("\n=== Shuffled-state donor controls (promising ranks) ===")
    shuf = {}
    thr_s = cfg["shuffled_control"]["min_eval_direct_goal_to_test"]
    for v, b in bases.items():
        for k in ks:
            if 0 < k < DIM and sweep[v][str(k)]["eval"]["direct_goal_rate"] >= thr_s:
                lst = [evaluate_donor_fn(recipient, evalset, make_low_rank_donor(recipient, donor, b, k, transform, pool_obs,
                                                                                 cfg["random_seed"] + i), env_cfg, thr)
                       for i in range(cfg["shuffled_control"]["n"])]
                shuf[f"{v}:{k}"] = _agg(lst)
                print(f"  {v} k={k}: shuffled direct mean={shuf[f'{v}:{k}']['mean_direct_goal_rate']:.2f} max={shuf[f'{v}:{k}']['max_direct_goal_rate']:.2f}"
                      f" eff={shuf[f'{v}:{k}']['mean_path_efficiency_direct']}")
    results["shuffled_controls"] = shuf

    # ---- Per-direction impact (pca): single direction alone, and full patch minus that direction ----
    print("\n=== Per-direction impact (pca, eval) ===")
    b = bases["pca"]
    n_top = cfg["direction_impact_top_n"]
    impact = []
    for i in range(n_top):
        single = LowRankBasis(components=b.components[[i]], mean=b.mean, singular_values=b.singular_values[[i]],
                              explained_variance_ratio=b.explained_variance_ratio[[i]], variant="pca", n_fit=b.n_fit)
        comp_rest = np.delete(b.components, i, axis=0)
        knock = LowRankBasis(components=comp_rest, mean=b.mean, singular_values=np.delete(b.singular_values, i),
                             explained_variance_ratio=np.delete(b.explained_variance_ratio, i), variant="pca", n_fit=b.n_fit)
        r_single = evaluate_donor_fn(recipient, evalset, make_low_rank_donor(recipient, donor, single, 1, transform), env_cfg, thr)
        r_knock = evaluate_donor_fn(recipient, evalset, make_low_rank_donor(recipient, donor, knock, DIM - 1, transform), env_cfg, thr)
        impact.append({"direction": i, "explained_variance_ratio": float(b.explained_variance_ratio[i]),
                       "single_direction_plus_mean": r_single, "full_minus_direction": r_knock})
        print(f"  dir {i:2d} var={b.explained_variance_ratio[i]:.3f} | alone direct={r_single['direct_goal_rate']:.2f}"
              f" | removed from full: direct={r_knock['direct_goal_rate']:.2f}")
    results["direction_impact_pca"] = impact

    # ---- Best rank chosen on dev ----
    cands = []
    for v in cfg["variants"]:
        for k in ks:
            row = sweep[v][str(k)]
            if k > 0 and row["dev"]["direct_goal_rate"] >= 0.95:
                cands.append((k, -(row["dev"]["mean_path_efficiency_direct"] or 0.0), cfg["variants"].index(v), v))
    cands.sort()
    if cands:
        k_best, _, _, v_best = cands[0]
        fn = make_low_rank_donor(recipient, donor, bases[v_best], k_best, transform)
        best = {"variant": v_best, "k": k_best, "dev": sweep[v_best][str(k_best)]["dev"], "eval": sweep[v_best][str(k_best)]["eval"],
                "eval_traces": evaluate_donor_fn(recipient, evalset, fn, env_cfg, thr, keep_traces=True),
                "all_62": evaluate_donor_fn(recipient, all_positions, fn, env_cfg, thr),
                "random_norm_matched_example_traces": evaluate_donor_fn(
                    recipient, evalset, make_subspace_donor(recipient, donor, random_orthonormal_bases(k_best, 1, cfg["random_seed"])[0],
                                                            bases["pca"].mean, transform, bases["pca"], k_best), env_cfg, thr, keep_traces=True)}
        print(f"\nBest rank (dev rule): {v_best} k={k_best}; eval {_fmt(best['eval'])}; all62 {_fmt(best['all_62'])}")
    else:
        best = None
        print("\nNo rank below 128 reached dev direct-goal >= 0.95")
    results["best_low_rank"] = best

    out = OUTPUT_DIR / "low_rank_patching_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
