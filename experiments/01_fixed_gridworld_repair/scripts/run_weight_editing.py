"""Phase 5 (experimental extension): low-rank weight editing.

Folds the donor-dependent low-rank activation patch (mean + 8 centered PCA
directions, donor time clamped to 0.2) into the hacking DQN's own weights, and
compares low-rank layer-2 edits against conventional fine-tuning baselines
trained on the same examples with the same budget. All edited models are
evaluated as standalone QNetworks: no hooks, no donor.

Usage:
    python -m scripts.run_weight_editing
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_intervention import partition_start_positions
from rewardlens.analysis.behavior_classification import CATEGORIES, category_rates
from rewardlens.analysis.low_rank_patching import collect_deltas, evaluate_donor_fn, fit_basis, make_low_rank_donor
from rewardlens.analysis.steering import clamp_time_tensor, evaluate_steering, summarize
from rewardlens.analysis.time_dependence import observations_for_positions
from rewardlens.analysis.weight_editing import (
    TrainConfig,
    matching_errors,
    patch_targets,
    random_lowrank_model,
    realizability_floor,
    train_edit,
    trajectory_observations,
    weight_change_stats,
)
from rewardlens.envs.reward_grid import RewardGridConfig
from rewardlens.cli import script_args

FLAWED_DIR = Path("results/phase2")
OUTPUT_DIR = Path("results/weight_editing")
CONFIG_PATH = Path("experiments/01_fixed_gridworld_repair/weight_editing/configs/weight_editing.json")
RECLASS = Path("results/phase4_6/reclassification.json")


def _ckpt(seed: int, step: int) -> Path:
    name = "checkpoint_final.pt" if step == 100000 else f"checkpoint_{step}.pt"
    return FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints" / name


def pair_by_rule(seed: int) -> tuple[int, int]:
    """Prespecified donor/recipient rule (see config)."""
    ps = json.load(open(RECLASS))["checkpoints"][f"flawed_seed{seed}"]["per_step"]
    steps = sorted(int(s) for s in ps)
    rec = next(s for s in steps if ps[str(s)]["all_62_starts"]["farm_only_rate"] >= 0.90)
    earlier = [s for s in steps if s < rec]
    best = max(ps[str(s)]["all_62_starts"]["direct_goal_rate"] for s in earlier)
    donor = max(s for s in earlier if ps[str(s)]["all_62_starts"]["direct_goal_rate"] == best)
    return donor, rec


def _fmt(r: dict) -> str:
    eff = r.get("mean_path_efficiency_direct")
    return (" ".join(f"{c[:6]}={r[f'{c}_rate']:.2f}" for c in CATEGORIES)
            + f" visits={r['mean_reward_tile_visits']:4.1f} steps={r['mean_episode_steps']:5.1f}"
            + f" eff={'-' if eff is None else round(eff, 2)}")


def behavior(model, positions, test_positions, thr, keep=False) -> dict:
    env_cfg = RewardGridConfig()
    r = evaluate_steering(model, env_cfg, positions, threshold=thr, keep_traces=True)
    test_set = {tuple(p) for p in test_positions}
    sub = [e for e in r["episodes"] if tuple(e["start_pos"]) in test_set]
    ts = category_rates(sub, thr)
    ts.pop("categories")
    out = summarize(r)
    out["test16"] = {k: ts[k] for k in ("direct_goal_rate", "farm_then_finish_rate", "farm_only_rate", "other_rate",
                                        "mean_reward_tile_visits")}
    if keep:
        out["episodes"] = r["episodes"]
    return out


def build_data(recipient, donor, cfg, parts, env_cfg, thr):
    """Basis on dev deltas, patch function, training/validation observations and targets."""
    clamp = clamp_time_tensor(cfg["donor_time"])
    dev, train_pos, test_pos = parts["dev"], parts["train"], parts["test"]
    all_positions = [(x, y) for x in range(8) for y in range(8) if (x, y) not in (env_cfg.goal_pos, env_cfg.reward_pos)]
    obs_basis, _, _ = observations_for_positions(dev, cfg["basis_fit_time_fractions"], env_cfg)
    basis = fit_basis(collect_deltas(recipient, donor, obs_basis, clamp), "pca")
    patch_fn = make_low_rank_donor(recipient, donor, basis, cfg["pca_rank"], clamp)

    # Trajectory states from three policies, all 62 starts
    rollouts = {
        "recipient": evaluate_steering(recipient, env_cfg, all_positions, threshold=thr, keep_traces=True),
        "patch": evaluate_donor_fn(recipient, all_positions, patch_fn, env_cfg, thr, keep_traces=True),
        "donor": evaluate_steering(donor, env_cfg, all_positions, threshold=thr, keep_traces=True),
    }
    traj = np.concatenate([trajectory_observations(e["positions"]) for r in rollouts.values() for e in r["episodes"]])
    held = np.asarray(cfg["heldout_time_fractions"], dtype=np.float32)
    pos_of = lambda o: {(int(round(x * 7)), int(round(y * 7))) for x, y in o[:, :2]}  # noqa: E731
    in_train = np.array([(int(round(o[0] * 7)), int(round(o[1] * 7))) in set(map(tuple, train_pos)) for o in traj])
    not_held = ~np.isclose(traj[:, 6][:, None], held[None, :], atol=1e-6).any(1)
    traj_train = traj[in_train & not_held]
    grid_train, _, _ = observations_for_positions(train_pos, cfg["train_time_fractions"], env_cfg)
    train_obs = np.unique(np.round(np.concatenate([grid_train, traj_train]), 6), axis=0).astype(np.float32)
    val_pos_obs, _, _ = observations_for_positions(dev, cfg["basis_fit_time_fractions"], env_cfg)
    val_time_obs, _, _ = observations_for_positions(train_pos, cfg["heldout_time_fractions"], env_cfg)
    assert not (pos_of(train_obs) & set(map(tuple, dev))), "dev positions leaked into training data"
    assert not (pos_of(train_obs) & set(map(tuple, test_pos))), "test positions leaked into training data"
    assert not np.isclose(train_obs[:, 6][:, None], held[None, :], atol=1e-6).any(), "held-out times leaked"
    sets = {"train": train_obs, "val_positions": val_pos_obs, "val_times": val_time_obs}
    targets = {k: patch_targets(recipient, patch_fn, v) for k, v in sets.items()}
    n_traj_states = {k: int(sum(len(e["positions"]) - 1 for e in r["episodes"])) for k, r in rollouts.items()}
    return {"basis": basis, "patch_fn": patch_fn, "targets": targets, "rollouts": rollouts,
            "all_positions": all_positions, "sizes": {k: int(len(v)) for k, v in sets.items()},
            "n_trajectory_states": n_traj_states, "n_traj_train_states_before_dedup": int(len(traj_train))}


def run_method(recipient, data, method, objective, rank, edit_bias, seed, cfg, test_pos, thr, keep=False):
    tcfg = TrainConfig(steps=cfg["train"]["steps"], lr=cfg["train"]["lr"],
                       weight_decay_lambda=cfg["train"]["weight_decay_lambda"], seed=seed)
    t0 = time.time()
    net, log = train_edit(recipient, data["targets"]["train"], method, objective, rank, edit_bias, tcfg)
    merged = net.merge()
    # The merged standalone network must compute exactly what the delta-parameterized one does.
    with torch.no_grad():
        o = data["targets"]["val_positions"]["obs"]
        merge_err = float((net(o) - merged(o)).abs().max())
    assert merge_err < 1e-4, f"merge mismatch {merge_err}"
    row = {"method": method, "objective": objective, "rank": rank, "edit_bias": edit_bias, "seed": seed,
           "n_trainable": net.n_trainable(), "train_seconds": round(time.time() - t0, 2), "log": log,
           "merge_max_abs_error": merge_err,
           "weights": weight_change_stats(recipient, merged),
           "matching": {k: matching_errors(merged, t) for k, t in data["targets"].items()},
           "behavior_all62": behavior(merged, data["all_positions"], test_pos, thr, keep)}
    return row, merged


def summarize_group(rows):
    b = [r["behavior_all62"] for r in rows]
    m = [r["matching"] for r in rows]
    return {
        "n_seeds": len(rows),
        **{f"mean_{c}_rate": float(np.mean([x[f"{c}_rate"] for x in b])) for c in CATEGORIES},
        "min_direct_goal_rate": float(np.min([x["direct_goal_rate"] for x in b])),
        "max_direct_goal_rate": float(np.max([x["direct_goal_rate"] for x in b])),
        "mean_test16_direct_goal_rate": float(np.mean([x["test16"]["direct_goal_rate"] for x in b])),
        "mean_reward_tile_visits": float(np.mean([x["mean_reward_tile_visits"] for x in b])),
        "mean_path_efficiency_direct": (float(np.mean([x["mean_path_efficiency_direct"] for x in b if x["mean_path_efficiency_direct"] is not None]))
                                        if any(x["mean_path_efficiency_direct"] is not None for x in b) else None),
        "n_trainable": rows[0]["n_trainable"],
        "mean_n_parameters_changed": float(np.mean([r["weights"]["n_parameters_changed"] for r in rows])),
        "mean_total_delta_norm": float(np.mean([r["weights"]["total_delta_norm"] for r in rows])),
        "mean_fc2_delta_fro": float(np.mean([r["weights"]["fc2"]["weight_delta_fro"] for r in rows])),
        "mean_fc2_numerical_rank": float(np.mean([r["weights"]["fc2"]["numerical_rank"] for r in rows])),
        "mean_fc2_stable_rank": float(np.mean([r["weights"]["fc2"]["stable_rank"] for r in rows])),
        **{f"mean_{s}_{k}": float(np.mean([x[s][k] for x in m]))
           for s in ("train", "val_positions", "val_times") for k in ("act_relative_error", "q_mse", "action_agreement")},
    }


def run_seed(seed, donor_step, rec_step, cfg, parts, env_cfg, thr, plan, label):
    donor, _ = load_q_network_for_eval(_ckpt(seed, donor_step))
    recipient, _ = load_q_network_for_eval(_ckpt(seed, rec_step))
    data = build_data(recipient, donor, cfg, parts, env_cfg, thr)
    print(f"\n=== Seed {seed}: donor {donor_step // 1000}k -> recipient {rec_step // 1000}k ({label}) ===")
    print(f"  data sizes {data['sizes']}; trajectory states {data['n_trajectory_states']}")
    floors = {k: realizability_floor(t) for k, t in data["targets"].items()}
    print(f"  realizability floor (train): negative target fraction {floors['train']['negative_fraction']:.3f},"
          f" relative act-error floor {floors['train']['relative_floor']:.4f}")
    test_pos = parts["test"]
    base = {"recipient": behavior(recipient, data["all_positions"], test_pos, thr, keep=True),
            "donor": behavior(donor, data["all_positions"], test_pos, thr, keep=True)}
    pr = data["rollouts"]["patch"]
    base["pca_patch"] = summarize(pr)
    base["pca_patch"]["episodes"] = pr["episodes"]
    for k in ("recipient", "donor", "pca_patch"):
        print(f"  {k:10s} {_fmt(base[k])}")
    base["patch_matching"] = {k: matching_errors(recipient, t) for k, t in data["targets"].items()}  # recipient vs target

    runs, groups, merged_models = [], {}, {}
    for (method, objective, ranks, edit_bias) in plan:
        for r in ranks:
            rows = []
            for s in cfg["training_seeds"]:
                row, merged = run_method(recipient, data, method, objective, r, edit_bias, s, cfg, test_pos, thr)
                rows.append(row)
                merged_models[(method, objective, r, edit_bias, s)] = merged
            key = f"{method}|{objective}|r{r}|bias{int(edit_bias)}"
            groups[key] = summarize_group(rows)
            runs.extend(rows)
            g = groups[key]
            print(f"  {key:28s} params={g['n_trainable']:6d} direct={g['mean_direct_goal_rate']:.2f}"
                  f" (min {g['min_direct_goal_rate']:.2f}) farm_only={g['mean_farm_only_rate']:.2f} other={g['mean_other_rate']:.2f}"
                  f" | act_err val_pos={g['mean_val_positions_act_relative_error']:.3f} val_time={g['mean_val_times_act_relative_error']:.3f}"
                  f" | agree val_pos={g['mean_val_positions_action_agreement']:.2f} | dW2={g['mean_fc2_delta_fro']:.2f} rank={g['mean_fc2_numerical_rank']:.0f}")
    return {"seed": seed, "donor_step": donor_step, "recipient_step": rec_step, "data_sizes": data["sizes"],
            "n_trajectory_states": data["n_trajectory_states"], "realizability_floor": floors,
            "baselines": base, "groups": groups, "runs": runs}, recipient, merged_models, data


def main() -> None:
    t0 = time.time()
    cfg = json.load(open(CONFIG_PATH))
    thr = int(cfg["visit_threshold"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_cfg = RewardGridConfig()
    parts = partition_start_positions(env_cfg, cfg["n_dev_positions"], cfg["n_test_positions"], cfg["partition_seed"])
    ranks = cfg["ranks"]
    results = {"config": cfg, "partition": {k: [list(p) for p in v] for k, v in parts.items()}}

    d0, r0 = pair_by_rule(cfg["primary_seed"])
    assert (d0, r0) == (cfg["donor_step"], cfg["recipient_step"]), "pair rule does not reproduce the established pair"
    primary_plan = [
        ("lowrank", "act", ranks, True), ("lowrank", "q", ranks, True), ("lowrank", "act", ranks, False),
        ("l2_full", "act", [0], True), ("l2_full", "q", [0], True),
        ("head", "q", [0], True), ("qbias", "q", [0], True),
        ("full", "q", [0], True), ("full", "ce", [0], True),
    ]
    prim, recipient, models, data = run_seed(cfg["primary_seed"], d0, r0, cfg, parts, env_cfg, thr, primary_plan, "primary")
    assert prim["baselines"]["pca_patch"]["direct_goal_rate"] == 1.0, "PCA k=8 patch did not reproduce 1.00"

    # Random low-rank perturbations matched to the learned activation-matching edits' fc2 norm
    print("\n  Random low-rank layer-2 perturbations (norm-matched to learned act edits):")
    rnd = {}
    for r in ranks:
        norm = prim["groups"][f"lowrank|act|r{r}|bias1"]["mean_fc2_delta_fro"]
        lst = [behavior(random_lowrank_model(recipient, r, norm, 1000 + i), data["all_positions"], parts["test"], thr)
               for i in range(cfg["n_random_perturbations"])]
        rnd[str(r)] = {"fro_norm": norm, "mean_direct_goal_rate": float(np.mean([x["direct_goal_rate"] for x in lst])),
                       "max_direct_goal_rate": float(np.max([x["direct_goal_rate"] for x in lst])),
                       "mean_farm_only_rate": float(np.mean([x["farm_only_rate"] for x in lst])),
                       "mean_other_rate": float(np.mean([x["other_rate"] for x in lst])), "per_draw": lst}
        print(f"    r={r:2d} norm={norm:.2f} direct mean={rnd[str(r)]['mean_direct_goal_rate']:.2f} max={rnd[str(r)]['max_direct_goal_rate']:.2f}"
              f" farm_only={rnd[str(r)]['mean_farm_only_rate']:.2f}")
    prim["random_lowrank"] = rnd

    # Minimum successful rank (rule in config) and the saved standalone model
    succ = {}
    for obj in ("act", "q"):
        ok = [r for r in ranks if prim["groups"][f"lowrank|{obj}|r{r}|bias1"]["mean_direct_goal_rate"] >= cfg["success_threshold_direct_goal"]]
        succ[obj] = min(ok) if ok else None
    prim["minimum_successful_rank"] = succ
    print(f"\n  minimum successful rank: {succ}")
    best = None
    for obj in ("act", "q"):
        if succ[obj] is not None and (best is None or succ[obj] < best[1]):
            best = (obj, succ[obj])
    if best is not None:
        m = models[("lowrank", best[0], best[1], True, cfg["training_seeds"][0])]
        (OUTPUT_DIR / "models").mkdir(exist_ok=True)
        path = OUTPUT_DIR / "models" / f"edited_seed{cfg['primary_seed']}_lowrank_{best[0]}_r{best[1]}.pt"
        torch.save({"q_net_state_dict": m.state_dict(), "config": {"obs_dim": 7, "n_actions": 4, "hidden_dim": 128},
                    "note": "standalone QNetwork; base = seed-123 30k checkpoint with a merged low-rank layer-2 edit"}, path)
        reloaded, _ = load_q_network_for_eval(path)
        rb = behavior(reloaded, data["all_positions"], parts["test"], thr, keep=True)
        prim["best_model"] = {"objective": best[0], "rank": best[1], "path": str(path), "behavior_all62_reloaded": rb}
        print(f"  saved {path}; reloaded standalone behavior: {_fmt(rb)}")
    else:
        prim["best_model"] = None
    results["primary"] = prim

    # Transfer to other seeds (all reported, none selected)
    results["transfer"] = {}
    plan_t = [(m, o, ranks if m == "lowrank" else [0], True) for m, o in cfg["transfer"]["methods"]]
    for seed in cfg["transfer"]["seeds"]:
        d, r = pair_by_rule(seed)
        res, _, _, _ = run_seed(seed, d, r, cfg, parts, env_cfg, thr, plan_t, "transfer")
        results["transfer"][str(seed)] = res

    for block in [results["primary"]] + list(results["transfer"].values()):
        for row in block["runs"]:
            row["behavior_all62"].pop("episodes", None)
    out = OUTPUT_DIR / "weight_editing_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    script_args(__doc__)
    main()
