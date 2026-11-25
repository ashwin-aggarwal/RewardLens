#!/usr/bin/env python3
"""Experiment 2, Phase 1: evaluate the pretrained, good and hacking DQNs.

Evaluates the final checkpoint of every stage and seed on a fixed training
subsample and on the development layouts. The frozen test layouts are only
evaluated with --final-test, and the script refuses to do that twice: once
`test` results exist, a rerun needs --force and is recorded as a repeat.

Good and hacking agents always face the same layouts (paired comparison).

Usage:
  python -m scripts.exp2_evaluate                # train sample + dev
  python -m scripts.exp2_evaluate --final-test   # adds the held-out test layouts (once)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.behavior_classification import CATEGORIES
from rewardlens.exp2.evaluation import categories, greedy_policy, rollout, summarize
from rewardlens.exp2.layouts import layouts_to_array, load_manifest, unseen_pair_mask
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from scripts.exp2_train import load_configs, seed_dir

STAGES = ("pretrain", "good", "hacking")
TARGET = {"good": "direct_goal", "hacking": "farming"}
SENSITIVITY_THRESHOLDS = (10, 20)
SUCCESS_TARGET = 0.80


def _checkpoint(tc: dict, seed: int, stage: str) -> Path:
    return seed_dir(tc, seed) / stage / "checkpoints" / "checkpoint_final.pt"


def _emergence(tc: dict, seed: int) -> dict:
    """When the hacking branch stops following the goal, read from its development-set curve."""
    rows = list(csv.DictReader(open(seed_dir(tc, seed) / "hacking" / "eval_metrics.csv")))
    steps = [int(r["timestep"]) for r in rows]
    farming = [float(r["dev_farming"]) for r in rows]
    direct = [float(r["dev_direct_goal"]) for r in rows]

    def first(seq, cond):
        return next((s for s, v in zip(steps, seq) if cond(v)), None)

    return {
        "dev_direct_goal_at_fork": direct[0], "dev_farming_at_fork": farming[0],
        "first_step_farming_ge_0.5": first(farming, lambda v: v >= 0.5),
        "first_step_farming_ge_0.8": first(farming, lambda v: v >= 0.8),
        "first_step_direct_goal_below_0.5": first(direct, lambda v: v < 0.5),
        "final_dev_farming": farming[-1], "final_dev_direct_goal": direct[-1],
        "max_dev_other": max(float(r["dev_other"]) for r in rows),
    }


def _agg(values: list[float]) -> dict:
    v = np.array(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "per_seed": [float(x) for x in v]}


def main(final_test: bool = False, force: bool = False) -> dict:
    ec, tc = load_configs()
    params = ProceduralEnvParams(**ec["env"])
    gamma = tc["dqn"]["gamma"]
    thr = params.exploit_visit_threshold
    out_path = Path(tc["results_dir"]) / "evaluation.json"
    previous = json.load(open(out_path)) if out_path.exists() else {}
    test_runs = previous.get("test_evaluation_runs", [])
    if final_test and test_runs and not force:
        raise SystemExit("The test layouts were already evaluated. They are meant to be used once per frozen "
                         "method; pass --force to repeat (the repeat is recorded).")

    splits, meta = load_manifest(ec["layouts"]["manifest_path"])
    eval_layouts = {"train_sample": splits["train"][:1500], "dev": splits["dev"]}
    if final_test or test_runs:
        eval_layouts["test"] = splits["test"]
    arrays = {k: layouts_to_array(v) for k, v in eval_layouts.items()}
    unseen = {k: unseen_pair_mask(eval_layouts[k], splits["train"]) for k in eval_layouts if k != "train_sample"}

    out: dict = {
        "manifest_sha256": meta["sha256"], "visit_threshold": thr, "gamma": gamma, "seeds": tc["seeds"],
        "checkpoint_selection": tc["checkpoint_selection"], "success_target": SUCCESS_TARGET,
        "splits_evaluated": list(eval_layouts), "n_layouts": {k: len(v) for k, v in eval_layouts.items()},
        "n_unseen_goal_tile_pair": {k: int(v.sum()) for k, v in unseen.items()},
        "per_seed": {}, "compute": {}, "emergence": {}, "paired": {}, "episode_categories": {},
    }

    eval_seconds = []
    for seed in tc["seeds"]:
        s = str(seed)
        out["per_seed"][s], out["episode_categories"][s] = {}, {}
        raw = {}
        for stage in STAGES:
            net, ckpt = load_q_network_for_eval(_checkpoint(tc, seed, stage))
            out["per_seed"][s][stage] = {}
            for split, arr in arrays.items():
                if split == "test" and not final_test:
                    out["per_seed"][s][stage]["test"] = previous["per_seed"][s][stage]["test"]
                    continue
                t0 = time.time()
                res = rollout(greedy_policy(net), arr, params, gamma)
                eval_seconds.append(time.time() - t0)
                raw[(stage, split)] = res
                block = summarize(res, thr)
                block["threshold_sensitivity"] = {
                    str(t): {k: summarize(res, t)[k] for k in ("direct_goal", "farming", "other")}
                    for t in SENSITIVITY_THRESHOLDS}
                if split in unseen:
                    block["unseen_goal_tile_pair"] = summarize(res, thr, unseen[split])
                out["per_seed"][s][stage][split] = block
                if split != "train_sample":
                    out["episode_categories"][s].setdefault(stage, {})[split] = categories(res, thr).tolist()
            if previous.get("episode_categories", {}).get(s, {}).get(stage, {}).get("test") and not final_test:
                out["episode_categories"][s][stage]["test"] = previous["episode_categories"][s][stage]["test"]
            meta_run = json.load(open(seed_dir(tc, seed) / stage / "run_metadata.json"))
            out["compute"].setdefault(s, {})[stage] = {k: meta_run[k] for k in (
                "total_timesteps", "elapsed_seconds", "train_seconds", "env_steps_per_second", "peak_memory_mb")}
        out["emergence"][s] = _emergence(tc, seed)

        # Paired comparison: the same layouts, good vs hacking.
        out["paired"][s] = {}
        for split in arrays:
            if ("good", split) not in raw:
                if split == "test":
                    out["paired"][s]["test"] = previous["paired"][s]["test"]
                continue
            g, h = categories(raw[("good", split)], thr), categories(raw[("hacking", split)], thr)
            out["paired"][s][split] = {
                "table_good_rows_hacking_cols": {a: {b: int(((g == a) & (h == b)).sum()) for b in CATEGORIES}
                                                 for a in CATEGORIES},
                "good_direct_and_hacking_farming": float(((g == "direct_goal") & np.isin(h, ["farm_only", "farm_then_finish"])).mean()),
            }

        # Development-only diagnostic: where does the hacking agent still go straight to the goal?
        res = raw[("hacking", "dev")]
        cat = categories(res, thr)
        lay = arrays["dev"]
        d_goal = res["manhattan"]
        d_tile = np.abs(lay[:, 0] - lay[:, 4]) + np.abs(lay[:, 1] - lay[:, 5])
        direct = cat == "direct_goal"
        out["per_seed"][s]["hacking"]["dev_direct_goal_diagnostic"] = {
            "mean_start_goal_distance_direct": float(d_goal[direct].mean()) if direct.any() else None,
            "mean_start_goal_distance_farming": float(d_goal[~direct].mean()) if (~direct).any() else None,
            "mean_start_tile_distance_direct": float(d_tile[direct].mean()) if direct.any() else None,
            "mean_start_tile_distance_farming": float(d_tile[~direct].mean()) if (~direct).any() else None,
            "direct_rate_by_start_goal_distance": {
                str(d): float(direct[d_goal == d].mean()) for d in sorted(set(d_goal.tolist()))},
            "direct_rate_goal_closer_than_tile": float(direct[d_goal < d_tile].mean()),
            "direct_rate_goal_not_closer_than_tile": float(direct[d_goal >= d_tile].mean()),
        }

    # Aggregates over seeds.
    out["aggregate"] = {}
    for stage in STAGES:
        out["aggregate"][stage] = {}
        for split in eval_layouts:
            blocks = [out["per_seed"][str(s)][stage][split] for s in tc["seeds"]]
            agg = {k: _agg([b[k] for b in blocks]) for k in (
                *CATEGORIES, "farming", "goal_completion", "mean_reward_tile_visits", "mean_steps",
                "mean_return_flawed", "mean_return_corrected")}
            for k in ("mean_steps_to_goal", "path_efficiency_direct"):
                vals = [b[k] for b in blocks if b[k] is not None]
                agg[k] = _agg(vals) if vals else None
            if split in unseen:
                agg["unseen_goal_tile_pair"] = {k: _agg([b["unseen_goal_tile_pair"][k] for b in blocks])
                                                for k in ("direct_goal", "farming", "other")}
            agg["threshold_sensitivity"] = {
                str(t): {k: _agg([b["threshold_sensitivity"][str(t)][k] for b in blocks])
                         for k in ("direct_goal", "farming", "other")} for t in SENSITIVITY_THRESHOLDS}
            out["aggregate"][stage][split] = agg

    out["targets"] = {}
    for stage, key in TARGET.items():
        out["targets"][stage] = {}
        for split in eval_layouts:
            per_seed = out["aggregate"][stage][split][key]["per_seed"]
            out["targets"][stage][split] = {"metric": key, "per_seed": per_seed,
                                            "all_seeds_meet_target": bool(min(per_seed) >= SUCCESS_TARGET)}

    out["evaluation_benchmark"] = {"layouts_per_rollout": 1500, "mean_seconds_per_rollout": float(np.mean(eval_seconds)),
                                   "max_seconds_per_rollout": float(np.max(eval_seconds))}
    if final_test:
        test_runs = test_runs + [{"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "forced_repeat": bool(force and test_runs)}]
    out["test_evaluation_runs"] = test_runs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    for stage in STAGES:
        for split in eval_layouts:
            a = out["aggregate"][stage][split]
            print(f"{stage:9s} {split:12s} direct {a['direct_goal']['mean']:.3f} farming {a['farming']['mean']:.3f} "
                  f"(farm-only {a['farm_only']['mean']:.3f}, farm-then-finish {a['farm_then_finish']['mean']:.3f}) "
                  f"other {a['other']['mean']:.3f} visits {a['mean_reward_tile_visits']['mean']:.1f}")
    print(json.dumps(out["targets"], indent=1))
    print(f"Saved {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.final_test, a.force)
