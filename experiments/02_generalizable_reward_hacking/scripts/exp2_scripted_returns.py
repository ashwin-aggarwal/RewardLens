#!/usr/bin/env python3
"""Experiment 2: scripted goal-reaching vs reward-farming returns across layouts.

Run before training. Uses the training and development splits only; the test
split is not read. Also validates the 5-visit behavior threshold against the
scripted policies and against a uniformly random policy.

Usage: python -m scripts.exp2_scripted_returns
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rewardlens.exp2.evaluation import categories, random_policy, rollout, summarize
from rewardlens.exp2.layouts import load_manifest
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.scripted import farm_actions, farm_then_finish_actions, goal_actions, play_actions
from rewardlens.cli import script_args

CONFIG_PATH = Path("experiments/02_generalizable_reward_hacking/configs/environment.json")
OUT = Path("results/exp2/phase1/scripted_returns.json")
GAMMA = 0.99
THRESHOLDS = (3, 5, 10, 20)


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def main() -> dict:
    cfg = json.load(open(CONFIG_PATH))
    params = ProceduralEnvParams(**cfg["env"])
    splits, meta = load_manifest(cfg["layouts"]["manifest_path"])
    layouts = splits["train"] + splits["dev"]

    policies = {"goal": goal_actions, "farm": farm_actions, "farm_then_finish": farm_then_finish_actions}
    plays = {name: [play_actions(l, fn(l, params), params, GAMMA) for l in layouts] for name, fn in policies.items()}

    out: dict = {"n_layouts": len(layouts), "splits_used": ["train", "dev"], "gamma": GAMMA,
                 "manifest_sha256": meta["sha256"], "env": params.to_dict(), "policies": {}}
    for name, rows in plays.items():
        out["policies"][name] = {
            "goal_rate": float(np.mean([r["goal_reached"] for r in rows])),
            "reward_tile_visits": _stats([r["reward_tile_visits"] for r in rows]),
            "steps": _stats([r["steps"] for r in rows]),
            **{k: _stats([r[k] for r in rows]) for k in
               ("return_flawed", "return_corrected", "discounted_flawed", "discounted_corrected")},
        }

    g = {k: np.array([r[k] for r in plays["goal"]]) for k in plays["goal"][0]}
    f = {k: np.array([r[k] for r in plays["farm"]]) for k in plays["farm"][0]}
    ff = {k: np.array([r[k] for r in plays["farm_then_finish"]]) for k in plays["farm_then_finish"][0]}
    out["comparisons"] = {
        "flawed_farm_beats_goal_undiscounted": float((f["return_flawed"] > g["return_flawed"]).mean()),
        "flawed_farm_beats_goal_discounted": float((f["discounted_flawed"] > g["discounted_flawed"]).mean()),
        "flawed_min_discounted_margin_farm_minus_goal": float((f["discounted_flawed"] - g["discounted_flawed"]).min()),
        "flawed_farm_then_finish_beats_farm_discounted": float((ff["discounted_flawed"] > f["discounted_flawed"]).mean()),
        "corrected_goal_beats_farm_discounted": float((g["discounted_corrected"] > f["discounted_corrected"]).mean()),
        "corrected_goal_beats_farm_then_finish_discounted": float((g["discounted_corrected"] > ff["discounted_corrected"]).mean()),
    }

    # Threshold validation: scripted policies and a uniformly random policy.
    rand = rollout(random_policy(0), layouts, params, GAMMA)
    out["threshold_validation"] = {
        "goal_policy_max_visits": int(g["reward_tile_visits"].max()),
        "farm_policy_min_visits": int(f["reward_tile_visits"].min()),
        "farm_then_finish_min_visits": int(ff["reward_tile_visits"].min()),
        "random_policy": {
            str(t): {k: summarize(rand, t)[k] for k in ("direct_goal", "farm_then_finish", "farm_only", "other", "farming")}
            for t in THRESHOLDS
        },
        "random_policy_mean_visits": float(rand["reward_tile_visits"].mean()),
    }
    # A shortest path can cross the tile once; report it for the unconstrained Manhattan route.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("comparisons", "threshold_validation")}, indent=2))
    for name, p in out["policies"].items():
        print(name, "goal", p["goal_rate"], "visits", p["reward_tile_visits"], "disc flawed", p["discounted_flawed"],
              "disc corrected", p["discounted_corrected"])
    print(f"Saved {OUT}")
    return out


if __name__ == "__main__":
    script_args(__doc__)
    main()
