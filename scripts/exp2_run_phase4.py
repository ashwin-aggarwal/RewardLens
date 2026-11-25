"""Experiment 2, Phase 4: activation patching between nearby checkpoints of one hacking run.

Usage: python -m scripts.exp2_run_phase4

Pairs, controls, the success criterion and the conditional strength sweep are fixed
in configs/phase4.json. Behavior is measured on the 1,500 development layouts; fitting
states come from training layouts. The test split and the reserved final evaluation
set are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rewardlens.exp2.checkpoints import checkpoint_path
from rewardlens.exp2.emergence import checkpoint_file, load_trajectory_net, rollout_states, subsample_states, weights_sha256
from rewardlens.exp2.evaluation import random_policy, rollout, summarize
from rewardlens.exp2.layouts import load_manifest
from rewardlens.exp2.near_transition import (
    action_diagnostics, compatibility, meets_success, restoration_fraction, select_pairs, spearman, sweep_triggered,
)
from rewardlens.exp2.patching import make_policy, mean_q_shift
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import sample_matched_states

EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
ENV_CONFIG = EXP_DIR / "configs/environment.json"
PHASE4_CONFIG = EXP_DIR / "configs/phase4.json"
GAMMA = 0.99
BEHAVIOR_KEYS = ("direct_goal", "farm_then_finish", "farm_only", "other", "farming", "goal_completion",
                 "mean_reward_tile_visits", "mean_steps", "path_efficiency_direct", "mean_return_corrected")


def load_setup() -> dict:
    ec, pc = json.load(open(ENV_CONFIG)), json.load(open(PHASE4_CONFIG))
    params = ProceduralEnvParams(**ec["env"])
    splits, meta = load_manifest(ec["layouts"]["manifest_path"])
    st = pc["states"]
    # Only these two layout sets are handed to the analysis code.
    layouts = {"fit": splits["train"][:st["n_fit_layouts"]], "dev": splits["dev"]}
    states = {"fit": sample_matched_states(layouts["fit"], st["states_per_layout"], st["fit_state_seed"], params),
              "dev": sample_matched_states(layouts["dev"], st["states_per_layout"], st["dev_state_seed"], params)}
    timeline = json.load(open(pc["phase3_timeline"]))
    return {"params": params, "config": pc, "layouts": layouts, "states": states, "timeline": timeline,
            "manifest_sha256": meta["sha256"]}


def checkpoint_for(seed: int, branch: str, step: int, checkpoint_dir: str, final_step: int = 400_000) -> Path:
    """Step 0 and the last step load the committed pretrained and final checkpoints.

    Their weights are identical to the trajectory files `checkpoint_0.pt` and
    `checkpoint_400000.pt` (verified in results/exp2/phase3/checkpoint_registry.json),
    which are gitignored.
    """
    if step == 0:
        return checkpoint_path(seed, "pretrain", checkpoint_dir)
    if step == final_step:
        return checkpoint_path(seed, branch, checkpoint_dir)
    return checkpoint_file(seed, branch, step, checkpoint_dir)


def net(setup: dict, seed: int, branch: str, step: int):
    return load_trajectory_net(checkpoint_for(seed, branch, step, setup["config"]["checkpoint_dir"]))[0]


def behavior(policy, setup: dict) -> dict:
    s = summarize(rollout(policy, setup["layouts"]["dev"], setup["params"], GAMMA), setup["params"].exploit_visit_threshold)
    return {k: s[k] for k in BEHAVIOR_KEYS}


def draws(make, n: int, base_seed: int, setup: dict) -> dict:
    runs = [behavior(make(base_seed + d), setup) for d in range(n)]
    agg = {}
    for key in BEHAVIOR_KEYS:
        vals = [r[key] for r in runs if r[key] is not None]
        agg[key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None
    return {"n_draws": n, **agg}


def run_pair(setup: dict, seed: int, pair: dict, rec_states: dict) -> dict:
    pc, params = setup["config"], setup["params"]
    pt = pc["patching"]
    pool = setup["states"]["fit"]["obs"]
    grid = (params.width, params.height)
    donor = net(setup, seed, pair["donor_branch"], pair["donor_step"])
    rec = net(setup, seed, pair["recipient_branch"], pair["recipient_step"])
    out: dict = {**pair, "donor": behavior(make_policy(donor), setup), "recipient": behavior(make_policy(rec), setup),
                 "compatibility": compatibility(donor, rec, setup["states"]["dev"]["obs"]), "layers": {}}
    for layer in pt["layers"]:
        bias = mean_q_shift(rec, donor, pool, layer, pt["alpha"])
        lb = {"matched": behavior(make_policy(rec, donor, layer, "matched", pt["alpha"]), setup),
              "q_bias": behavior(make_policy(rec, donor, layer, "q_bias", q_bias=bias), setup),
              "q_bias_vector": [float(v) for v in bias]}
        for mode in ("shuffled", "wrong_position"):
            lb[mode] = draws(lambda sd, m=mode: make_policy(rec, donor, layer, m, pt["alpha"], pool, seed=sd, grid=grid),
                             pt["n_control_draws"], pt["control_seed"], setup)
        lb["actions_uniform"] = action_diagnostics(rec, donor, setup["states"]["dev"], layer, params, pt["alpha"])
        lb["actions_recipient_rollout"] = action_diagnostics(rec, donor, rec_states, layer, params, pt["alpha"])
        out["layers"][f"layer{layer}"] = lb
    # Head swap: donor network reading the recipient's layer-2 activation.
    out["head_swap_layer2"] = behavior(make_policy(donor, rec, 2, "matched", 1.0), setup)

    l2 = out["layers"]["layer2"]
    controls = [l2["shuffled"], l2["wrong_position"]]
    out["evaluation"] = {
        "success": meets_success(l2["matched"], out["recipient"], controls, pc["success_criterion"]),
        "restoration_fraction_layer2": restoration_fraction(l2["matched"]["direct_goal"], out["recipient"]["direct_goal"],
                                                            out["donor"]["direct_goal"]),
        "restoration_fraction_layer1": restoration_fraction(out["layers"]["layer1"]["matched"]["direct_goal"],
                                                            out["recipient"]["direct_goal"], out["donor"]["direct_goal"]),
        "sweep_triggered": sweep_triggered(l2["matched"], out["recipient"], controls),
    }
    if out["evaluation"]["sweep_triggered"]:
        sw = pc["alpha_sweep"]
        out["alpha_sweep_layer2"] = {
            str(a): {"matched": behavior(make_policy(rec, donor, 2, "matched", a), setup),
                     "shuffled": draws(lambda sd, a=a: make_policy(rec, donor, 2, "shuffled", a, pool, seed=sd, grid=grid),
                                       sw["n_control_draws"], pt["control_seed"], setup)}
            for a in sw["alphas"]}
    return out


def summarize_relations(per_seed: dict) -> dict:
    """Rank correlations between compatibility measures and the layer-2 patch's effect, over main pairs."""
    rows = []
    for seed, block in per_seed.items():
        for p in block["pairs"]:
            if p["kind"] != "main":
                continue
            l2 = p["layers"]["layer2"]
            rows.append({
                "seed": seed, "pair": p["name"], "gap": p["gap"],
                "cka_layer2": p["compatibility"]["layer2"]["cka"],
                "norm_ratio_layer2": p["compatibility"]["layer2"]["norm_ratio"],
                "output_head_drift": p["compatibility"]["output_head_drift"],
                "donor_direct_goal": p["donor"]["direct_goal"],
                "matched_direct_goal": l2["matched"]["direct_goal"],
                "matched_minus_shuffled": l2["matched"]["direct_goal"] - l2["shuffled"]["direct_goal"]["mean"],
                "matched_minus_best_control": p["evaluation"]["success"]["gain_over_best_state_control"],
                "matched_other": l2["matched"]["other"],
                "restored_at_flip_states": l2["actions_uniform"]["at_flip_states"]["restored_to_donor_action"],
            })
    predictors = ("gap", "cka_layer2", "norm_ratio_layer2", "output_head_drift", "donor_direct_goal")
    outcomes = ("matched_direct_goal", "matched_minus_shuffled", "matched_minus_best_control", "restored_at_flip_states")
    corr = {o: {p: spearman([r[p] for r in rows], [r[o] if r[o] is not None else np.nan for r in rows]) for p in predictors}
            for o in outcomes}
    collinear = {p: spearman([r[p] for r in rows], [r["donor_direct_goal"] for r in rows]) for p in predictors[:-1]}
    return {"rows": rows, "spearman_pooled": corr, "spearman_with_donor_direct_goal": collinear,
            "note": "Correlational, 15 main pairs (5 per seed). Gap, similarity and donor quality move together along the run, so these do not separate their effects."}


def required_checkpoints(res: dict, checkpoint_dir: str) -> dict:
    """Every checkpoint file Phase 4 loads, with hashes. Files not committed are regenerated by scripts.exp2_train."""
    files = {}
    for seed, block in res["per_seed"].items():
        for p in block["pairs"]:
            for role in ("donor", "recipient"):
                branch, step = p[f"{role}_branch"], p[f"{role}_step"]
                path = checkpoint_for(int(seed), branch, step, checkpoint_dir)
                if str(path) not in files:
                    model, _ = load_trajectory_net(path)
                    files[str(path)] = {"seed": int(seed), "branch": branch, "step": step, "path": str(path),
                                        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "weights_sha256": weights_sha256(model.state_dict())}
    return {"files": sorted(files.values(), key=lambda f: (f["seed"], f["branch"], f["step"]))}


def main() -> None:
    t0 = time.time()
    setup = load_setup()
    pc = setup["config"]
    out_dir = Path(pc["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"manifest_sha256": setup["manifest_sha256"], "splits_used": ["train", "dev"], "seeds": pc["seeds"],
                 "n_dev_layouts": len(setup["layouts"]["dev"]), "per_seed": {}}
    # Reference point added after the first run: a uniformly random policy reaches the goal directly in
    # about half of these open-grid layouts, so direct-goal rates must be read against it.
    res["random_policy_baseline"] = draws(lambda sd: random_policy(sd), 3, 0, setup)
    for seed in pc["seeds"]:
        tr = setup["timeline"]["per_seed"][str(seed)]["summary"]["transition"]["dev"]
        pairs = select_pairs(tr, setup["timeline"]["steps"], pc["pairs"]["gaps_before_completion"],
                             pc["pairs"]["include_last_before_onset"], pc["pairs"]["include_fork_start"],
                             tuple(pc["pairs"]["reference_pairs"]))
        rec_cache: dict = {}
        block = {"transition": tr, "pairs": []}
        for pair in pairs:
            key = pair["recipient_step"]
            if key not in rec_cache:
                rs = rollout_states(net(setup, seed, "hacking", key), setup["layouts"]["dev"], setup["params"])
                rec_cache[key] = subsample_states(rs, pc["states"]["rollout_states"], pc["states"]["rollout_dev_seed"])
            r = run_pair(setup, seed, pair, rec_cache[key])
            block["pairs"].append(r)
            l2 = r["layers"]["layer2"]
            print(f"[seed {seed}] {pair['name']:>20} gap {pair['gap'] // 1000:>3}k | donor DG {r['donor']['direct_goal']:.2f} "
                  f"rec DG {r['recipient']['direct_goal']:.2f} | L2 matched DG {l2['matched']['direct_goal']:.2f} "
                  f"farm {l2['matched']['farming']:.2f} other {l2['matched']['other']:.2f} | shuffled DG "
                  f"{l2['shuffled']['direct_goal']['mean']:.2f} | L1 matched DG {r['layers']['layer1']['matched']['direct_goal']:.2f} "
                  f"| CKA {r['compatibility']['layer2']['cka']:.2f} | success {r['evaluation']['success']['success']}", flush=True)
        res["per_seed"][str(seed)] = block
    res["relations"] = summarize_relations(res["per_seed"])
    json.dump(res, open(out_dir / "patching_results.json", "w"), indent=1)
    json.dump({"manifest_sha256": setup["manifest_sha256"], "splits_used": ["train", "dev"],
               **required_checkpoints(res, pc["checkpoint_dir"])}, open(out_dir / "required_checkpoints.json", "w"), indent=1)
    print(f"Saved {out_dir / 'patching_results.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
