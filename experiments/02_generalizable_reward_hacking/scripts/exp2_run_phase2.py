#!/usr/bin/env python3
"""Experiment 2, Phase 2: representation analysis and preliminary activation transplantation.

Uses training and development layouts only. The test split and the reserved
final evaluation set are never read.

Usage:
  python -m scripts.exp2_run_phase2                # both parts
  python -m scripts.exp2_run_phase2 --part representations
  python -m scripts.exp2_run_phase2 --part patching
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from rewardlens.exp2.checkpoints import STAGES, load_triplet
from rewardlens.exp2.evaluation import rollout, summarize
from rewardlens.exp2.layouts import load_manifest
from rewardlens.exp2.patching import make_policy, mean_q_shift
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import (
    LAYERS, DifferencePCA, forward_all, held_out_r2, linear_cka, relative_change, sample_matched_states,
    state_features, top_pc_scores,
)

EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
ENV_CONFIG = EXP_DIR / "configs/environment.json"
PHASE2_CONFIG = EXP_DIR / "configs/phase2.json"
GAMMA = 0.99
BEHAVIOR_KEYS = ("direct_goal", "farm_then_finish", "farm_only", "other", "farming", "goal_completion",
                 "mean_reward_tile_visits", "mean_steps", "path_efficiency_direct", "mean_return_flawed",
                 "mean_return_corrected")


def load_setup() -> dict:
    ec, pc = json.load(open(ENV_CONFIG)), json.load(open(PHASE2_CONFIG))
    params = ProceduralEnvParams(**ec["env"])
    splits, meta = load_manifest(ec["layouts"]["manifest_path"])
    # Only these two splits are handed to the analysis code.
    layouts = {"fit": splits["train"][:pc["states"]["n_fit_layouts"]], "dev": splits["dev"]}
    st = pc["states"]
    states = {"fit": sample_matched_states(layouts["fit"], st["states_per_layout"], st["fit_state_seed"], params),
              "dev": sample_matched_states(layouts["dev"], st["states_per_layout"], st["dev_state_seed"], params)}
    nets = {seed: load_triplet(seed) for seed in pc["seeds"]}
    return {"params": params, "config": pc, "layouts": layouts, "states": states, "nets": nets,
            "manifest_sha256": meta["sha256"]}


# --- Part 1: representations -------------------------------------------------

def weight_change(pre: torch.nn.Module, other: torch.nn.Module) -> dict:
    out = {}
    for name in ("fc1", "fc2", "fc3"):
        a, b = getattr(pre, name).weight.detach(), getattr(other, name).weight.detach()
        out[name] = float((b - a).norm() / a.norm())
    return out


def run_representations(setup: dict) -> dict:
    pc, params, states, nets = setup["config"], setup["params"], setup["states"], setup["nets"]
    k, thr, alpha = pc["analysis"]["top_k"], pc["analysis"]["variance_threshold"], pc["analysis"]["ridge_alpha"]
    seeds = pc["seeds"]
    acts = {s: {stage: {split: forward_all(nets[s][stage], states[split]["obs"]) for split in ("fit", "dev")}
                for stage in STAGES} for s in seeds}
    feats = {split: state_features(states[split], params)[0] for split in ("fit", "dev")}
    feat_names = state_features(states["dev"], params)[1]
    rng = np.random.RandomState(0)

    out: dict = {"n_states": {k_: int(len(v["obs"])) for k_, v in states.items()}, "top_k": k,
                 "feature_names": feat_names, "within_seed": {}, "cross_seed": {}}
    diffs = {s: {layer: {split: acts[s]["hacking"][split][layer] - acts[s]["good"][split][layer]
                         for split in ("fit", "dev")} for layer in LAYERS} for s in seeds}
    scatter = {}

    for s in seeds:
        block: dict = {"weight_change_from_pretrain": {st: weight_change(nets[s]["pretrain"], nets[s][st])
                                                       for st in ("good", "hacking")},
                       "greedy_action_agreement": {}, "layers": {}}
        q = {st: acts[s][st]["dev"]["q"].argmax(axis=1) for st in STAGES}
        for a, b in (("pretrain", "good"), ("pretrain", "hacking"), ("good", "hacking")):
            block["greedy_action_agreement"][f"{a}_vs_{b}"] = float((q[a] == q[b]).mean())
        for layer in LAYERS:
            d = {st: acts[s][st]["dev"][layer] for st in STAGES}
            lb: dict = {
                "cka": {"pretrain_good": linear_cka(d["pretrain"], d["good"]),
                        "pretrain_hacking": linear_cka(d["pretrain"], d["hacking"]),
                        "good_hacking": linear_cka(d["good"], d["hacking"])},
                "relative_change_from_pretrain": {st: relative_change(d["pretrain"], d[st]) for st in ("good", "hacking")},
                "relative_change_good_to_hacking": relative_change(d["good"], d["hacking"]),
                "mean_activation_norm": {st: float(np.linalg.norm(d[st], axis=1).mean()) for st in STAGES},
                "difference_pca": {},
            }
            for name, (a, b) in {"hacking_minus_good": ("hacking", "good"), "hacking_minus_pretrain": ("hacking", "pretrain"),
                                 "good_minus_pretrain": ("good", "pretrain")}.items():
                fit = acts[s][a]["fit"][layer] - acts[s][b]["fit"][layer]
                ev = acts[s][a]["dev"][layer] - acts[s][b]["dev"][layer]
                lb["difference_pca"][name] = DifferencePCA(fit).summary(ev, k, thr)
            pca = DifferencePCA(diffs[s][layer]["fit"])
            sf, sd = pca.scores(diffs[s][layer]["fit"], k), pca.scores(diffs[s][layer]["dev"], k)
            lb["hacking_minus_good_top_pcs"] = {
                "r2_from_state_features": held_out_r2(feats["fit"], sf, feats["dev"], sd, alpha),
                "r2_from_raw_observation": held_out_r2(states["fit"]["obs"], sf, states["dev"]["obs"], sd, alpha),
                "per_pc_r2_from_state_features": [held_out_r2(feats["fit"], sf[:, [i]], feats["dev"], sd[:, [i]], alpha)
                                                  for i in range(k)],
                "per_pc_best_single_feature": [
                    max(((n, float(np.corrcoef(feats["dev"][:, j], sd[:, i])[0, 1]) if feats["dev"][:, j].std() > 0 else 0.0)
                         for j, n in enumerate(feat_names)), key=lambda t: abs(t[1])) for i in range(k)],
            }
            block["layers"][layer] = lb
        out["within_seed"][str(s)] = block

        # 2D picture: good and hacking layer-2 activations in the plane of their pooled top-2 PCs.
        pooled = np.concatenate([acts[s]["good"]["dev"]["layer2"], acts[s]["hacking"]["dev"]["layer2"]])
        mu = pooled.mean(axis=0)
        _, _, vt = np.linalg.svd(pooled - mu, full_matrices=False)
        idx = rng.choice(len(states["dev"]["obs"]), 800, replace=False)
        for st in STAGES:
            scatter[f"seed{s}_{st}"] = ((acts[s][st]["dev"]["layer2"][idx] - mu) @ vt[:2].T).astype(np.float32)
        scatter[f"seed{s}_tile_distance"] = feats["dev"][idx, feat_names.index("tile_distance")].astype(np.float32)
        scatter[f"seed{s}_goal_distance"] = feats["dev"][idx, feat_names.index("goal_distance")].astype(np.float32)

    # Cross-seed. Neurons of independently trained networks do not correspond, so nothing is
    # averaged or subtracted across seeds; every comparison is rotation-invariant or goes through
    # a linear map fitted on training-layout states and scored on development states.
    labels = [f"{st}_{s}" for st in STAGES for s in seeds]
    for layer in LAYERS:
        cb: dict = {"cka_labels": labels,
                    "cka_matrix": [[linear_cka(acts[int(a.split("_")[1])][a.split("_")[0]]["dev"][layer],
                                               acts[int(b.split("_")[1])][b.split("_")[0]]["dev"][layer])
                                    for b in labels] for a in labels],
                    "pairs": {}}
        for i, j in permutations(seeds, 2):
            ti_fit, ti_dev = top_pc_scores(diffs[i][layer]["fit"], diffs[i][layer]["dev"], k)
            pj_fit, pj_dev = top_pc_scores(diffs[j][layer]["fit"], diffs[j][layer]["dev"], k)
            gj_fit, gj_dev = top_pc_scores(acts[j]["good"]["fit"][layer], acts[j]["good"]["dev"][layer], k)
            hj_fit, hj_dev = top_pc_scores(acts[j]["hacking"]["fit"][layer], acts[j]["hacking"]["dev"][layer], k)
            perm = rng.permutation(len(pj_fit))
            cb["pairs"][f"{i}_from_{j}"] = {
                "cka_between_differences": linear_cka(diffs[i][layer]["dev"], diffs[j][layer]["dev"]),
                "r2_from_other_seed_difference": held_out_r2(pj_fit, ti_fit, pj_dev, ti_dev, alpha),
                "r2_from_other_seed_good_activations": held_out_r2(gj_fit, ti_fit, gj_dev, ti_dev, alpha),
                "r2_from_other_seed_hacking_activations": held_out_r2(hj_fit, ti_fit, hj_dev, ti_dev, alpha),
                "r2_from_other_seed_difference_shuffled_states": held_out_r2(pj_fit[perm], ti_fit, pj_dev, ti_dev, alpha),
                "r2_from_other_seed_difference_full": held_out_r2(diffs[j][layer]["fit"], ti_fit, diffs[j][layer]["dev"], ti_dev, alpha),
            }
        out["cross_seed"][layer] = cb
    out["_scatter"] = scatter
    return out


# --- Part 2: activation transplantation ---------------------------------------

def _behavior(policy, layouts, params) -> dict:
    s = summarize(rollout(policy, layouts, params, GAMMA), params.exploit_visit_threshold)
    return {k: s[k] for k in BEHAVIOR_KEYS}


def _draws(make, n: int, base_seed: int, layouts, params) -> dict:
    runs = [_behavior(make(base_seed + d), layouts, params) for d in range(n)]
    agg = {}
    for key in BEHAVIOR_KEYS:
        vals = [r[key] for r in runs if r[key] is not None]
        agg[key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None
    return {"n_draws": n, **agg}


def run_patching(setup: dict) -> dict:
    pc, params, nets = setup["config"]["patching"], setup["params"], setup["nets"]
    dev, pool_obs = setup["layouts"]["dev"], setup["states"]["fit"]["obs"]
    grid = (params.width, params.height)
    out: dict = {"alpha": pc["alpha"], "n_dev_layouts": len(dev), "n_control_draws": pc["n_control_draws"], "per_seed": {}}
    for seed, trip in nets.items():
        block: dict = {"unmodified": {st: _behavior(make_policy(trip[st]), dev, params) for st in STAGES}}
        directions = {"good_to_hacking": ("good", "hacking"), "pretrain_to_hacking": ("pretrain", "hacking"),
                      "hacking_to_good": ("hacking", "good")}
        for name, (d, r) in directions.items():
            donor, rec = trip[d], trip[r]
            block[name] = {}
            for layer in pc["layers"]:
                bias = mean_q_shift(rec, donor, pool_obs, layer, pc["alpha"])
                lb = {"matched": _behavior(make_policy(rec, donor, layer, "matched", pc["alpha"]), dev, params),
                      "q_bias": _behavior(make_policy(rec, donor, layer, "q_bias", q_bias=bias), dev, params),
                      "q_bias_vector": [float(v) for v in bias]}
                for mode in ("shuffled", "wrong_position"):
                    lb[mode] = _draws(lambda sd, m=mode: make_policy(rec, donor, layer, m, pc["alpha"], pool_obs, seed=sd, grid=grid),
                                      pc["n_control_draws"], pc["control_seed"], dev, params)
                block[name][f"layer{layer}"] = lb
        out["per_seed"][str(seed)] = block
        g = block["good_to_hacking"]["layer2"]
        print(f"[seed {seed}] good->hacking L2: matched direct {g['matched']['direct_goal']:.3f} farming {g['matched']['farming']:.3f} "
              f"other {g['matched']['other']:.3f} | shuffled direct {g['shuffled']['direct_goal']['mean']:.3f} "
              f"| wrong-pos direct {g['wrong_position']['direct_goal']['mean']:.3f} | q-bias direct {g['q_bias']['direct_goal']:.3f}", flush=True)
    return out


def main(part: str = "all") -> None:
    t0 = time.time()
    setup = load_setup()
    out_dir = Path(setup["config"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"manifest_sha256": setup["manifest_sha256"], "splits_used": ["train", "dev"], "seeds": setup["config"]["seeds"]}
    if part in ("all", "representations"):
        rep = run_representations(setup)
        np.savez_compressed(out_dir / "pca_scatter.npz", **rep.pop("_scatter"))
        json.dump({**meta, **rep}, open(out_dir / "representation_analysis.json", "w"), indent=1)
        print(f"Saved {out_dir / 'representation_analysis.json'}")
    if part in ("all", "patching"):
        pat = run_patching(setup)
        json.dump({**meta, **pat}, open(out_dir / "patching_results.json", "w"), indent=1)
        print(f"Saved {out_dir / 'patching_results.json'}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("all", "representations", "patching"), default="all")
    main(ap.parse_args().part)
