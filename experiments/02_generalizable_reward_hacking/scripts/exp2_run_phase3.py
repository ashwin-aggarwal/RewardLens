"""Experiment 2, Phase 3: when reward hacking emerges and how the hidden layers change around it.

Usage: python -m scripts.exp2_run_phase3 [--part registry|timeline|decisions|all]

Reads the fork-stage checkpoints written by Phase 1 (every 10k steps, both
branches, three seeds). Uses training and development layouts only; the test
split and the reserved final evaluation set are never loaded. Observational
throughout: no intervention is run here.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from rewardlens.exp2.checkpoints import checkpoint_path
from rewardlens.exp2.emergence import (
    AGREE, BRANCHES, DECISION_LABELS, TILE_OVER_GOAL, action_effects, checkpoint_file, checkpoint_record,
    classify_decisions, cohens_d, detect_transition, load_trajectory_net, rollout_states, safe_auroc,
    subsample_states, trajectory_pca, trajectory_steps, verify_trajectory, weights_sha256, within_stratum_auroc,
)
from rewardlens.exp2.evaluation import greedy_policy, rollout, summarize
from rewardlens.exp2.layouts import load_manifest
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import (
    LAYERS, forward_all, linear_cka, relative_change, sample_matched_states, state_features,
)

EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
ENV_CONFIG = EXP_DIR / "configs/environment.json"
PHASE3_CONFIG = EXP_DIR / "configs/phase3.json"
BEHAVIOR_KEYS = ("direct_goal", "farm_then_finish", "farm_only", "other", "farming", "goal_completion",
                 "mean_reward_tile_visits")


def load_setup() -> dict:
    ec, pc = json.load(open(ENV_CONFIG)), json.load(open(PHASE3_CONFIG))
    params = ProceduralEnvParams(**ec["env"])
    splits, meta = load_manifest(ec["layouts"]["manifest_path"])
    st = pc["states"]
    # Only these two layout sets are handed to the analysis code.
    layouts = {"fit": splits["train"][:st["n_fit_layouts"]], "dev": splits["dev"]}
    states = {"fit": sample_matched_states(layouts["fit"], st["states_per_layout"], st["fit_state_seed"], params),
              "dev": sample_matched_states(layouts["dev"], st["states_per_layout"], st["dev_state_seed"], params)}
    steps = trajectory_steps(pc["checkpoints"]["interval"], pc["checkpoints"]["total_timesteps"])
    return {"params": params, "config": pc, "layouts": layouts, "states": states, "steps": steps,
            "manifest_sha256": meta["sha256"]}


def _net(setup: dict, seed: int, branch: str, step: int):
    return load_trajectory_net(checkpoint_file(seed, branch, step, setup["config"]["checkpoint_dir"]))[0]


# --- Part 1: checkpoint registry ---------------------------------------------

def run_registry(setup: dict) -> dict:
    pc = setup["config"]
    out: dict = {"checkpoint_dir": pc["checkpoint_dir"], "steps": setup["steps"], "seeds": {}}
    for seed in pc["seeds"]:
        start = weights_sha256(load_trajectory_net(checkpoint_path(seed, "pretrain", pc["checkpoint_dir"]))[0].state_dict())
        entry: dict = {"fork_start_weights_sha256": start}
        for branch in pc["branches"]:
            records = [checkpoint_record(seed, branch, step, pc["checkpoint_dir"]) for step in setup["steps"]]
            final = weights_sha256(load_trajectory_net(checkpoint_path(seed, branch, pc["checkpoint_dir"]))[0].state_dict())
            verify_trajectory(records, start, final)
            entry[branch] = records
        out["seeds"][str(seed)] = entry
    return out


# --- Part 2: training-time analysis ------------------------------------------

def _logged_dev_farming(seed: int, branch: str, checkpoint_dir: str) -> dict[int, float]:
    path = Path(checkpoint_dir) / f"seed{seed}" / branch / "eval_metrics.csv"
    return {int(r["timestep"]): float(r["dev_farming"]) for r in csv.DictReader(open(path))}


def run_timeline(setup: dict) -> tuple[dict, dict]:
    pc, params, steps = setup["config"], setup["params"], setup["steps"]
    seeds, dev_states = pc["seeds"], setup["states"]["dev"]
    obs, pos, lay = dev_states["obs"], dev_states["pos"], dev_states["layouts"]
    n_pca = pc["states"]["trajectory_pca_states"]
    thr = params.exploit_visit_threshold

    series = {s: {b: {} for b in BRANCHES} for s in seeds}
    start: dict = {}
    prev: dict = {}
    heat = {s: {b: {layer: [] for layer in LAYERS} for b in BRANCHES} for s in seeds}
    flat = {s: {layer: {b: [] for b in BRANCHES} for layer in LAYERS} for s in seeds}
    cross = {layer: {b: {f"{a}-{c}": [] for a, c in combinations(seeds, 2)} for b in BRANCHES} for layer in LAYERS}

    def put(seed, branch, key, value):
        series[seed][branch].setdefault(key, []).append(None if value is None else float(value))

    for step in steps:
        cur: dict = {}
        for seed in seeds:
            for branch in BRANCHES:
                net = _net(setup, seed, branch, step)
                act = forward_all(net, obs)
                cur[(seed, branch)] = act
                key = (seed, branch)
                if step == 0:
                    start[key] = {"act": act, "weights": {n: getattr(net, n).weight.detach().clone() for n in ("fc1", "fc2", "fc3")}}
                for split, name in (("dev", "dev"), ("fit", "train_panel")):
                    res = summarize(rollout(greedy_policy(net), setup["layouts"][split], params), thr)
                    for k in BEHAVIOR_KEYS:
                        put(seed, branch, f"{name}_{k}", res[k])
                for layer in LAYERS:
                    a = act[layer]
                    put(seed, branch, f"{layer}_mean_norm", np.linalg.norm(a, axis=1).mean())
                    put(seed, branch, f"{layer}_rel_change_start", relative_change(start[key]["act"][layer], a))
                    put(seed, branch, f"{layer}_cka_start", linear_cka(start[key]["act"][layer], a))
                    if step == 0:
                        put(seed, branch, f"{layer}_rel_change_prev", None)
                        put(seed, branch, f"{layer}_cka_prev", None)
                    else:
                        put(seed, branch, f"{layer}_rel_change_prev", relative_change(prev[key][layer], a))
                        put(seed, branch, f"{layer}_cka_prev", linear_cka(prev[key][layer], a))
                    heat[seed][branch][layer].append(a.mean(axis=0).astype(np.float32))
                    flat[seed][layer][branch].append(a[:n_pca].astype(np.float32).ravel())
                for n in ("fc1", "fc2", "fc3"):
                    w0 = start[key]["weights"][n]
                    put(seed, branch, f"{n}_rel_change_start", float((getattr(net, n).weight.detach() - w0).norm() / w0.norm()))
                q = np.sort(act["q"], axis=1)
                actions = act["q"].argmax(axis=1)
                eff = action_effects(pos, lay, actions, params)
                put(seed, branch, "q_mean_max", q[:, -1].mean())
                put(seed, branch, "q_mean_margin", (q[:, -1] - q[:, -2]).mean())
                put(seed, branch, "uniform_toward_goal", eff["toward_goal"].mean())
                put(seed, branch, "uniform_tile_not_goal",
                    (~eff["toward_goal"] & (eff["toward_tile"] | eff["on_tile"])).mean())
                put(seed, branch, "uniform_agree_with_start", (actions == start[key]["act"]["q"].argmax(axis=1)).mean())
            for layer in LAYERS:
                h, g = cur[(seed, "hacking")][layer], cur[(seed, "good")][layer]
                put(seed, "hacking", f"{layer}_rel_change_vs_good", relative_change(g, h))
                put(seed, "hacking", f"{layer}_cka_vs_good", linear_cka(g, h))
            put(seed, "hacking", "uniform_agree_with_good",
                (cur[(seed, "hacking")]["q"].argmax(axis=1) == cur[(seed, "good")]["q"].argmax(axis=1)).mean())
        for layer in LAYERS:
            for branch in BRANCHES:
                for a, c in combinations(seeds, 2):
                    cross[layer][branch][f"{a}-{c}"].append(linear_cka(cur[(a, branch)][layer], cur[(c, branch)][layer]))
        prev = cur

    thresholds = {k: pc["transition"][k] for k in ("onset", "midpoint", "completion")}
    out: dict = {"steps": steps, "n_dev_layouts": len(setup["layouts"]["dev"]), "n_states": int(len(obs)),
                 "transition_thresholds": thresholds, "per_seed": {}, "cross_seed_cka_same_step": cross}
    arrays: dict = {"steps": np.array(steps)}
    for seed in seeds:
        hk, gd = series[seed]["hacking"], series[seed]["good"]
        tr = {"dev": detect_transition(steps, hk["dev_farming"], thresholds),
              "train_panel": detect_transition(steps, hk["train_panel_farming"], thresholds)}
        logged = _logged_dev_farming(seed, "hacking", pc["checkpoint_dir"])
        summary: dict = {
            "transition": tr,
            "good_branch_max_farming": max(gd["dev_farming"]), "good_branch_min_direct_goal": min(gd["dev_direct_goal"]),
            "max_abs_diff_vs_phase1_log": max(abs(hk["dev_farming"][i] - logged[s]) for i, s in enumerate(steps)),
            "layers": {},
        }
        onset, done = tr["dev"]["onset"], tr["dev"]["completion"]
        i_pre, i_on, i_done = steps.index(tr["dev"]["last_before_onset"]), steps.index(onset), steps.index(done)
        for layer in LAYERS:
            hs, gs = hk[f"{layer}_rel_change_start"], gd[f"{layer}_rel_change_start"]
            hp, gp = hk[f"{layer}_rel_change_prev"], gd[f"{layer}_rel_change_prev"]
            # First step where the hacking branch has moved more than twice as far as the good branch ever had so far.
            rep_onset = next((steps[i] for i in range(1, len(steps)) if hs[i] > 2 * max(gs[1:i + 1])), None)
            peak = int(np.argmax([v or 0.0 for v in hp]))

            def window(v, lo, hi):
                vals = [x for x in v[lo:hi] if x is not None]
                return float(np.mean(vals)) if vals else None

            summary["layers"][layer] = {
                "representation_onset_step": rep_onset,
                "at_last_step_before_onset": {
                    "step": steps[i_pre], "dev_farming": hk["dev_farming"][i_pre],
                    "hacking_rel_change_start": hs[i_pre], "good_rel_change_start": gs[i_pre],
                    "hacking_cka_start": hk[f"{layer}_cka_start"][i_pre], "good_cka_start": gd[f"{layer}_cka_start"][i_pre],
                    "hacking_cka_vs_good": hk[f"{layer}_cka_vs_good"][i_pre],
                    "share_of_final_change": hs[i_pre] / hs[-1],
                },
                "share_of_final_change_at_completion": hs[i_done] / hs[-1],
                "cka_start_at_completion": hk[f"{layer}_cka_start"][i_done], "cka_start_final": hk[f"{layer}_cka_start"][-1],
                "peak_consecutive_change": {"step": steps[peak], "value": hp[peak],
                                            "inside_transition_window": bool(i_on <= peak <= i_done)},
                # Mean change per 10k steps, hacking and good branch over the same intervals.
                "consecutive_change_mean": {
                    "before_onset": {"hacking": window(hp, 1, i_on), "good": window(gp, 1, i_on)},
                    "during_transition": {"hacking": window(hp, i_on, i_done + 1), "good": window(gp, i_on, i_done + 1)},
                    "after_completion": {"hacking": window(hp, i_done + 1, len(steps)), "good": window(gp, i_done + 1, len(steps))},
                },
            }
        summary["uniform_tile_not_goal_at_last_step_before_onset"] = {
            "hacking": hk["uniform_tile_not_goal"][i_pre], "good": gd["uniform_tile_not_goal"][i_pre]}
        summary["q_mean_max_at_last_step_before_onset"] = {"hacking": hk["q_mean_max"][i_pre], "good": gd["q_mean_max"][i_pre],
                                                           "start": hk["q_mean_max"][0]}
        pca = {}
        for layer in LAYERS:
            rows = np.stack(flat[seed][layer]["hacking"] + flat[seed][layer]["good"])
            coords, ratio = trajectory_pca(rows, 3)
            n = len(steps)
            pca[layer] = {"explained_variance_ratio": [float(v) for v in ratio],
                          "hacking": coords[:n].tolist(), "good": coords[n:].tolist()}
            for branch in BRANCHES:
                arrays[f"mean_activation_seed{seed}_{branch}_{layer}"] = np.stack(heat[seed][branch][layer])
        out["per_seed"][str(seed)] = {"summary": summary, "series": series[seed], "trajectory_pca": pca}
    return out, arrays


# --- Part 3: decision-level analysis -----------------------------------------

def _pair(nets: dict, states: dict, params: ProceduralEnvParams) -> dict:
    """Both checkpoints of the hacking branch, and of the good branch, on one set of states."""
    act = {k: forward_all(n, states["obs"]) for k, n in nets.items()}
    feats, names = state_features(states, params)
    a_pre, a_post = act["pre"]["q"].argmax(axis=1), act["post"]["q"].argmax(axis=1)
    out = {"act": act, "features": feats, "feature_names": names, "a_pre": a_pre, "a_post": a_post,
           "labels": classify_decisions(a_pre, a_post, states["pos"], states["layouts"], params), "delta": {}, "rel": {},
           "good_rel": {}}
    qs = np.sort(act["pre"]["q"], axis=1)
    out["pre_q_margin"] = qs[:, -1] - qs[:, -2]
    for layer in LAYERS:
        d = act["post"][layer] - act["pre"][layer]
        out["delta"][layer] = d
        out["rel"][layer] = np.linalg.norm(d, axis=1) / (np.linalg.norm(act["pre"][layer], axis=1) + 1e-8)
        g = act["good_post"][layer] - act["good_pre"][layer]
        out["good_rel"][layer] = np.linalg.norm(g, axis=1) / (np.linalg.norm(act["good_pre"][layer], axis=1) + 1e-8)
    return out


def _gbc_auroc(x_fit, y_fit, x_dev, y_dev) -> float:
    model = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(x_fit, y_fit)
    return safe_auroc(y_dev, model.predict_proba(x_dev)[:, 1])


def analyze_distribution(fit: dict, dev: dict, cfg: dict) -> dict:
    k, min_count = cfg["top_k"], cfg["min_class_count"]
    labels = dev["labels"]
    out: dict = {"n_fit": int(len(fit["labels"])), "n_dev": int(len(labels)),
                 "label_fraction": {name: float((labels == name).mean()) for name in DECISION_LABELS},
                 "disagreement_rate": float((labels != AGREE).mean()), "layers": {}}
    y_fit, y_dev = fit["labels"] == TILE_OVER_GOAL, labels == TILE_OVER_GOAL
    enough = min(y_fit.sum(), (~y_fit).sum(), y_dev.sum(), (~y_dev).sum()) >= min_count
    out["enough_of_both_classes"] = bool(enough)

    for layer in LAYERS:
        rel = dev["rel"][layer]
        lb: dict = {"activation_change_by_label": {}, "good_branch_change_by_label": {}}
        for name in DECISION_LABELS:
            m = labels == name
            lb["activation_change_by_label"][name] = float(rel[m].mean()) if m.any() else None
            lb["good_branch_change_by_label"][name] = float(dev["good_rel"][layer][m].mean()) if m.any() else None
        if enough:
            # Raw gap, then the gap left after removing what position, distances and time explain.
            lb["tile_over_goal_vs_rest_cohens_d"] = cohens_d(rel[y_dev], rel[~y_dev])
            reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(fit["features"], fit["rel"][layer])
            pred = reg.predict(dev["features"])
            resid = rel - pred
            lb["variance_explained_by_state_features"] = float(1 - resid.var() / rel.var())
            lb["residual_tile_over_goal_vs_rest_cohens_d"] = cohens_d(resid[y_dev], resid[~y_dev])
            lb["residual_mean"] = {"tile_over_goal": float(resid[y_dev].mean()), "rest": float(resid[~y_dev].mean())}
        out["layers"][layer] = lb

    if not enough:
        return out

    # A direction in the layer-2 change, fitted on training-layout states.
    d2f, d2d = fit["delta"]["layer2"], dev["delta"]["layer2"]
    direction = d2f[y_fit].mean(axis=0) - d2f[~y_fit].mean(axis=0)
    direction /= np.linalg.norm(direction)
    score = d2d @ direction
    names = dev["feature_names"]
    col = {n: dev["features"][:, names.index(n)] for n in ("tile_distance", "goal_distance", "time_left")}
    strata = np.stack([col["tile_distance"], col["goal_distance"], np.digitize(col["time_left"], [1 / 3, 2 / 3])], axis=1)
    i_td = names.index("tile_distance")
    out["direction"] = {
        "auroc": safe_auroc(y_dev, score),
        "auroc_within_distance_and_time_strata": within_stratum_auroc(score, y_dev, strata, cfg["stratum_min_per_class"]),
        "auroc_tile_distance_only": _gbc_auroc(fit["features"][:, [i_td]], y_fit, dev["features"][:, [i_td]], y_dev),
        "auroc_pre_activation_on_direction": safe_auroc(y_dev, dev["act"]["pre"]["layer2"] @ direction),
        "auroc_post_activation_on_direction": safe_auroc(y_dev, dev["act"]["post"]["layer2"] @ direction),
        "auroc_good_branch_change_on_direction": safe_auroc(
            y_dev, (dev["act"]["good_post"]["layer2"] - dev["act"]["good_pre"]["layer2"]) @ direction),
    }

    # Does the size of the activation change mark these states beyond what simpler quantities already tell us?
    def stack(d, *extra):
        return np.column_stack([d["features"], *extra])

    def pre_q(d):
        return np.column_stack([d["act"]["pre"]["q"], d["pre_q_margin"]])

    def mags(d):
        return np.column_stack([d["rel"]["layer1"], d["rel"]["layer2"]])

    mean = d2f.mean(axis=0)
    _, _, vt = np.linalg.svd(d2f - mean, full_matrices=False)

    def pcs(d):
        return (d["delta"]["layer2"] - mean) @ vt[:k].T

    out["predict_tile_over_goal_auroc"] = {
        "state_features": _gbc_auroc(fit["features"], y_fit, dev["features"], y_dev),
        "plus_pre_checkpoint_q_values": _gbc_auroc(stack(fit, pre_q(fit)), y_fit, stack(dev, pre_q(dev)), y_dev),
        "plus_activation_change_magnitude": _gbc_auroc(stack(fit, pre_q(fit), mags(fit)), y_fit,
                                                       stack(dev, pre_q(dev), mags(dev)), y_dev),
        "activation_change_magnitude_alone": _gbc_auroc(mags(fit), y_fit, mags(dev), y_dev),
        # The later checkpoint's layer 2 linearly determines its Q-values, so this one is close to circular.
        f"plus_top{k}_change_directions_near_circular": _gbc_auroc(stack(fit, pre_q(fit), pcs(fit)), y_fit,
                                                                   stack(dev, pre_q(dev), pcs(dev)), y_dev),
    }
    out["_direction_score"] = score
    return out


def run_decisions(setup: dict, timeline: dict) -> tuple[dict, dict]:
    pc, params = setup["config"], setup["params"]
    st, cfg = pc["states"], pc["decision"]
    out: dict = {"label_definitions": {
        "agree": "same greedy action",
        "equivalent_route": "different actions, both closer to the goal",
        "tile_over_goal": "earlier checkpoint moves closer to the goal; later one does not and moves closer to the tile, or steps off it",
        "other_disagreement": "any other difference"}, "per_seed": {}, "cross_seed_uniform": {}}
    arrays: dict = {}
    keep: dict = {}
    for seed in pc["seeds"]:
        tr = timeline["per_seed"][str(seed)]["summary"]["transition"]["dev"]
        pre_step, post_step = tr["last_before_onset"], tr["completion"]
        nets = {"pre": _net(setup, seed, "hacking", pre_step), "post": _net(setup, seed, "hacking", post_step),
                "good_pre": _net(setup, seed, "good", pre_step), "good_post": _net(setup, seed, "good", post_step)}
        sets = {"uniform": {split: setup["states"][split] for split in ("fit", "dev")}}
        for name, key in (("rollout_goal_following", "pre"), ("rollout_hacking", "post")):
            sets[name] = {split: subsample_states(rollout_states(nets[key], setup["layouts"][split], params),
                                                  st["rollout_states_per_policy"], st[f"rollout_{split}_seed"])
                          for split in ("fit", "dev")}
        block: dict = {"goal_following_checkpoint_step": pre_step, "hacking_checkpoint_step": post_step,
                       "sampling": {}}
        for name, pair in sets.items():
            fit, dev = _pair(nets, pair["fit"], params), _pair(nets, pair["dev"], params)
            res = analyze_distribution(fit, dev, cfg)
            score = res.pop("_direction_score", None)
            block["sampling"][name] = res
            s = pair["dev"]
            arrays[f"seed{seed}_{name}_obs"] = s["obs"]
            arrays[f"seed{seed}_{name}_layout"] = s["layouts"].astype(np.int8)
            arrays[f"seed{seed}_{name}_pos"] = s["pos"].astype(np.int8)
            arrays[f"seed{seed}_{name}_elapsed_steps"] = s["steps"].astype(np.int16)
            arrays[f"seed{seed}_{name}_action_pre"] = dev["a_pre"].astype(np.int8)
            arrays[f"seed{seed}_{name}_action_post"] = dev["a_post"].astype(np.int8)
            arrays[f"seed{seed}_{name}_q_pre"] = dev["act"]["pre"]["q"].astype(np.float32)
            arrays[f"seed{seed}_{name}_q_post"] = dev["act"]["post"]["q"].astype(np.float32)
            arrays[f"seed{seed}_{name}_label"] = np.array([DECISION_LABELS.index(l) for l in dev["labels"]], dtype=np.int8)
            for layer in LAYERS:
                arrays[f"seed{seed}_{name}_{layer}_rel_change"] = dev["rel"][layer].astype(np.float32)
            if name == "uniform":
                keep[seed] = {"dev": dev, "score": score, "states": s}
        out["per_seed"][str(seed)] = block

    # One worked example for the figure, chosen by a fixed rule rather than by eye.
    seed0 = pc["seeds"][0]
    dev, s = keep[seed0]["dev"], keep[seed0]["states"]
    names = dev["feature_names"]
    f = {n: dev["features"][:, names.index(n)] for n in ("goal_distance", "tile_distance", "time_left")}
    idx = int(np.flatnonzero((dev["labels"] == TILE_OVER_GOAL) & (f["goal_distance"] >= 3) & (f["tile_distance"] >= 2)
                             & (f["time_left"] >= 0.5))[0])
    out["example"] = {
        "rule": "first uniformly sampled development state labelled tile_over_goal with goal distance >= 3, tile distance >= 2 and at least half the episode left",
        "seed": seed0, "state_index": idx, "pos": s["pos"][idx].tolist(), "layout": s["layouts"][idx].tolist(),
        "elapsed_steps": int(s["steps"][idx]), "obs": s["obs"][idx].tolist(),
        "action_pre": int(dev["a_pre"][idx]), "action_post": int(dev["a_post"][idx]),
        "q_pre": dev["act"]["pre"]["q"][idx].tolist(), "q_post": dev["act"]["post"]["q"][idx].tolist(),
        "q_good_pre": dev["act"]["good_pre"]["q"][idx].tolist(), "q_good_post": dev["act"]["good_post"]["q"][idx].tolist(),
        "layer2_change": dev["delta"]["layer2"][idx].tolist(),
        "layer2_change_good_branch": (dev["act"]["good_post"]["layer2"][idx] - dev["act"]["good_pre"]["layer2"][idx]).tolist(),
    }

    # Across seeds, on the identical uniformly sampled development observations. No neuron indices are compared.
    for a, c in combinations(pc["seeds"], 2):
        da, dc = keep[a]["dev"], keep[c]["dev"]
        ya, yc = da["labels"] == TILE_OVER_GOAL, dc["labels"] == TILE_OVER_GOAL
        pair: dict = {
            "tile_over_goal_rate": [float(ya.mean()), float(yc.mean())],
            "tile_over_goal_both": float((ya & yc).mean()), "expected_if_independent": float(ya.mean() * yc.mean()),
            "label_phi_correlation": float(np.corrcoef(ya, yc)[0, 1]),
            "direction_score_correlation": (float(np.corrcoef(keep[a]["score"], keep[c]["score"])[0, 1])
                                            if keep[a]["score"] is not None and keep[c]["score"] is not None else None),
            "relative_change_correlation": {layer: float(np.corrcoef(da["rel"][layer], dc["rel"][layer])[0, 1]) for layer in LAYERS},
            "cka": {layer: {"change_hacking_branch": linear_cka(da["delta"][layer], dc["delta"][layer]),
                            "goal_following_checkpoints": linear_cka(da["act"]["pre"][layer], dc["act"]["pre"][layer]),
                            "hacking_checkpoints": linear_cka(da["act"]["post"][layer], dc["act"]["post"][layer])}
                    for layer in LAYERS},
        }
        out["cross_seed_uniform"][f"{a}-{c}"] = pair
    return out, arrays


def required_checkpoints(decisions: dict, registry: dict) -> dict:
    """The intermediate checkpoints the decision-level analysis and the GIF load, with their hashes.

    The timeline needs all 41 checkpoints of both branches; those are regenerated by
    `python -m scripts.exp2_train --all-seeds` and checked against checkpoint_registry.json.
    """
    files = []
    for seed, block in decisions["per_seed"].items():
        steps = sorted({block["goal_following_checkpoint_step"], block["hacking_checkpoint_step"]})
        for branch in BRANCHES:
            by_step = {r["step"]: r for r in registry["seeds"][seed][branch]}
            files += [{"seed": int(seed), "branch": branch, "role": "decision pair", **by_step[s]} for s in steps]
    return {"note": "Committed so the decision-level analysis can be rerun without retraining. All other intermediate "
                    "checkpoints stay local and are listed with hashes in checkpoint_registry.json.", "files": files}


# --- driver --------------------------------------------------------------------

def main(part: str = "all") -> None:
    setup = load_setup()
    out_dir = Path(setup["config"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"manifest_sha256": setup["manifest_sha256"], "splits_used": ["train", "dev"], "seeds": setup["config"]["seeds"]}
    if part in ("registry", "all"):
        json.dump({**meta, **run_registry(setup)}, open(out_dir / "checkpoint_registry.json", "w"), indent=1)
        print("registry verified")
    if part in ("timeline", "all"):
        timeline, arrays = run_timeline(setup)
        json.dump({**meta, **timeline}, open(out_dir / "timeline.json", "w"))
        np.savez_compressed(out_dir / "mean_activations.npz", **arrays)
        for seed, block in timeline["per_seed"].items():
            print(seed, block["summary"]["transition"])
    if part in ("decisions", "all"):
        timeline = json.load(open(out_dir / "timeline.json"))
        decisions, arrays = run_decisions(setup, timeline)
        json.dump({**meta, **decisions}, open(out_dir / "decision_analysis.json", "w"), indent=1)
        np.savez_compressed(out_dir / "decision_points.npz", **arrays)
        for seed, block in decisions["per_seed"].items():
            print(seed, {k: round(v["disagreement_rate"], 3) for k, v in block["sampling"].items()})
        json.dump({**meta, **required_checkpoints(decisions, json.load(open(out_dir / "checkpoint_registry.json")))},
                  open(out_dir / "required_checkpoints.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["registry", "timeline", "decisions", "all"])
    main(ap.parse_args().part)
