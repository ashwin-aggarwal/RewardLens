"""Experiment 2, Phase 5: where is reward hacking encoded? Weight-level interventions.

Usage:
  python -m scripts.exp2_run_phase5                 # all parts
  python -m scripts.exp2_run_phase5 --part swaps    # baselines | swaps | finetune | distill | weights

Everything is fixed in configs/phase5.json. Behavior is measured on the 1,500
development layouts (and a 1,500-layout training panel for the baselines);
fine-tuning interacts with training layouts only. The test split and the reserved
final evaluation set are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from rewardlens.exp2.checkpoints import checkpoint_path
from rewardlens.exp2.emergence import checkpoint_file
from rewardlens.exp2.evaluation import greedy_policy, random_policy
from rewardlens.exp2.layouts import layouts_to_array, load_manifest
from rewardlens.exp2.near_transition import spearman
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import sample_matched_states
from rewardlens.exp2.training import EVAL_KEYS
from rewardlens.exp2.weight_level import (
    LAYERS, behavior, closed_form_head, conflict_masks, count_parameters, decision_diagnostics, distill, finetune, first_sustained,
    frozen_unchanged, load_net, meets_repair, q_preference, random_edit, resolve_step, scripted_shortest_path,
    select_checkpoint, swap_layers, weight_difference, weights_sha256,
)

EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
ENV_CONFIG = EXP_DIR / "configs/environment.json"
PHASE5_CONFIG = EXP_DIR / "configs/phase5.json"
GAMMA = 0.99


def load_setup() -> dict:
    ec, pc = json.load(open(ENV_CONFIG)), json.load(open(PHASE5_CONFIG))
    params = ProceduralEnvParams(**ec["env"])
    splits, meta = load_manifest(ec["layouts"]["manifest_path"])
    # Only training and development layouts are handed to the analysis code.
    layouts = {"train": splits["train"], "train_panel": splits["train"][:pc["baselines"]["n_train_layouts"]], "dev": splits["dev"]}
    states = sample_matched_states(layouts["dev"], pc["states"]["states_per_layout"], pc["states"]["dev_state_seed"], params)
    timeline = json.load(open(pc["phase3_timeline"]))
    return {"params": params, "config": pc, "layouts": layouts, "states": states, "timeline": timeline,
            "steps": timeline["steps"], "manifest_sha256": meta["sha256"], "manifest_path": ec["layouts"]["manifest_path"],
            "env": ec["env"]}


def transition(setup: dict, seed: int) -> dict:
    return setup["timeline"]["per_seed"][str(seed)]["summary"]["transition"]["dev"]


def net_at(setup: dict, seed: int, ref) -> torch.nn.Module:
    branch, point = ref
    step = resolve_step(point, transition(setup, seed), setup["steps"][-1])
    return load_net(seed, branch, step, setup["config"]["checkpoint_dir"], setup["steps"][-1])


def dev_behavior(net_or_policy, setup: dict, split: str = "dev") -> dict:
    policy = greedy_policy(net_or_policy) if isinstance(net_or_policy, torch.nn.Module) else net_or_policy
    return behavior(policy, setup["layouts"][split], setup["params"], GAMMA)


def aggregate(runs: list[dict]) -> dict:
    out = {"n_draws": len(runs)}
    for key in EVAL_KEYS:
        vals = [r[key] for r in runs if r[key] is not None]
        out[key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None
    return out


def header(setup: dict) -> dict:
    return {"manifest_sha256": setup["manifest_sha256"], "splits_used": ["train", "dev"], "seeds": setup["config"]["seeds"],
            "n_dev_layouts": len(setup["layouts"]["dev"]), "repair_criterion": setup["config"]["repair_criterion"]}


def save(setup: dict, name: str, data: dict) -> None:
    out_dir = Path(setup["config"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({**header(setup), **data}, open(out_dir / name, "w"), indent=1)
    print(f"Saved {out_dir / name}", flush=True)


# --- 1. baselines ------------------------------------------------------------

def run_baselines(setup: dict) -> dict:
    bc, crit = setup["config"]["baselines"], setup["config"]["repair_criterion"]
    points = {"pretrained": ("pretrain", "start"), "good_final": ("good", "final"),
              "hacking_pre_onset": ("hacking", "last_before_onset"), "hacking_completion": ("hacking", "completion"),
              "hacking_final": ("hacking", "final")}
    out: dict = {"references": {}, "per_seed": {}}
    for split in ("dev", "train_panel"):
        ref = {"random_policy": aggregate([dev_behavior(random_policy(bc["random_seed"] + d), setup, split)
                                           for d in range(bc["n_random_draws"])]),
               "scripted_shortest_path": scripted_shortest_path(setup["layouts"][split], setup["params"], GAMMA)}
        out["references"][split] = ref
    for seed in setup["config"]["seeds"]:
        block = {"transition": transition(setup, seed), "networks": {}}
        for name, ref in points.items():
            net = net_at(setup, seed, ref)
            b = {split: dev_behavior(net, setup, split) for split in ("dev", "train_panel")}
            block["networks"][name] = {"branch": ref[0], "step": resolve_step(ref[1], block["transition"], setup["steps"][-1]),
                                       **b, "meets_repair_criterion": meets_repair(b["dev"], crit)}
            d = b["dev"]
            print(f"[baseline seed {seed}] {name:>18} DG {d['direct_goal']:.3f} farm {d['farming']:.3f} other {d['other']:.3f} "
                  f"eff {d['path_efficiency_direct']} visits {d['mean_reward_tile_visits']:.1f}", flush=True)
        out["per_seed"][str(seed)] = block
    return out


# --- 2. weight swaps ---------------------------------------------------------

def run_swap(setup: dict, recipient, donor, good, layers: list[str]) -> dict:
    wc, crit = setup["config"]["weight_swap"], setup["config"]["repair_criterion"]
    swapped = swap_layers(recipient, donor, layers)
    diff = weight_difference(recipient, swapped, layers)
    b = dev_behavior(swapped, setup)
    rc = wc["random_control"]
    controls = [dev_behavior(random_edit(recipient, swapped, layers, rc["seed"] + d), setup) for d in range(rc["n_draws"])]
    return {"layers": layers, "n_parameters": count_parameters(layers),
            "update_relative_norm": {l: diff[l]["relative"] for l in layers}, "behavior": b,
            "meets_repair_criterion": meets_repair(b, crit),
            "decisions": decision_diagnostics(swapped, recipient, good, setup["states"], setup["params"]),
            "random_control": aggregate(controls)}


def run_swaps(setup: dict) -> dict:
    wc, crit = setup["config"]["weight_swap"], setup["config"]["repair_criterion"]
    out: dict = {"per_seed": {str(s): {"pairs": {}} for s in setup["config"]["seeds"]}, "followup": {}}
    for pair, spec in wc["pairs"].items():
        nets = {}
        for seed in setup["config"]["seeds"]:
            tr, final = transition(setup, seed), setup["steps"][-1]
            donor, rec, good = net_at(setup, seed, spec["donor"]), net_at(setup, seed, spec["recipient"]), net_at(setup, seed, ("good", "final"))
            nets[seed] = (rec, donor, good)
            block = {"donor": {"branch": spec["donor"][0], "step": resolve_step(spec["donor"][1], tr, final), "behavior": dev_behavior(donor, setup),
                               "decisions": decision_diagnostics(donor, rec, good, setup["states"], setup["params"])},
                     "recipient": {"branch": spec["recipient"][0], "step": resolve_step(spec["recipient"][1], tr, final),
                                   "behavior": dev_behavior(rec, setup),
                                   "decisions": decision_diagnostics(rec, rec, good, setup["states"], setup["params"])},
                     "weight_difference": weight_difference(rec, donor), "swaps": {}}
            for name, layers in wc["primary_swaps"].items():
                block["swaps"][name] = run_swap(setup, rec, donor, good, layers)
            out["per_seed"][str(seed)]["pairs"][pair] = block
        # Follow-up subsets only when no primary swap repairs every seed (rule fixed in the config).
        repaired = [name for name in wc["primary_swaps"]
                    if all(out["per_seed"][str(s)]["pairs"][pair]["swaps"][name]["meets_repair_criterion"] for s in setup["config"]["seeds"])]
        out["followup"][pair] = {"primary_swaps_repairing_all_seeds": repaired, "triggered": not repaired}
        if not repaired:
            for seed, (rec, donor, good) in nets.items():
                for name, layers in wc["followup_swaps"].items():
                    out["per_seed"][str(seed)]["pairs"][pair]["swaps"][name] = run_swap(setup, rec, donor, good, layers)
        for seed in setup["config"]["seeds"]:
            block = out["per_seed"][str(seed)]["pairs"][pair]
            for name, s in block["swaps"].items():
                b, c = s["behavior"], s["random_control"]
                print(f"[swap seed {seed}] {pair:>24} {name:>14} DG {b['direct_goal']:.2f} farm {b['farming']:.2f} other {b['other']:.2f} "
                      f"eff {b['path_efficiency_direct'] if b['path_efficiency_direct'] is None else round(b['path_efficiency_direct'], 2)} "
                      f"| random DG {c['direct_goal']['mean']:.2f} farm {c['farming']['mean']:.2f} | agree good {s['decisions']['agree_with_good']:.2f} "
                      f"| repair {s['meets_repair_criterion']}", flush=True)
    return out


# --- 3. constrained fine-tuning ----------------------------------------------

def _summarize_training(setup: dict, run: dict, start, good, layers: list[str], cfg: dict, out_path: Path, extra: dict) -> dict:
    """Shared reporting for RL fine-tuning and distillation: selection, repair check, weight change, controls."""
    pc, params, crit = setup["config"], setup["params"], setup["config"]["repair_criterion"]
    evals, snaps = run["evals"], run["snapshots"]
    init = {k: v.clone() for k, v in start.state_dict().items()}
    flags = [meets_repair(e["dev"], crit) for e in evals]
    sustained = first_sustained(flags)
    sel = select_checkpoint(evals, "dev_select", cfg["selection"]["metric"])

    def describe(i: int) -> dict:
        net = type(start)()
        net.load_state_dict(snaps[i])
        net.eval()
        diff = weight_difference(start, net)
        rc = cfg["random_control"]
        controls = [dev_behavior(random_edit(start, net, layers, rc["seed"] + d), setup) for d in range(rc["n_draws"])] if i > 0 else []
        return {"step": evals[i]["step"], "dev": evals[i]["dev"], "dev_report": evals[i]["dev_report"],
                "dev_select": evals[i]["dev_select"], "meets_repair_criterion_dev": flags[i],
                "meets_repair_criterion_dev_report": meets_repair(evals[i]["dev_report"], crit),
                "weight_change": {l: {"relative": diff[l]["relative"], "l2": diff[l]["l2"], "n_changed": diff[l]["n_changed"]} for l in LAYERS},
                "n_parameters_changed": int(sum(diff[l]["n_changed"] for l in LAYERS)),
                "frozen_layers_unchanged": frozen_unchanged(init, snaps[i], layers),
                "decisions": decision_diagnostics(net, start, good, setup["states"], params),
                "random_control": aggregate(controls) if controls else None,
                "weights_sha256": weights_sha256(snaps[i])}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"q_net_state_dict": snaps[sel], "trainable_layers": layers, "selected_step": evals[sel]["step"],
                "start": cfg["start"], **extra}, out_path)
    return {**extra, "trainable_layers": layers, "n_trainable_parameters": run["n_trainable"], "train_seconds": run["train_seconds"],
            "curve": [{"step": e["step"], **({"train_loss": e["train_loss"]} if "train_loss" in e else {}),
                       **{k: e["dev"][k] for k in EVAL_KEYS}} for e in evals],
            "first_step_meeting_criterion": next((evals[i]["step"] for i, f in enumerate(flags) if f), None),
            "sustained_from_step": evals[sustained]["step"] if sustained is not None else None,
            "selected": describe(sel), "final": describe(len(evals) - 1), "selected_weights_file": str(out_path)}


def _finetune_job(job: dict) -> dict:
    """One seed and condition, in its own process. Loads its own layouts (training and development only)."""
    setup = load_setup()
    pc, fc, params = setup["config"], setup["config"]["finetune"], setup["params"]
    seed, cond, layers = job["seed"], job["condition"], job["layers"]
    tc = json.load(open(pc["training_config"]))["dqn"]
    hp = {**{k: tc[k] for k in ("learning_rate", "gamma", "buffer_capacity", "batch_size", "learning_starts",
                                "train_frequency", "target_update_frequency")},
          **{k: fc[k] for k in ("total_timesteps", "eval_interval", "epsilon_start", "epsilon_end", "epsilon_fraction")}}
    start, good = net_at(setup, seed, fc["start"]), net_at(setup, seed, ("good", "final"))
    dev = layouts_to_array(setup["layouts"]["dev"])
    select = np.arange(len(dev)) < fc["selection"]["n_dev_select"]
    run = finetune({k: v.clone() for k, v in start.state_dict().items()}, layers, setup["layouts"]["train"], dev, params, hp,
                   seed, fc["reward_mode"], select)
    out = _summarize_training(setup, run, start, good, layers, fc, Path(pc["results_dir"]) / "finetuned" / f"seed{seed}_{cond}.pt",
                              {"seed": seed, "condition": cond})
    return {**out, "hyperparameters": hp}


def distill_states(setup: dict) -> dict:
    dc = setup["config"]["distillation"]
    return sample_matched_states(setup["layouts"]["train"][:dc["n_train_layouts"]], dc["states_per_layout"], dc["state_seed"], setup["params"])


def _distill_job(job: dict) -> dict:
    setup = load_setup()
    pc, dc, params = setup["config"], setup["config"]["distillation"], setup["params"]
    seed, cond, layers = job["seed"], job["condition"], job["layers"]
    hp = {k: dc[k] for k in ("n_updates", "eval_interval_updates", "batch_size", "learning_rate")}
    start, teacher = net_at(setup, seed, dc["start"]), net_at(setup, seed, dc["teacher"])
    dev = layouts_to_array(setup["layouts"]["dev"])
    select = np.arange(len(dev)) < dc["selection"]["n_dev_select"]
    run = distill({k: v.clone() for k, v in start.state_dict().items()}, teacher, layers, distill_states(setup)["obs"], dev,
                  params, hp, seed, select)
    out = _summarize_training(setup, run, start, teacher, layers, dc, Path(pc["results_dir"]) / "distilled" / f"seed{seed}_{cond}.pt",
                              {"seed": seed, "condition": cond})
    return {**out, "hyperparameters": hp}


def _report(kind: str, r: dict) -> None:
    s, f = r["selected"], r["final"]
    eff = s["dev_report"]["path_efficiency_direct"]
    print(f"[{kind} seed {r['seed']}] {r['condition']:>13} ({r['n_trainable_parameters']:>5} params) selected {s['step']:>6}: "
          f"report DG {s['dev_report']['direct_goal']:.3f} farm {s['dev_report']['farming']:.3f} other {s['dev_report']['other']:.3f} "
          f"eff {eff if eff is None else round(eff, 3)} repair {s['meets_repair_criterion_dev_report']} | final DG {f['dev']['direct_goal']:.3f} "
          f"| sustained from {r['sustained_from_step']} | agree good {s['decisions']['agree_with_good']:.2f} | frozen ok "
          f"{f['frozen_layers_unchanged']} | {r['train_seconds']}s", flush=True)


def run_finetune(setup: dict) -> dict:
    fc = setup["config"]["finetune"]
    jobs = [{"seed": s, "condition": c, "layers": l} for s in setup["config"]["seeds"] for c, l in fc["conditions"].items()]
    out: dict = {"start": fc["start"], "per_seed": {str(s): {} for s in setup["config"]["seeds"]}}
    with ProcessPoolExecutor(max_workers=fc["max_workers"]) as pool:
        for r in pool.map(_finetune_job, jobs):
            out["per_seed"][str(r["seed"])][r["condition"]] = r
            _report("finetune", r)
    return out


def run_distill(setup: dict) -> dict:
    """Supervised distillation from the same-seed good network. Reported separately from RL fine-tuning."""
    dc, crit = setup["config"]["distillation"], setup["config"]["repair_criterion"]
    jobs = [{"seed": s, "condition": c, "layers": l} for s in setup["config"]["seeds"] for c, l in dc["conditions"].items()]
    states = distill_states(setup)
    out: dict = {"kind": "supervised distillation, not reinforcement learning", "start": dc["start"], "teacher": dc["teacher"],
                 "n_training_states": int(len(states["obs"])), "per_seed": {str(s): {} for s in setup["config"]["seeds"]}}
    with ProcessPoolExecutor(max_workers=dc["max_workers"]) as pool:
        for r in pool.map(_distill_job, jobs):
            out["per_seed"][str(r["seed"])][r["condition"]] = r
            _report("distill", r)
    for seed in setup["config"]["seeds"]:
        start, teacher = net_at(setup, seed, dc["start"]), net_at(setup, seed, dc["teacher"])
        net = closed_form_head(start, teacher, states["obs"])
        b = dev_behavior(net, setup)
        diff = weight_difference(start, net, ["fc3"])
        rc = dc["random_control"]
        out["per_seed"][str(seed)]["head_closed_form"] = {
            "trainable_layers": ["fc3"], "n_trainable_parameters": count_parameters(["fc3"]), "dev": b,
            "meets_repair_criterion_dev": meets_repair(b, crit), "weight_change": {"fc3": {"relative": diff["fc3"]["relative"]}},
            "decisions": decision_diagnostics(net, start, teacher, setup["states"], setup["params"]),
            "random_control": aggregate([dev_behavior(random_edit(start, net, ["fc3"], rc["seed"] + d), setup) for d in range(rc["n_draws"])])}
        print(f"[distill seed {seed}] head_closed_form DG {b['direct_goal']:.3f} farm {b['farming']:.3f} other {b['other']:.3f} "
              f"eff {b['path_efficiency_direct']} agree good {out['per_seed'][str(seed)]['head_closed_form']['decisions']['agree_with_good']:.2f}", flush=True)
    return out


# --- 4. weights along the training run ---------------------------------------

def run_weights(setup: dict) -> dict:
    steps, final = setup["steps"], setup["steps"][-1]
    masks = conflict_masks(setup["states"], setup["params"])
    out: dict = {"steps": steps, "layers": list(LAYERS), "parameters_per_layer": {l: count_parameters([l]) for l in LAYERS},
                 "per_seed": {}}
    for seed in setup["config"]["seeds"]:
        tr = transition(setup, seed)
        nets = {b: [load_net(seed, b, s, setup["config"]["checkpoint_dir"], final) for s in steps] for b in ("good", "hacking")}
        block: dict = {"transition": tr, "series": {}, "summary": {}}
        for branch, seq in nets.items():
            ser: dict = {"q_preference_mean": [], "q_preference_fraction_tile": []}
            for net in seq:
                qp = q_preference(net, setup["states"], masks)
                ser["q_preference_mean"].append(qp["mean_tile_minus_goal"])
                ser["q_preference_fraction_tile"].append(qp["fraction_prefer_tile"])
            block["n_conflict_states"] = qp["n_conflict_states"]
            for layer in LAYERS:
                from_start = [weight_difference(seq[0], n, [layer])[layer] for n in seq]
                step_change = [0.0] + [weight_difference(seq[i - 1], seq[i], [layer])[layer]["l2"] / from_start[0]["reference_norm"]
                                       for i in range(1, len(seq))]
                total = from_start[-1]["l2"]
                ser[layer] = {"relative_change_from_start": [d["relative"] for d in from_start],
                              "cosine_with_start": [d["cosine"] for d in from_start],
                              "norm": [d["other_norm"] for d in from_start],
                              "change_per_interval": step_change,
                              "share_of_final_change": [d["l2"] / total for d in from_start]}
            block["series"][branch] = ser
        block["series"]["branch_difference"] = {
            layer: [weight_difference(g, h, [layer])[layer]["l2"] / weight_difference(nets["good"][0], nets["good"][0], [layer])[layer]["reference_norm"]
                    for g, h in zip(nets["good"], nets["hacking"])] for layer in LAYERS}

        farming = setup["timeline"]["per_seed"][str(seed)]["series"]["hacking"]["dev_farming"]
        d_farm = [0.0] + [abs(farming[i] - farming[i - 1]) for i in range(1, len(steps))]
        i_pre, i_on, i_done = (steps.index(tr[k]) for k in ("last_before_onset", "onset", "completion"))
        hack, good = block["series"]["hacking"], block["series"]["good"]
        for layer in LAYERS:
            rate = hack[layer]["change_per_interval"]
            block["summary"][layer] = {
                "final_relative_change": {"hacking": hack[layer]["relative_change_from_start"][-1], "good": good[layer]["relative_change_from_start"][-1]},
                "relative_change_at": {k: hack[layer]["relative_change_from_start"][i] for k, i in
                                       (("last_before_onset", i_pre), ("onset", i_on), ("completion", i_done))},
                "good_relative_change_at": {k: good[layer]["relative_change_from_start"][i] for k, i in
                                            (("last_before_onset", i_pre), ("completion", i_done))},
                "share_of_final_change_at": {k: hack[layer]["share_of_final_change"][i] for k, i in
                                             (("last_before_onset", i_pre), ("completion", i_done))},
                "change_during_transition": weight_difference(nets["hacking"][i_pre], nets["hacking"][i_done], [layer])[layer]["relative"],
                "good_change_same_steps": weight_difference(nets["good"][i_pre], nets["good"][i_done], [layer])[layer]["relative"],
                "mean_rate": {"before_onset": float(np.mean(rate[1:i_pre + 1])), "during_transition": float(np.mean(rate[i_pre + 1:i_done + 1])),
                              "after_completion": float(np.mean(rate[i_done + 1:]))},
                "good_mean_rate": {"before_onset": float(np.mean(good[layer]["change_per_interval"][1:i_pre + 1])),
                                   "during_transition": float(np.mean(good[layer]["change_per_interval"][i_pre + 1:i_done + 1])),
                                   "after_completion": float(np.mean(good[layer]["change_per_interval"][i_done + 1:]))},
                "step_of_peak_rate": steps[int(np.argmax(rate))],
                "spearman_rate_vs_abs_farming_change": spearman(rate[1:], d_farm[1:]),
                "spearman_drift_vs_farming": spearman(hack[layer]["relative_change_from_start"], farming),
            }
        block["summary"]["q_preference"] = {
            "hacking_at": {k: hack["q_preference_mean"][i] for k, i in (("start", 0), ("last_before_onset", i_pre), ("onset", i_on),
                                                                         ("completion", i_done), ("final", len(steps) - 1))},
            "hacking_fraction_tile_at": {k: hack["q_preference_fraction_tile"][i] for k, i in
                                         (("start", 0), ("last_before_onset", i_pre), ("completion", i_done), ("final", len(steps) - 1))},
            "good_range": [float(min(good["q_preference_mean"])), float(max(good["q_preference_mean"]))],
            "good_max_fraction_tile": float(max(good["q_preference_fraction_tile"])),
            "first_step_mean_preference_positive": next((s for s, v in zip(steps, hack["q_preference_mean"]) if v > 0), None),
        }
        out["per_seed"][str(seed)] = block
        print(f"[weights seed {seed}] " + " | ".join(
            f"{l}: final {block['summary'][l]['final_relative_change']['hacking']:.2f} (good {block['summary'][l]['final_relative_change']['good']:.2f}), "
            f"pre-onset {block['summary'][l]['relative_change_at']['last_before_onset']:.2f}, completion {block['summary'][l]['relative_change_at']['completion']:.2f}"
            for l in LAYERS), flush=True)
    return out


def required_checkpoints(setup: dict) -> dict:
    """Every checkpoint file Phase 5 reads, with hashes. The weight-trajectory part reads all saved steps."""
    pc, final = setup["config"], setup["steps"][-1]
    files = {}

    def add(seed, branch, step, part):
        if branch == "pretrain" or step == 0:
            path, key = checkpoint_path(seed, "pretrain", pc["checkpoint_dir"]), ("pretrain", 0)
        elif step == final:
            path, key = checkpoint_path(seed, branch, pc["checkpoint_dir"]), (branch, step)
        else:
            path, key = checkpoint_file(seed, branch, step, pc["checkpoint_dir"]), (branch, step)
        rec = files.setdefault(str(path), {"seed": seed, "branch": key[0], "step": key[1], "path": str(path),
                                           "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "used_by": []})
        if part not in rec["used_by"]:
            rec["used_by"].append(part)

    for seed in pc["seeds"]:
        tr = transition(setup, seed)
        for branch, point in [("pretrain", "start"), ("good", "final"), ("hacking", "last_before_onset"), ("hacking", "completion"), ("hacking", "final")]:
            add(seed, branch, resolve_step(point, tr, final), "interventions")
        for branch in ("good", "hacking"):
            for step in setup["steps"]:
                add(seed, branch, step, "weight_trajectory")
    recs = sorted(files.values(), key=lambda f: (f["seed"], f["branch"], f["step"]))
    return {"n_files": len(recs), "n_for_interventions": sum("interventions" in r["used_by"] for r in recs), "files": recs}


PARTS = {"baselines": (run_baselines, "baselines.json"), "swaps": (run_swaps, "weight_swaps.json"),
         "finetune": (run_finetune, "finetune.json"), "distill": (run_distill, "distill.json"),
         "weights": (run_weights, "weight_trajectory.json")}


def main(part: str = "all") -> None:
    t0 = time.time()
    setup = load_setup()
    for name, (fn, filename) in PARTS.items():
        if part in ("all", name):
            save(setup, filename, fn(setup))
    if part == "all":
        save(setup, "required_checkpoints.json", required_checkpoints(setup))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("all",) + tuple(PARTS), default="all")
    main(ap.parse_args().part)
