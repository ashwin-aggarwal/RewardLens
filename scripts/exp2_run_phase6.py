"""Experiment 2, Phase 6: coordinated low-rank weight editing of the hacking networks.

Usage: python -m scripts.exp2_run_phase6

Design in configs/phase6.json. Supervision, budget, checkpoint selection and the
repair criterion are Phase 5's. Training states come from training layouts;
behavior is measured on development layouts. The test split and the reserved
final evaluation set are never loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.exp2.layouts import layouts_to_array
from rewardlens.exp2.low_rank import (
    distill_low_rank, parameter_counts, random_low_rank_edit, truncate_update, update_summary,
)
from rewardlens.exp2.representations import forward_all, linear_cka
from rewardlens.exp2.training import EVAL_KEYS
from rewardlens.exp2.weight_level import (
    LAYERS, count_parameters, decision_diagnostics, distill, first_sustained, meets_repair, random_edit,
    select_checkpoint, weights_sha256,
)
from scripts.exp2_run_phase5 import aggregate, dev_behavior, distill_states, load_setup, net_at

PHASE6_CONFIG = Path("experiments/02_generalizable_reward_hacking/configs/phase6.json")
CURVE_KEYS = ("direct_goal", "farming", "other", "path_efficiency_direct")


def load_config() -> dict:
    return json.load(open(PHASE6_CONFIG))


def run_id(job: dict) -> str:
    return f"seed{job['dqn_seed']}_{job['method']}_r{job['rank']}_{job['lr_name']}_opt{job['opt_seed']}"


def build_jobs(cfg: dict) -> list[dict]:
    jobs = []
    for seed in cfg["seeds"]:
        for k in cfg["training"]["optimization_seeds"]:
            for method in ("coordinated", "head_only", "layer2_only"):
                spec = cfg["low_rank"][method]
                for lr_name in spec["learning_rates"]:
                    for rank in spec["ranks"]:
                        jobs.append({"dqn_seed": seed, "opt_seed": k, "method": method, "layers": spec["layers"], "rank": rank, "lr_name": lr_name})
            for lr_name in cfg["full_network_control"]["learning_rates"]:
                jobs.append({"dqn_seed": seed, "opt_seed": k, "method": "full_network", "layers": list(LAYERS), "rank": None, "lr_name": lr_name})
    return jobs


def similarity(net, hacking, good, obs: np.ndarray) -> dict:
    """How much of the hacking network's hidden representation survives the edit (linear CKA on fixed states)."""
    a, h, g = forward_all(net, obs), forward_all(hacking, obs), forward_all(good, obs)
    return {layer: {"cka_with_hacking": linear_cka(a[layer], h[layer]), "cka_with_good": linear_cka(a[layer], g[layer]),
                    "mean_activation_norm": float(np.linalg.norm(a[layer], axis=1).mean())} for layer in ("layer1", "layer2")}


def describe(setup: dict, cfg: dict, state: dict, start, teacher, layers: list[str], rank: int | None) -> dict:
    """Everything reported about one merged network."""
    crit = setup["config"]["repair_criterion"]
    net = QNetwork()
    net.load_state_dict(state)
    net.eval()
    base = start.state_dict()
    rc = cfg["random_controls"]
    out = {"update": update_summary(base, state),
           "decisions": decision_diagnostics(net, start, teacher, setup["states"], setup["params"]),
           "similarity": similarity(net, start, teacher, setup["states"]["obs"]),
           "weights_sha256": weights_sha256(state),
           "random_full_rank": aggregate([dev_behavior(random_edit(start, net, layers, rc["seed"] + d), setup) for d in range(rc["n_draws"])])}
    if rank is not None:
        out["random_low_rank"] = aggregate([dev_behavior(random_low_rank_edit(base, state, layers, rank, rc["seed"] + d), setup)
                                            for d in range(rc["n_draws"])])
    out["meets_repair_criterion_dev"] = meets_repair(dev_behavior(net, setup), crit)
    return out


def _job(job: dict) -> dict:
    setup, cfg = load_setup(), load_config()
    dc, params, crit = setup["config"]["distillation"], setup["params"], setup["config"]["repair_criterion"]
    seed, k = job["dqn_seed"], job["opt_seed"]
    hp = {"n_updates": dc["n_updates"], "eval_interval_updates": dc["eval_interval_updates"], "batch_size": dc["batch_size"],
          "learning_rate": cfg["training"]["learning_rates"][job["lr_name"]]}
    start, teacher = net_at(setup, seed, dc["start"]), net_at(setup, seed, dc["teacher"])
    base = {key: v.clone() for key, v in start.state_dict().items()}
    obs = distill_states(setup)["obs"]
    dev = layouts_to_array(setup["layouts"]["dev"])
    select = np.arange(len(dev)) < dc["selection"]["n_dev_select"]
    batch_seed, init_seed = seed + 1000 * k, 1000 * k + seed
    if job["method"] == "full_network":
        run = distill(base, teacher, job["layers"], obs, dev, params, hp, batch_seed, select)
        counts = {"factor_parameters": 0, "bias_parameters": 260, "trainable_parameters": count_parameters(job["layers"]),
                  "weights_in_edited_matrices": count_parameters(job["layers"]) - 260, "per_layer": None}
        base_unchanged = True
    else:
        run = distill_low_rank(base, teacher, job["layers"], job["rank"], obs, dev, params, hp, init_seed, batch_seed, select)
        counts = parameter_counts(job["layers"], job["rank"])
        base_unchanged = run["base_unchanged"] and all(torch.equal(base[key], start.state_dict()[key]) for key in base)
        assert run["n_trainable"] == counts["trainable_parameters"]
    evals, snaps = run["evals"], run["snapshots"]
    flags = [meets_repair(e["dev"], crit) for e in evals]
    sustained = first_sustained(flags)
    sel = select_checkpoint(evals, "dev_select", dc["selection"]["metric"])
    selected = {"step": evals[sel]["step"], "train_loss": evals[sel]["train_loss"], "dev": evals[sel]["dev"],
                "dev_select": evals[sel]["dev_select"], "dev_report": evals[sel]["dev_report"],
                "meets_repair_criterion_dev_report": meets_repair(evals[sel]["dev_report"], crit),
                **describe(setup, cfg, snaps[sel], start, teacher, job["layers"], job["rank"])}
    out = {**job, "run_id": run_id(job), "learning_rate": hp["learning_rate"], "parameters": counts,
           "original_weights_unchanged": base_unchanged, "train_seconds": run["train_seconds"],
           "curve": [{"step": e["step"], "train_loss": e["train_loss"], **{key: e["dev"][key] for key in CURVE_KEYS}} for e in evals],
           "first_step_meeting_criterion": next((evals[i]["step"] for i, f in enumerate(flags) if f), None),
           "sustained_from_step": evals[sustained]["step"] if sustained is not None else None,
           "selected": selected,
           "final": {"step": evals[-1]["step"], "train_loss": evals[-1]["train_loss"], "dev": evals[-1]["dev"], "meets_repair_criterion_dev": flags[-1]}}
    if job["method"] != "full_network":
        path = Path(cfg["results_dir"]) / "edits" / f"{out['run_id']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"factors": run["factors"][sel], "layers": job["layers"], "rank": job["rank"], "dqn_seed": seed, "opt_seed": k,
                    "base": dc["start"], "base_weights_sha256": weights_sha256(base), "selected_step": evals[sel]["step"],
                    "merged_weights_sha256": selected["weights_sha256"]}, path)
        out["edit_file"] = str(path)
    return out


def stats(values: list) -> dict | None:
    vals = [v for v in values if v is not None]
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "std": float(np.std(vals))} if vals else None


def summarize_runs(runs: list[dict], cfg: dict) -> dict:
    """Per method, rank and learning rate: every run's outcome, counts of repaired runs, and spread."""
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["method"], r["rank"], r["lr_name"]), []).append(r)
    out = []
    for (method, rank, lr_name), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1], kv[0][2])):
        rep = lambda r: r["selected"]["dev_report"]
        per_seed = {}
        for seed in cfg["seeds"]:
            sr = sorted([r for r in rs if r["dqn_seed"] == seed], key=lambda r: r["opt_seed"])
            per_seed[str(seed)] = {"direct_goal": [rep(r)["direct_goal"] for r in sr],
                                   "path_efficiency": [rep(r)["path_efficiency_direct"] for r in sr],
                                   "farming": [rep(r)["farming"] for r in sr], "other": [rep(r)["other"] for r in sr],
                                   "repaired": [r["selected"]["meets_repair_criterion_dev_report"] for r in sr],
                                   "n_repaired": int(sum(r["selected"]["meets_repair_criterion_dev_report"] for r in sr))}
        n_rep = int(sum(v["n_repaired"] for v in per_seed.values()))
        best = max(rs, key=lambda r: (rep(r)["direct_goal"], rep(r)["path_efficiency_direct"] or 0))
        out.append({
            "method": method, "rank": rank, "lr_name": lr_name, "learning_rate": rs[0]["learning_rate"], "n_runs": len(rs),
            "trainable_parameters": rs[0]["parameters"]["trainable_parameters"],
            "factor_parameters": rs[0]["parameters"]["factor_parameters"], "bias_parameters": rs[0]["parameters"]["bias_parameters"],
            "n_repaired": n_rep, "reliably_successful": n_rep == len(rs), "seeds_with_all_runs_repaired": int(sum(v["n_repaired"] == len(v["repaired"]) for v in per_seed.values())),
            "per_seed": per_seed,
            "direct_goal": stats([rep(r)["direct_goal"] for r in rs]), "path_efficiency": stats([rep(r)["path_efficiency_direct"] for r in rs]),
            "farming": stats([rep(r)["farming"] for r in rs]), "other": stats([rep(r)["other"] for r in rs]),
            "farm_then_finish": stats([rep(r)["farm_then_finish"] for r in rs]), "farm_only": stats([rep(r)["farm_only"] for r in rs]),
            "tile_visits": stats([rep(r)["mean_reward_tile_visits"] for r in rs]), "steps_to_goal": stats([rep(r)["mean_steps_to_goal"] for r in rs]),
            "agree_with_teacher": stats([r["selected"]["decisions"]["agree_with_good"] for r in rs]),
            "toward_goal": stats([r["selected"]["decisions"]["toward_goal"] for r in rs]),
            "network_relative_norm": stats([r["selected"]["update"]["network"]["relative_norm"] for r in rs]),
            "layer_relative_norm": {l: stats([r["selected"]["update"][l]["relative_norm"] for r in rs]) for l in LAYERS},
            "effective_rank": {l: sorted({r["selected"]["update"][l]["effective_rank"] for r in rs}) for l in LAYERS},
            "weights_changed": stats([r["selected"]["update"]["network"]["weights_changed"] for r in rs]),
            "layer2_cka_with_hacking": stats([r["selected"]["similarity"]["layer2"]["cka_with_hacking"] for r in rs]),
            "layer2_cka_with_good": stats([r["selected"]["similarity"]["layer2"]["cka_with_good"] for r in rs]),
            "final_train_loss": stats([r["final"]["train_loss"] for r in rs]),
            "random_full_rank_direct_goal": stats([r["selected"]["random_full_rank"]["direct_goal"]["mean"] for r in rs]),
            "random_low_rank_direct_goal": stats([r["selected"]["random_low_rank"]["direct_goal"]["mean"] for r in rs]) if rank is not None else None,
            "random_low_rank_farming": stats([r["selected"]["random_low_rank"]["farming"]["mean"] for r in rs]) if rank is not None else None,
            "best_single_run": {"run_id": best["run_id"], "direct_goal": rep(best)["direct_goal"], "path_efficiency": rep(best)["path_efficiency_direct"]},
            "sustained_from_step": [r["sustained_from_step"] for r in sorted(rs, key=lambda r: (r["dqn_seed"], r["opt_seed"]))],
        })
    return {"groups": out}


def phase5_baselines(setup: dict, cfg: dict) -> dict:
    """Existing baselines, read from the Phase 5 result files instead of being rerun."""
    p5 = Path(cfg["phase5_results"])
    base, ds = json.load(open(p5 / "baselines.json")), json.load(open(p5 / "distill.json"))
    out: dict = {"source": str(p5), "random_policy": base["references"]["dev"]["random_policy"],
                 "scripted_shortest_path": base["references"]["dev"]["scripted_shortest_path"], "per_seed": {}}
    for seed in cfg["seeds"]:
        s = str(seed)
        out["per_seed"][s] = {
            "hacking": base["per_seed"][s]["networks"]["hacking_final"]["dev"],
            "good": base["per_seed"][s]["networks"]["good_final"]["dev"],
            "distillation": {c: {"trainable_parameters": r["n_trainable_parameters"], "dev_report": r["selected"]["dev_report"],
                                 "meets_repair_criterion_dev_report": r["selected"]["meets_repair_criterion_dev_report"],
                                 "weight_change": r["selected"]["weight_change"], "agree_with_teacher": r["selected"]["decisions"]["agree_with_good"],
                                 "weights_sha256": r["selected"]["weights_sha256"],
                                 "random_full_rank_direct_goal": r["selected"]["random_control"]["direct_goal"]["mean"]}
                             for c, r in ds["per_seed"][s].items() if "curve" in r}}
    return out


def svd_truncation(setup: dict, cfg: dict) -> dict:
    """Truncate the Phase 5 full-network distillation update to its top singular directions and evaluate."""
    dc, crit = setup["config"]["distillation"], setup["config"]["repair_criterion"]
    out = {}
    for seed in cfg["seeds"]:
        start, teacher = net_at(setup, seed, dc["start"]), net_at(setup, seed, dc["teacher"])
        full = torch.load(Path(cfg["phase5_results"]) / "distilled" / f"seed{seed}_full_network.pt", weights_only=False)["q_net_state_dict"]
        base = start.state_dict()
        block = {"full_update": update_summary(base, full), "ranks": {}}
        for rank in cfg["svd_truncation"]["ranks"]:
            net = truncate_update(base, full, rank)
            b = dev_behavior(net, setup)
            block["ranks"][str(rank)] = {"dev": b, "meets_repair_criterion_dev": meets_repair(b, crit),
                                         "update": update_summary(base, net.state_dict()),
                                         "decisions": decision_diagnostics(net, start, teacher, setup["states"], setup["params"])}
            print(f"[svd seed {seed}] rank {rank}: DG {b['direct_goal']:.3f} farm {b['farming']:.3f} other {b['other']:.3f} "
                  f"eff {b['path_efficiency_direct']}", flush=True)
        out[str(seed)] = block
    return out


def main() -> None:
    t0 = time.time()
    setup, cfg = load_setup(), load_config()
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(cfg)
    print(f"{len(jobs)} training runs", flush=True)
    runs = []
    with ProcessPoolExecutor(max_workers=cfg["training"]["max_workers"]) as pool:
        for r in pool.map(_job, jobs):
            runs.append(r)
            s = r["selected"]
            eff = s["dev_report"]["path_efficiency_direct"]
            print(f"[{r['run_id']:>44}] {r['parameters']['trainable_parameters']:>5} params | sel {s['step']:>5} DG {s['dev_report']['direct_goal']:.3f} "
                  f"farm {s['dev_report']['farming']:.3f} eff {eff if eff is None else round(eff, 3)} repair {s['meets_repair_criterion_dev_report']} "
                  f"| norm {s['update']['network']['relative_norm']:.2f} | loss {r['final']['train_loss']:.4f} | {r['train_seconds']}s", flush=True)
    header = {"manifest_sha256": setup["manifest_sha256"], "splits_used": ["train", "dev"], "seeds": cfg["seeds"],
              "n_dev_layouts": len(setup["layouts"]["dev"]), "repair_criterion": setup["config"]["repair_criterion"],
              "training": {**{k: setup["config"]["distillation"][k] for k in ("n_updates", "eval_interval_updates", "batch_size", "loss", "n_train_layouts",
                                                                              "states_per_layout", "state_seed")},
                           "learning_rates": cfg["training"]["learning_rates"], "optimization_seeds": cfg["training"]["optimization_seeds"]}}
    json.dump({**header, "runs": runs}, open(out_dir / "runs.json", "w"), indent=1)
    summary = {**header, **summarize_runs(runs, cfg), "baselines": phase5_baselines(setup, cfg), "svd_truncation": svd_truncation(setup, cfg)}
    # The full-network control at the Phase 5 rate and optimization seed 0 must be the Phase 5 run.
    summary["reproduces_phase5_full_network"] = {
        str(seed): next(r for r in runs if r["method"] == "full_network" and r["lr_name"] == "primary" and r["opt_seed"] == 0
                        and r["dqn_seed"] == seed)["selected"]["weights_sha256"]
        == summary["baselines"]["per_seed"][str(seed)]["distillation"]["full_network"]["weights_sha256"] for seed in cfg["seeds"]}
    coord = [g for g in summary["groups"] if g["method"] == "coordinated"]
    summary["minimum_reliable_rank"] = {lr: next((g["rank"] for g in sorted(coord, key=lambda g: g["rank"])
                                                   if g["lr_name"] == lr and g["reliably_successful"]), None)
                                        for lr in cfg["low_rank"]["coordinated"]["learning_rates"]}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    for g in summary["groups"]:
        print(f"{g['method']:>12} rank {str(g['rank']):>4} {g['lr_name']:>9} | {g['trainable_parameters']:>5} params | repaired {g['n_repaired']}/{g['n_runs']} "
              f"| DG {g['direct_goal']['mean']:.3f} [{g['direct_goal']['min']:.3f}, {g['direct_goal']['max']:.3f}] | eff {g['path_efficiency']['mean']:.3f} "
              f"| norm {g['network_relative_norm']['mean']:.2f}", flush=True)
    print("reproduces Phase 5 full network:", summary["reproduces_phase5_full_network"], "| minimum reliable rank:", summary["minimum_reliable_rank"])
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
