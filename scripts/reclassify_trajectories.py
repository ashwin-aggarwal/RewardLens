"""Phase 4.6: retrospective behavioral reclassification of Phase 1-4.5 trajectories.

Applies the four-category classification (direct_goal / farm_then_finish /
farm_only / other) to every trajectory the repository can reproduce:

  stored      Phase 1 scripted trajectories, Phase 2 / 2.5 final greedy
              trajectories, Phase 4 held-out test episodes (all conditions and
              seeds), Phase 4.5 matched-patch test episodes.
  regenerated Phase 2 / 2.5 greedy episodes at every stored checkpoint (default
              start and all 62 valid starts), and the Phase 4.5 test conditions
              whose per-episode detail was not stored (baseline, controls,
              donor, cross-model). Policies and the environment are
              deterministic, so regeneration reproduces the original runs; the
              script asserts that regenerated goal rates match the stored
              summaries.

Original classifications (`is_reward_hacking`) are kept alongside the new ones.
Nothing in results/phase2*, results/phase4 or results/phase4_5 is modified.

Usage:
    python -m scripts.reclassify_trajectories
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_extraction import build_matched_state_dataset, extract_activations
from rewardlens.analysis.activation_patching import (
    evaluate_patching,
    fit_linear_mapping,
    identity_mapping,
    make_matched_donor,
    make_shuffled_donor,
)
from rewardlens.analysis.behavior_classification import (
    CATEGORIES,
    category_rates,
    classify_trajectory,
    goal_step,
    threshold_sensitivity,
    visits_vs_completion,
)
from rewardlens.analysis.steering import evaluate_steering
from rewardlens.envs.reward_grid import RewardGridConfig
from rewardlens.envs.trajectory import Trajectory
from scripts.run_interventions import _position_mask

OUT_DIR = Path("results/phase4_6")
CONFIG_PATH = Path("configs/phase4_6.json")
FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")
SEEDS = (42, 123, 456)
CHECKPOINT_STEPS = list(range(10000, 100001, 10000))


def _rates(eps, thr):
    r = category_rates(eps, thr)
    r.pop("categories", None)
    return r


def _traj_row(path: Path, label: str, thr: int) -> dict | None:
    if not path.exists():
        return None
    tr = Trajectory.load(path)
    return {
        "label": label, "source": "stored", "path": str(path),
        "goal_reached": tr.goal_reached, "reward_tile_visits": tr.final_reward_tile_visits,
        "episode_steps": tr.total_steps, "goal_step": goal_step(tr),
        "original_is_reward_hacking": bool(tr.truncated and not tr.goal_reached
                                           and tr.final_reward_tile_visits >= thr),
        "category": classify_trajectory(tr, thr),
    }


def _ckpt_path(cond: str, seed: int, step: int) -> Path:
    base = FLAWED_DIR / f"baseline_seed{seed}" if cond == "flawed" else CORRECTED_DIR / f"goal_seed{seed}"
    name = "checkpoint_final.pt" if step == 100000 else f"checkpoint_{step}.pt"
    return base / "checkpoints" / name


def reclassify_stored_trajectories(thr: int) -> dict:
    rows = []
    for name in ("goal", "exploit"):
        r = _traj_row(Path("results/phase1") / f"{name}_trajectory.json", f"phase1_scripted_{name}", thr)
        if r:
            rows.append(r)
    for seed in SEEDS:
        r = _traj_row(FLAWED_DIR / f"baseline_seed{seed}" / "evaluation_trajectories" / "final_greedy_trajectory.json",
                      f"phase2_flawed_seed{seed}_final", thr)
        if r:
            rows.append(r)
        r = _traj_row(CORRECTED_DIR / f"goal_seed{seed}" / "evaluation_trajectories" / "final_greedy_trajectory.json",
                      f"phase2_5_corrected_seed{seed}_final", thr)
        if r:
            rows.append(r)
    return {"rows": rows}


def reclassify_checkpoints(thr: int) -> dict:
    """Regenerate greedy episodes at each checkpoint (deterministic)."""
    env_cfg = RewardGridConfig()
    all_starts = [(x, y) for x in range(8) for y in range(8)
                  if (x, y) not in (env_cfg.goal_pos, env_cfg.reward_pos)]
    out = {}
    for cond in ("flawed", "corrected"):
        for seed in SEEDS:
            per_step = {}
            for step in CHECKPOINT_STEPS:
                p = _ckpt_path(cond, seed, step)
                if not p.exists():
                    continue
                net, _ = load_q_network_for_eval(p)
                default = evaluate_steering(net, env_cfg, [env_cfg.agent_start], threshold=thr)
                allpos = evaluate_steering(net, env_cfg, all_starts, threshold=thr)
                per_step[str(step)] = {
                    "default_start": {"category": default["episodes"][0]["category"],
                                      "reward_tile_visits": default["episodes"][0]["reward_tile_visits"],
                                      "goal_step": default["episodes"][0]["goal_step"],
                                      "original_is_reward_hacking": default["episodes"][0]["is_reward_hacking"]},
                    "all_62_starts": {k: v for k, v in allpos.items() if k != "episodes"},
                }
            out[f"{cond}_seed{seed}"] = {"source": "regenerated_from_checkpoints", "per_step": per_step}
            if per_step:
                last = per_step[str(max(int(s) for s in per_step))]["all_62_starts"]
                print(f"  {cond} seed {seed} final (62 starts): "
                      + " ".join(f"{c}={last[f'{c}_rate']:.2f}" for c in CATEGORIES))
    return out


def reclassify_phase4(thr: int) -> dict:
    p = Path("results/phase4/intervention_results.json")
    if not p.exists():
        return {"available": False}
    d = json.load(open(p))
    out = {"available": True, "source": "stored_test_episodes", "flawed": {}}
    for seed, blk in d["flawed"].items():
        t = blk["test"]
        row = {}
        for cond in ("baseline", "probe_subtract", "probe_add", "probe_project_out", "contrastive_subtract"):
            if cond in t and "episodes" in t[cond]:
                row[cond] = _rates(t[cond]["episodes"], thr)
        if "random_subtract" in t:
            rs = [_rates(r["episodes"], thr) for r in t["random_subtract"] if "episodes" in r]
            row["random_subtract"] = {
                "n": len(rs),
                **{f"mean_{c}_rate": float(np.mean([r[f"{c}_rate"] for r in rs])) for c in CATEGORIES},
                "mean_reward_tile_visits": float(np.mean([r["mean_reward_tile_visits"] for r in rs])),
            }
        out["flawed"][seed] = {"selected_alpha": blk.get("selected_alpha"), "test": row}
        ps = row.get("probe_subtract", {})
        print(f"  Phase 4 seed {seed} probe_subtract: "
              + " ".join(f"{c}={ps.get(f'{c}_rate', float('nan')):.2f}" for c in CATEGORIES))
    return out


def reclassify_phase4_5(thr: int, cfg46: dict) -> dict:
    p = Path("results/phase4_5/patching_results.json")
    if not p.exists():
        return {"available": False}
    d = json.load(open(p))
    cfg = d["config"]
    parts = {k: [tuple(x) for x in v] for k, v in d["partition"].items()}
    env_cfg = RewardGridConfig()
    observations = build_matched_state_dataset()
    pool_obs = observations[_position_mask(observations, parts["train"])]
    ck = FLAWED_DIR / "baseline_seed123" / "checkpoints"
    out = {"available": True, "partition_seed": cfg["partition_seed"], "same_run": {}, "cross_model": {}}

    # Stored per-episode detail for the matched 30k condition
    ep_path = Path("results/phase4_5/same_run_30k_matched_test_episodes.json")
    if ep_path.exists():
        ep = json.load(open(ep_path))
        out["same_run"]["30000_matched_stored"] = {"source": "stored", "alpha": ep["alpha"],
                                                   **_rates(ep["episodes"], thr),
                                                   "rows": visits_vs_completion(ep["episodes"], thr)}

    def block(label, recipient, donor, mapping, blk):
        sel = blk["selected_alpha"]
        donor_fn = make_matched_donor(donor, mapping)
        res = {"source": "regenerated", "selected_alpha": sel}
        conds = {
            "baseline": evaluate_patching(recipient, env_cfg, parts["test"], 0.0),
            "matched": evaluate_patching(recipient, env_cfg, parts["test"], sel, donor_fn, "matched"),
            "donor_independent": evaluate_patching(donor, env_cfg, parts["test"], 0.0),
        }
        # constant Q-bias control vector: mean dev delta, as in Phase 4.5
        import torch
        obs_dev = torch.as_tensor(observations[_position_mask(observations, parts["dev"])], dtype=torch.float32)
        with torch.no_grad():
            h_rec = recipient.relu2(recipient.fc2(recipient.relu1(recipient.fc1(obs_dev))))
            h_don = donor_fn(obs_dev)
        cdelta = (h_don - h_rec).mean(0).numpy()
        conds["constant_qbias"] = evaluate_patching(recipient, env_cfg, parts["test"], sel, None, "constant",
                                                    constant_delta=cdelta)
        for name, r in conds.items():
            stored = blk["test"][name]["goal_rate"]
            assert abs(r["goal_rate"] - stored) < 1e-9, f"{label}/{name}: regenerated {r['goal_rate']} != stored {stored}"
            res[name] = {**_rates(r["episodes"], thr), "stored_goal_rate": stored,
                         "rows": visits_vs_completion(r["episodes"], thr)}
        rnd = [evaluate_patching(recipient, env_cfg, parts["test"], sel, donor_fn, "random",
                                 random_seed=cfg["control_seed"] + i) for i in range(cfg["n_controls"])]
        shf = [evaluate_patching(recipient, env_cfg, parts["test"], sel,
                                 make_shuffled_donor(donor, pool_obs, cfg["control_seed"] + i, mapping), "matched")
               for i in range(cfg["n_controls"])]
        for name, lst in (("random", rnd), ("shuffled", shf)):
            stored = [r["goal_rate"] for r in blk["test"][name]]
            got = [r["goal_rate"] for r in lst]
            assert np.allclose(stored, got), f"{label}/{name}: regenerated controls differ from stored"
            rates = [_rates(r["episodes"], thr) for r in lst]
            res[name] = {"n": len(rates),
                         **{f"mean_{c}_rate": float(np.mean([x[f"{c}_rate"] for x in rates])) for c in CATEGORIES},
                         **{f"max_{c}_rate": float(np.max([x[f"{c}_rate"] for x in rates])) for c in CATEGORIES},
                         "mean_reward_tile_visits": float(np.mean([x["mean_reward_tile_visits"] for x in rates]))}
        res["matched_sweep"] = {}
        for a in cfg["alphas"]:
            r = evaluate_patching(recipient, env_cfg, parts["test"], float(a), donor_fn, "matched")
            res["matched_sweep"][str(a)] = {**_rates(r["episodes"], thr),
                                            "threshold_sensitivity": threshold_sensitivity(
                                                r["episodes"], cfg46["threshold_sensitivity"])}
        m = res["matched"]
        print(f"  {label}: matched a={sel} " + " ".join(f"{c}={m[f'{c}_rate']:.2f}" for c in CATEGORIES)
              + f" visits={m['mean_reward_tile_visits']:.1f}")
        return res

    donor, _ = load_q_network_for_eval(ck / f"checkpoint_{cfg['same_run']['donor_step']}.pt")
    for rstep in cfg["same_run"]["recipient_steps"]:
        name = "checkpoint_final.pt" if rstep == 100000 else f"checkpoint_{rstep}.pt"
        recipient, _ = load_q_network_for_eval(ck / name)
        out["same_run"][str(rstep)] = block(f"Phase 4.5 same-run 20k->{rstep}", recipient, donor, None,
                                            d["same_run"][str(rstep)])

    seed = cfg["seed"]
    cpath = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints" / "checkpoint_final.pt"
    donor_c, _ = load_q_network_for_eval(cpath)
    recipient_f, _ = load_q_network_for_eval(ck / "checkpoint_final.pt")
    rec_d = extract_activations(cpath, observations, seed, "corrected", 100000)
    rec_r = extract_activations(ck / "checkpoint_final.pt", observations, seed, "flawed", 100000)
    fit_mask = _position_mask(observations, parts["train"])
    agree = rec_d.greedy_actions == rec_r.greedy_actions
    mappings = {
        "identity": identity_mapping(),
        "ridge_all": fit_linear_mapping(rec_d.hidden2[fit_mask], rec_r.hidden2[fit_mask],
                                        cfg["cross_model"]["ridge_alpha"], "ridge_all"),
        "ridge_agreement": fit_linear_mapping(rec_d.hidden2[fit_mask & agree], rec_r.hidden2[fit_mask & agree],
                                              cfg["cross_model"]["ridge_alpha"], "ridge_agreement"),
    }
    for name in cfg["cross_model"]["mappings"]:
        out["cross_model"][name] = block(f"Phase 4.5 cross-model [{name}]", recipient_f, donor_c, mappings[name],
                                         d["cross_model"][name])
    return out


def main() -> None:
    cfg46 = json.load(open(CONFIG_PATH))
    thr = int(cfg46["visit_threshold"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reclassifying with visit threshold {thr}")
    results = {"visit_threshold": thr, "categories": list(CATEGORIES)}
    print("Stored step-level trajectories:")
    results["stored_trajectories"] = reclassify_stored_trajectories(thr)
    for r in results["stored_trajectories"]["rows"]:
        print(f"  {r['label']:34s} {r['category']:17s} visits={r['reward_tile_visits']:3d} goal_step={r['goal_step']}")
    print("Checkpoint regeneration:")
    results["checkpoints"] = reclassify_checkpoints(thr)
    print("Phase 4 stored test episodes:")
    results["phase4"] = reclassify_phase4(thr)
    print("Phase 4.5 (stored + regenerated):")
    results["phase4_5"] = reclassify_phase4_5(thr, cfg46)
    out = OUT_DIR / "reclassification.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
