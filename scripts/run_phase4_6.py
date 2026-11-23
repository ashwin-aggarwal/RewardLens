"""Phase 4.6: behavioral reclassification, time-dependence analysis, and
direct-goal steering experiments.

Donor = seed-123 flawed-run 20k checkpoint (goal-following).
Recipient = seed-123 flawed-run 30k checkpoint (reward-hacking).

A fresh prespecified start-state partition (partition_seed=13) is used for
selection (dev) and reporting (test), because the Phase 4.5 held-out test
states (partition_seed=11) were reported at every patch strength and have
therefore already informed this phase's design. With only 62 valid start
positions the new test set necessarily overlaps the old splits; the overlap is
recorded in the output.

Selection on dev uses the Phase 4.6 primary metric: direct-goal rate
(tie-break: fewer tile visits, then less total intervention).

Usage:
    python -m scripts.run_phase4_6
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rewardlens.agents.dqn import load_q_network_for_eval
from rewardlens.analysis.activation_extraction import build_matched_state_dataset
from rewardlens.analysis.activation_intervention import partition_start_positions
from rewardlens.analysis.behavior_classification import CATEGORIES
from rewardlens.analysis.steering import (
    PatchController,
    clamp_time_tensor,
    evaluate_steering,
    make_matched_donor_layer,
    make_schedule,
    make_shuffled_donor_layer,
    mean_layer_delta,
    scheduled_patching,
    schedule_constant,
    summarize,
)
from rewardlens.analysis.time_dependence import (
    clamp_time,
    earliest_goal_preferring_time,
    goal_preference_by_time,
    patched_q_components,
)
from rewardlens.envs.reward_grid import RewardGridConfig
from scripts.run_interventions import _position_mask

FLAWED_DIR = Path("results/phase2")
OUTPUT_DIR = Path("results/phase4_6")
CONFIG_PATH = Path("configs/phase4_6.json")


def _ckpt(step: int) -> Path:
    name = "checkpoint_final.pt" if step == 100000 else f"checkpoint_{step}.pt"
    return FLAWED_DIR / "baseline_seed123" / "checkpoints" / name


def _fmt(r: dict) -> str:
    eff = r.get("mean_path_efficiency_direct")
    return (" ".join(f"{c[:6]}={r[f'{c}_rate']:.2f}" for c in CATEGORIES)
            + f" visits={r['mean_reward_tile_visits']:4.1f} steps={r['mean_episode_steps']:5.1f}"
            + f" eff={eff if eff is None else round(eff, 2)}"
            + f" norm={r['mean_applied_norm']:5.1f} ratio={r['mean_norm_ratio']:.2f}"
            + f" neg={r['mean_negative_fraction']:.3f}")


def _spec_label(spec: dict, alpha: float, tclamp=None) -> str:
    params = ",".join(f"{k}={v}" for k, v in spec.items() if k != "name")
    s = f"{spec['name']}({params})@a={alpha}" if params else f"{spec['name']}@a={alpha}"
    return s + (f"+donor_t={tclamp}" if tclamp is not None else "")


def _donor_transform(tclamp):
    return clamp_time_tensor(tclamp) if tclamp is not None else None


def control_suite(recipient, donor, env_cfg, positions, schedule, layer, cfg, pool_obs, cdelta, thr, tclamp=None):
    """All Phase 4.6 controls under the SAME schedule, layer and donor transform as the intervention."""
    tr = _donor_transform(tclamp)
    donor_fn = make_matched_donor_layer(donor, layer, tr)
    rnd = [summarize(evaluate_steering(recipient, env_cfg, positions, donor_fn, schedule, layer, "random",
                                       random_seed=cfg["control_seed"] + i, threshold=thr))
           for i in range(cfg["n_controls"])]
    shf_full = [evaluate_steering(recipient, env_cfg, positions,
                                  make_shuffled_donor_layer(donor, pool_obs, cfg["control_seed"] + i, layer, tr),
                                  schedule, layer, "matched", threshold=thr, keep_traces=(i == 0))
                for i in range(cfg["n_controls"])]
    shf = [summarize(r) for r in shf_full]
    const = summarize(evaluate_steering(recipient, env_cfg, positions, None, schedule, layer, "constant",
                                        constant_delta=cdelta, threshold=thr))

    def agg(lst):
        eff = [r["mean_path_efficiency_direct"] for r in lst if r.get("mean_path_efficiency_direct") is not None]
        return {"n": len(lst),
                **{f"mean_{c}_rate": float(np.mean([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
                **{f"max_{c}_rate": float(np.max([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
                **{f"std_{c}_rate": float(np.std([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
                "mean_reward_tile_visits": float(np.mean([r["mean_reward_tile_visits"] for r in lst])),
                "mean_episode_steps": float(np.mean([r["mean_episode_steps"] for r in lst])),
                "mean_no_move_steps": float(np.mean([r["mean_no_move_steps"] for r in lst])),
                "mean_path_efficiency_direct": float(np.mean(eff)) if eff else None,
                "mean_action_fractions": {str(a): float(np.mean([r["action_fractions"][str(a)] for r in lst]))
                                          for a in range(4)},
                "per_control": lst}
    out = {"random": agg(rnd), "shuffled": agg(shf), "constant_qbias": const,
           "shuffled_example_episodes": shf_full[0]["episodes"]}
    return out


def _margins(matched: dict, test: dict) -> dict:
    """Matched-patch direct-goal rate minus each control's mean (and max) direct-goal rate."""
    return {
        "direct_goal_minus_random_mean": matched["direct_goal_rate"] - test["random"]["mean_direct_goal_rate"],
        "direct_goal_minus_random_max": matched["direct_goal_rate"] - test["random"]["max_direct_goal_rate"],
        "direct_goal_minus_shuffled_mean": matched["direct_goal_rate"] - test["shuffled"]["mean_direct_goal_rate"],
        "direct_goal_minus_shuffled_max": matched["direct_goal_rate"] - test["shuffled"]["max_direct_goal_rate"],
        "direct_goal_minus_constant_qbias": matched["direct_goal_rate"] - test["constant_qbias"]["direct_goal_rate"],
    }


def run_block(name, spec_block, recipient, donor, env_cfg, parts, cfg, pool_obs, obs_dev, thr):
    """Dev sweep over (schedule, alpha[, donor time clamp]), selection by direct-goal rate,
    held-out test with controls under the identical schedule/layer/transform."""
    layer = spec_block["layer"]
    tclamps = spec_block.get("donor_time_clamps", [None])
    print(f"\n=== Block {name} (layer {layer}) ===")
    dev = {}
    for tclamp in tclamps:
        donor_fn = make_matched_donor_layer(donor, layer, _donor_transform(tclamp))
        for spec in spec_block["schedules"]:
            for a in spec_block["alphas"]:
                sched = make_schedule(spec["name"], a, **{k: v for k, v in spec.items() if k != "name"})
                r = summarize(evaluate_steering(recipient, env_cfg, parts["dev"], donor_fn, sched, layer, "matched",
                                                threshold=thr))
                label = _spec_label(spec, a, tclamp)
                dev[label] = {"spec": spec, "alpha": a, "donor_time_clamp": tclamp, **r}
                print(f"  dev {label:40s} {_fmt(r)}")
    best = min(dev, key=lambda k: (-dev[k]["direct_goal_rate"], dev[k]["mean_reward_tile_visits"],
                                   dev[k]["mean_applied_norm"] * dev[k]["mean_active_steps"]))
    spec, a, tclamp = dev[best]["spec"], dev[best]["alpha"], dev[best]["donor_time_clamp"]
    sched = make_schedule(spec["name"], a, **{k: v for k, v in spec.items() if k != "name"})
    donor_fn = make_matched_donor_layer(donor, layer, _donor_transform(tclamp))
    print(f"  selected (dev direct-goal): {best}")

    cdelta = mean_layer_delta(recipient, donor_fn, obs_dev, layer)
    test_matched = evaluate_steering(recipient, env_cfg, parts["test"], donor_fn, sched, layer, "matched",
                                     threshold=thr, keep_traces=True)
    test = {"matched": test_matched,
            **control_suite(recipient, donor, env_cfg, parts["test"], sched, layer, cfg, pool_obs, cdelta, thr, tclamp)}
    test["margins"] = _margins(test_matched, test)
    print(f"  TEST matched   {_fmt(test_matched)}")
    for k in ("random", "shuffled"):
        t = test[k]
        print(f"  TEST {k:9s} direct={t['mean_direct_goal_rate']:.2f} (max {t['max_direct_goal_rate']:.2f})"
              f" ftf={t['mean_farm_then_finish_rate']:.2f} farm_only={t['mean_farm_only_rate']:.2f}"
              f" other={t['mean_other_rate']:.2f} visits={t['mean_reward_tile_visits']:.1f}"
              f" eff={t['mean_path_efficiency_direct']} actions={t['mean_action_fractions']}")
    print(f"  TEST qbias     {_fmt(test['constant_qbias'])}")
    print(f"  margins: {test['margins']}")
    return {"layer": layer, "dev_sweep": dev, "selected": best, "selected_spec": spec, "selected_alpha": a,
            "selected_donor_time_clamp": tclamp, "test": test, "constant_delta_norm": float(np.linalg.norm(cdelta))}


def time_dependence_analysis(recipient, donor, positions, cfg, pool_obs, env_cfg, thr):
    tf = cfg["time_fractions"]
    a45 = cfg["phase4_5_alpha"]
    donor_fn = make_matched_donor_layer(donor, 2)

    def patched_forward(alpha, mode="matched", fn=donor_fn, seed=0):
        def f(obs):
            with scheduled_patching(recipient, fn, schedule_constant(alpha), PatchController(), 2, mode,
                                    random_seed=seed), torch.no_grad():
                return recipient(obs)
        return f

    def donor_via_recipient_head(obs):
        with torch.no_grad():
            return recipient.fc3(donor.relu2(donor.fc2(donor.relu1(donor.fc1(obs)))))

    conditions = {
        "recipient_unmodified": lambda obs: recipient(obs),
        "donor_own_head": lambda obs: donor(obs),
        "donor_via_recipient_head": donor_via_recipient_head,
        f"patched_a{a45}": patched_forward(a45),
        "patched_a1.0": patched_forward(1.0),
        "patched_a1.5": patched_forward(1.5),
        f"patched_a{a45}_donor_t0.05": patched_forward(a45, "matched", make_matched_donor_layer(donor, 2, clamp_time_tensor(0.05))),
        f"random_a{a45}": patched_forward(a45, "random"),
        f"shuffled_a{a45}": patched_forward(a45, "matched", make_shuffled_donor_layer(donor, pool_obs, 0, 2)),
    }
    out = {"conditions": {}}
    print("\n=== Time dependence (matched positions x remaining time) ===")
    for name, fn in conditions.items():
        r = goal_preference_by_time(fn, positions, tf, env_cfg)
        r["earliest_goal_preferring_time"] = earliest_goal_preferring_time(r)
        out["conditions"][name] = r
        ga = r["goal_approach_fraction"]
        gq = r["mean_goal_minus_reward_q"]
        print(f"  {name:28s} goal-approach frac t=1.0:{ga[0]:.2f} t=0.5:{ga[tf.index(0.5)]:.2f} "
              f"t=0.1:{ga[tf.index(0.1)]:.2f} t=0.0:{ga[-1]:.2f} | Qgap t=1.0:{gq[0]:+.2f} t=0.0:{gq[-1]:+.2f}")

    # Exact affine decomposition check on the same observation grid
    from rewardlens.analysis.time_dependence import observations_for_positions
    obs, _, tvals = observations_for_positions(positions, tf, env_cfg)
    comp = patched_q_components(recipient, donor, torch.as_tensor(obs), a45)
    recon_err = float((comp["q_patched"] - comp["q_affine_reconstruction"]).abs().max())
    by_time = {}
    for t in tf:
        sel = np.isclose(tvals, float(t))
        by_time[str(t)] = {
            "mean_max_q_recipient": float(comp["q_recipient"][sel].max(1).values.mean()),
            "mean_max_q_donor_via_recipient_head": float(comp["q_donor_via_recipient_head"][sel].max(1).values.mean()),
            "mean_max_q_patched": float(comp["q_patched"][sel].max(1).values.mean()),
            "mean_q_range_recipient": float((comp["q_recipient"][sel].max(1).values - comp["q_recipient"][sel].min(1).values).mean()),
            "mean_q_range_donor_via_recipient_head": float((comp["q_donor_via_recipient_head"][sel].max(1).values - comp["q_donor_via_recipient_head"][sel].min(1).values).mean()),
        }
    out["affine_decomposition"] = {"alpha": a45, "max_reconstruction_error": recon_err, "by_time": by_time}
    print(f"  affine decomposition max error: {recon_err:.2e}")

    # Input-clamp diagnostic: fix the time feature the NETWORK sees
    print("  time-clamp rollouts (network sees fixed remaining time; env unchanged):")
    clamp = {}
    for tau in cfg["time_clamps"]:
        row = {}
        for name, (net, fn, alpha) in {"recipient": (recipient, None, 0.0), "donor": (donor, None, 0.0),
                                       f"patched_a{a45}": (recipient, donor_fn, a45),
                                       "patched_a1.0": (recipient, donor_fn, 1.0)}.items():
            r = summarize(evaluate_steering(net, env_cfg, positions, fn, schedule_constant(alpha), 2,
                                            "matched" if fn else "none", obs_transform=clamp_time(tau),
                                            threshold=thr))
            row[name] = r
        clamp[str(tau)] = row
        print(f"    tau={tau:<4} " + " | ".join(
            f"{n}: direct={r['direct_goal_rate']:.2f} ftf={r['farm_then_finish_rate']:.2f} "
            f"farm={r['farm_only_rate']:.2f} visits={r['mean_reward_tile_visits']:.0f}" for n, r in row.items()))
    out["time_clamp_rollouts"] = clamp
    return out


def main() -> None:
    cfg = json.load(open(CONFIG_PATH))
    thr = int(cfg["visit_threshold"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_cfg = RewardGridConfig()

    parts = partition_start_positions(env_cfg, n_dev=cfg["n_dev_positions"], n_test=cfg["n_test_positions"],
                                      seed=cfg["partition_seed"])
    prev = partition_start_positions(env_cfg, n_dev=cfg["n_dev_positions"], n_test=cfg["n_test_positions"],
                                     seed=cfg["previous_partition_seed"])
    overlap = {f"new_test_in_prev_{k}": len(set(parts["test"]) & set(prev[k])) for k in ("test", "dev", "train")}
    print(f"Partition seed {cfg['partition_seed']}: {len(parts['train'])} fit / {len(parts['dev'])} dev / "
          f"{len(parts['test'])} test. Overlap of new test with previous split: {overlap}")

    observations = build_matched_state_dataset()
    pool_obs = observations[_position_mask(observations, parts["train"])]
    obs_dev = observations[_position_mask(observations, parts["dev"])]

    donor, _ = load_q_network_for_eval(_ckpt(cfg["donor_step"]))
    recipient, _ = load_q_network_for_eval(_ckpt(cfg["recipient_step"]))
    results = {"config": cfg, "partition": {k: [list(p) for p in v] for k, v in parts.items()},
               "partition_overlap_with_previous": overlap}

    # ---- 1. Phase 4.5 patch and all controls, reclassified on the new test states ----
    a45 = cfg["phase4_5_alpha"]
    donor_fn2 = make_matched_donor_layer(donor, 2)
    const_sched = schedule_constant(a45)
    cdelta = mean_layer_delta(recipient, donor_fn2, obs_dev, 2)
    print("\n=== Reclassified baseline comparison (new test states) ===")
    base = {
        "unmodified": evaluate_steering(recipient, env_cfg, parts["test"], threshold=thr, keep_traces=True),
        "donor": evaluate_steering(donor, env_cfg, parts["test"], threshold=thr, keep_traces=True),
        "phase4_5_patch": evaluate_steering(recipient, env_cfg, parts["test"], donor_fn2, const_sched, 2, "matched",
                                            threshold=thr, keep_traces=True),
    }
    base.update(control_suite(recipient, donor, env_cfg, parts["test"], const_sched, 2, cfg, pool_obs, cdelta, thr))
    base["margins"] = _margins(base["phase4_5_patch"], base)
    for k in ("unmodified", "donor", "phase4_5_patch", "constant_qbias"):
        print(f"  {k:16s} {_fmt(base[k])}")
    for k in ("random", "shuffled"):
        t = base[k]
        print(f"  {k:16s} direct={t['mean_direct_goal_rate']:.2f} ftf={t['mean_farm_then_finish_rate']:.2f}"
              f" farm_only={t['mean_farm_only_rate']:.2f} other={t['mean_other_rate']:.2f} visits={t['mean_reward_tile_visits']:.1f}")
    results["baseline_comparison"] = base

    # ---- 2. Time dependence ----
    results["time_dependence"] = time_dependence_analysis(recipient, donor, parts["test"], cfg, pool_obs, env_cfg, thr)

    # ---- 3. Steering blocks ----
    blocks = {}
    for name, spec_block in cfg["blocks"].items():
        blocks[name] = run_block(name, spec_block, recipient, donor, env_cfg, parts, cfg, pool_obs, obs_dev, thr)
    results["blocks"] = blocks

    # ---- 4. Most promising new intervention (chosen on DEV) ----
    best_block = min(blocks, key=lambda b: (-blocks[b]["dev_sweep"][blocks[b]["selected"]]["direct_goal_rate"],
                                            blocks[b]["dev_sweep"][blocks[b]["selected"]]["mean_reward_tile_visits"]))
    results["most_promising"] = {"block": best_block, "selected": blocks[best_block]["selected"],
                                 "dev": summarize(blocks[best_block]["dev_sweep"][blocks[best_block]["selected"]]),
                                 "test": summarize(blocks[best_block]["test"]["matched"]),
                                 "test_margins": blocks[best_block]["test"]["margins"]}
    mp = results["most_promising"]
    p45 = base["phase4_5_patch"]
    print(f"\nMost promising (by dev direct-goal): {best_block} / {mp['selected']}")
    print(f"  TEST direct-goal: Phase 4.5 patch {p45['direct_goal_rate']:.2f} -> new {mp['test']['direct_goal_rate']:.2f}; "
          f"visits {p45['mean_reward_tile_visits']:.1f} -> {mp['test']['mean_reward_tile_visits']:.1f}; "
          f"margins vs controls {mp['test_margins']}")

    # ---- 5. Extra check on positions never used for diagnostics or selection ----
    # The time-dependence diagnostics were computed on the TEST positions and
    # motivated block D, so the test set is not fully naive with respect to
    # block D's design. The 30 fit positions were used only as the shuffled-
    # donor pool; the selected intervention and its controls are re-evaluated
    # there and on all 62 valid start positions.
    bb = blocks[best_block]
    spec, a, tclamp, layer = bb["selected_spec"], bb["selected_alpha"], bb["selected_donor_time_clamp"], bb["layer"]
    sched = make_schedule(spec["name"], a, **{k: v for k, v in spec.items() if k != "name"})
    donor_fn = make_matched_donor_layer(donor, layer, _donor_transform(tclamp))
    cdelta_b = mean_layer_delta(recipient, donor_fn, obs_dev, layer)
    all_positions = [(x, y) for x in range(env_cfg.width) for y in range(env_cfg.height)
                     if (x, y) not in (env_cfg.goal_pos, env_cfg.reward_pos)]
    extra = {}
    for label, positions in (("fit_positions_30", parts["train"]), ("all_positions_62", all_positions)):
        m = summarize(evaluate_steering(recipient, env_cfg, positions, donor_fn, sched, layer, "matched", threshold=thr))
        ctr = control_suite(recipient, donor, env_cfg, positions, sched, layer, cfg, pool_obs, cdelta_b, thr, tclamp)
        ctr.pop("shuffled_example_episodes")
        row = {"matched": m, "unmodified": summarize(evaluate_steering(recipient, env_cfg, positions, threshold=thr)),
               "donor": summarize(evaluate_steering(donor, env_cfg, positions, threshold=thr)),
               "phase4_5_patch": summarize(evaluate_steering(recipient, env_cfg, positions, donor_fn2, const_sched, 2,
                                                             "matched", threshold=thr)), **ctr}
        row["margins"] = _margins(m, row)
        extra[label] = row
        print(f"  EXTRA {label}: matched {_fmt(m)}")
        print(f"        shuffled direct={row['shuffled']['mean_direct_goal_rate']:.2f} (max {row['shuffled']['max_direct_goal_rate']:.2f})"
              f" random direct={row['random']['mean_direct_goal_rate']:.2f} donor direct={row['donor']['direct_goal_rate']:.2f}"
              f" phase4_5 direct={row['phase4_5_patch']['direct_goal_rate']:.2f}")
    results["most_promising"]["extra_position_checks"] = extra

    out = OUTPUT_DIR / "steering_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
