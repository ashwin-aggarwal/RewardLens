"""Phase 4.7: controlling for time in activation patching.

Question: does the Phase 4.6 intervention (donor layer-2 activation computed
with the donor's remaining-time input clamped to 0.2, injected at alpha = 1
into the 30k hacking recipient) transfer goal-directed representation, or
does it succeed mainly by changing the agent's representation of time?

Conditions (all prespecified in configs/phase4_7.json; no selection):
  A  original recipient, real time
  B  recipient time clamp only (network input; env unchanged), sweep
  C  donor alone with time clamp, sweep
  D  matched-time patch: donor and recipient see real time (alpha 1.0; 0.75 ref)
  E  time-shifted patch (Phase 4.6), donor time sweep
  F  time-only transfer: h_rec(s,t) + [h_donor(s,t_clamp) - h_donor(s,t)], sweep
  G  head swap: donor's output head reads the recipient's layer-2 activation
Controls at E(0.2) and F(0.2): 20 norm-matched random, 20 shuffled-state,
constant Q-bias, donor alone, unmodified recipient.

Usage:
    python -m scripts.run_phase4_7
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
    make_shuffled_donor_layer,
    make_time_delta_donor,
    mean_layer_delta,
    scheduled_patching,
    schedule_constant,
    summarize,
)
from rewardlens.analysis.time_dependence import (
    clamp_time,
    compare_to_reference,
    goal_preference_by_time,
)
from rewardlens.envs.reward_grid import RewardGridConfig
from scripts.run_interventions import _position_mask

FLAWED_DIR = Path("results/phase2")
OUTPUT_DIR = Path("results/phase4_7")
CONFIG_PATH = Path("configs/phase4_7.json")


def _ckpt(step: int) -> Path:
    name = "checkpoint_final.pt" if step == 100000 else f"checkpoint_{step}.pt"
    return FLAWED_DIR / "baseline_seed123" / "checkpoints" / name


def _fmt(r: dict) -> str:
    eff = r.get("mean_path_efficiency_direct")
    af = r["action_fractions"]
    return (" ".join(f"{c[:6]}={r[f'{c}_rate']:.2f}" for c in CATEGORIES)
            + f" visits={r['mean_reward_tile_visits']:4.1f} steps={r['mean_episode_steps']:5.1f}"
            + f" eff={'-' if eff is None else round(eff, 2)} norm={r['mean_applied_norm']:4.1f}"
            + f" U/R/D/L={af['0']:.2f}/{af['1']:.2f}/{af['2']:.2f}/{af['3']:.2f}")


def _agg(lst):
    eff = [r["mean_path_efficiency_direct"] for r in lst if r.get("mean_path_efficiency_direct") is not None]
    return {"n": len(lst),
            **{f"mean_{c}_rate": float(np.mean([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
            **{f"max_{c}_rate": float(np.max([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
            **{f"std_{c}_rate": float(np.std([r[f"{c}_rate"] for r in lst])) for c in CATEGORIES},
            "mean_reward_tile_visits": float(np.mean([r["mean_reward_tile_visits"] for r in lst])),
            "mean_episode_steps": float(np.mean([r["mean_episode_steps"] for r in lst])),
            "mean_path_efficiency_direct": float(np.mean(eff)) if eff else None,
            "mean_applied_norm": float(np.mean([r["mean_applied_norm"] for r in lst])),
            "per_control": lst}


def run_conditions(recipient, donor, positions, cfg, pool_obs, obs_bias, thr, keep_traces):
    env_cfg = RewardGridConfig()
    a46, t46, a45 = cfg["phase4_6_alpha"], cfg["phase4_6_donor_time"], cfg["phase4_5_alpha"]
    sched = schedule_constant(a46)
    donor_fn_real = make_matched_donor_layer(donor, 2)
    out = {}

    def ev(net, donor_fn=None, mode="none", alpha=a46, obs_transform=None, seed=0, cdelta=None, traces=False):
        return evaluate_steering(net, env_cfg, positions, donor_fn, schedule_constant(alpha), 2, mode,
                                 random_seed=seed, constant_delta=cdelta, obs_transform=obs_transform,
                                 threshold=thr, keep_traces=traces)

    # A
    out["A_recipient"] = ev(recipient, traces=keep_traces)
    print(f"  A recipient                 {_fmt(out['A_recipient'])}")
    # B: recipient input clamp sweep
    out["B_recipient_clamp"] = {}
    for tau in cfg["clamp_sweep"]:
        r = ev(recipient, obs_transform=clamp_time(tau), traces=(keep_traces and tau == t46))
        out["B_recipient_clamp"][str(tau)] = r
        print(f"  B recipient clamp t={tau:<4} {_fmt(r)}")
    # C: donor clamp sweep (+ unclamped)
    out["C_donor"] = {"real": ev(donor, traces=keep_traces)}
    print(f"  C donor real time           {_fmt(out['C_donor']['real'])}")
    for tau in cfg["clamp_sweep"]:
        r = ev(donor, obs_transform=clamp_time(tau))
        out["C_donor"][str(tau)] = r
        print(f"  C donor clamp t={tau:<4}      {_fmt(r)}")
    # D: matched-time patch
    out["D_matched_time"] = {str(a46): ev(recipient, donor_fn_real, "matched", a46, traces=keep_traces),
                             str(a45): ev(recipient, donor_fn_real, "matched", a45)}
    for a, r in out["D_matched_time"].items():
        print(f"  D matched-time a={a:<4}       {_fmt(r)}")
    # E: time-shifted patch sweep (alpha 1.0 and 0.75)
    out["E_time_shifted"] = {}
    for a in (a46, a45):
        for tau in cfg["donor_time_sweep"]:
            fn = make_matched_donor_layer(donor, 2, clamp_time_tensor(tau))
            r = ev(recipient, fn, "matched", a, traces=(keep_traces and a == a46 and tau == t46))
            out["E_time_shifted"][f"a={a},t={tau}"] = r
            print(f"  E shifted a={a:<4} donor t={tau:<4} {_fmt(r)}")
    # F: time-only transfer sweep
    out["F_time_only"] = {}
    for a in (a46, a45):
        for tau in cfg["donor_time_sweep"]:
            fn = make_time_delta_donor(recipient, donor, 2, tau)
            r = ev(recipient, fn, "matched", a, traces=(keep_traces and a == a46 and tau == t46))
            out["F_time_only"][f"a={a},t={tau}"] = r
            print(f"  F time-only a={a:<4} t={tau:<4}   {_fmt(r)}")
    # G: head swap (donor network, layer 2 replaced by recipient activation; recipient sees real/clamped time)
    out["G_head_swap"] = {
        "recipient_h2_real_time": ev(donor, make_matched_donor_layer(recipient, 2), "matched", 1.0),
        f"recipient_h2_time_{t46}": ev(donor, make_matched_donor_layer(recipient, 2, clamp_time_tensor(t46)), "matched", 1.0),
    }
    for k, r in out["G_head_swap"].items():
        print(f"  G donor head reads {k:26s} {_fmt(r)}")

    # Controls at E(0.2) and F(0.2), alpha = 1.0, same magnitudes
    ctrl = {}
    for name, fn_builder, shuf_builder in (
        ("E", lambda: make_matched_donor_layer(donor, 2, clamp_time_tensor(t46)),
         lambda i: make_shuffled_donor_layer(donor, pool_obs, cfg["control_seed"] + i, 2, clamp_time_tensor(t46))),
        ("F", lambda: make_time_delta_donor(recipient, donor, 2, t46),
         lambda i: make_time_delta_donor(recipient, donor, 2, t46, pool_obs, cfg["control_seed"] + i)),
    ):
        fn = fn_builder()
        cdelta = mean_layer_delta(recipient, fn, obs_bias, 2)
        rnd = [summarize(ev(recipient, fn, "random", a46, seed=cfg["control_seed"] + i)) for i in range(cfg["n_controls"])]
        shf = [summarize(ev(recipient, shuf_builder(i), "matched", a46)) for i in range(cfg["n_controls"])]
        ctrl[name] = {"random": _agg(rnd), "shuffled": _agg(shf),
                      "constant_qbias": summarize(ev(recipient, None, "constant", a46, cdelta=cdelta)),
                      "constant_delta_norm": float(np.linalg.norm(cdelta))}
        for k in ("random", "shuffled"):
            t = ctrl[name][k]
            print(f"  ctrl {name} {k:9s} direct={t['mean_direct_goal_rate']:.2f} (max {t['max_direct_goal_rate']:.2f})"
                  f" farm_only={t['mean_farm_only_rate']:.2f} other={t['mean_other_rate']:.2f}"
                  f" visits={t['mean_reward_tile_visits']:.1f} eff={t['mean_path_efficiency_direct']} norm={t['mean_applied_norm']:.1f}")
        print(f"  ctrl {name} qbias     {_fmt(ctrl[name]['constant_qbias'])}")
    out["controls"] = ctrl
    return out


def q_analysis(recipient, donor, positions, cfg, pool_obs):
    tf = cfg["time_fractions_q"]
    t46, a46 = cfg["phase4_6_donor_time"], cfg["phase4_6_alpha"]
    env_cfg = RewardGridConfig()

    def patched(net, donor_fn, alpha=a46):
        def f(obs):
            with scheduled_patching(net, donor_fn, schedule_constant(alpha), PatchController(), 2, "matched"), \
                    torch.no_grad():
                return net(obs)
        return f

    def clamped(net, tau):
        return lambda obs: net(clamp_time_tensor(tau)(obs))

    conds = {
        "A_recipient": lambda obs: recipient(obs),
        f"B_recipient_clamp_{t46}": clamped(recipient, t46),
        "C_donor": lambda obs: donor(obs),
        f"C_donor_clamp_{t46}": clamped(donor, t46),
        "D_matched_time": patched(recipient, make_matched_donor_layer(donor, 2)),
        f"E_time_shifted_{t46}": patched(recipient, make_matched_donor_layer(donor, 2, clamp_time_tensor(t46))),
        f"F_time_only_{t46}": patched(recipient, make_time_delta_donor(recipient, donor, 2, t46)),
        "G_head_swap": patched(donor, make_matched_donor_layer(recipient, 2)),
        f"E_shuffled_{t46}": patched(recipient, make_shuffled_donor_layer(donor, pool_obs, 0, 2, clamp_time_tensor(t46))),
        f"F_shuffled_{t46}": patched(recipient, make_time_delta_donor(recipient, donor, 2, t46, pool_obs, 0)),
    }
    res = {}
    for name, fn in conds.items():
        res[name] = goal_preference_by_time(fn, positions, tf, env_cfg)
    ref = res["A_recipient"]
    for name in res:
        res[name]["vs_recipient"] = compare_to_reference(res[name], ref)
        r = res[name]
        print(f"  Q {name:26s} goal-approach t=1:{r['goal_approach_fraction'][0]:.2f} t=0.2:{r['goal_approach_fraction'][tf.index(0.2)]:.2f}"
              f" t=0:{r['goal_approach_fraction'][-1]:.2f} | gap t=1:{r['mean_goal_minus_reward_q'][0]:+.2f}"
              f" | flip vs A t=1:{r['vs_recipient']['greedy_flip_rate'][0]:.2f}"
              f" | dominant-action frac t=1:{r['vs_recipient']['dominant_action_fraction'][0]:.2f}")
    # Decomposition magnitudes on the same grid: ||matched delta||, ||time delta||, ||full delta||, cosine
    from rewardlens.analysis.time_dependence import observations_for_positions
    obs, _, _ = observations_for_positions(positions, tf, env_cfg)
    ot = torch.as_tensor(obs)
    with torch.no_grad():
        h_rec = recipient.relu2(recipient.fc2(recipient.relu1(recipient.fc1(ot))))
        h_don = donor.relu2(donor.fc2(donor.relu1(donor.fc1(ot))))
        h_don_c = donor.relu2(donor.fc2(donor.relu1(donor.fc1(clamp_time_tensor(t46)(ot)))))
        h_rec_c = recipient.relu2(recipient.fc2(recipient.relu1(recipient.fc1(clamp_time_tensor(t46)(ot)))))
    d_match, d_time, d_full = h_don - h_rec, h_don_c - h_don, h_don_c - h_rec
    d_rec_time = h_rec_c - h_rec
    cos = lambda a, b: float((torch.nn.functional.cosine_similarity(a, b, dim=1)).mean())
    res["decomposition"] = {
        "max_abs_error_full_minus_sum": float((d_full - (d_match + d_time)).abs().max()),
        "mean_norm_matched_delta": float(d_match.norm(dim=1).mean()),
        "mean_norm_time_delta_donor": float(d_time.norm(dim=1).mean()),
        "mean_norm_time_delta_recipient": float(d_rec_time.norm(dim=1).mean()),
        "mean_norm_full_delta": float(d_full.norm(dim=1).mean()),
        "mean_cosine_time_delta_donor_vs_recipient": cos(d_time, d_rec_time),
        "mean_cosine_matched_vs_time_delta": cos(d_match, d_time),
        "mean_norm_h_rec": float(h_rec.norm(dim=1).mean()),
        "mean_norm_h_donor": float(h_don.norm(dim=1).mean()),
    }
    print(f"  decomposition: |matched|={res['decomposition']['mean_norm_matched_delta']:.2f} "
          f"|time(donor)|={res['decomposition']['mean_norm_time_delta_donor']:.2f} "
          f"|time(recipient)|={res['decomposition']['mean_norm_time_delta_recipient']:.2f} "
          f"|full|={res['decomposition']['mean_norm_full_delta']:.2f} "
          f"cos(time donor, time rec)={res['decomposition']['mean_cosine_time_delta_donor_vs_recipient']:.2f} "
          f"cos(matched, time)={res['decomposition']['mean_cosine_matched_vs_time_delta']:.2f}")
    return res


def main() -> None:
    cfg = json.load(open(CONFIG_PATH))
    thr = int(cfg["visit_threshold"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_cfg = RewardGridConfig()
    parts = partition_start_positions(env_cfg, cfg["n_dev_positions"], cfg["n_test_positions"], cfg["phase4_6_partition_seed"])
    all_positions = [(x, y) for x in range(env_cfg.width) for y in range(env_cfg.height)
                     if (x, y) not in (env_cfg.goal_pos, env_cfg.reward_pos)]
    observations = build_matched_state_dataset()
    pool_obs = observations[_position_mask(observations, parts["train"])]
    obs_bias = observations[_position_mask(observations, parts["dev"])]
    donor, _ = load_q_network_for_eval(_ckpt(cfg["donor_step"]))
    recipient, _ = load_q_network_for_eval(_ckpt(cfg["recipient_step"]))

    results = {"config": cfg, "position_sets": {"phase4_6_test_16": [list(p) for p in parts["test"]],
                                                "all_62": [list(p) for p in all_positions]},
               "untouched_validation_states": "none: all 62 valid start positions were used in Phase 4.6"}
    for label, positions in (("phase4_6_test_16", parts["test"]), ("all_62", all_positions)):
        print(f"\n=== Conditions on {label} ({len(positions)} starts) ===")
        results[label] = run_conditions(recipient, donor, positions, cfg, pool_obs, obs_bias, thr,
                                        keep_traces=(label == "phase4_6_test_16"))
    print("\n=== Q-value analysis (Phase 4.6 test positions x time inputs) ===")
    results["q_analysis_test_16"] = q_analysis(recipient, donor, parts["test"], cfg, pool_obs)
    print("\n=== Q-value analysis (all 62 positions x time inputs) ===")
    results["q_analysis_all_62"] = q_analysis(recipient, donor, all_positions, cfg, pool_obs)

    # Reproduction check against Phase 4.6
    p46 = Path("results/phase4_6/steering_results.json")
    if p46.exists():
        d46 = json.load(open(p46))
        ref = d46["blocks"]["D_time_shifted_donor"]["test"]["matched"]
        mine = results["phase4_6_test_16"]["E_time_shifted"][f"a={cfg['phase4_6_alpha']},t={cfg['phase4_6_donor_time']}"]
        same = all(abs(ref[k] - mine[k]) < 1e-9 for k in ("direct_goal_rate", "mean_reward_tile_visits", "mean_episode_steps"))
        results["phase4_6_reproduction_exact"] = bool(same)
        print(f"\nPhase 4.6 block D reproduced exactly: {same}")

    out = OUTPUT_DIR / "time_control_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
