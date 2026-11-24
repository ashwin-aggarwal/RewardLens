"""Experiment 2, Phase 4: activation patching between nearby checkpoints of one hacking run.

Pair selection, donor/recipient compatibility measures and decision-level
diagnostics. The patching itself reuses `rewardlens.exp2.patching`.
"""

from __future__ import annotations

import numpy as np
import torch

from rewardlens.exp2.emergence import AGREE, EQUIVALENT, TILE_OVER_GOAL, action_effects, classify_decisions
from rewardlens.exp2.patching import donor_activation, patched_q
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import forward_all, linear_cka, relative_change


def select_pairs(transition: dict, steps: list[int], gaps: list[int], include_last_before_onset: bool = True,
                 include_fork_start: bool = True, reference_pairs: tuple[str, ...] = ()) -> list[dict]:
    """Donor/recipient checkpoints for one seed, by a fixed rule that does not look at patching results.

    `transition` holds the seed's onset/completion steps (Phase 3). The main recipient is the
    hacking checkpoint at completion; donors are earlier checkpoints of the same run.
    """
    done, final = transition["completion"], steps[-1]
    pre = transition["last_before_onset"]
    pairs = []

    def add(name, donor, recipient, kind="main"):
        d_branch, d_step = donor
        r_branch, r_step = recipient
        if d_step not in steps or r_step not in steps:
            raise ValueError(f"pair {name}: checkpoint step not available")
        pairs.append({"name": name, "kind": kind, "donor_branch": d_branch, "donor_step": d_step,
                      "recipient_branch": r_branch, "recipient_step": r_step, "gap": r_step - d_step})

    for gap in sorted(gaps):
        if done - gap >= 0:
            add(f"minus_{gap // 1000}k", ("hacking", done - gap), ("hacking", done))
    if include_last_before_onset:
        add("pre_onset", ("hacking", pre), ("hacking", done))
    if include_fork_start:
        add("fork_start", ("hacking", 0), ("hacking", done))
    if "pre_onset_to_final" in reference_pairs:
        add("pre_onset_to_final", ("hacking", pre), ("hacking", final), "reference")
    if "good_same_step_to_completion" in reference_pairs:
        add("good_same_step", ("good", done), ("hacking", done), "reference")

    seen = set()
    for p in pairs:
        key = (p["donor_branch"], p["donor_step"], p["recipient_branch"], p["recipient_step"])
        if key in seen:
            raise ValueError(f"duplicate pair {key}")
        seen.add(key)
        if p["donor_branch"] == p["recipient_branch"] and p["gap"] <= 0:
            raise ValueError(f"pair {p['name']}: donor is not earlier than recipient")
    return pairs


def compatibility(donor: torch.nn.Module, recipient: torch.nn.Module, obs: np.ndarray) -> dict:
    """How far apart two networks are, on identical observations and in their weights."""
    d, r = forward_all(donor, obs), forward_all(recipient, obs)
    out: dict = {}
    for layer in ("layer1", "layer2"):
        nd, nr = np.linalg.norm(d[layer], axis=1).mean(), np.linalg.norm(r[layer], axis=1).mean()
        out[layer] = {"cka": linear_cka(d[layer], r[layer]), "donor_norm": float(nd), "recipient_norm": float(nr),
                      "norm_ratio": float(nr / nd), "relative_change": relative_change(d[layer], r[layer])}

    def drift(name):
        a, b = getattr(donor, name), getattr(recipient, name)
        return {"weight": float((b.weight - a.weight).norm() / a.weight.norm()),
                "bias": float((b.bias - a.bias).norm() / (a.bias.norm() + 1e-8))}

    with torch.no_grad():
        out["weight_drift"] = {n: drift(n) for n in ("fc1", "fc2", "fc3")}
    out["output_head_drift"] = out["weight_drift"]["fc3"]["weight"]
    out["greedy_action_agreement"] = float((d["q"].argmax(1) == r["q"].argmax(1)).mean())
    out["mean_max_q"] = {"donor": float(d["q"].max(1).mean()), "recipient": float(r["q"].max(1).mean())}
    return out


def action_diagnostics(recipient: torch.nn.Module, donor: torch.nn.Module, states: dict, layer: int,
                       params: ProceduralEnvParams, alpha: float = 1.0) -> dict:
    """What the matched patch does to individual decisions.

    States are labelled by comparing the donor's greedy action (reference) with the
    recipient's (new). At "tile_over_goal" states the donor approaches the goal and the
    recipient approaches the tile: a restoring patch should move the recipient back to
    a goal-ward action there, and should leave "agree" states alone.
    """
    obs = torch.from_numpy(np.asarray(states["obs"], dtype=np.float32))
    with torch.no_grad():
        q_r, q_d = recipient(obs).numpy(), donor(obs).numpy()
        q_p = patched_q(recipient, obs, donor_activation(donor, obs, layer), layer, alpha).numpy()
    a_r, a_d, a_p = q_r.argmax(1), q_d.argmax(1), q_p.argmax(1)
    pos, lay = states["pos"], states["layouts"]
    labels = classify_decisions(a_d, a_r, pos, lay, params)
    eff = action_effects(pos, lay, a_p, params)
    tile_not_goal = ~eff["toward_goal"] & (eff["toward_tile"] | eff["on_tile"])

    def frac(mask, values):
        return float(values[mask].mean()) if mask.any() else None

    flip, agree = labels == TILE_OVER_GOAL, labels == AGREE

    def margin(q):
        s = np.sort(q, axis=1)
        return float((s[:, -1] - s[:, -2]).mean())

    return {
        "n_states": int(len(a_r)),
        "label_fraction": {k: float((labels == k).mean()) for k in (AGREE, EQUIVALENT, TILE_OVER_GOAL)},
        "patched_action_changed": float((a_p != a_r).mean()),
        "patched_matches_donor": float((a_p == a_d).mean()),
        "patched_toward_goal": float(eff["toward_goal"].mean()),
        "patched_tile_not_goal": float(tile_not_goal.mean()),
        "recipient_toward_goal": float(action_effects(pos, lay, a_r, params)["toward_goal"].mean()),
        "donor_toward_goal": float(action_effects(pos, lay, a_d, params)["toward_goal"].mean()),
        "at_flip_states": {"n": int(flip.sum()), "restored_to_donor_action": frac(flip, a_p == a_d),
                           "toward_goal": frac(flip, eff["toward_goal"]), "still_tile_not_goal": frac(flip, tile_not_goal)},
        "at_agree_states": {"n": int(agree.sum()), "action_changed": frac(agree, a_p != a_r),
                            "toward_goal": frac(agree, eff["toward_goal"])},
        "mean_max_q": {"recipient": float(q_r.max(1).mean()), "donor": float(q_d.max(1).mean()), "patched": float(q_p.max(1).mean())},
        "mean_q_margin": {"recipient": margin(q_r), "donor": margin(q_d), "patched": margin(q_p)},
        "mean_abs_q_change": float(np.abs(q_p - q_r).mean()),
    }


def restoration_fraction(patched: float, recipient: float, donor: float, min_headroom: float = 0.2) -> float | None:
    """Share of the donor's advantage over the recipient that the patch recovers; None if the donor has no advantage."""
    if donor - recipient < min_headroom:
        return None
    return float((patched - recipient) / (donor - recipient))


def meets_success(matched: dict, recipient: dict, controls: list[dict], criterion: dict) -> dict:
    """Apply the preregistered goal-restoration criterion to one pair's layer-2 results."""
    best_control = max(c["direct_goal"]["mean"] for c in controls)
    gain_r = matched["direct_goal"] - recipient["direct_goal"]
    gain_c = matched["direct_goal"] - best_control
    checks = {"gain_over_recipient": gain_r >= criterion["min_gain_over_recipient"],
              "gain_over_best_state_control": gain_c >= criterion["min_gain_over_best_state_control"],
              "other_at_most": matched["other"] <= criterion["max_other"]}
    return {"gain_over_recipient": float(gain_r), "gain_over_best_state_control": float(gain_c),
            "best_state_control_direct_goal": float(best_control), "checks": checks, "success": all(checks.values())}


def sweep_triggered(matched: dict, recipient: dict, controls: list[dict], min_gain: float = 0.1) -> bool:
    best_control = max(c["direct_goal"]["mean"] for c in controls)
    return (matched["direct_goal"] - recipient["direct_goal"] >= min_gain
            and matched["direct_goal"] - best_control >= min_gain)


def spearman(x: list[float], y: list[float]) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(x[ok], y[ok]).statistic)
