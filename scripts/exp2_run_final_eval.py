"""Experiment 2 confirmatory evaluation on the reserved final evaluation set.

The protocol is frozen in `experiments/02_generalizable_reward_hacking/configs/final_eval.json` and was committed
before this script was ever run against the reserved layouts. The script evaluates the nine rank-8 coordinated
edits selected in Phase 6, plus the unedited hacking and good networks, once. It trains nothing and selects nothing.

    python -m scripts.exp2_run_final_eval --check   # verify the nine models and reproduce their development numbers;
                                                    # never opens the reserved set
    python -m scripts.exp2_run_final_eval           # the one evaluation; refuses to run if the results file exists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rewardlens.exp2.emergence import weights_sha256
from rewardlens.exp2.evaluation import greedy_policy
from rewardlens.exp2.layouts import load_layout_set
from rewardlens.exp2.low_rank import merged_network
from rewardlens.exp2.training import EVAL_KEYS
from rewardlens.exp2.weight_level import behavior, meets_repair, scripted_shortest_path
from scripts.exp2_run_phase5 import EXP_DIR, GAMMA, load_setup, net_at

PROTOCOL = EXP_DIR / "configs/final_eval.json"
PHASE6_RUNS = Path("results/exp2/phase6/runs.json")
CRITERION_KEYS = ("direct_goal", "path_efficiency_direct", "farming")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_interval(p: float, n: int, z: float = 1.959964) -> list[float]:
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def load_models(protocol: dict, setup: dict) -> list[dict]:
    """The nine frozen networks, each checked against the hashes recorded in the protocol."""
    models = []
    for spec in protocol["models"]["runs"]:
        if file_sha256(spec["edit_file"]) != spec["edit_file_sha256"]:
            raise ValueError(f"{spec['edit_file']} does not match the hash frozen in the protocol")
        edit = torch.load(spec["edit_file"], weights_only=False)
        base = net_at(setup, spec["dqn_seed"], tuple(protocol["baselines"]["hacking"])).state_dict()
        if weights_sha256(base) != edit["base_weights_sha256"]:
            raise ValueError(f"{spec['run_id']}: hacking network differs from the one the edit was trained on")
        net = merged_network(base, edit["factors"])
        if weights_sha256(net.state_dict()) != spec["merged_weights_sha256"] or edit["selected_step"] != spec["selected_step"]:
            raise ValueError(f"{spec['run_id']}: merged network is not the checkpoint selected in Phase 6")
        models.append({**spec, "net": net})
    return models


def evaluate(net_or_policy, layouts, params, criterion: dict) -> dict:
    policy = greedy_policy(net_or_policy) if isinstance(net_or_policy, torch.nn.Module) else net_or_policy
    b = behavior(policy, layouts, params, GAMMA)
    return {**b, "meets_repair_criterion": meets_repair(b, criterion),
            "direct_goal_wilson_95": wilson_interval(b["direct_goal"], len(layouts))}


def stats(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def aggregate(rows: list[dict], split: str) -> dict:
    out = {key: stats([r[split][key] for r in rows]) for key in EVAL_KEYS if all(r[split][key] is not None for r in rows)}
    seeds = sorted({r["dqn_seed"] for r in rows})
    out["per_dqn_seed_mean"] = {str(s): {key: float(np.mean([r[split][key] for r in rows if r["dqn_seed"] == s])) for key in CRITERION_KEYS}
                                for s in seeds}
    return out


def development_numbers(protocol: dict) -> dict:
    runs = {r["run_id"]: r for r in json.load(open(PHASE6_RUNS))["runs"]}
    return {spec["run_id"]: {"dev_report_750": runs[spec["run_id"]]["selected"]["dev_report"], "dev_all_1500": runs[spec["run_id"]]["selected"]["dev"]}
            for spec in protocol["models"]["runs"]}


def check(protocol: dict, setup: dict, models: list[dict]) -> None:
    """Reproduce the Phase 6 development numbers with this script's evaluation path. Development layouts only."""
    dev, recorded = setup["layouts"]["dev"], development_numbers(protocol)
    for m in models:
        got = behavior(greedy_policy(m["net"]), dev, setup["params"], GAMMA)
        want = recorded[m["run_id"]]["dev_all_1500"]
        for key in EVAL_KEYS:
            if want[key] is not None and abs(got[key] - want[key]) > 1e-9:
                raise ValueError(f"{m['run_id']}: {key} {got[key]} differs from the Phase 6 record {want[key]}")
        print(f"{m['run_id']}: hashes match, development numbers reproduced "
              f"(direct goal {got['direct_goal']:.3f}, efficiency {got['path_efficiency_direct']:.3f}, farming {got['farming']:.3f})")
    print(f"Check passed for {len(models)} models. The reserved set was not opened.")


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(check_only: bool = False) -> None:
    protocol = json.load(open(PROTOCOL))
    out_path = Path(protocol["evaluation"]["results_file"])
    setup = load_setup()
    models = load_models(protocol, setup)
    if check_only:
        check(protocol, setup, models)
        return
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists. The reserved set is evaluated once; this script does not overwrite the result.")

    criterion, params = protocol["repair_criterion"], setup["params"]
    layouts, meta = load_layout_set(protocol["evaluation"]["layout_set"])
    assert len(layouts) == protocol["evaluation"]["n_layouts"]
    dev = development_numbers(protocol)

    rows = []
    for m in models:
        final = evaluate(m["net"], layouts, params, criterion)
        rows.append({k: v for k, v in m.items() if k != "net"} | {"final_eval": final, **dev[m["run_id"]]})
        print(f"{m['run_id']}: direct goal {final['direct_goal']:.3f}, efficiency {final['path_efficiency_direct']:.3f}, "
              f"farming {final['farming']:.3f}, meets criterion {final['meets_repair_criterion']}", flush=True)

    baselines = {"per_seed": {}, "scripted_shortest_path": {**scripted_shortest_path(layouts, params, GAMMA)}}
    for seed in setup["config"]["seeds"]:
        baselines["per_seed"][str(seed)] = {name: evaluate(net_at(setup, seed, tuple(protocol["baselines"][name])), layouts, params, criterion)
                                            for name in ("hacking", "good")}

    n_pass = sum(r["final_eval"]["meets_repair_criterion"] for r in rows)
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": file_sha256(PROTOCOL),
        "script_sha256": file_sha256(__file__), "git_commit": git_commit(),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layout_set": protocol["evaluation"]["layout_set"], "layout_set_sha256": meta["sha256"], "n_layouts": len(layouts),
        "repair_criterion": criterion,
        "models": rows,
        "aggregate": {"final_eval": aggregate(rows, "final_eval"), "dev_report_750": aggregate(rows, "dev_report_750"),
                      "dev_all_1500": aggregate(rows, "dev_all_1500")},
        "n_models": len(rows), "n_meeting_criterion": int(n_pass), "confirmed": bool(n_pass == len(rows)),
        "baselines": baselines,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=1)
    print(f"{n_pass}/{len(rows)} models meet the criterion. Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify models on development layouts only; never opens the reserved set")
    main(ap.parse_args().check)
