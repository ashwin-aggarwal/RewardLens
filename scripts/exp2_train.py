#!/usr/bin/env python3
"""Experiment 2, Phase 1: navigation pretraining and the good/hacking fork.

Stages for one seed:
  pretrain  corrected reward from random initialization
  good      corrected reward, continued from the pretrained checkpoint
  hacking   flawed reward, continued from the same pretrained checkpoint

Usage:
  python -m scripts.exp2_train --seed 42                 # all three stages
  python -m scripts.exp2_train --seed 42 --stage pretrain
  python -m scripts.exp2_train --all-seeds               # every configured seed, in parallel
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rewardlens.exp2.training import Exp2TrainConfig, train

EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
ENV_CONFIG = EXP_DIR / "configs/environment.json"
TRAIN_CONFIG = EXP_DIR / "configs/training.json"
STAGES = ("pretrain", "good", "hacking")


def load_configs(train_config: Path = TRAIN_CONFIG) -> tuple[dict, dict]:
    return json.load(open(ENV_CONFIG)), json.load(open(train_config))


def seed_dir(tc: dict, seed: int, tag: str | None = None) -> Path:
    base = Path(tc["results_dir"])
    return base / tag / f"seed{seed}" if tag else base / f"seed{seed}"


def pretrained_checkpoint(tc: dict, seed: int, tag: str | None = None) -> Path:
    return seed_dir(tc, seed, tag) / "pretrain" / "checkpoints" / "checkpoint_final.pt"


def stage_config(stage: str, seed: int, ec: dict, tc: dict, tag: str | None = None) -> Exp2TrainConfig:
    common = dict(stage=stage, seed=seed, env=ec["env"], manifest_path=ec["layouts"]["manifest_path"],
                  run_dir=str(seed_dir(tc, seed, tag) / stage), **tc["dqn"])
    if stage == "pretrain":
        p = tc["pretrain"]
        return Exp2TrainConfig(reward_mode=p["reward_mode"], total_timesteps=p["total_timesteps"],
                               epsilon_start=p["epsilon_start"], epsilon_end=p["epsilon_end"],
                               epsilon_fraction=p["epsilon_fraction"], **common)
    f = tc["fork"]
    return Exp2TrainConfig(reward_mode=f[f"{stage}_reward_mode"], total_timesteps=f["total_timesteps"],
                           epsilon_start=f["epsilon_start"], epsilon_end=f["epsilon_end"],
                           epsilon_fraction=f["epsilon_fraction"],
                           init_checkpoint=str(pretrained_checkpoint(tc, seed, tag)), **common)


def run(seed: int, stages: tuple[str, ...], train_config: Path = TRAIN_CONFIG, tag: str | None = None) -> None:
    ec, tc = load_configs(train_config)
    for stage in stages:
        meta = train(stage_config(stage, seed, ec, tc, tag))
        print(f"[{stage} seed {seed}] done in {meta['elapsed_seconds']}s, "
              f"{meta['env_steps_per_second']} steps/s, peak {meta['peak_memory_mb']} MB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stage", choices=STAGES + ("fork", "all"), default="all")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--config", type=Path, default=TRAIN_CONFIG)
    ap.add_argument("--tag", default=None, help="subfolder of the results dir, e.g. pilot runs")
    args = ap.parse_args()
    stages = STAGES if args.stage == "all" else ("good", "hacking") if args.stage == "fork" else (args.stage,)
    if args.all_seeds:
        _, tc = load_configs(args.config)
        procs = []
        for seed in tc["seeds"]:
            log = seed_dir(tc, seed, args.tag)
            log.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "scripts.exp2_train", "--seed", str(seed), "--stage", args.stage,
                   "--config", str(args.config)] + (["--tag", args.tag] if args.tag else [])
            procs.append(subprocess.Popen(cmd, stdout=open(log / f"train_{args.stage}.log", "w"), stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"training failed with exit codes {codes}")
        return
    if args.seed is None:
        ap.error("--seed or --all-seeds is required")
    run(args.seed, stages, args.config, args.tag)


if __name__ == "__main__":
    main()
