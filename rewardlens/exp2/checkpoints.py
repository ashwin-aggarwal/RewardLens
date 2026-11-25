"""Registry of the committed Experiment 2 checkpoints.

Each reporting seed has three final checkpoints that belong together:

    pretrain  corrected reward, 300k steps from random initialization
    good      corrected reward, 400k more steps starting from that pretrain checkpoint
    hacking   flawed reward, 400k more steps starting from the same pretrain checkpoint

`results/exp2/phase1/checkpoints.json` records the file hash of each and the
hash of the checkpoint each fork branch started from, so later analyses can
confirm they compare the three networks of one seed on identical states.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rewardlens.agents.dqn import QNetwork, load_checkpoint, load_q_network_for_eval

STAGES = ("pretrain", "good", "hacking")
REGISTRY_PATH = Path("results/exp2/phase1/checkpoints.json")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_path(seed: int, stage: str, results_dir: str | Path = "results/exp2/phase1") -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return Path(results_dir) / f"seed{seed}" / stage / "checkpoints" / "checkpoint_final.pt"


def build_registry(seeds: list[int], results_dir: str | Path = "results/exp2/phase1") -> dict:
    registry: dict = {"results_dir": str(results_dir), "seeds": {}}
    for seed in seeds:
        entry = {}
        for stage in STAGES:
            path = checkpoint_path(seed, stage, results_dir)
            ckpt = load_checkpoint(path)
            cfg = ckpt["config"]
            init = cfg.get("init_checkpoint")
            entry[stage] = {
                "path": str(path), "sha256": file_sha256(path), "training_steps": int(ckpt["step"]),
                "reward_mode": cfg["reward_mode"], "init_checkpoint": init,
                "init_checkpoint_sha256": file_sha256(init) if init else None,
            }
        registry["seeds"][str(seed)] = entry
    return registry


def verify_registry(registry: dict) -> None:
    """Raise if a file changed or a fork branch did not start from its seed's pretrain checkpoint."""
    for seed, entry in registry["seeds"].items():
        for stage in STAGES:
            rec = entry[stage]
            if file_sha256(rec["path"]) != rec["sha256"]:
                raise ValueError(f"checkpoint {rec['path']} does not match its recorded hash")
        for stage in ("good", "hacking"):
            if entry[stage]["init_checkpoint"] != entry["pretrain"]["path"]:
                raise ValueError(f"seed {seed} {stage} was not initialized from its own pretrain checkpoint")
            if entry[stage]["init_checkpoint_sha256"] != entry["pretrain"]["sha256"]:
                raise ValueError(f"seed {seed} {stage} started from different weights than the committed pretrain checkpoint")


def load_triplet(seed: int, registry_path: str | Path = REGISTRY_PATH, verify: bool = True) -> dict[str, QNetwork]:
    """The pretrained, good and hacking networks of one seed, in eval mode."""
    registry = json.load(open(registry_path))
    if verify:
        verify_registry({"seeds": {str(seed): registry["seeds"][str(seed)]}})
    return {stage: load_q_network_for_eval(registry["seeds"][str(seed)][stage]["path"])[0] for stage in STAGES}
