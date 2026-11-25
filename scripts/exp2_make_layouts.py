#!/usr/bin/env python3
"""Experiment 2: generate the layout manifest and freeze the train/dev/test split.

Run once, before any training. Refuses to overwrite an existing manifest unless
--force is given, because the test split must stay frozen.

Usage:
  python -m scripts.exp2_make_layouts               # train/dev/test manifest
  python -m scripts.exp2_make_layouts --final-eval  # reserved final evaluation set, disjoint from all of the above
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rewardlens.exp2.layouts import (
    assert_disjoint, generate_layouts, is_valid_layout, load_manifest, save_layout_set, save_manifest,
    split_layouts, unseen_pair_mask,
)

CONFIG_PATH = Path("experiments/02_generalizable_reward_hacking/configs/environment.json")


def main(force: bool = False) -> dict:
    cfg = json.load(open(CONFIG_PATH))
    env, lay = cfg["env"], cfg["layouts"]
    path = Path(lay["manifest_path"])
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists. The split is frozen; pass --force only to regenerate deliberately.")
    layouts = generate_layouts(lay["n_layouts"], lay["generation_seed"], env["width"], env["height"])
    assert all(is_valid_layout(l, env["width"], env["height"]) for l in layouts)
    splits = split_layouts(layouts, tuple(lay["fractions"]), lay["split_seed"])
    assert_disjoint(splits)
    manifest = save_manifest(path, splits, {
        "experiment": "02_generalizable_reward_hacking", "width": env["width"], "height": env["height"],
        "generation_seed": lay["generation_seed"], "split_seed": lay["split_seed"], "fractions": lay["fractions"],
    })
    for name in ("dev", "test"):
        frac = float(unseen_pair_mask(splits[name], splits["train"]).mean())
        print(f"{name}: {len(splits[name])} layouts, {frac:.3f} with a goal/tile pair never seen in training")
    print(f"train: {len(splits['train'])} layouts")
    print(f"Saved {path} (sha256 {manifest['sha256'][:16]}...)")
    return manifest


def make_final_eval(force: bool = False) -> dict:
    """Reserve a fresh evaluation set for the final steering experiment.

    It shares no layout with the train, development or test splits and must be
    frozen before any intervention is developed.
    """
    cfg = json.load(open(CONFIG_PATH))
    env, fe = cfg["env"], cfg["final_eval"]
    path = Path(fe["manifest_path"])
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists and is frozen; pass --force only to regenerate deliberately.")
    splits, meta = load_manifest(cfg["layouts"]["manifest_path"])
    existing = [l for v in splits.values() for l in v]
    layouts = generate_layouts(fe["n_layouts"], fe["generation_seed"], env["width"], env["height"], exclude=existing)
    assert not {l.as_tuple() for l in layouts} & {l.as_tuple() for l in existing}
    manifest = save_layout_set(path, layouts, {
        "experiment": "02_generalizable_reward_hacking", "width": env["width"], "height": env["height"],
        "generation_seed": fe["generation_seed"], "purpose": fe["purpose"],
        "disjoint_from_manifest_sha256": meta["sha256"],
    })
    frac = float(unseen_pair_mask(layouts, splits["train"]).mean())
    print(f"final_eval: {len(layouts)} layouts, {frac:.3f} with a goal/tile pair never seen in training")
    print(f"Saved {path} (sha256 {manifest['sha256'][:16]}...)")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--final-eval", action="store_true")
    a = ap.parse_args()
    make_final_eval(a.force) if a.final_eval else main(a.force)
