"""Collect activations from trained DQN checkpoints on a matched-state dataset.

Usage:
    python -m scripts.collect_activations
    python -m scripts.collect_activations --seed 42 --condition flawed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rewardlens.analysis.activation_extraction import (
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
    verify_extraction_preserves_predictions,
)


FLAWED_DIR = Path("results/phase2")
CORRECTED_DIR = Path("results/phase2_5")
OUTPUT_DIR = Path("results/phase3")


def collect_for_checkpoint(
    ckpt_path: Path,
    observations: np.ndarray,
    seed: int,
    condition: str,
    step: int,
) -> dict:
    """Extract activations and return serializable record."""
    ok = verify_extraction_preserves_predictions(ckpt_path, observations)
    if not ok:
        raise RuntimeError(f"Extraction verification failed for {ckpt_path}")

    rec = extract_activations(ckpt_path, observations, seed, condition, step)
    return {
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "condition": condition,
        "step": step,
        "n_observations": len(observations),
        "hidden1_shape": list(rec.hidden1.shape),
        "hidden2_shape": list(rec.hidden2.shape),
        "q_values_shape": list(rec.q_values.shape),
        "greedy_action_counts": {
            int(a): int(c)
            for a, c in zip(*np.unique(rec.greedy_actions, return_counts=True))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DQN activations")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to process")
    parser.add_argument("--condition", choices=["flawed", "corrected", "both"], default="both")
    parser.add_argument("--checkpoint", default="final", help="'final', 'all', or step number")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed else [42, 123, 456]
    conditions = ["flawed", "corrected"] if args.condition == "both" else [args.condition]

    observations = build_matched_state_dataset()
    metadata = compute_state_metadata(observations)
    print(f"Matched-state dataset: {len(observations)} observations "
          f"({8*8} positions × {5} time fractions)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "matched_observations.npy", observations)
    meta_save = {k: v.tolist() for k, v in metadata.items()}
    with open(OUTPUT_DIR / "observation_metadata.json", "w") as f:
        json.dump(meta_save, f)
    print(f"Saved observations and metadata to {OUTPUT_DIR}/")

    summary = []
    for seed in seeds:
        for condition in conditions:
            if condition == "flawed":
                base = FLAWED_DIR / f"baseline_seed{seed}" / "checkpoints"
            else:
                base = CORRECTED_DIR / f"goal_seed{seed}" / "checkpoints"

            if not base.exists():
                print(f"  SKIP: {base} not found")
                continue

            if args.checkpoint == "all":
                ckpts = sorted(base.glob("checkpoint_*.pt"))
            elif args.checkpoint == "final":
                ckpts = [base / "checkpoint_final.pt"]
            else:
                ckpts = [base / f"checkpoint_{args.checkpoint}.pt"]

            for ckpt_path in ckpts:
                if not ckpt_path.exists():
                    print(f"  SKIP: {ckpt_path} not found")
                    continue

                name = ckpt_path.stem
                step = 100000 if name == "checkpoint_final" else int(name.split("_")[1])
                print(f"  Extracting: seed={seed} condition={condition} step={step}...")

                rec = extract_activations(ckpt_path, observations, seed, condition, step)
                ok = verify_extraction_preserves_predictions(ckpt_path, observations)

                out_name = f"activations_{condition}_seed{seed}_step{step}"
                np.savez_compressed(
                    OUTPUT_DIR / f"{out_name}.npz",
                    observation=rec.observation,
                    hidden1=rec.hidden1,
                    hidden2=rec.hidden2,
                    q_values=rec.q_values,
                    greedy_actions=rec.greedy_actions,
                )

                entry = {
                    "file": f"{out_name}.npz",
                    "seed": seed,
                    "condition": condition,
                    "step": step,
                    "checkpoint": str(ckpt_path),
                    "n_obs": len(observations),
                    "extraction_verified": ok,
                    "greedy_action_counts": {
                        int(a): int(c) for a, c in
                        zip(*np.unique(rec.greedy_actions, return_counts=True))
                    },
                }
                summary.append(entry)
                print(f"    -> {out_name}.npz (verified={ok})")

    with open(OUTPUT_DIR / "collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. {len(summary)} activation files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
