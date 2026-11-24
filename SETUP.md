# Setup and Reproduction

Every command runs from the repository root. All experiments run on CPU.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.7, scikit-learn 1.4 and Gymnasium 1.4.

## Tests

```bash
python -m pytest -q
```

Tests that need trained checkpoints skip themselves when the checkpoints are missing. Checkpoints are not committed; they are produced by the training commands below.

## Reproduce Experiment 1: Fixed Gridworld Reward-Hacking Repair

Report: [experiments/01_fixed_gridworld_repair/README.md](experiments/01_fixed_gridworld_repair/README.md). Experiment 2 has its own section below.

The phases depend on each other in order. Training (Phase 2 and 2.5) must run first, because every later phase loads its checkpoints from `results/phase2/` and `results/phase2_5/`.

### Phase 1: environment and scripted exploit

```bash
python -m scripts.verify_environment
```

### Phase 2: reward-hacking DQN (flawed reward)

```bash
python -m scripts.discounted_returns
python -m scripts.train_dqn --experiment-id baseline_seed42 --seed 42 --device cpu
python -m scripts.train_dqn --experiment-id baseline_seed123 --seed 123 --device cpu
python -m scripts.train_dqn --experiment-id baseline_seed456 --seed 456 --device cpu
python -m scripts.visualize_phase2 --run-dir results/phase2/baseline_seed42
```

### Phase 2.5: goal-following DQN (corrected reward)

```bash
python -m scripts.train_dqn --config configs/dqn_corrected.json
python -m scripts.train_dqn --config configs/dqn_corrected.json --seed 123 --experiment-id goal_seed123
python -m scripts.train_dqn --config configs/dqn_corrected.json --seed 456 --experiment-id goal_seed456
python -m scripts.evaluate_phase2_5
python -m scripts.visualize_phase2_5
```

When you pass `--seed`, also pass `--experiment-id`. Otherwise the run is named `baseline_seed<N>`.

### Phase 3: activation probing

```bash
python -m scripts.collect_activations --checkpoint final
python -m scripts.probe_activations
python -m scripts.analyze_training_transition
python -m scripts.visualize_phase3
```

### Phase 4: probe-direction interventions

```bash
python -m scripts.run_interventions
python -m scripts.evaluate_interventions
python -m scripts.visualize_phase4
```

### Phase 4.5: goal-activation patching

```bash
python -m scripts.run_phase4_5
python -m scripts.visualize_phase4_5
```

### Phase 4.6: behavior categories, time dependence, steering

```bash
python -m scripts.reclassify_trajectories
python -m scripts.run_phase4_6
python -m scripts.visualize_phase4_6
```

### Phase 4.7: time controls

```bash
python -m scripts.run_phase4_7
python -m scripts.visualize_phase4_7
```

### Extension: sparse patching (branch `experiment/sparse-patching`)

```bash
python -m scripts.run_sparse_patching
python -m scripts.visualize_sparse_patching --gif
```

Report: [experiments/01_fixed_gridworld_repair/sparse_patching/README.md](experiments/01_fixed_gridworld_repair/sparse_patching/README.md).

### Extension: low-rank patching (branch `experiment/low-rank-patching`)

```bash
python -m scripts.run_low_rank_patching    # about 12 minutes on CPU
python -m scripts.visualize_low_rank_patching --gif
```

Report: [experiments/01_fixed_gridworld_repair/low_rank_patching/README.md](experiments/01_fixed_gridworld_repair/low_rank_patching/README.md).

### Phase 5: low-rank weight editing (branch `experiment/weight-editing`)

```bash
python -m scripts.run_weight_editing         # about 19 minutes on CPU; needs Phase 2 checkpoints and results/phase4_6/
python -m scripts.visualize_weight_editing --gif
```

Report: [experiments/01_fixed_gridworld_repair/weight_editing/README.md](experiments/01_fixed_gridworld_repair/weight_editing/README.md). The standalone edited model is saved to `results/weight_editing/models/` and loads with `load_q_network_for_eval`. Like the training checkpoints, it is gitignored; the runner above regenerates it.

## Reproduce Experiment 2: Generalizable Reward-Hacking Interventions

Report: [experiments/02_generalizable_reward_hacking/README.md](experiments/02_generalizable_reward_hacking/README.md). Independent of Experiment 1's checkpoints.

### Phase 1: procedural gridworld and behavioral baselines

```bash
python -m scripts.exp2_scripted_returns       # scripted goal vs farming returns (train + dev layouts)
python -m scripts.exp2_train --all-seeds      # pretrain, then good and hacking forks; about 9 minutes on CPU
python -m scripts.exp2_evaluate               # training sample + development layouts
python -m scripts.exp2_evaluate --final-test  # adds the held-out test layouts; refuses to run twice
python -m scripts.exp2_visualize --gif
python -m pytest tests/test_exp2_phase1.py -q
```

### Phase 2: representation analysis and preliminary transplantation

```bash
python -m scripts.exp2_run_phase2             # about 2.5 minutes on CPU; needs the nine committed checkpoints
python -m scripts.exp2_visualize_phase2 --gif
python -m pytest tests/test_exp2_phase2.py -q
```

### Phase 3: tracking the emergence of reward hacking

```bash
python -m scripts.exp2_train --all-seeds      # only if the intermediate checkpoints are missing; about 9 minutes on CPU
python -m scripts.exp2_run_phase3             # about 13 minutes on CPU
python -m scripts.exp2_visualize_phase3 --gif
python -m pytest tests/test_exp2_phase3.py -q
```

Phase 3 reads every fork-stage checkpoint (`checkpoint_<step>.pt`, every 10k steps, good and hacking branch). These 246 files are gitignored; `results/exp2/phase3/checkpoint_registry.json` records the hash of each, and `python -m scripts.exp2_run_phase3 --part registry` re-verifies a regenerated set (it raises if steps, reward modes or the fork-start and final weights do not line up). The parts can be run one at a time with `--part registry`, `--part timeline` or `--part decisions`; `decisions` needs `timeline.json` and only the 12 checkpoints listed in `results/exp2/phase3/required_checkpoints.json`. Retraining overwrites the committed final checkpoints, so restore them afterwards with `git checkout -- results/exp2/phase1` if the hashes differ on your hardware.

### Phase 4: near-transition causal patching

```bash
python -m scripts.exp2_run_phase4             # about 3.3 minutes on CPU
python -m scripts.exp2_visualize_phase4 --gif
python -m pytest tests/test_exp2_phase4.py -q
```

Phase 4 needs `results/exp2/phase3/timeline.json` and the 24 checkpoints listed with hashes in `results/exp2/phase4/required_checkpoints.json`. All 24 are committed, including the nine hacking checkpoints 10k, 20k and 50k before each seed's completion step (2.7 MB); `python -m pytest tests/test_exp2_phase4.py -q` checks their hashes. Pairs, controls and the success criterion are fixed in `experiments/02_generalizable_reward_hacking/configs/phase4.json`.

Phase 2 reads training and development layouts only. `--part representations` or `--part patching` runs one half. Settings are in `experiments/02_generalizable_reward_hacking/configs/phase2.json`; results go to `results/exp2/phase2/`.

### Layouts and checkpoints

The layout split is frozen in `experiments/02_generalizable_reward_hacking/layouts/manifest.json`. `python -m scripts.exp2_make_layouts` regenerates the identical file from the seeds in `configs/environment.json` and refuses to overwrite an existing one. One seed or stage can be run with `python -m scripts.exp2_train --seed 42 --stage pretrain` (stages: `pretrain`, `good`, `hacking`, `fork`). Checkpoints are written to `results/exp2/phase1/seed<N>/<stage>/checkpoints/`. The nine `checkpoint_final.pt` files (pretrain, good, hacking for each seed) are committed, so the evaluation and visualization commands work without retraining; intermediate checkpoints and per-episode training logs are gitignored. Retraining overwrites the committed files, and may not reproduce them bit-for-bit on other hardware, so after retraining rebuild the registry or restore the files with `git checkout -- results/exp2`.

```python
from rewardlens.exp2.checkpoints import load_triplet
nets = load_triplet(42)   # {"pretrain": ..., "good": ..., "hacking": ...}, hashes verified
```

The test split has been used once for the Phase 1 baselines and must not guide Phase 2 choices. A separate final evaluation set is frozen in `layouts/final_eval_manifest.json` (`python -m scripts.exp2_make_layouts --final-eval` regenerates the identical file and refuses to overwrite it); do not evaluate on it until the final steering experiment.

## Where things live

| Path | Contents |
|---|---|
| `rewardlens/envs/` | Gridworld environment, scripted policies, trajectory recording |
| `rewardlens/agents/dqn.py` | Q-network, replay buffer, checkpointing |
| `rewardlens/analysis/` | Activation extraction, probing, interventions, patching, behavior classification |
| `scripts/` | One runner and one visualizer per phase |
| `configs/`, `experiments/01_fixed_gridworld_repair/*/configs/`, `experiments/02_generalizable_reward_hacking/configs/` | Fixed experiment configurations |
| `rewardlens/exp2/` | Experiment 2 code: layouts, procedural environment, scripted policies, batched evaluation, training |
| `experiments/` | One folder per experiment: report, methods, progress journal, extension reports |
| `results/` | Machine-readable results; checkpoints and trajectories are gitignored |
| `assets/figures/`, `assets/gifs/` | Generated figures and GIFs |
| `tests/` | One test file per phase or extension |
