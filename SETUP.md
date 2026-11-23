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

Report: [experiments/01_fixed_gridworld_repair/README.md](experiments/01_fixed_gridworld_repair/README.md). Experiment 2 is planned and has no code yet.

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

## Where things live

| Path | Contents |
|---|---|
| `rewardlens/envs/` | Gridworld environment, scripted policies, trajectory recording |
| `rewardlens/agents/dqn.py` | Q-network, replay buffer, checkpointing |
| `rewardlens/analysis/` | Activation extraction, probing, interventions, patching, behavior classification |
| `scripts/` | One runner and one visualizer per phase |
| `configs/`, `experiments/01_fixed_gridworld_repair/*/configs/` | Fixed experiment configurations |
| `experiments/` | One folder per experiment: report, methods, progress journal, extension reports |
| `results/` | Machine-readable results; checkpoints and trajectories are gitignored |
| `assets/figures/`, `assets/gifs/` | Generated figures and GIFs |
| `tests/` | One test file per phase or extension |
