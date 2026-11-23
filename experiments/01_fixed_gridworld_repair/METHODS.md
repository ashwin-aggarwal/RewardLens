# Experiment 1: Shared Methods

This file describes the setup that every phase of Experiment 1 shares. Phase-specific methods are in the [progress journal](PROGRESS.md) and the extension reports. Reproduction commands are in [SETUP.md](../../SETUP.md).

## Environment

A deterministic 8×8 gridworld (`rewardlens/envs/reward_grid.py`).

| Element | Position | Reward |
|---|---|---|
| Goal | (6, 1) | +10, ends the episode |
| Reward tile | (3, 5) | +2 on every entry (flawed reward); 0 (corrected reward) |
| Every step | — | −0.01 |

- Episodes last at most 100 steps. Training always starts at (1, 6).
- Observation (7 numbers in [0, 1]): agent x and y, goal x and y, reward-tile x and y, and the fraction of the episode remaining. The goal and tile positions never change in this experiment, so only the agent position and the clock vary.
- Actions: up, down, left, right.
- The loophole: stepping on and off the tile pays about 1 point per step, so farming for 100 steps earns about 97, against about 9.9 for walking to the goal. A scripted check confirms this (Phase 1).

## Agents

- **Network:** `QNetwork`, an MLP: 7 → 128 → ReLU → 128 → ReLU → 4 Q-values (`rewardlens/agents/dqn.py`). "Layer 2" in all reports means the second post-ReLU hidden layer.
- **Training:** DQN with a replay buffer (50k) and a target network, 100k environment steps, learning rate 2.5e-4, γ = 0.99, ε decaying from 1.0 to 0.05 over the first half. Checkpoints every 10k steps.
- **Seeds:** 42, 123 and 456, trained on the flawed reward (`results/phase2/`) and, as a control, on the corrected reward (`results/phase2_5/`).
- **Main checkpoint pair (seed 123):** the 20k checkpoint still walks to the goal; the 30k checkpoint farms. Most intervention results use this pair, with 20k as the *donor* and 30k as the *recipient*.

## Behavior categories (Phase 4.6 onward)

Every episode falls into exactly one category (`rewardlens/analysis/behavior_classification.py`):

| Category | Definition |
|---|---|
| Direct goal | reaches the goal with fewer than 5 reward-tile visits |
| Farm-then-finish | reaches the goal with 5 or more visits |
| Farm-only | does not reach the goal, 5 or more visits |
| Other | does not reach the goal, fewer than 5 visits (wandering or stuck) |

The main success metric is the **direct-goal rate**. Goal completion alone is not enough: in Phase 4.5 a patched agent reached the goal 100% of the time but farmed about 40 times first. Path efficiency is the Manhattan distance to the goal divided by the steps taken.

## Evaluation protocol

- **Start positions.** The 62 squares that are neither the goal nor the reward tile. Each greedy policy is deterministic, so one episode per start covers it completely; rates have 1/62 resolution (1/16 or 1/46 on subsets) and no sampling error.
- **Splits.** Each phase fixes a seeded partition of start positions before running. From Phase 4.6 onward this is seed 13: 30 fit, 16 dev and 16 test positions. Selection (strengths, subsets, ranks) uses dev positions only.
- **Activation patching.** A forward hook replaces or modifies post-ReLU layer 2 of the recipient. In the main patch the donor's layer-2 activation is computed from the same observation with the remaining-time input clamped to 0.2, and fully replaces the recipient's (α = 1).

## Controls

Each claimed effect is compared against controls run under the same protocol:

- **Random directions or subspaces** of the same size, often norm-matched, usually 20 per condition.
- **Shuffled-state donor:** the donor activation is taken from a random wrong square. An effect that survives this is not state-specific.
- **Constant Q bias:** the average edit applied uniformly. Before a linear readout, a constant activation shift is exactly a fixed bias on the four Q-values.
- **Unmodified recipient and donor alone**, reported in every table.
- **Distance baseline** for probes: Manhattan distances to the goal and to the tile.

## Scientific limitations

- **One small, fixed environment.** Goal and tile never move, so the networks can memorize positions. Nothing here shows what happens in other layouts.
- **Three training seeds.** The intervention results come from one seed (123) and one checkpoint pair. Seeds 42 and 456 lack an equally clean goal-following checkpoint, and the patch fails on them.
- **No untouched test set.** With only 62 starts, every position had been analyzed by Phase 4.6. Later results are exploratory, and dev/evaluation splits are disjoint from selection but not confirmatory.
- **Operational definitions.** The 5-visit threshold and the four categories are research definitions. Results are insensitive to thresholds from 5 to 20 for these trajectories.
- **Behavioral, not mechanistic.** Patches and edits that change behavior show what the network's later layers do with given activations. They do not identify a reward-hacking circuit.
