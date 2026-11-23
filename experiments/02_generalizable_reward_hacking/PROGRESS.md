# Experiment 2 Progress Journal: Generalizable Reward-Hacking Interventions

Chronological record of Experiment 2. The summary is in [README.md](README.md). Experiment 1's journal is [here](../01_fixed_gridworld_repair/PROGRESS.md).

## Phase 1: Procedural Gridworld and Behavioral Baselines

**Date:** 2026-11-23. Branch: `experiment/generalizable-reward-hacking`.

**Goal.** Build a gridworld with a new layout every episode and train reliable good and hacking DQNs on it, evaluated on unseen layouts. No steering, patching or weight editing in this phase.

**Hypotheses going in.**
- H1: the Experiment 1 DQN, given goal and tile coordinates, learns to reach moving goals from plain normalized coordinates.
- H2: continuing that agent on the flawed reward produces reward farming; this was not assumed.
- H3: both behaviors hold on layouts never used in training.

### 1. Design decisions

- **Environment.** `ProceduralRewardGridEnv` subclasses the Experiment 1 `RewardGridEnv` and only replaces the layout on reset, so dynamics, rewards and observation code are the Experiment 1 ones. 8×8, 100 steps, step penalty −0.01, goal +10 (terminal), tile +2 per entry (flawed) or 0 (corrected). No obstacles.
- **Observation.** `[agent_x, agent_y, goal_x, goal_y, tile_x, tile_y, remaining_time]`, each in [0, 1]; coordinates divided by 7. Same ordering as Experiment 1. No relative-distance features: pretraining reached 1.00 direct goal without them, so they were not justified. Nothing privileged is included.
- **Layout validity.** Start, goal and tile are distinct in-bounds cells; the goal is reachable without crossing the tile and the tile without crossing the goal; the tile has a non-goal neighbor so it can be re-entered. On an open 8×8 grid every distinct triple passes, but the generator checks rather than assumes.
- **Split by whole layouts.** 10,000 distinct layouts (of 249,984 possible) drawn with seed 20261123 and split 7,000 / 1,500 / 1,500 with seed 20261124. Saved to `layouts/manifest.json` with a SHA-256 over the splits; loading verifies the hash and disjointness. The generation script refuses to overwrite it.
- **Leakage rules.** Training reads only the training split (a test checks that training runs with an empty test split). Periodic evaluation uses development layouts and a fixed 1,500-layout training sample. Hyperparameters and budgets are chosen on development layouts with a separate pilot seed (7). The test split is evaluated by one command that refuses to run twice.
- **Stricter subset.** Because many held-out layouts share their goal/tile pair with a training layout, results are also reported on the held-out layouts whose pair never occurs in training (272 of dev, 282 of test).
- **Paired evaluation.** Every agent is evaluated greedily on the same layouts, one deterministic episode each, and scored under both reward specifications.
- **Metrics.** Good agent: direct-goal rate. Hacking agent: farming rate = farm-only + farm-then-finish. "Other" is reported separately and never counted as hacking.
- **Fork.** Both branches load the same pretrained weights (checked bit-for-bit in a test and saved as each branch's step-0 checkpoint), use the same seed, a fresh optimizer and an empty replay buffer, the same layouts and the same budget. The reward is the only difference. Exploration restarts at ε = 0.3 and decays to 0.05 over the first 100k fork steps.
- **Checkpoint rule.** Final checkpoint of each stage, fixed after the pilot and before the reporting seeds were trained.

### 2. Scripted returns before training

Script: `scripts/exp2_scripted_returns.py`, on the 8,500 train + development layouts (test not read). Discount γ = 0.99, the DQN's.

| Scripted policy | Goal reached | Tile visits | Discounted return, flawed (mean, min–max) | Discounted return, corrected (mean) |
|---|---|---|---|---|
| Walk to goal (avoiding the tile) | 100% | 0 | 9.5 (8.6–10.0) | 9.5 |
| Farm until time runs out | 0% | 44–50 | 59.0 (51.1–63.1) | −0.6 |
| Farm, then finish on the last step | 100% | 38–50 | 60.7 (50.2–66.8) | 3.1 |

- Flawed reward: farming beats the goal in 100% of layouts, by at least 41.4 discounted points. Farm-then-finish beats pure farming in 90% of layouts, by a small margin.
- Corrected reward: the goal beats both farming policies in 100% of layouts.
- So hacking is never mechanically impossible and reaching the goal is never optimal under the flawed reward.

**A bug caught here.** The first farm-then-finish script left for the goal one step late in about half the layouts (goal rate 0.50). Fixed by checking the distance after the next farming step.

**Threshold validation.** The goal walker makes 0 tile visits and the farmers at least 38, so any threshold from 1 to 38 separates them. A uniformly random policy reaches the goal with few visits in 48% of episodes and is counted as farming in 5.7% at threshold 5, 0.3% at 10 and 0% at 20. Decision: keep 5 as the primary threshold (the Experiment 1 convention) and report 10 and 20 as sensitivity checks.

### 3. Pilot (seed 7, development layouts only)

Configuration 1: Experiment 1 DQN hyperparameters, pretrain 300k, fork 200k.

- Pretraining: direct goal 0.86 at 20k steps, 1.00 from 40k on.
- Good branch: 1.00 throughout.
- Hacking branch: direct goal 1.00 at the fork, farming 0.25 at 60k, 0.80 at 100k, 0.93 at 200k.
- No tuning was needed. H1 and H2 held on the pilot.

### 4. Reporting seeds, first budget: one seed short

Seeds 42, 123, 456 with configuration 1. Development layouts at the end of the 200k fork:

| Seed | Good, direct goal | Hacking, farming | Hacking, other |
|---|---|---|---|
| 42 | 1.000 | 0.932 | 0.000 |
| 123 | 1.000 | **0.715** | 0.140 |
| 456 | 1.000 | 0.882 | 0.040 |

**The hacking target was missed for seed 123.** Its transition started late (farming 0.03 at 100k, 0.40 at 160k) and was still rising when the budget ended. Results preserved in `results/exp2/phase1/fork_200k/`.

**Diagnosis.** Training duration, not reward magnitude or observation: the same seed's curve had the same shape as the others, shifted later, and "other" was elevated mid-transition (0.14–0.30), which is what an unfinished switch looks like. Onset time varying by seed also happened in Experiment 1.

**Adjustment (configuration 2, the only change made).** Fork budget 200k → 400k for both branches and all seeds. Exploration still decays over the first 100k steps, so the first 200k steps are the same run: every branch's step-200k checkpoint is bit-identical to the configuration-1 final checkpoint (verified). Decided from development curves; the test set had not been evaluated. Tuning budget used: 2 of 3 configurations.

### 5. Final results (configuration 2)

**When hacking emerged (development layouts, steps after the fork):**

| Seed | Farming ≥ 0.5 | Farming ≥ 0.8 | Peak "other" during the switch | Final farming |
|---|---|---|---|---|
| 42 | 100k | 160k | 0.24 | 0.917 |
| 123 | 180k | 220k | 0.30 | 0.919 |
| 456 | 140k | 190k | 0.29 | 0.908 |

All three start the fork at direct goal 1.00 and farming 0.00. During the switch up to 30% of episodes are "other" (neither goal nor farming); this falls to under 1% by the end. After about 250k steps the rates are flat.

**Rates, mean ± std over 3 seeds:**

| Agent | Split | Direct goal | Farm only | Farm then finish | Other | Farming |
|---|---|---|---|---|---|---|
| Good | train sample | 1.000 | 0 | 0 | 0 | 0 |
| Good | development | 1.000 | 0 | 0 | 0 | 0 |
| Good | test | 1.000 | 0 | 0 | 0 | 0 |
| Hacking | train sample | 0.075 | 0.913 | 0.010 | 0.001 | 0.924 |
| Hacking | development | 0.082 ± 0.005 | 0.907 ± 0.004 | 0.008 ± 0.002 | 0.004 ± 0.003 | 0.914 ± 0.005 |
| Hacking | test | 0.076 ± 0.002 | 0.907 ± 0.002 | 0.014 ± 0.003 | 0.003 ± 0.002 | 0.921 ± 0.001 |

- Test, per seed: good direct goal 1.000 / 1.000 / 1.000; hacking farming 0.923 / 0.921 / 0.919. Both 0.80 targets met on every seed.
- Unseen goal/tile pairs (282 test layouts): good 1.000, hacking farming 0.918.
- Threshold sensitivity (test, hacking farming): 0.921 at 5, 0.920 at 10, 0.914 at 20.
- Good agent on test: 5.46 steps to goal, path efficiency 1.00, 0.08 tile visits (shortest paths sometimes cross the tile once), return 9.9 under the corrected reward.
- Hacking agent on test: 43.7 tile visits, return 87.5 under the flawed reward and 0.0 under the corrected one.
- Paired (seed 42, test): of 1,500 layouts where the good agent goes directly to the goal, the hacking agent farms in 1,384, goes directly in 111 and fails in 5.
- The pretrained and good agents are behaviorally identical at this resolution. Their weights differ (400k more training steps).
- The test split was evaluated once (2026-11-23T22:30Z), recorded in `evaluation.json`.

**Residual direct-goal behavior in the hacking agent (development layouts, diagnostic only).** About 8% of layouts. It is concentrated where the goal is closer to the start than the tile: direct rate 0.15–0.17 there against 0.01–0.015 elsewhere, and 0.23–0.27 when the goal is one step away. Farming pays more in those layouts too, so this is leftover goal-following. The hacking agent rarely farms and then finishes (about 1%), although the scripted comparison says that would pay slightly more in 90% of layouts.

### 6. Compute

CPU only (8 cores, PyTorch 2.7.1, one thread per process).

| Item | Measured |
|---|---|
| Environment + training speed, one process alone | about 5,700–10,000 steps/s |
| Same, three seeds in parallel | about 2,300–2,700 steps/s per process |
| Pretraining (300k steps), per seed | about 2 minutes |
| Each fork branch (400k steps), per seed | about 3 minutes |
| All seeds, all stages, in parallel | about 9 minutes wall clock |
| Peak memory per process | about 290 MB |
| Greedy evaluation of 1,500 layouts | about 0.13 s |

No GPU or RunPod is needed at this scale.

### 7. Hypotheses, outcome

- H1 held: direct goal on development layouts is at least 0.999 from 50k–70k pretraining steps onward in every reporting seed, and 1.000 on test.
- H2 held, with a caveat: hacking emerged in all 3 reporting seeds and the pilot, but the step at which farming passed 0.5 varied from 80k (pilot) to 180k (seed 123), and the first budget was too short for one seed.
- H3 held: no measurable gap between training, development and test layouts.

### 8. Limitations

- Unseen layouts are unseen combinations from the same generator on an open grid. This is interpolation, not structural generalization.
- Three reporting seeds; one architecture; one reward magnitude (+2); one exploit type; one grid size.
- The fork budget was extended once after seeing development curves. The extension applied to every seed and both branches and preceded any test evaluation, but it is a tuning step and is counted as one.
- The pretrained and good agents are indistinguishable behaviorally, so "good vs hacking" comparisons in later phases should also use the pretrained checkpoint as a reference.
- Training is deterministic on this machine (step-200k checkpoints reproduced bit-for-bit across runs), but bit-exact reproduction on other hardware is not guaranteed. The nine final checkpoints are therefore committed; see section 10.

### 9. Artifacts and commands

- Figures: `assets/figures/exp2_{scripted_returns,training_curves,behavior_by_seed,generalization}.png`. GIFs: `assets/gifs/exp2_{good,hacking}_dqn.gif` (seed 42, test layout index 3; details in `results/exp2/phase1/gif_episode.json`).
- Results: `results/exp2/phase1/` (`scripted_returns.json`, `evaluation.json`, per-seed `eval_metrics.csv`, `training_episodes.csv`, `run_metadata.json`; `pilot/`; `fork_200k/`).
- Commands: see [SETUP.md](../../SETUP.md).

### 10. Finalization (2026-11-23)

Decisions made before Phase 2, at the project owner's direction:

- **Exact weights are committed.** The nine final checkpoints (pretrain, good, hacking for seeds 42, 123, 456; about 2.7 MB) are tracked in git, because activation analysis needs these exact networks and bit-exact retraining on other hardware is not guaranteed. Intermediate checkpoints remain gitignored.
- **Same-seed correspondence is recorded.** `results/exp2/phase1/checkpoints.json` holds a SHA-256 for every checkpoint and, for each fork branch, the hash of the checkpoint it started from. A test checks that both branches of a seed started from that seed's committed pretrain checkpoint. `load_triplet(seed)` returns the three networks.
- **Results kept lightweight.** Committed: summary metrics, per-evaluation curves (`eval_metrics.csv`), run metadata, configs, `evaluation.json`, `scripted_returns.json`, and the 200k and pilot summaries. Everything needed to regenerate the figures is included. Not committed: raw per-episode training logs (`training_episodes.csv`, about 17 MB) and training console logs, which stay on the local machine and are rewritten by retraining.
- **The test split is spent for selection purposes.** It was evaluated once for the Phase 1 baselines. Phase 2 must not use it to choose representations, interventions or hyperparameters; only training and development layouts may be used for that.
- **A fresh final evaluation set is reserved.** 1,500 layouts generated with a new seed (20261125), disjoint from all 10,000 existing layouts, saved to `layouts/final_eval_manifest.json` with a hash. It has not been evaluated by anything. 17.6% of it has a goal/tile pair never seen in training. It is for the final steering experiment, once that method is frozen.
- **Scope of the generalization claim, stated plainly.** Phase 1 shows generalization to unseen combinations of start, goal and tile positions on an open grid. It does not show generalization to new maze structures (there are no obstacles) or to new exploit mechanisms (there is one).
- **Reproducibility check.** A test reloads the committed checkpoints and reproduces the reported development-layout rates exactly. The test split is not re-evaluated by any test.

### 11. Follow-up questions

- Where in the network do the good and hacking agents diverge, given a shared starting point, and does that location or direction stay the same across layouts and seeds?
- Is the residual 8% direct-goal behavior a distinct internal state (goal nearby) that an intervention could widen?
- Does the "other" phase during the switch correspond to a measurable intermediate representation?
- Harder generalization tests for later: obstacles, unseen reward magnitudes, a different exploit, more seeds.
