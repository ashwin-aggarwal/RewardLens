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


## Phase 2: Representation Analysis and Preliminary Transplantation

**Date:** 2026-11-23. Branch: `experiment/generalizable-reward-hacking`. Script: `scripts/exp2_run_phase2.py`. Settings: `configs/phase2.json`.

**Question.** Do reward-hacking DQNs develop consistent internal representations across randomized layouts and independent seeds, and does transplanting goal-following activations change their behavior?

**Hypotheses, written before running.**
- H1: the good branch stays close to the pretrained network; the hacking branch diverges, more in layer 2 than layer 1.
- H2: the matched-state difference between hacking and good activations is low-dimensional.
- H3: the change is similar across seeds once neuron misalignment is accounted for.
- H4: a same-seed full-layer transplant (good → hacking) restores direct goal pursuit in a state-dependent way, beating shuffled-state and constant-bias controls.

### 1. Methods

- **Data rules.** Training and development layouts only. The test split and `final_eval_manifest.json` are not read; a test scans the Phase 2 source files for any reference to them and checks that the setup exposes only training and development layouts.
- **Matched states.** For each layout, 4 states with the agent on a uniformly random non-goal cell and a uniformly random elapsed time. 6,000 "fit" states from the first 1,500 training layouts (seed 101) and 6,000 evaluation states from the 1,500 development layouts (seed 202). Every network sees the same observations. These states are policy-independent, so many are states a trained agent would rarely visit.
- **Networks.** The nine committed checkpoints, loaded and hash-verified with `load_triplet`.
- **Activations.** Post-ReLU layer 1 and layer 2 (128 units each), computed without hooks.
- **Within a seed.** Linear CKA; mean per-state activation change relative to the pretrained norm; relative weight change per layer; PCA of matched-state differences (fitted on fit states, variance measured on development states); the share of the difference that is a constant offset; linear regression of the top-8 difference components on 14 hand-written state features.
- **Across seeds.** Raw differences are never averaged or subtracted across seeds, because neurons do not correspond. Two tools: (a) CKA, which ignores neuron order and rotation; (b) held-out R² of a ridge map (α = 0.001) from one seed's representation to another seed's top-8 difference components, fitted on fit states and scored on development states. Comparison inputs of equal size: the other seed's top-8 difference components, its top-8 good-activation components, its top-8 hacking-activation components, and the difference components with states shuffled.
- **Transplant.** `h ← h_recipient + α(h_donor − h_recipient)` at α = 1, at layer 1 or layer 2, on all 1,500 development layouts. At layer 2 this is the recipient's output layer reading the donor's activation. Both layers are reported; nothing was selected or tuned.
- **Controls.** Unmodified agents; donor activation from a random training-layout state (5 draws); donor seeing the right goal and tile but a random agent position and time (5 draws); a constant Q-value bias equal to the mean Q change the matched patch causes on fit states. Extra directions: pretrained → hacking, and hacking → good.
- **Fixed in advance:** k = 8, α = 1, 5 control draws, both layers. One run; no settings were changed after seeing results.

### 2. Results: representations

**Within seed (development states; seeds 42 / 123 / 456):**

| Quantity | Layer 1 | Layer 2 |
|---|---|---|
| CKA pretrained vs good | 0.99 / 1.00 / 1.00 | 0.99 / 0.99 / 0.99 |
| CKA pretrained vs hacking | 0.88 / 0.76 / 0.90 | 0.37 / 0.41 / 0.46 |
| CKA good vs hacking | 0.86 / 0.74 / 0.89 | 0.36 / 0.40 / 0.41 |
| Activation change vs pretrained, good | 0.16 / 0.14 / 0.11 | 0.06 / 0.07 / 0.06 |
| Activation change vs pretrained, hacking | 0.71 / 0.64 / 0.67 | 1.84 / 2.44 / 2.05 |
| Mean activation norm, good → hacking | 4.0 → 5.9 / 4.7 → 6.7 / 4.3 → 6.3 | 11.3 → 31.4 / 11.5 → 39.1 / 13.0 → 39.3 |

- Weights, relative change from pretrained: good 0.21–0.29 (fc1), 0.27–0.38 (fc2), 0.14–0.18 (fc3); hacking 0.75–0.97 (fc1), 1.77–1.82 (fc2), 1.87–2.06 (fc3).
- **H1 held.** The good branch is almost unchanged; the hacking branch changed most in layer 2 and in the output layer.
- Greedy actions on matched states agree between pretrained and good in only 67–85% of states although both reach the goal directly everywhere. Many states have two equally short moves, and many matched states are off-policy, so action agreement understates behavioral similarity.

**The (hacking − good) difference:**

| Quantity | Layer 1 | Layer 2 |
|---|---|---|
| Share that is a constant offset | 0.79 / 0.72 / 0.78 | 0.88 / 0.93 / 0.91 |
| Components for 90% of the remaining variance | 7 / 6 / 8 | 4 / 5 / 4 |
| Variance in top 8 components | 0.93 / 0.95 / 0.91 | 0.96 / 0.96 / 0.96 |
| R² of top-8 components from state features | 0.98 / 0.98 / 0.98 | 0.96 / 0.96 / 0.96 |

- Layer 2, strongest single correlate of the first component: time remaining (r = −0.98 / −0.78 / +0.87; the sign of a component is arbitrary). Components 2–3 correlate with the tile's horizontal or vertical offset from the agent (|r| 0.74–0.91).
- **H2 held, with a qualification.** The state-dependent part is low-dimensional, but most of the difference is a constant shift that carries no state information.

### 3. Results: across seeds

- Layer-2 CKA: hacking vs hacking 0.77–0.86; good vs good 0.86–0.93; pretrained vs pretrained 0.84–0.90; hacking vs good or pretrained (any seeds) 0.32–0.48.
- CKA between difference representations of two seeds: layer 2 0.74–0.85, layer 1 0.66–0.86.
- Held-out R² for one seed's top-8 layer-2 difference components, over the 6 ordered seed pairs:

| Predictor (other seed, 8 components) | Layer 1 | Layer 2 |
|---|---|---|
| Difference components | 0.97 (0.97–0.97) | 0.94 (0.92–0.96) |
| Hacking activations | 0.98 (0.97–0.98) | 0.94 (0.93–0.96) |
| Good activations | 0.97 (0.97–0.98) | 0.76 (0.64–0.82) |
| Difference components, states shuffled | 0.00 | 0.00 |

- **H3 held in a limited sense.** The state-dependent change at layer 2 is the same function of the state across seeds, and good networks express it less directly. But another seed's ordinary hacking activations predict it equally well, and at layer 1 any network does. Since the components are almost linear in the clock and the tile offset (R² 0.96), the shared structure may simply be what the farming task requires. This is similarity, not evidence of a common causal mechanism.

### 4. Results: transplantation (1,500 development layouts)

Unmodified: pretrained and good 1.00 direct goal in every seed; hacking 0.08 / 0.08 / 0.09 direct, 0.92 / 0.92 / 0.91 farming.

**Good → hacking** (direct goal / farming / other):

| Condition | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Layer 2, matched | 0.44 / 0.03 / 0.54 | 0.11 / 0.00 / 0.89 | 0.30 / 0.07 / 0.63 |
| Layer 2, random-state donor (mean of 5) | 0.27 / 0.07 / 0.67 | 0.13 / 0.00 / 0.87 | 0.19 / 0.04 / 0.78 |
| Layer 2, wrong-position donor (mean of 5) | 0.34 / 0.03 / 0.62 | 0.11 / 0.00 / 0.89 | 0.16 / 0.05 / 0.79 |
| Layer 2, constant Q bias | 0.08 / 0.92 / 0.00 | 0.08 / 0.85 / 0.07 | 0.08 / 0.90 / 0.02 |
| Layer 1, matched | 0.17 / 0.25 / 0.59 | 0.10 / 0.01 / 0.89 | 0.14 / 0.15 / 0.70 |
| Layer 1, random-state donor (mean of 5) | 0.26 / 0.05 / 0.69 | 0.16 / 0.04 / 0.80 | 0.36 / 0.05 / 0.59 |
| Layer 1, constant Q bias | 0.05 / 0.75 / 0.20 | 0.07 / 0.80 / 0.13 | 0.09 / 0.91 / 0.01 |

- Layer 2 matched patch: reward-tile visits 0.8 / 0.1 / 1.4 per episode; path efficiency of the direct-goal episodes 0.77 / 0.78 / 0.77, against 0.36 / 0.73 / 0.52 for random-state donors.
- **Pretrained → hacking, layer 2, matched:** 0.52 / 0.08 / 0.53 direct goal.
- **Hacking → good, layer 2, matched:** direct 0.10 / 0.06 / 0.08, farming 0.04 / 0.00 / 0.02, other 0.86 / 0.94 / 0.91. Transplanting hacking activations does not make the good network farm.

**H4 failed.**
- The transplant reliably removes farming, and so do donors from the wrong state. That part is disruption.
- Direct goal pursuit is restored in at most 44% of layouts. The matched patch beats the wrong-state controls by 0.10–0.17 for seeds 42 and 456, with more direct paths, so a state-dependent component exists there. For seed 123 it is indistinguishable from a random-state donor (0.11 vs 0.13).
- At layer 1 the matched patch is worse than a random-state donor in every seed.
- The constant Q bias changes almost nothing at layer 2, so the effect is not a uniform action-preference shift.

**Negative results, preserved:** H4 at both layers and in all three seeds; the pretrained donor for seed 123; the reverse transplant; layer-1 patches doing worse than their shuffled control.

### 5. Interpretation

- Unlike Experiment 1, where donor and recipient were 10k steps apart on one training run and the recipient's output layer could read the donor's layer 2, here the branches trained apart for 400k steps. Layer-2 activation norms tripled and the output layer moved by about twice its original weight norm. A good activation of norm about 11 is far outside what the hacking output layer was trained to read (about 31–39).
- Representation similarity across seeds says the hacking networks compute similar features. The transplant result says that having the good network's features in place of the hacking network's is not enough to make the hacking network's later weights pursue the goal. These are different questions, and only the second is causal.
- "Other" is the dominant outcome of every α = 1 patch. In Experiment 1 terms these patches are closer to the wandering produced by direction subtraction than to the time-shifted transplant.

### 6. Limitations

- Matched states are uniform over positions and time, not the states agents actually visit.
- One transplant strength (α = 1) and full-layer replacement only. No partial, low-rank or norm-matched patches were tried, deliberately: this phase was meant to be preliminary and untuned.
- Only final checkpoints were used. The fork checkpoints nearer the transition exist on the local machine (every 10k steps) but are not committed.
- Three seeds; one environment family; behavior measured on development layouts, which were also used for Phase 1 budgeting decisions.
- Linear tools only (PCA, linear CKA, ridge). Nonlinear shared structure would be missed.
- Feature correlations describe what difference directions co-vary with; they do not show what the network uses them for.

### 7. Compute and artifacts

- `python -m scripts.exp2_run_phase2`: about 2.5 minutes on CPU. Visualization: under a minute.
- Results: `results/exp2/phase2/representation_analysis.json`, `patching_results.json`, `pca_scatter.npz`, `gif_episode.json`.
- Figures: `assets/figures/exp2_phase2_{cka,difference_pca,cross_seed,patching}.png`. GIFs: `assets/gifs/exp2_phase2_{hacking,patched}.gif` (seed 42, development layout index 0; the patched episode ends as "other").
- Tests: `tests/test_exp2_phase2.py` (17).

### 8. Recommended next steps

1. **Patch across a smaller training gap.** Use hacking-branch checkpoints from just before and just after the switch (as Experiment 1 did), where the output layer may still read goal-following activations. This needs the intermediate fork checkpoints committed or regenerated.
2. **Patch only the state-dependent directions.** Leave the large constant offset and the activation scale of the hacking network in place, and replace or remove only the 4–5 state-dependent components.
3. **Translate between the spaces.** Fit a linear map from good to hacking activation space on training-layout states, so the hacking output layer receives inputs in its own scale.
4. **Keep the evaluation rules.** Develop on training and development layouts; open the reserved final evaluation set once, for the frozen method.

## Phase 3: Tracking the Emergence of Reward Hacking

**Date:** 2026-11-23. **Branch:** `experiment/generalizable-reward-hacking`. **Layouts used:** training and development only. The test split and the reserved final evaluation set were not loaded.

**Question.** When do the DQNs begin reward hacking, how do their hidden layers change during that transition, and which changes go with individual decisions? This phase observes; it runs no intervention. Phase 2 was committed and pushed before it started (commits `c62fb11` to `7effa21`).

**Hypotheses, written before the run.**

- P1: the hacking branch's hidden layers begin to diverge from the good branch's before the greedy policy changes.
- P2: the fastest representation change coincides with the behavioral transition.
- P3: activation change between a goal-following and a hacking checkpoint is larger at states where their decisions differ than where they agree.
- P4: the pattern is the same across seeds once timing is aligned.

### 1. Checkpoints

- Source: the fork-stage checkpoints written by `scripts.exp2_train` in Phase 1, every 10k steps from 0 to 400k, for the good and hacking branch of seeds 42, 123 and 456. 41 per branch, 246 files, 73 MB. Nothing was retrained.
- Integrity (`results/exp2/phase3/checkpoint_registry.json`): each file's stored step, reward mode and seed match its path; steps strictly increase; step 0 of both branches has the weights of the seed's committed pretrained checkpoint; step 400k has the weights of the committed final checkpoint; no two checkpoints of a trajectory share weights. All 246 passed.
- Behavior recomputed from the checkpoints matches the development farming rates logged during Phase 1 training exactly (largest difference 0.0 in every seed).
- What is kept: the 246 files stay local and gitignored. The registry lists a file hash and a weights hash for each. `python -m scripts.exp2_train --all-seeds` regenerates them in about 9 minutes; in Phase 1 a rerun reproduced earlier checkpoints bit for bit on this machine, and the registry hashes let anyone check that. The 12 checkpoints the decision-level analysis loads (both branches at the last step before onset and at completion, per seed; `required_checkpoints.json`, 3.6 MB) are un-ignored in `.gitignore` and committed; the other 234 stay local.

### 2. Methods

Settings are in `configs/phase3.json` and were fixed before the run. One run; nothing was tuned afterwards.

- **Behavior panel:** all 1,500 development layouts, one greedy episode each, at every checkpoint. The first 1,500 training layouts were scored as well.
- **Transition steps:** onset, midpoint and completion are the first checkpoints from which the development farming rate stays at or above 0.1, 0.5 and 0.8 for the rest of training. A brief early crossing that falls back does not count.
- **Matched states:** the Phase 2 states (6,000 uniform states from development layouts, 6,000 from the first 1,500 training layouts, same seeds). Every checkpoint sees identical observations.
- **Per checkpoint, layers 1 and 2:** change from the previous checkpoint and from the fork start (mean per-state distance over mean norm), linear CKA with the previous checkpoint, the fork start and the good branch at the same step, mean activation norm, weight change per layer, mean maximum Q-value, and the share of uniform states whose greedy action approaches the tile and not the goal.
- **Trajectory PCA:** each checkpoint is one point, its activations on 2,000 fixed states flattened; PCA over the 82 points of a seed (both branches). Coordinates are per seed and are not compared across seeds.
- **Representation onset rule (uses no later checkpoint):** first checkpoint where the hacking branch's layer-2 change from the fork start exceeds twice the largest change the good branch has shown so far.
- **Decision pair:** the last hacking-branch checkpoint before onset ("goal-following") and the one at completion ("hacking"): 60k and 160k (seed 42), 110k and 220k (seed 123), 90k and 190k (seed 456). Control: the good branch at the same two steps.
- **Decision labels** for a state: agree; equivalent route (different actions, both closer to the goal); tile over goal (the earlier action approaches the goal, the later one does not and approaches the tile or steps off it); other disagreement.
- **Three sampling distributions, kept separate:** uniform states; states visited by the goal-following checkpoint's greedy rollouts; states visited by the hacking checkpoint's rollouts (6,000 sampled from each, development layouts for evaluation, training layouts for fitting).
- **Controls for position, distances and time:** a gradient-boosted model of the activation-change size from 14 state features (fitted on training-layout states), with the gap between flipped and other states recomputed on residuals; AUROC inside strata of tile distance, goal distance and time tercile; gradient-boosted classifiers of "tile over goal" from state features, then adding the earlier checkpoint's Q-values, then adding the size of the activation change.
- **Across seeds:** CKA between seeds at the same step; overlap of flipped states on identical observations; correlation of direction scores on identical observations. No neuron indices are compared.

### 3. Results: timing

| Hacking branch | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Onset / midpoint / completion (development) | 70k / 100k / 160k | 120k / 180k / 220k | 100k / 140k / 190k |
| Same, first 1,500 training layouts | identical | identical | identical |
| Peak share of "other" episodes (step) | 0.24 (130k) | 0.30 (180k) | 0.29 (140k) |
| Good branch, maximum farming over all checkpoints | 0.00 | 0.00 | 0.00 |

Direct goal falls as farming rises, with a wandering stage in between that disappears by completion.

### 4. Results: representation change over training

| Layer 2, hacking branch (good branch in brackets) | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Change from start at 10k | 0.067 (0.011) | 0.093 (0.014) | 0.047 (0.017) |
| Representation onset by the fixed rule | 10k | 10k | 10k |
| Last step before behavioral onset | 60k | 110k | 90k |
| Change from start at that step | 0.13 (0.02) | 0.22 (0.04) | 0.20 (0.03) |
| CKA with start at that step | 0.93 (1.00) | 0.77 (1.00) | 0.76 (1.00) |
| Share of final change at that step | 0.07 | 0.09 | 0.10 |
| Mean change per 10k: before / during / after | 0.043 / 0.113 / 0.039 | 0.052 / 0.134 / 0.040 | 0.047 / 0.116 / 0.041 |
| Good branch, same intervals | 0.008 / 0.006 / 0.005 | 0.008 / 0.005 / 0.005 | 0.007 / 0.005 / 0.004 |
| Step of largest change per 10k | 160k | 220k | 180k |
| Share of final change at completion | 0.53 | 0.48 | 0.46 |
| CKA with start at completion / at 400k | 0.34 / 0.37 | 0.28 / 0.41 | 0.42 / 0.46 |
| Mean activation norm, start to 400k | 11.6 to 31.4 | 11.7 to 39.1 | 13.3 to 39.3 |

- **P1 held.** The hacking branch's layer 2 leaves the good branch's from the first checkpoint, 60k to 110k steps before the greedy policy farms in 10% of layouts. This is expected rather than surprising: the flawed reward changes the training targets from step 0, and exploration (ε starts at 0.3) reaches the tile long before the greedy policy prefers it. The good branch, with the same exploration schedule, barely moves, so the drift is not ordinary continued learning.
- **P2 held in part.** The largest per-step change falls inside the transition window in every seed, but at its end (at or 10k before completion), not at onset or midpoint, and the rate is elevated over the whole window. There is no sharp spike.
- **About half of the change comes after behavior has settled.** Activation norm starts growing at about the midpoint and continues until 260k to 360k steps, while the mean maximum Q-value rises from about 9 to 34 / 59 / 49. Before that rise the mean maximum Q-value sags (to 7.7 / 4.9 / 5.8 at 100k / 150k / 120k).
- **Layer 1** changes less: CKA with start ends at 0.88 / 0.76 / 0.90, change from start 0.71 / 0.64 / 0.67 (layer 2: 1.84 / 2.44 / 2.05). Weight change from start at 400k, fc1 / fc2 / fc3: 0.75 / 1.80 / 1.87, 0.96 / 1.77 / 1.94, 0.86 / 1.82 / 2.06.
- **Trajectory PCA:** the first component holds 95–96% of the layer-2 trajectory's variance (layer 1: 84–88%). It mostly follows activation size, so the picture is dominated by the post-transition growth. The good branch's 41 checkpoints sit on the starting point.
- **Uniform-state actions:** the share of states where the greedy action approaches the tile and not the goal is 0.04 / 0.04 / 0.03 at the last step before onset and 0.35 / 0.31 / 0.35 at 400k.

### 5. Results: decisions

Share of development states by label (seeds 42 / 123 / 456):

| Sampling | Agree | Equivalent route | Tile over goal | Other |
|---|---|---|---|---|
| Uniform | 0.48 / 0.54 / 0.49 | 0.18 / 0.16 / 0.18 | 0.28 / 0.24 / 0.29 | 0.06 / 0.05 / 0.04 |
| Visited by the goal-following checkpoint | 0.23 / 0.27 / 0.33 | 0.08 / 0.09 / 0.08 | 0.33 / 0.22 / 0.25 | 0.36 / 0.42 / 0.35 |
| Visited by the hacking checkpoint | 0.26 / 0.35 / 0.37 | 0.23 / 0.17 / 0.13 | 0.43 / 0.42 / 0.43 | 0.08 / 0.07 / 0.07 |

Layer-2 activation change between the two checkpoints, relative to the earlier activation's norm, uniform states:

| | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Agree states | 1.05 | 1.24 | 1.01 |
| Tile-over-goal states | 1.00 | 1.23 | 1.02 |
| Good branch, same steps, any label | 0.02 | 0.02 | 0.02 |
| Cohen's d, tile over goal against the rest | −0.07 | −0.08 | +0.10 |
| Same, after removing what state features explain | −0.03 | −0.11 | −0.03 |
| Variance of the change size explained by state features | 0.99 | 0.99 | 0.98 |

- **P3 failed.** Across the three seeds, three sampling distributions and both layers, Cohen's d for the size of the activation change at tile-over-goal states against all other states lies between −0.16 and +0.17, and between −0.13 and +0.18 after the state-feature control. No sign is consistent. The layer changes by about its own size at every state; decision points are not special.
- **Which decisions flip is a function of the state.** Held-out AUROC for predicting tile over goal on uniform states: state features 0.98 / 0.97 / 0.99; plus the earlier checkpoint's Q-values 0.98 / 0.97 / 0.98; plus the size of the activation change 0.98 / 0.97 / 0.98; size of the change alone 0.58 / 0.56 / 0.54. On rollout states the feature baseline is 0.94–0.97 and the additions change it by at most 0.01.
- **A single direction does not capture the flip.** The difference between the mean layer-2 change at tile-over-goal states and at other states, fitted on training-layout states, gives AUROC 0.55 / 0.59 / 0.67 on uniform development states (0.52–0.58 on rollout states). Tile distance alone gives 0.58 / 0.61 / 0.57. Inside strata of tile distance, goal distance and time the direction scores 0.59 / 0.63 / 0.65, so the little it has is not only tile proximity, but it is weak. One reason is built in: "toward the tile" means a different action in different states, and one direction cannot encode four.
- **Near-circular upper bound, reported for completeness:** adding the top 8 components of the layer-2 change to features and Q-values gives 0.94–0.99, no better than features alone. The later checkpoint's layer 2 determines its Q-values linearly, so this could not have been evidence either way.
- **Worked example** (`exp2_phase3_decision_divergence.png`; seed 42, first uniform state meeting a fixed rule): at step 39 with the goal 3 cells up and the tile up and to the right, the 60k checkpoint moves up (Q margin 0.09) and the 160k checkpoint moves right (margin 0.64, all Q-values near 23). 83 of the 128 layer-2 units change by more than 0.05 at this state; the good branch's largest per-unit change is 0.05.

### 6. Results: across seeds

- **Timing** differs by up to 60k steps; the order of events (early drift, wandering stage, late peak, post-transition growth) is the same in all three. P4 held for the timeline.
- **Same-step CKA between hacking branches, layer 2:** starts at 0.84–0.90, falls to 0.50 (42 vs 123, at 150k), 0.27 (42 vs 456, at 190k) and 0.32 (123 vs 456, at 230k), and ends at 0.86 / 0.86 / 0.77. Good branches stay at 0.84–0.93. Layer 1 between hacking branches: minimum 0.78–0.87, end 0.83–0.91. The dip is partly a timing effect, since the seeds are at different stages at the same step.
- **At the decision-pair checkpoints,** layer-2 CKA between seeds is 0.79–0.86 for the goal-following checkpoints and 0.76 / 0.25 / 0.34 for the hacking checkpoints (42 vs 123, 42 vs 456, 123 vs 456). So seeds are still dissimilar when the behavior has just completed and converge later.
- **Flipped decisions overlap:** both seeds flip in 0.185 / 0.165 / 0.174 of identical uniform states, against 0.069 / 0.083 / 0.071 expected if independent (phi 0.60 / 0.40 / 0.53).
- **The flip direction is not shared:** direction scores on identical states correlate 0.74 / −0.05 / −0.33. P4 failed at the level of directions. The size of the layer-2 change per state correlates 0.79 / 0.07 / 0.18.

### 7. Interpretation

- The network does not acquire a discrete "farming feature" at the moment behavior flips. Layer 2 drifts from the first checkpoint, reorganizes over about 100k steps, and then mostly rescales as value estimates grow. Behavior flips partway through.
- At the level of single decisions the change is global. Whether a decision flips is predicted almost perfectly by where the agent, goal and tile are and how much time is left, and the amount of activation change adds nothing to that.
- The Phase 2 transplant failure fits this picture: by 400k most of the layer-2 difference is post-transition growth that the good network never underwent. The checkpoints either side of the transition differ far less in scale (norm about 11 against 21 / 23 / 24 at completion), which makes them better candidates for Phase 4.
- None of this is causal. CKA, PCA and the decision statistics describe what changed, not what produces the behavior.

### 8. Negative results and limitations

- **Negative:** P3 (no larger change at flipped decisions, with or without controls); no shared flip direction across seeds; no single-direction account of the flip within a seed; no sharp spike at onset or midpoint.
- **Resolution:** checkpoints are 10k steps apart. Onset and completion are known to that precision only, and "precedes" means by at least one checkpoint.
- **The early drift is unsurprising.** It shows that the representation changes before the greedy policy does, not that it predicts when or whether hacking will follow. No early-warning claim is made: every hacking run here did hack, so there is no run to tell apart.
- **Thresholds** (0.1, 0.5, 0.8) are conventions. The curves are in `timeline.json` for other choices.
- **Display choices use later information:** the heatmap sorts units by their final change and the PCA is fitted on the whole trajectory. Neither feeds a timing claim.
- **Decision labels** depend on greedy actions, which flip on small Q differences; 16–18% of uniform states are "equivalent route" disagreements that are not behavior changes. The pair spans 100k–110k steps, so it compares before and after, not the moment of change.
- **Uniform states** over-represent positions and times the agents rarely reach; rollout states are dominated by the area around the tile (hacker) or short goal paths (goal-follower). Results are reported for each separately and agree.
- **Linear and magnitude-based tools only.** A nonlinear or multi-directional code for the flip would be missed.
- Three seeds, one architecture, one environment family.

### 9. Compute and artifacts

- **Reproducibility check (2026-11-24):** a full rerun of `python -m scripts.exp2_run_phase3` (9.8 minutes) reproduced all four JSON files and both `.npz` files exactly, and regenerating the figures and GIF gave byte-identical files.

- `python -m scripts.exp2_run_phase3`: about 13 minutes on CPU (registry 5 s, timeline 9 min, decisions 4 min). Visualization with the GIF: under a minute.
- Results (`results/exp2/phase3/`, 2.9 MB): `checkpoint_registry.json`, `required_checkpoints.json`, `timeline.json`, `mean_activations.npz` (per-unit mean activation at every checkpoint), `decision_analysis.json`, `decision_points.npz` (observation, layout, elapsed steps, both actions, both sets of Q-values, label and activation-change size for every analyzed development state), `gif_episode.json`.
- Figures: `assets/figures/exp2_phase3_{behavior_timeline,activation_timeline,neuron_heatmap,pca_trajectory,decision_divergence,decision_stats,cross_seed}.png`.
- GIF: `assets/gifs/exp2_phase3_training_progression.gif` (seed 42, development layout 0, checkpoints 0 / 60k / 100k / 160k / 400k; goal in 9 steps, goal in 19 steps with one tile entry, goal in 17 steps, 100 steps of wandering, 47 tile entries).
- Tests: `tests/test_exp2_phase3.py` (31).

### 10. Recommended Phase 4 experiments

1. **Transplant across the transition within the hacking branch.** Donor: the last checkpoint before onset. Recipient: the completion checkpoint. Same controls as Phase 2 (random-state donor, wrong-position donor, constant Q bias), plus the good branch at the donor step as a second donor.
2. **Locate the flip by layer.** Combine the earlier checkpoint's hidden layers with the later checkpoint's output layer and the reverse, to see whether the flipped decisions follow the features or the readout.
3. **Scale-matched patches.** Rescale or linearly translate donor activations to the recipient's scale before patching, since activation size is the largest change after the midpoint.
4. **Sweep the recipient along training.** Apply one fixed donor to recipients at 160k, 200k, 300k and 400k to measure when a transplant stops being readable.
5. Keep development on training and development layouts and open the final evaluation set once, for the frozen method.

## Phase 4: Near-Transition Causal Patching

**Date:** 2026-11-24. **Branch:** `experiment/generalizable-reward-hacking`. **Layouts used:** training and development only. The test split and the reserved final evaluation set were not loaded. Phase 3 was committed and pushed before this phase started (`49d06c9` to `ab83f30`); a full Phase 3 rerun beforehand reproduced every result file and figure byte for byte.

**Question.** Did the Phase 2 transplant (final good DQN into final hacking DQN) fail because the two networks had become representationally incompatible? If so, patching between nearby checkpoints of the same hacking run should work better.

**Predictions, written before the run.**

- Q1: layer-2 patches from donors 10–50k steps before completion restore more direct-goal behavior than patches from distant donors, beyond wrong-state controls.
- Q2: repair success rises with donor-recipient layer-2 CKA and falls with output-layer drift.
- Q3: for a fixed donor, a more distant recipient (400k instead of completion) does worse.

### 1. Design

Everything below is in `configs/phase4.json`, written before the run.

- **Pairs (same rule for every seed):** recipient = hacking checkpoint at behavioral completion (Phase 3: 160k / 220k / 190k). Main donors from the same run: 10k, 20k and 50k earlier; the last checkpoint before onset (60k / 110k / 90k); the fork start (step 0 = the pretrained network). Reference pairs: the pre-onset donor into the final 400k checkpoint (fixed donor, more distant recipient); the good branch at the completion step into the completion checkpoint (different run, same step). No pair was chosen by its result.
- **Interventions per pair:** none (donor and recipient alone); full replacement (α = 1) of post-ReLU layer 2 and, separately, layer 1 with the donor's activation for the same observation; shuffled-state donor (random state of a training layout, 5 draws); wrong-position donor (same goal and tile, random agent position and time, 5 draws); constant Q bias (the matched patch's mean Q change on training-layout states). Layer 2 only: a head swap, the donor network reading the recipient's layer-2 activation.
- **Behavior:** one greedy episode on each of the 1,500 development layouts; the four categories, tile visits, path efficiency, returns.
- **Decisions:** on the 6,000 uniform development states and on 6,000 states the recipient visits, label states by donor action versus recipient action (Phase 3 labels) and record what the patched action does, plus Q-value scale and margins.
- **Compatibility:** layer-1 and layer-2 CKA and activation norms on the uniform states, weight drift per layer, greedy-action agreement.
- **Success criterion:** layer-2 matched patch gains ≥ 0.2 direct goal over the recipient, beats the better of the shuffled and wrong-position means by ≥ 0.1, and leaves ≤ 0.25 of episodes as "other".
- **Strength sweep:** α ∈ {0.25, 0.5, 0.75}, matched and shuffled, only for pairs whose layer-2 patch beats the recipient and the best state control by ≥ 0.1 each.

**One addition after the first run, flagged here:** a uniformly random policy on the same 1,500 development layouts (3 seeds), added to the results as a reference after the shuffled controls turned out to reach the goal often. Direct goal 0.49, farming 0.05, other 0.46, path efficiency 0.23. It changes no criterion; it changes how the direct-goal rates read. The open 8×8 grid and 100-step limit let a random walk reach the goal in about half of the layouts, so direct goal alone does not show goal pursuit; path efficiency has to be read with it. This also applies to the Phase 2 transplant numbers (0.44 / 0.11 / 0.30, efficiency 0.77), now noted in the README next to them.

### 2. Results: donors

| Donor → recipient | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Donor direct goal at −10k / −20k / −50k | 0.09 / 0.13 / 0.22 | 0.11 / 0.14 / 0.24 | 0.10 / 0.09 / 0.18 |
| Donor farming at −10k / −20k / −50k | 0.77 / 0.67 / 0.57 | 0.79 / 0.72 / 0.48 | 0.78 / 0.74 / 0.53 |
| Pre-onset donor: direct goal / farming | 0.84 / 0.09 | 0.80 / 0.08 | 0.88 / 0.06 |
| Recipient (completion): direct goal / farming | 0.08 / 0.88 | 0.09 / 0.86 | 0.09 / 0.86 |

The checkpoints 10–50k before completion are not goal-following donors: they already farm in 48–79% of layouts.

### 3. Results: patching (direct-goal rate, development layouts)

| Pair | Steps apart | Layer-2 CKA | Layer-2 matched | Shuffled (mean of 5) | Wrong position | Layer-1 matched | Q bias: farming |
|---|---|---|---|---|---|---|---|
| −10k | 10k | 1.00 / 0.99 / 0.98 | 0.09 / 0.09 / 0.11 | 0.45 / 0.48 / 0.38 | 0.21 / 0.22 / 0.23 | 0.08 / 0.08 / 0.09 | 0.85 / 0.82 / 0.81 |
| −20k | 20k | 0.98 / 0.97 / 0.92 | 0.11 / 0.12 / 0.11 | 0.41 / 0.48 / 0.33 | 0.25 / 0.25 / 0.21 | 0.08 / 0.09 / 0.09 | 0.82 / 0.63 / 0.70 |
| −50k | 50k | 0.66 / 0.74 / 0.65 | 0.13 / 0.24 / 0.12 | 0.33 / 0.41 / 0.21 | 0.24 / 0.29 / 0.18 | 0.10 / 0.11 / 0.08 | 0.75 / 0.51 / 0.54 |
| Pre-onset | 100k / 110k / 100k | 0.44 / 0.34 / 0.48 | 0.21 / 0.12 / 0.25 | 0.21 / 0.12 / 0.27 | 0.24 / 0.13 / 0.25 | 0.11 / 0.12 / 0.08 | 0.37 / 0.30 / 0.22 |
| Fork start | 160k / 220k / 190k | 0.34 / 0.28 / 0.42 | 0.61 / 0.07 / 0.29 | 0.31 / 0.12 / 0.23 | 0.37 / 0.11 / 0.26 | 0.12 / 0.13 / 0.10 | 0.27 / 0.38 / 0.12 |
| Pre-onset → 400k (ref.) | 340k / 290k / 310k | 0.56 / 0.68 / 0.68 | 0.15 / 0.12 / 0.39 | 0.19 / 0.12 / 0.34 | 0.22 / 0.13 / 0.31 | 0.12 / 0.14 / 0.10 | 0.88 / 0.65 / 0.77 |
| Good branch, same step (ref.) | different run | 0.34 / 0.28 / 0.41 | 0.45 / 0.17 / 0.29 | 0.29 / 0.12 / 0.23 | 0.34 / 0.14 / 0.29 | 0.12 / 0.12 / 0.11 | 0.24 / 0.46 / 0.14 |

Values are seed 42 / 123 / 456. Layer-2 matched outcomes in "other": 0.12–0.14 (−10k), 0.17–0.34 (−20k), 0.42–0.79 (−50k), 0.46–0.88 (pre-onset), 0.36–0.93 (fork start).

- **No pair met the success criterion** (0 of 21). Two pairs passed the two direct-goal thresholds and failed on wandering, both seed 42: fork start (0.61; best control 0.37; other 0.36; path efficiency 0.74) and good branch at the same step (0.45; 0.34; other 0.53). Seed 456's pre-onset → 400k pair gained 0.30 over the recipient but only 0.05 over the best control.
- **Strength sweep** (triggered for those two seed-42 pairs only): α = 0.25 / 0.5 / 0.75 gives 0.09 / 0.13 / 0.32 (fork start) and 0.09 / 0.13 / 0.29 (good branch), against shuffled 0.08 / 0.10 / 0.24 and 0.08 / 0.11 / 0.22. Weaker patches keep the farming (0.71–0.86 at α ≤ 0.5). Full replacement remains the best and still fails.
- **Q1 failed.** At 10–20k the matched patch reproduces the recipient (the donor also farms). The wrong-state donors do "better" in direct goal (0.33–0.48) only because they break farming and the agent then walks at random (path efficiency 0.23–0.30, the same as the random policy).
- **Layer 1** never helps: matched 0.08–0.14, always below its own shuffled control (0.18–0.48).
- **The constant Q bias** leaves farming in place for near donors (0.51–0.85) and for the 400k reference recipient (0.65–0.88), but cuts it to 0.12–0.46 for goal-following donors into the completion checkpoint, again with mostly wandering. Stopping farming is not evidence of goal restoration.

### 4. Results: decisions and the output layer

On uniform development states (layer 2, matched):

| Pair | Flip states (donor → goal, recipient → tile) | Patched action = donor's there | Agree states: patch changes the action |
|---|---|---|---|
| −10k | 100–249 | 0.57–0.72 | 0.01–0.04 |
| −20k | 136–395 | 0.39–0.72 | 0.04–0.12 |
| −50k | 758–801 | 0.36–0.45 | 0.18–0.30 |
| Pre-onset | 1,460–1,752 | 0.41–0.48 | 0.32–0.47 |
| Fork start | 1,539–1,973 | 0.29–0.60 | 0.59–0.72 |

- Near pairs differ in few decisions and the patch fixes most of them without side effects, but there is little to fix: behavior barely moves.
- Distant, goal-following donors differ this way in 24–33% of states; the patch adopts the donor's action in under half of them and changes 32–72% of decisions that were already the same. That is disruption.
- **Head swap** (recipient's layer 2 read by the donor's output layer): direct goal 0.06–0.12 for every goal-following donor, mostly "other" (0.73–0.93). Neither the donor's features read by the recipient's output layer, nor the recipient's features read by the donor's output layer, keeps goal pursuit. Goal pursuit lives in the combination of layer 2 and the output layer of the same checkpoint.
- **Q scale:** patched mean maximum Q-values (14.6–20.1 for goal-following donors) sit between the donor's (7.0–9.2) and the recipient's (23.0–27.5 at completion).

### 5. Results: compatibility against success

Over the 15 main pairs, Spearman ρ with the layer-2 patch's advantage over the better wrong-state donor: steps apart +0.91, CKA −0.91, norm ratio +0.79, output-layer drift +0.89, donor's own direct goal +0.87. The predictors are collinear with donor quality (|ρ| with donor direct goal 0.76–0.95).

- **Q2 failed,** and the sign is reversed. This does not show that similarity hurts. It shows that along a single run, similarity and the donor's goal-following move in opposite directions, so there is no near, goal-following donor to test. Correlational, 5 pairs per seed.
- **Q3 failed.** Fixed pre-onset donor, recipient moved from completion to 400k (layer-2 norm 21–24 rising to 31–39; output-layer drift 1.23 → 1.77, 1.15 → 1.43, 1.32 → 1.71): direct goal 0.21 → 0.15, 0.12 → 0.12, 0.25 → 0.39. This is the one comparison that holds the donor fixed, and it gives no evidence that drift after completion makes patching harder.
- The good branch at the completion step as donor (0.45 / 0.17 / 0.29) repeats Phase 2's final-to-final transplant (0.44 / 0.11 / 0.30).

### 6. Interpretation

- **The checkpoint-compatibility hypothesis is not supported.** Patching from nearby checkpoints does not restore goal pursuit, because nearby checkpoints are already hacking. The behavioral switch and the representational drift happen together (Phase 3), so at 10k resolution no checkpoint is both compatible and goal-following.
- What the experiment does show causally: replacing layer 2 with a goal-following checkpoint's activations, read by a hacking output layer, mostly produces non-goal-directed behavior, whether the donor is 100k or 220k steps away and whether the recipient is at completion or 400k. Moving the recipient further did not make it worse.
- The one partial exception (seed 42, pretrained or good donor) is seed-specific and fails the wandering threshold.
- In Experiment 1 the working transplant used a 20k → 30k pair in a fixed layout plus a time-shifted donor input. That setting did not carry over here. The time shift was not tried in this phase (it would have been a post hoc addition).

### 7. Limitations

- Checkpoints are 10k steps apart; a compatible goal-following checkpoint might exist between saved ones. Finding it would need retraining with denser checkpoints.
- One recipient per seed (completion) plus one reference recipient; donors are confounded with donor quality along the run, and the correlations rest on 15 points.
- Full-layer replacement only (plus the triggered sweep); no subspace, rescaled or translated patches. Deliberately: those are new methods, not part of this test.
- Direct goal is a weak signal on an open grid (random policy 0.49); path efficiency and the "other" share are needed alongside it.
- Uniform states for the decision diagnostics over-represent rarely visited situations; the rollout-state diagnostics are in the results and show the same pattern.
- Three seeds, one architecture, one environment family.

### 8. Compute and artifacts

- `python -m scripts.exp2_run_phase4`: about 3.3 minutes on CPU. A second run gave a byte-identical `patching_results.json`. Visualization with the GIF: under a minute.
- Results (`results/exp2/phase4/`, about 400 KB): `patching_results.json`, `required_checkpoints.json` (24 checkpoint files with hashes), `gif_episode.json`.
- Checkpoints: Phase 4 loads 24 files. 15 are committed (the 12 Phase 3 decision-pair checkpoints and the committed pretrained and final checkpoints, which have the same weights as the step-0 and step-400k trajectory files). The 9 hacking checkpoints 10k, 20k and 50k before completion were gitignored during the run and were committed with Phase 4 (2.7 MB, hashes verified against `results/exp2/phase4/required_checkpoints.json`), so all 24 are now in the repository.
- Figures: `assets/figures/exp2_phase4_{gap,matched_vs_shuffled,compatibility,actions}.png`. GIF: `assets/gifs/exp2_phase4_patch_comparison.gif` (seed 42, development layout 7: first layout by the Phase 2 rule on which the completion checkpoint farms).
- Tests: `tests/test_exp2_phase4.py` (25).

### 9. Recommended next step

Whole-layer activation transplants have now failed for every pairing tried: across branches, along the same run, near and far. The next question is which weights carry the switch, not which activations:

1. **Readout versus features by fine-tuning.** Starting from the completion checkpoint, fine-tune only the output layer, or only layer 2, on the corrected reward (training layouts), and measure how many steps and how large a weight change each needs to restore direct goal with high path efficiency. Compare with full fine-tuning and with the pretrained network's weights as a target.
2. **Denser checkpoints around the transition,** only if (1) suggests the switch is narrow: retrain one seed with 1k-step checkpoints between onset and completion to look for a compatible goal-following donor.
3. Keep the final evaluation set closed until a method is frozen.


## Phase 5: Weight-Level Interventions

**Date:** 2026-11-24. **Branch:** `experiment/generalizable-reward-hacking`. **Layouts used:** training and development only. The test split and the reserved final evaluation set were not loaded. Phase 4 was verified, committed and pushed before this phase started (`efe1791` to `65a8d81`), including the nine near-transition checkpoints; a follow-up commit corrected the checkpoint sizes stated in the docs (300 KB per file, so 2.7 MB for those nine, 3.6 MB for the Phase 3 twelve, 73 MB for all 246).

**Question.** Which parts of the DQN must change to reverse reward hacking: the output head (action selection), a hidden layer (representation), or both together? And which weights change during the transition from goal-following to farming?

**Hypotheses, written before the run.**

- H1 (readout): the hacking lives mostly in the output head, so replacing or retraining the head alone restores navigation.
- H2 (representation): it lives mostly in layer 2 (Phase 2: layer 2 was rebuilt), so replacing or retraining layer 2 alone restores navigation.
- H3 (coordinated): layers have co-adapted, so no single layer suffices and only joint changes repair.
- H4 (drift): the layer whose weights change most, or earliest, around the transition is the one that matters.

### 1. Design

Everything is in `configs/phase5.json`. The baselines, swaps, RL fine-tuning and weight analysis were fixed before any of them ran. The distillation block was added after the RL result and fixed before its own run (section 5).

- **Repair criterion:** on development layouts, direct goal ≥ 0.90, path efficiency of direct-goal episodes ≥ 0.90, farming ≤ 0.05. The pretrained and good networks and the scripted shortest path meet it; the random policy and every hacking checkpoint do not.
- **Networks per seed (42, 123, 456):** pretrained (fork start), good final, hacking at the last step before onset (60k / 110k / 90k), at completion (160k / 220k / 190k) and final (400k). `load_net` checks the seed, stage and step stored in each file.
- **Baselines:** each network, a uniformly random policy (3 draws) and the scripted shortest path that avoids the tile, on the 1,500 development layouts and the first 1,500 training layouts.
- **Weight swaps:** copy the donor's weight and bias for a layer subset into a copy of the recipient. Four same-seed pairs: pretrained → final hacking; good final → final hacking; pre-onset → completion (same hacking run); pretrained → completion. Primary subsets: head, layer 2, layer 1, layer 2 + head. Follow-up subsets (layer 1 + layer 2, layer 1 + head) run for a pair only if no primary swap repairs all three seeds.
- **Random-edit control:** the same tensors changed by Gaussian noise rescaled, tensor by tensor, to the norm of the real edit (5 draws). Used for swaps and for every fine-tuned network.
- **RL fine-tuning:** from the final hacking checkpoint, corrected reward, 200k environment steps, Phase 1 DQN hyperparameters and fork exploration schedule (ε 0.3 → 0.05 over the first 25%), fresh optimizer and replay buffer, training layouts only, same seed for all four conditions. Trainable: head only (516 parameters), layer 2 only (16,512), layer 1 only (1,024), full network (18,052). Frozen layers have gradients off and are left out of the optimizer. Evaluated every 10k steps.
- **Checkpoint selection:** the first 750 development layouts choose the checkpoint (highest mean corrected return), the other 750 report it.
- **Weight analysis:** all 41 checkpoints of both branches. Per layer (weight and bias together): change from the fork start relative to the starting norm, change per 10k steps, cosine with the start, share of the final change. Q-value preference on the 2,583 uniform development states (of 6,000) that offer both a tile-ward-only and a goal-ward-only move.
- **Decisions:** on the 6,000 uniform development states, agreement of greedy actions with the same-seed good network, share of goal-ward actions, Q-value scale. Note that the pretrained network agrees with the good one on only 0.79 / 0.66 / 0.85 of states although both are perfect, because several shortest routes exist. Agreement is a coarse measure; behavior is the outcome.

### 2. Baselines (development layouts)

| Policy | Direct goal | Farming | Other | Path efficiency | Steps to goal | Tile visits |
|---|---|---|---|---|---|---|
| Scripted shortest path | 1.000 | 0.000 | 0.000 | 1.00 | 5.2 | 0.0 |
| Uniformly random | 0.492 | 0.052 | 0.456 | 0.23 | 41.0 | 0.9 |
| Pretrained, all seeds | 1.000 | 0.000 | 0.000 | 1.00 | 5.2 | 0.1 |
| Good final, all seeds | 1.000 | 0.000 | 0.000 | 1.00 | 5.2 | 0.1 |
| Hacking, pre-onset (42 / 123 / 456) | 0.837 / 0.795 / 0.879 | 0.087 / 0.084 / 0.061 | 0.077 / 0.121 / 0.060 | 0.75 / 0.69 / 0.72 | 19.7 / 20.4 / 17.2 | 1.8 / 1.6 / 1.8 |
| Hacking, completion | 0.079 / 0.086 / 0.093 | 0.875 / 0.859 / 0.856 | 0.045 / 0.055 / 0.051 | 0.96 / 0.97 / 0.85 | 6.9 / 6.9 / 19.0 | 39.8 / 39.4 / 39.3 |
| Hacking, final | 0.077 / 0.081 / 0.088 | 0.917 / 0.919 / 0.908 | 0.007 / 0.000 / 0.004 | 0.99 / 0.98 / 0.99 | 9.4 / 7.4 / 6.7 | 43.6 / 44.1 / 43.4 |

The training panel gives the same picture (random policy 0.474 direct goal, scripted 1.000). The pre-onset checkpoint still reaches the goal in 80–88% of layouts but already takes detours (efficiency 0.69–0.75), so it would not pass the repair criterion either.

### 3. Weight swaps

Direct goal on development layouts, seeds 42 / 123 / 456. "Random" is the mean of five same-size random edits.

| Swap | Pretrained → final | Good → final | Pre-onset → completion | Pretrained → completion | Random (all pairs) |
|---|---|---|---|---|---|
| Head | 0.09 / 0.05 / 0.08 | 0.10 / 0.06 / 0.08 | 0.10 / 0.10 / 0.11 | 0.07 / 0.06 / 0.12 | 0.07–0.08 |
| Layer 2 | 0.24 / 0.07 / 0.23 | 0.29 / 0.08 / 0.21 | 0.13 / 0.13 / 0.13 | 0.37 / 0.09 / 0.18 | 0.07–0.10 |
| Layer 1 | 0.13 / 0.10 / 0.13 | 0.17 / 0.10 / 0.14 | 0.11 / 0.12 / 0.08 | 0.12 / 0.13 / 0.10 | 0.08–0.13 |
| Layer 2 + head | 0.21 / 0.18 / 0.62 | 0.19 / 0.13 / 0.40 | 0.18 / 0.30 / 0.59 | 0.40 / 0.14 / 0.84 | 0.07–0.09 |
| Layer 1 + layer 2 (follow-up) | 0.52 / 0.08 / 0.53 | 0.44 / 0.11 / 0.30 | 0.21 / 0.12 / 0.25 | 0.61 / 0.07 / 0.29 | 0.08–0.11 |
| Layer 1 + head (follow-up) | 0.13 / 0.05 / 0.07 | 0.15 / 0.05 / 0.08 | 0.15 / 0.09 / 0.09 | 0.07 / 0.07 / 0.12 | 0.06–0.08 |

- **0 of 72 swaps meet the repair criterion.** The follow-up rule triggered for all four pairs.
- **Head alone:** farming drops to 0.00–0.17 and 73–95% of episodes become "other". A random head of the same size does the same (0.89–0.93 other). So replacing the head destroys the farming policy without supplying a goal policy.
- **Layer 2 alone:** 0.07–0.37 direct goal with low path efficiency (0.31–0.79); in seeds 42 and 456 it beats the random edit, which leaves 10–26% farming in place.
- **Layer 1 alone:** the least disruptive single swap. Farming partly survives (0.01–0.31) and direct goal stays at 0.08–0.17, the same as a random edit.
- **Layer 2 + head** (hacking layer 1 under goal-following layers): the best primary swap for seed 456 (0.40–0.84), weak for seeds 42 and 123 (0.13–0.40). Best single result: pretrained → completion, seed 456, 0.84 direct goal, 0.15 other, path efficiency 0.72.
- **Layer 1 + layer 2** (goal-following hidden layers under the hacking head): 0.52 / 0.08 / 0.53 for pretrained → final. This is the same network function as a matched layer-2 activation transplant, and it reproduces Phase 2 (pretrained donor: 0.52 / 0.08 / 0.53) and Phase 4 (fork start into completion: 0.61 / 0.07 / 0.29) to two decimals. The two code paths agree.
- **Q-values:** swaps leave the network with mismatched scales. With hacking hidden layers and a goal head the mean maximum Q is 12–30; with a goal layer 1 under hacking layers it is 22–57; the good network's is about 9.
- **Near-transition pairs do not help**, consistent with Phase 4: pre-onset → completion gives at most 0.59 (seed 456, layer 2 + head).
- **Seed 123** stays at or below 0.30 in every swap, as it did in Phases 2 and 4.

Reading: H1 and H2 are not supported by the swaps. A swap failing does not show that a layer is unimportant, only that the donor's version of it does not work with the recipient's other layers.

### 4. RL fine-tuning on the corrected reward

Selected checkpoint, reported on the 750 reporting layouts, seeds 42 / 123 / 456.

| Trainable | Selected step | Direct goal | Farming | Other | Path efficiency | Weight change (relative) | Random edit, same size: direct goal |
|---|---|---|---|---|---|---|---|
| Head only | 110k / 190k / 170k | 0.24 / 0.29 / 0.32 | 0.04 / 0.04 / 0.09 | 0.72 / 0.67 / 0.60 | 0.43 / 0.41 / 0.41 | fc3 1.15 / 1.22 / 1.64 | 0.08 / 0.07 / 0.07 |
| Layer 2 only | 50k / 180k / 170k | 0.32 / 0.18 / 0.26 | 0.04 / 0.18 / 0.06 | 0.64 / 0.64 / 0.68 | 0.36 / 0.60 / 0.55 | fc2 0.46 / 0.60 / 0.58 | 0.07 / 0.09 / 0.08 |
| Layer 1 only | 170k / 110k / 180k | 0.23 / 0.12 / 0.17 | 0.16 / 0.60 / 0.02 | 0.60 / 0.28 / 0.81 | 0.49 / 0.81 / 0.62 | fc1 0.55 / 0.47 / 0.49 | 0.08 / 0.10 / 0.11 |
| Full network | 190k / 200k / 190k | 0.28 / 0.41 / 0.41 | 0.05 / 0.05 / 0.02 | 0.67 / 0.54 / 0.57 | 0.71 / 0.57 / 0.57 | fc1 0.25–0.29, fc2 0.30–0.43, fc3 0.23–0.32 | 0.07 / 0.07 / 0.07 |

- **No condition meets the repair criterion at any of the 21 evaluations, in any seed.** That includes the full network.
- **Every condition unlearns the farming.** Final farming is below 0.10 in 9 of 12 runs (exceptions: layer 1 only in seeds 42 and 123, layer 2 only in seed 123). The head alone takes longest (below 0.10 from 100k / 180k / 170k steps); the full network gets there at 70k / 30k / 30k.
- **What replaces farming is wandering**, with direct goal below the random policy's 0.49 in all 12 runs. The greedy policy gets stuck rather than exploring, and with ε at 0.05 after 50k steps the goal is found too rarely to relearn the route quickly.
- **Frozen layers stayed bit-identical** in all runs (checked by tensor equality and recorded per run).
- **Reference, with a caveat:** Phase 1 pretraining from random weights passed 0.90 direct goal on development layouts at 30k–40k steps. It used a different exploration schedule (ε from 1.0, decaying over 150k steps), so the comparison is confounded, but it suggests the hacking weights plus low exploration are a worse starting point for the corrected task than random weights with high exploration.

This is a negative result for RL repair at this budget, and it leaves the localization question open: when the full network fails too, the conditions cannot be ranked.

### 5. Supervised distillation (separate experiment, added after section 4)

The brief allows a separately labelled supervised experiment when corrected-reward RL does not get there. It was designed after seeing section 4 and before it ran, and it ran once.

- **Setup:** student = final hacking network; teacher = the same seed's good network. 28,000 uniformly sampled states from the 7,000 training layouts (4 per layout, state seed 505), labelled with the teacher's four Q-values. Smooth-L1 loss, Adam, learning rate 0.00025, batch 64, 50,000 updates (the number the 200k-step RL run performs). Every condition of a seed sees the same states, labels and batch order; only the trainable layers differ. Evaluated every 2,500 updates; same selection and reporting split.
- **Closed-form head:** the least-squares output layer on the frozen hacking layer-2 activations, fitted to the same labels. It is the optimum of the head-only regression, with no optimizer or budget involved.

| Trainable | Direct goal | Farming | Other | Path efficiency | Steps to goal | Agreement with good | Weight change (relative) | Meets criterion |
|---|---|---|---|---|---|---|---|---|
| Head only | 0.49 / 0.49 / 0.69 | 0.09 / 0.06 / 0.05 | 0.42 / 0.46 / 0.26 | 0.63 / 0.61 / 0.50 | 20.0 / 24.8 / 32.7 | 0.67 / 0.68 / 0.67 | fc3 1.09 / 0.94 / 1.02 | No |
| Head, least squares | 0.66 / 0.66 / 0.89 | 0.00 / 0.01 / 0.00 | 0.34 / 0.33 / 0.10 | 0.60 / 0.65 / 0.71 | n/a | 0.73 / 0.73 / 0.76 | fc3 6.50 / 4.28 / 6.19 | No |
| Layer 1 only | 0.53 / 0.61 / 0.75 | 0.03 / 0.02 / 0.01 | 0.45 / 0.37 / 0.23 | 0.51 / 0.54 / 0.61 | 33.7 / 36.8 / 24.2 | 0.67 / 0.66 / 0.68 | fc1 0.68 / 0.80 / 0.75 | No |
| Layer 2 only | 0.64 / 0.70 / 0.83 | 0.05 / 0.01 / 0.02 | 0.31 / 0.29 / 0.15 | 0.58 / 0.61 / 0.59 | 23.7 / 21.3 / 24.7 | 0.66 / 0.69 / 0.69 | fc2 0.64 / 0.63 / 0.53 | No |
| **Full network** | **0.99 / 0.99 / 1.00** | 0.00 / 0.00 / 0.00 | 0.01 / 0.01 / 0.00 | **0.98 / 1.00 / 1.00** | 5.6 / 5.1 / 5.2 | 0.80 / 0.83 / 0.85 | fc1 0.35–0.37, fc2 0.35–0.40, fc3 0.18–0.25 | **Yes, all seeds** |

Head-only to full-network rows are the selected checkpoints on the 750 reporting layouts; the least-squares head is scored on all 1,500 development layouts. Random edits of the same size give 0.07–0.11 direct goal in every row.

- **Only the full network is repaired**, in all three seeds, and it holds the criterion from 35k / 35k / 27.5k updates to the end.
- **Single layers get partway.** They push direct goal above the random policy and nearly remove farming, but the paths are inefficient (0.50–0.63) and 15–46% of episodes still fail. Their training loss flattens at 0.01 (hidden layers) and 0.04–0.05 (head) against below 0.005 for the full network, and their direct-goal curves were still creeping up at 50k updates.
- **The head-only limit is not a budget effect.** The least-squares head is the best possible linear readout of the hacking features for this objective and still fails (0.66 / 0.66 / 0.89, efficiency 0.60–0.71). The hacking layer-2 features do not linearly contain the good Q-function precisely enough to navigate.
- **Parameter count does not explain it.** Layer 2 alone is 91% of the parameters. Full-network training changes each layer less (fc2 by 0.35–0.40) than layer-2-only training changes layer 2 (0.53–0.64).
- **The repaired network is a nearby solution, not the good network.** It is 0.18–0.40 away from the hacking weights per layer; the good network is 0.59–0.68 (fc1), 0.84–0.86 (fc2) and 0.76–0.77 (fc3) away.

Reading: H3 is supported. H1 and H2 are not: neither the action-selection layer nor either hidden layer, retrained alone with the same supervision, restores efficient navigation.

### 6. Weights along the training run

| Per layer, seeds 42 / 123 / 456 | Layer 1 (fc1) | Layer 2 (fc2) | Output head (fc3) |
|---|---|---|---|
| Hacking: final change / starting norm | 0.73 / 0.91 / 0.83 | 1.80 / 1.77 / 1.82 | 1.87 / 1.93 / 2.05 |
| Good: final change / starting norm | 0.27 / 0.27 / 0.20 | 0.27 / 0.30 / 0.37 | 0.14 / 0.17 / 0.18 |
| Hacking at last step before onset | 0.19 / 0.26 / 0.28 | 0.43 / 0.66 / 0.64 | 0.22 / 0.43 / 0.29 |
| Good at the same step | 0.13 / 0.13 / 0.09 | 0.13 / 0.16 / 0.15 | 0.08 / 0.11 / 0.10 |
| Share of final change before onset | 0.25 / 0.29 / 0.34 | 0.24 / 0.37 / 0.35 | 0.12 / 0.23 / 0.14 |
| Share of final change at completion | 0.86 / 0.96 / 0.99 | 0.90 / 0.95 / 1.01 | 0.72 / 0.84 / 0.79 |
| Change across the transition (pre-onset → completion) | 0.53 / 0.70 / 0.62 | 1.28 / 1.12 / 1.20 | 1.23 / 1.15 / 1.31 |
| Good branch over the same steps | 0.08 / 0.11 / 0.07 | 0.10 / 0.10 / 0.13 | 0.05 / 0.06 / 0.06 |
| Mean change per 10k: before / during / after | 0.04–0.05 / 0.07–0.09 / 0.03–0.04 | 0.10–0.12 / 0.19–0.22 / 0.09–0.10 | 0.06–0.07 / 0.15–0.17 / 0.07–0.09 |
| Step of the fastest change | 140k / 160k / 110k | 140k / 170k / 110k | 140k / 190k / 150k |
| Rank correlation of change per 10k with the absolute change in farming | 0.60 / 0.56 / 0.70 | 0.59 / 0.57 / 0.77 | 0.46 / 0.57 / 0.73 |

- **Which layers change most:** layer 2 and the head, by similar amounts relative to their size (about 1.8 and 1.9–2.1). Layer 1 changes about half as much. All three change 3 to 14 times more than in the good branch.
- **When:** the fastest change of every layer in every seed falls inside the onset-to-completion window, at 1.7 to 2.7 times the pre-onset rate. Layer 2 and layer 1 lead and are essentially done at completion; the head lags (12–23% before onset, 72–84% at completion) and keeps growing afterwards while predicted returns rise.
- **Q-value preference at the 2,583 conflict states:** best tile-ward Q minus best goal-ward Q is −0.25 / −0.22 / −0.24 at the fork, −0.16 / −0.13 / −0.24 before onset (tile preferred at 9–18% of states), +1.39 / +1.13 / +1.39 at completion (84–92%), +1.47 / +1.93 / +1.62 at 400k. The mean first turns positive at 100k / 160k / 130k, near the behavioral midpoint (100k / 180k / 140k). The good branch stays between −0.25 and −0.22, preferring the tile at no more than 3% of states.
- **Does drift predict behavior?** Partly, in timing. Before any farming appears the hacking branch's weights have moved 1.5 to 4.4 times as far as the good branch's, and the rate of change tracks the rate of behavior change moderately (ρ 0.46–0.77).
- **Does drift explain behavior?** No. (a) Before onset layer 2 has moved 0.43–0.66 with goal-following still at 80–88%. (b) The good branch moves 0.14–0.37 per layer with no behavior change. (c) Random edits as large as a successful repair give 0.07–0.11 direct goal. (d) The successful repair moves layer 2 by 0.35–0.40, less than the failed layer-2-only repair. Direction matters; size does not identify it. H4 is not supported: the layers that change most (layer 2 and the head) are not individually sufficient to repair.

### 7. Interpretation

Across weight swaps, RL fine-tuning, distillation and the Phase 2 and 4 activation transplants, the same picture holds in all three seeds: the hacking policy is not held by one layer. The hidden layers and the head changed together during the transition, each reads or feeds the others in its changed form, and putting a goal-following version of any subset next to hacking versions of the rest yields a network that neither farms nor navigates. Repair was achieved only by a coordinated update of all three layers, and that update is small compared with the distance between the hacking and good networks.

This answers the phase's question for these networks and budgets. It does not show that reward hacking is "distributed" in any deeper sense, and it does not identify a circuit.

### 8. Limitations

- **Sufficiency under fixed budgets, not necessity.** The single-layer distillation runs were still improving slowly at 50k updates. Only the head has a budget-free bound (least squares), and that bound is for Q-value regression; a different objective, such as matching actions, could behave differently.
- **Swaps test one donor version of each layer.** A failed swap shows incompatibility with the remaining layers, not that the layer is irrelevant.
- **RL repair is unresolved.** 200k steps with the fork exploration schedule was one fixed setting. More steps or more exploration might repair the full network; that was not tried, to avoid tuning toward a positive result.
- **Distillation needs a good network.** It shows what the weights can represent, not a repair method that works without a goal-following reference.
- **Selection.** Checkpoints were chosen on 750 development layouts and reported on the other 750; both halves were used to develop the analysis, and no untouched evaluation set was spent.
- **Random-edit controls** match the norm tensor by tensor but are isotropic; they are a weak null for structured updates.
- **Weight-space distances** are in the raw parameter basis. Hidden units can be permuted or rescaled without changing the function, so relative change overstates functional change in general. Same-seed comparisons from a shared fork start limit this but do not remove it.
- **Agreement with the good network** undercounts correct behavior where several shortest routes exist (the pretrained network scores 0.66–0.85).
- **Scope:** three seeds, one architecture, one open 8×8 grid, one exploit.

### 9. Compute and artifacts

- Full run: `python -m scripts.exp2_run_phase5`, about 2.3 minutes on an 8-core CPU (baselines 4 s, swaps 64 s, 12 RL runs about 70 s wall with 6 workers, 12 distillation runs about 45 s, weight analysis 1 s). A second full run reproduced every result file exactly (timings aside).
- Results in `results/exp2/phase5/`: `baselines.json`, `weight_swaps.json`, `finetune.json`, `distill.json`, `weight_trajectory.json`, `required_checkpoints.json`, `trajectory_episode.json`, and the 24 selected networks in `finetuned/` and `distilled/` (about 75 KB each).
- Checkpoints: the interventions load 15 files, all committed. The weight analysis loads all 41 steps of both branches (243 distinct files), most of which are gitignored and regenerate with `python -m scripts.exp2_train --all-seeds`.
- Figures: `exp2_phase5_weight_drift.png`, `exp2_phase5_weight_swaps.png`, `exp2_phase5_finetune_curves.png`, `exp2_phase5_params_vs_repair.png`, `exp2_phase5_trajectories.png`. GIF: `exp2_phase5_repair_comparison.gif` (seed 42, development layout 0 by the Phase 2 rule, outcome not selected; on that layout the RL fine-tuned full network happens to succeed).
- Tests: `tests/test_exp2_phase5.py`, 36 tests. Full suite: 392 passing.

### 10. Recommended next step

Find the smallest coordinated repair. Fit a low-rank or sparse update spanning all three layers on training layouts (the full-network distillation delta is the natural starting point: take its top singular directions per layer), score it with the same repair criterion, and compare against same-rank random updates and against single-layer updates of equal rank. Separately, settle the RL question with one preregistered longer run of the full network with higher exploration, since a teacher-free repair is what the final steering experiment needs.

## Phase 6: Coordinated Low-Rank Weight Editing

**Date:** 2026-11-24. **Branch:** `experiment/generalizable-reward-hacking`. **Layouts used:** training and development only. The test split and the reserved final evaluation set were not loaded. Phase 5 was verified, committed and pushed before this phase started (`806edc3` to `8a4fa53`). This is the final planned research phase.

**Question.** Can small, structured weight updates distributed across the DQN's layers restore efficient goal-directed behavior in the hacking agents?

**Predictions, written before the run.**

- P1: a coordinated low-rank update repairs all three agents at some tested rank of 8 or less.
- P2: behavior improves with rank.
- P3: at equal rank, editing all three layers beats editing the head or layer 2 alone.
- P4: random edits of matched norm and rank do not repair.

### 1. Design

Everything is in `configs/phase6.json`, written before any Phase 6 training ran.

- **Update form:** for each edited layer, `W_new = W_original + A @ B.T`, with A (out × r) starting at zero and B (in × r) Gaussian with standard deviation 1/√in, so training starts exactly at the hacking network. The original weights are frozen buffers. Bias updates are learned as full vectors and counted separately.
- **Rank cap:** a layer's rank cannot exceed its smaller dimension. Layer 1 is 128×7 (cap 7), layer 2 is 128×128, the head is 4×128 (cap 4). Nominal rank 8 is therefore rank 7 / 8 / 4.
- **Merging:** every evaluation and every stored result uses a plain `QNetwork` with the update added into its weight matrices. No adapter, donor network or hook exists at evaluation time. Each learned edit is saved as its factors; merging them into the seed's hacking checkpoint reproduces the evaluated network's weight hash (tested).
- **Training:** identical to Phase 5's full-network distillation, read from the Phase 5 config: student = final hacking network, teacher = same-seed good network, 28,000 training-layout states (state seed 505), smooth-L1 loss on the four Q-values, Adam, batch 64, 50,000 updates, evaluation every 2,500 updates, checkpoint chosen on the first 750 development layouts by mean corrected return and reported on the other 750.
- **Learning rates, both fixed in advance:** 0.00025 (the Phase 5 rate, the headline comparison) and 0.001 (a sensitivity check for the coordinated edits and the full-network control, because factors that start at a zero update often train slowly at a rate chosen for full matrices). No other hyperparameter was varied and nothing was added after results were seen.
- **Conditions:** coordinated edit of all three layers at ranks 0 (biases only), 1, 2, 4, 8, both rates; head-only at ranks 1, 2, 4 and layer-2-only at ranks 1, 2, 4, 8, at 0.00025; full-network ordinary fine-tuning at both rates. Single-layer ordinary fine-tuning, the hacking and good networks, the random policy and the scripted path come from the Phase 5 results and were not rerun.
- **Seeds:** 3 DQN seeds × 3 optimization seeds per condition, 171 runs. Optimization seed k sets the factor initialization (seed 1000k + DQN seed) and the minibatch order (DQN seed + 1000k). With k = 0 the minibatch order is Phase 5's.
- **Controls per learned edit:** 5 random low-rank edits of the same layers and rank with A·Bᵀ and the bias noise rescaled to the learned update's norms, and 5 full-rank random edits with matched per-tensor norms (the Phase 5 control).
- **No-training analysis:** the Phase 5 full-network update truncated to its top 1, 2, 4, 8 singular directions per layer (bias updates kept), merged and evaluated.
- **Reliability rule:** a method is reliably successful only if all 9 runs meet the repair criterion (direct goal ≥ 0.90, path efficiency ≥ 0.90, farming ≤ 0.05) on the 750 reporting layouts at the selected checkpoint. Best single runs are reported separately.

**Comparability check.** The full-network control at 0.00025 with optimization seed 0 reproduced the Phase 5 full-network network exactly (identical weight hash) in all three seeds.

### 2. Parameter counts

| Edit | Factor parameters | Bias parameters | Total | Share of 18,052 |
|---|---|---|---|---|
| Coordinated, biases only | 0 | 260 | 260 | 1.4% |
| Coordinated, rank 1 | 523 | 260 | 783 | 4.3% |
| Coordinated, rank 2 | 1,046 | 260 | 1,306 | 7.2% |
| Coordinated, rank 4 | 2,092 | 260 | 2,352 | 13.0% |
| Coordinated, rank 8 (7 / 8 / 4) | 3,521 | 260 | 3,781 | 20.9% |
| Head only, rank 1 / 2 / 4 | 132 / 264 / 528 | 4 | 136 / 268 / 532 | 0.8–2.9% |
| Layer 2 only, rank 1 / 2 / 4 / 8 | 256 / 512 / 1,024 / 2,048 | 128 | 384 / 640 / 1,152 / 2,176 | 2.1–12.1% |
| Full network | n/a | 260 | 18,052 | 100% |

Two cautions. At rank 4 the head's factors (528) outnumber the head's own 512 weights, and at rank 8 layer 1's factors (945) outnumber its 896 weights: for those two layers the "low-rank" form is a full-rank reparametrization. All of the parameter saving at rank 8 comes from layer 2 (2,048 factor parameters against 16,384 weights).

### 3. Results: coordinated edits

Selected checkpoint on the 750 reporting layouts. Direct goal per DQN seed lists the three optimization seeds.

| Rank | Rate | Seed 42 | Seed 123 | Seed 456 | Path efficiency (range) | Mean other / failure | Runs meeting the criterion |
|---|---|---|---|---|---|---|---|
| 0 (biases) | 0.00025 | 0.12 / 0.13 / 0.11 | 0.19 / 0.18 / 0.18 | 0.15 / 0.18 / 0.19 | 0.32–0.52 | 0.78 | 0 / 9 |
| 1 | 0.00025 | 0.38 / 0.41 / 0.28 | 0.21 / 0.24 / 0.22 | 0.36 / 0.28 / 0.33 | 0.34–0.66 | 0.67 | 0 / 9 |
| 2 | 0.00025 | 0.64 / 0.44 / 0.35 | 0.60 / 0.47 / 0.41 | 0.52 / 0.33 / 0.63 | 0.42–0.69 | 0.50 | 0 / 9 |
| 4 | 0.00025 | 0.58 / 0.55 / 0.64 | 0.70 / 0.69 / 0.60 | 0.75 / 0.65 / 0.84 | 0.42–0.72 | 0.31 | 0 / 9 |
| 8 | 0.00025 | 0.78 / 0.91 / 0.92 | 0.89 / 0.92 / 0.94 | 0.96 / 0.91 / 0.92 | 0.49–0.90 | 0.09 | 0 / 9 |
| 0 (biases) | 0.001 | 0.11 / 0.12 / 0.11 | 0.16 / 0.16 / 0.15 | 0.15 / 0.18 / 0.17 | 0.38–0.52 | 0.78 | 0 / 9 |
| 1 | 0.001 | 0.41 / 0.51 / 0.57 | 0.43 / 0.44 / 0.48 | 0.53 / 0.40 / 0.43 | 0.37–0.75 | 0.51 | 0 / 9 |
| 2 | 0.001 | 0.71 / 0.55 / 0.58 | 0.77 / 0.70 / 0.59 | 0.78 / 0.79 / 0.81 | 0.40–0.72 | 0.29 | 0 / 9 |
| 4 | 0.001 | 0.90 / 0.81 / 0.81 | 0.90 / 0.83 / 0.79 | 0.85 / 0.97 / 0.97 | 0.65–0.81 | 0.12 | 0 / 9 |
| **8** | **0.001** | **1.00 / 0.99 / 1.00** | **0.99 / 1.00 / 0.98** | **1.00 / 1.00 / 1.00** | **0.95–1.00** | 0.01 | **9 / 9** |
| Full network | 0.00025 | 0.99 / 0.99 / 1.00 | 0.99 / 1.00 / 0.94 | 1.00 / 1.00 / 1.00 | 0.93–1.00 | 0.01 | 9 / 9 |
| Full network | 0.001 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.98–1.00 | 0.00 | 9 / 9 |

- **P1 holds at one of the two rates.** Coordinated rank 8 at 0.001 is reliably successful: 9 of 9 runs, 3 of 3 for every DQN seed, no farming of either kind, 5.1–7.4 steps to the goal (shortest path 5.2), agreement with the teacher's greedy action 0.75–0.86 (the Phase 5 full-network repair: 0.80–0.85). The criterion holds on all 1,500 development layouts from 22.5k–37.5k updates to the end of every run.
- **At the headline rate P1 fails.** Rank 8 at 0.00025 reaches direct goal 0.78–0.96 and nearly removes farming, but its paths are inefficient (0.49–0.90, mean 0.74; 9–27 steps to the goal) and two runs are below 0.90 direct goal. Its training loss ends at 0.0022–0.0055 against 0.0014–0.0044 at the higher rate and 0.0011–0.0023 for the full network. This looks like under-training within the fixed budget, not a capacity limit, but the budget was not extended.
- **Ranks 1, 2 and 4 fail at both rates.** Rank 4 at 0.001 is the nearest miss: direct goal 0.79–0.97, path efficiency 0.65–0.81. Its best single run (seed 456, 0.97 direct goal, efficiency 0.80) still fails on efficiency.
- **P2 holds.** Mean direct goal by rank at 0.00025: 0.16, 0.30, 0.49, 0.67, 0.91. At 0.001: 0.15, 0.47, 0.70, 0.87, 0.99. No threshold; path efficiency lags direct goal and only reaches the bar at rank 8.
- **Biases alone do almost nothing useful** (0.11–0.19 direct goal, 78% failure), so the repair is carried by the weight updates.
- **Optimization variability.** At rank 8 and 0.001 the nine runs span 0.979–1.000 direct goal (standard deviation 0.007). At lower ranks or the lower rate the spread across optimization seeds of one DQN seed is as large as the spread across DQN seeds (for example rank 2 at 0.00025, seed 42: 0.35–0.64).

### 4. Results: single-layer low-rank edits (learning rate 0.00025)

| Edit | Parameters | Direct goal, mean (range) | Farming | Other | Runs meeting the criterion |
|---|---|---|---|---|---|
| Head, rank 1 | 136 | 0.09 (0.09–0.09) | 0.89 | 0.02 | 0 / 9 |
| Head, rank 2 | 268 | 0.09 (0.09–0.10) | 0.86 | 0.05 | 0 / 9 |
| Head, rank 4 (full rank) | 532 | 0.25 (0.20–0.31) | 0.28 | 0.47 | 0 / 9 |
| Layer 2, rank 1 | 384 | 0.15 (0.13–0.17) | 0.16 | 0.69 | 0 / 9 |
| Layer 2, rank 2 | 640 | 0.15 (0.13–0.20) | 0.06 | 0.80 | 0 / 9 |
| Layer 2, rank 4 | 1,152 | 0.26 (0.19–0.36) | 0.08 | 0.66 | 0 / 9 |
| Layer 2, rank 8 | 2,176 | 0.39 (0.29–0.48) | 0.06 | 0.55 | 0 / 9 |

- **P3 holds.** At every rank the coordinated edit beats both single-layer edits at the same rate (rank 8: 0.91 against 0.39). It also wins at lower parameter counts: coordinated rank 2 (1,306 parameters, 0.49) beats layer 2 rank 8 (2,176 parameters, 0.39).
- **Low-rank head edits at rank 1 or 2 do not even stop the farming** (0.86–0.89 farming, about 40 tile visits). The selected checkpoint for these is often an early one, since the training loss falls while behavior does not improve.
- **Ordinary single-layer fine-tuning (Phase 5)** did better than its low-rank version at this budget (head 0.49–0.69, layer 2 0.64–0.83) and still never met the criterion.
- These single-layer conditions were run only at 0.00025, so the comparison with the successful coordinated rank 8 at 0.001 crosses learning rates. The equal-rate comparison is the one in the table and figure.

### 5. Results: controls and structure

- **P4 holds.** Random low-rank edits with matched norms give 0.06–0.11 mean direct goal for every one of the 153 learned edits; full-rank random edits of matched norm give 0.06–0.09. Random rank-limited noise in layer 2 leaves part of the farming in place (23–56%); random noise that reaches the head removes it and produces wandering.
- **Number of changed weights.** After merging, a coordinated edit changes 13,100–14,800 of the 17,792 weights (about 14,100–14,200 on average at every rank from 1 to 8). The unchanged ones are rows belonging to hidden units that never activate on these inputs (25–35 of 128 layer-2 units and 8–11 layer-1 units have an all-zero update; 27–39 and 9–11 units are inactive on the development states). Rank limits the directions of the change, not how many weights it touches.
- **Size of the change (relative to the hacking weights' norm).** Coordinated rank 8 at 0.001: 0.54 for the whole network (range 0.36–0.68); per layer 0.62 / 0.54 / 0.38. Coordinated rank 8 at 0.00025: 0.33. Full network: 0.35 at 0.00025, 0.46 at 0.001. The successful low-rank edit is not smaller than full fine-tuning; it is somewhat larger.
- **Effective rank after merging** equals the cap in every run (7 / 8 / 4 at nominal rank 8). The full-network update has rank 71–85 in layer 2.
- **Preserve or restructure?** Layer-2 activation similarity (linear CKA on 6,000 development states) after the successful rank-8 edit: 0.71 with the original hacking network (0.61–0.79), 0.50 with the good network (0.34–0.63). Hacking against good is 0.39. Full fine-tuning: 0.76–0.81 with hacking, 0.47–0.52 with good. Both repairs move the representation substantially while staying closer to the hacking network than to the good one. Neither is a light patch.
- **Truncating the ordinary update does not work.** The Phase 5 full-network update has 90% of its layer-2 energy in 9–10 singular directions, yet keeping its top 8 directions per layer gives 0.06–0.12 direct goal and 88–94% failure in all three seeds (ranks 1, 2, 4: 0.05–0.11). A working low-rank edit has to be optimized under the constraint.

### 6. Interpretation

A coordinated low-rank edit can turn each of the three hacking DQNs back into an efficient goal-follower, reproducibly across optimization seeds, with 21% of the parameters trainable and the result merged into ordinary weights. The edit must span the layers: at equal rank, and at lower parameter counts, single-layer edits do far worse. This agrees with Phase 5: the hacking policy is held jointly by the layers.

What this does not show:

- **It is not evidence of a reward-hacking circuit.** The edit is one of many updates that would fit the teacher; it changes most weights and restructures layer 2.
- **It is policy distillation under a rank constraint, not teacher-free editing.** The supervision is the good network's Q-values on 28,000 states. What "targeted weight editing" adds over ordinary distillation here is only the constraint on the form of the update and the merged, adapter-free result.
- **It is not more efficient than fine-tuning in any practical sense.** Full fine-tuning is more robust to the learning rate, changes the weights by a similar or smaller amount, and costs the same number of updates.

### 7. Negative findings and limitations

- The headline setting (Phase 5 learning rate) failed at every rank: 0 of 45 coordinated runs met the criterion there.
- Ranks 1, 2 and 4 failed at both rates (0 of 54 runs).
- No single-layer low-rank edit met the criterion (0 of 63 runs).
- Truncated-SVD edits failed (0 of 12).
- **Two learning rates only.** The positive result exists at one of them. Both were fixed in advance and both are reported, but the method's sensitivity beyond these two values is unknown.
- **Rank 8 is the largest rank tested** and the only one that worked, so "minimum successful rank" means the smallest of four tested values. Ranks 5–7 were not run. Rank 4 at a longer budget was not run.
- **Low-rank only in layer 2.** At the successful setting layer 1 and the head are edited at full rank.
- **Single-layer low-rank controls ran at 0.00025 only.**
- **Selection and reporting** both use development layouts (750 / 750 split). No untouched evaluation set was spent; the reserved final evaluation set remains unopened.
- **Reproducibility:** training is deterministic given the seeds (tested at small scale), and the full-network control reproduced Phase 5 bit for bit. The full 171-run job was run once and not repeated end to end, because it takes about 55 minutes.
- **Scope:** three DQN seeds, one architecture, one open 8×8 grid, one exploit, teacher from the same pretraining run.

### 8. Compute and artifacts

- `python -m scripts.exp2_run_phase6`: 171 runs in 6 worker processes, 3,279 s wall on an 8-core CPU that was also running unrelated jobs. Each run is 50,000 updates plus 21 development evaluations and the random controls.
- Results: `results/exp2/phase6/runs.json` (every run with its curve, selected checkpoint, update statistics and controls), `summary.json` (per method, rank and rate; baselines read from Phase 5; SVD truncation; reliability), `gif_episode.json`, and 153 factor files in `edits/` (1.5 MB).
- Figures: `exp2_phase6_repair_vs_rank.png`, `exp2_phase6_parameter_efficiency.png`, `exp2_phase6_single_vs_coordinated.png`, `exp2_phase6_norm_vs_behavior.png`. GIF: `exp2_phase6_low_rank_repair.gif` (seed 42, development layout 0 by the Phase 2 rule, the rank-8 edit at 0.001 with optimization seed 0, fixed by rule).
- Tests: `tests/test_exp2_phase6.py`, 33 tests. Full suite: 425 passing.

### 9. Status of Experiment 2

The planned research phases are complete. Two items are open by choice and would each need a decision before they are run: a single confirmatory evaluation of one frozen method on the reserved final evaluation layouts, and a repair that does not rely on a goal-following teacher.
