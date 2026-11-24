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
