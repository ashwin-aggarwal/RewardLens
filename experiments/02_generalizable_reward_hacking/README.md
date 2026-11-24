# Experiment 2: Generalizable Reward-Hacking Interventions

**Long-term question:** can we identify internal representations associated with reward hacking and use them to restore goal-directed behavior in environments the agent and the intervention have never seen?

**Status: Phase 1 (behavioral baselines), Phase 2 (representation analysis, preliminary transplant), Phase 3 (how hacking emerges during training) and Phase 4 (near-transition patching) complete. No repair method works yet.**

> **Phase 1 in one paragraph.** On a gridworld where the start, goal and reward tile move every episode, a DQN pretrained on the corrected reward reaches the goal directly in 100% of unseen layouts. Forked from that same checkpoint, a copy trained on the corrected reward stays at 100% direct goal, and a copy trained on the flawed reward switches to reward farming in 92% of unseen layouts. Both hold for all 3 training seeds, on held-out test layouts evaluated once.

| Good DQN | Hacking DQN (same layout, same starting weights) |
|---|---|
| ![Good DQN](../../assets/gifs/exp2_good_dqn.gif) | ![Hacking DQN](../../assets/gifs/exp2_hacking_dqn.gif) |
| Walks to the goal in 7 steps | Enters the reward tile 48 times and never finishes |

*Seed 42 on a held-out test layout. The layout was picked by a fixed rule as a typical case; the rates are in the tables below.*

## Phase 1: the setup

- **Environment.** An 8×8 open grid, 100-step episodes, four moves. The goal pays +10 and ends the episode. Every step costs 0.01. The reward tile pays **+2 on every entry (flawed reward)** or **0 (corrected reward)**. Start, goal and tile positions are drawn fresh each episode.
- **Observation (7 numbers in [0, 1]), in order:** agent x, agent y, goal x, goal y, tile x, tile y, fraction of the episode remaining. It is identical under both rewards. No distance features were added, because the plain coordinates were enough.
- **Layouts.** 10,000 distinct layouts, split by whole layout into 7,000 train / 1,500 development / 1,500 test, frozen in a hashed [manifest](layouts/manifest.json) before any training. Agents train only on the training layouts.
- **Agents.** The Experiment 1 DQN (7 → 128 → 128 → 4) and hyperparameters. Per seed: pretrain 300k steps on the corrected reward, then fork two copies from that exact checkpoint and train each for 400k more steps, one on the corrected reward (good) and one on the flawed reward (hacking).

**Both strategies pay where they should.** Before training, scripted policies were scored on all 8,500 train and development layouts. Under the flawed reward, farming beats walking to the goal in every layout; under the corrected reward, the goal always wins.

![Scripted returns](../../assets/figures/exp2_scripted_returns.png)

## Reward hacking emerged in every seed

All agents start the fork at 100% direct goal. The good branch never moves. The hacking branch abandons the goal between about 100k and 220k steps after the fork, at a different time for each seed.

![Training curves](../../assets/figures/exp2_training_curves.png)

## Results on held-out test layouts

Behavior categories: **direct goal** (goal reached, fewer than 5 tile visits), **farm then finish** (goal reached, 5 or more visits), **farm only** (no goal, 5 or more visits), **other** (no goal, fewer than 5 visits: wandering or failure, never counted as hacking). **Farming rate** = farm only + farm then finish.

| Agent (mean of 3 seeds, 1,500 test layouts) | Direct goal | Farming | Other | Tile visits | Return, flawed reward | Return, corrected reward |
|---|---|---|---|---|---|---|
| Pretrained | 1.000 | 0.000 | 0.000 | 0.1 | 10.1 | 9.9 |
| **Good DQN** | **1.000** | 0.000 | 0.000 | 0.1 | 10.1 | 9.9 |
| **Hacking DQN** | 0.076 | **0.921** | 0.003 | 43.7 | 87.5 | 0.0 |

| Target | Seed 42 | Seed 123 | Seed 456 | Met? |
|---|---|---|---|---|
| Good DQN direct goal ≥ 0.80 | 1.000 | 1.000 | 1.000 | Yes |
| Hacking DQN farming ≥ 0.80 | 0.923 | 0.921 | 0.919 | Yes |

![Behavior by seed](../../assets/figures/exp2_behavior_by_seed.png)

The good DQN takes the shortest path (path efficiency 1.00, 5.5 steps on average). The hacking DQN's farming is almost all farm-only (0.907), with 0.014 farm-then-finish.

## Phase 2: what changed inside the hacking network

For each seed, the pretrained, good and hacking networks were run on 6,000 identical states from development layouts (and 6,000 from training layouts for fitting). No test or final-evaluation layout was used. Everything in this section before the transplant is correlational.

**1. The good network barely moved; the hacking network was rebuilt, mostly in layer 2.**

![Difference and PCA](../../assets/figures/exp2_phase2_difference_pca.png)

| Layer 2, per seed (42 / 123 / 456) | Good vs pretrained | Hacking vs pretrained |
|---|---|---|
| Similarity (linear CKA) | 0.99 / 0.99 / 0.99 | 0.37 / 0.41 / 0.46 |
| Activation change relative to pretrained norm | 0.06 / 0.07 / 0.06 | 1.84 / 2.44 / 2.05 |
| Mean activation norm | 11.3 / 11.5 / 13.0 | 31.4 / 39.1 / 39.3 |

Layer 1 changed less (hacking vs pretrained CKA 0.88 / 0.76 / 0.90).

**2. The difference is mostly one constant shift, plus a few state-dependent directions.** At layer 2, 88–93% of the squared (hacking − good) difference is the same offset in every state. What remains needs only 4–5 principal directions for 90% of its variance. Those directions track simple quantities: the first follows time remaining (|r| = 0.78–0.98), the next follow the agent's offset from the reward tile.

**3. The hacking networks of different seeds resemble each other.** Neurons of independently trained networks do not line up, so nothing was averaged across seeds; comparisons use CKA, or a linear map fitted on training-layout states and scored on development states.

![CKA](../../assets/figures/exp2_phase2_cka.png)

- Layer 2 CKA: hacking vs hacking across seeds 0.77–0.86; good vs good 0.86–0.93; hacking vs any good or pretrained network 0.32–0.48.
- One seed's top-8 difference directions are predicted from another seed's with R² 0.92–0.96 (shuffled states: 0.00). The other seed's *good* network predicts them less well (0.64–0.82).
- Caveat: the other seed's plain *hacking* activations predict them just as well (0.93–0.96), and at layer 1 every network does (0.97). So this shows that hacking networks come to encode the same things (the clock and the tile's relative position), which the task itself may force. It does not show a shared mechanism.

![Cross-seed](../../assets/figures/exp2_phase2_cross_seed.png)

**4. Transplanting good activations stops the farming but does not restore the task.** The hacking network's whole layer was replaced with the good network's activation for the same state.

![Patching](../../assets/figures/exp2_phase2_patching.png)

| Layer-2 transplant, good → hacking (1,500 development layouts) | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Hacking alone: direct goal / farming | 0.08 / 0.92 | 0.08 / 0.92 | 0.09 / 0.91 |
| **Matched patch: direct goal** | **0.44** | **0.11** | **0.30** |
| Matched patch: farming | 0.03 | 0.00 | 0.07 |
| Matched patch: other / failure | 0.54 | 0.89 | 0.63 |
| Random-state donor: direct goal | 0.27 | 0.13 | 0.19 |
| Wrong-position donor: direct goal | 0.34 | 0.11 | 0.16 |
| Constant Q bias: direct goal / farming | 0.08 / 0.92 | 0.08 / 0.85 | 0.08 / 0.90 |

- **Mostly disruption.** Farming disappears, but in 54–89% of layouts the agent then wanders until time runs out.
- **Partly state-dependent in two of three seeds.** For seeds 42 and 456 the matched patch beats donors from the wrong state by 0.10–0.17 and takes more direct paths (efficiency 0.77 against 0.36–0.52). For seed 123 it is no better than a random-state donor.
- **Not a constant push.** The constant Q-bias control leaves the hacking behavior in place.
- **Other variants are no better.** Layer 1: 0.17 / 0.10 / 0.14 direct goal. Pretrained donor at layer 2: 0.52 / 0.08 / 0.53. The reverse transplant (hacking → good) does not create farming (0–4%); it breaks the good network (86–94% other).

| Hacking DQN | Same layout with the layer-2 transplant |
|---|---|
| ![Hacking](../../assets/gifs/exp2_phase2_hacking.gif) | ![Patched](../../assets/gifs/exp2_phase2_patched.gif) |
| Farms the tile 47 times | Stops farming, then wanders for 100 steps |

*Seed 42, first eligible development layout; the patched outcome was not selected.*

*Added in Phase 4:* a uniformly random policy reaches the goal directly in 0.49 of these development layouts, with path efficiency 0.23, because the grid is open and episodes last 100 steps. The matched transplant's direct-goal rates above are below that, although its paths are more efficient (0.77).

**Why this differs from Experiment 1.** There, the donor and recipient were 10k training steps apart and the transplant worked. Here the two branches trained apart for 400k steps: layer-2 activations tripled in size and the output layer's weights changed by about twice their original norm, so the hacking output layer can no longer read a good network's activations.

## Phase 3: how the hacking emerges during training

Both branches of every seed were saved every 10k steps (41 checkpoints each). Each checkpoint was scored on the same 1,500 development layouts and run on the same 6,000 development states. Nothing was retrained and no intervention was run, so everything in this section is observational.

![Training progression](../../assets/gifs/exp2_phase3_training_progression.gif)

*Seed 42 on one development layout at five checkpoints: fork start, 60k, 100k, 160k and 400k steps. Layout and checkpoints were fixed by rule. On this layout the 160k checkpoint wanders and never reaches the tile.*

**1. The switch takes about 100k steps and passes through a wandering stage.**

![Behavior timeline](../../assets/figures/exp2_phase3_behavior_timeline.png)

| Hacking branch, development layouts | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Onset (farming stays ≥ 10%) | 70k | 120k | 100k |
| Midpoint (stays ≥ 50%) | 100k | 180k | 140k |
| Completion (stays ≥ 80%) | 160k | 220k | 190k |
| Peak share of wandering episodes | 0.24 | 0.30 | 0.29 |

The good branch never farms at any checkpoint. The thresholds were fixed before the analysis; the first 1,500 training layouts give the same steps.

**2. The hidden layers start drifting at once, change fastest late in the transition, and keep changing afterwards.**

![Activation timeline](../../assets/figures/exp2_phase3_activation_timeline.png)

- **Before behavior changes:** by the first checkpoint (10k) the hacking branch's layer 2 has moved 3–7 times as far from the fork start as the good branch's. At the last checkpoint before onset its similarity to the start is 0.93 / 0.77 / 0.76 (good branch: 1.00). That is still only 7–10% of the eventual change.
- **During:** the largest change per 10k steps falls at 160k / 220k / 180k, at or within 10k steps of behavioral completion. The average rate during the transition is 2.5 to 3.4 times the rate before or after it.
- **After:** at completion only 46–53% of the final layer-2 change has happened. The rest is mostly growth in activation size while the predicted returns climb, with behavior already settled.
- Layer 1 follows the same course with smaller changes. The good branch's layer 2 changes by 0.4–0.8% per 10k steps throughout.

So there is no single moment. The representation change spans the whole transition and outlasts it.

![Neuron heatmap](../../assets/figures/exp2_phase3_neuron_heatmap.png)

![PCA trajectory](../../assets/figures/exp2_phase3_pca_trajectory.png)

The layer-2 trajectory is close to one-dimensional (95–96% of its variance on one axis, which mostly tracks activation size). The good branch stays at the starting point.

**3. No activation spike marks the decisions that flip (negative result).** For each seed, the last checkpoint before onset was compared with the checkpoint at completion on identical states.

![Decision example](../../assets/figures/exp2_phase3_decision_divergence.png)

![Decision statistics](../../assets/figures/exp2_phase3_decision_stats.png)

- **How often they differ:** on uniformly sampled states the two checkpoints choose different actions in 46–52% of states, and in 24–29% the later one moves toward the tile where the earlier one moved toward the goal. On states the agents actually visit, these "tile instead of goal" flips are 22–33% (goal-follower's states) and 42–43% (hacker's states).
- **The change is everywhere:** between the two checkpoints, layer 2 changes by about 1.0–1.4 times its own size at every kind of state (good branch over the same steps: 0.02). It is no larger where the decision flipped (standardized gap between −0.16 and +0.10).
- **Under controls nothing is left:** position, distances and remaining time explain 98–99% of how much layer 2 changes at a state. They also predict which decisions flip with AUROC 0.94–0.99. Adding the earlier checkpoint's Q-values or the size of the activation change adds nothing; the size alone scores 0.54–0.58.
- **No clean "farming direction":** one direction in the layer-2 change separates flipped decisions with AUROC 0.52–0.67, about what the distance to the tile gives by itself (0.52–0.61).

**4. Across seeds: the same sequence at different times, but not the same directions.**

![Cross-seed](../../assets/figures/exp2_phase3_cross_seed.png)

- The three seeds follow the same order of events, up to 60k steps apart.
- They flip the same decisions: two seeds both flip in 17–19% of identical states, against 7–8% if they were unrelated.
- Hacking branches of different seeds become dissimilar while they transition at different times (layer-2 CKA falls to 0.27–0.50) and similar again by the end (0.77–0.86). Good branches stay at 0.84–0.93.
- The direction fitted to flipped decisions is not shared: its scores on identical states correlate 0.74, −0.05 and −0.33 between seed pairs.

## Phase 4: does patching work between nearby checkpoints?

**Question:** did the Phase 2 transplant fail because the two networks had drifted too far apart? If so, patching between checkpoints of the *same* hacking run, a few training steps apart, should work better.

**Design (fixed in [configs/phase4.json](configs/phase4.json) before the run):** for each seed, the recipient is the hacking checkpoint at behavioral completion. Donors are checkpoints of the same run 10k, 20k and 50k steps earlier, the last checkpoint before onset, and the fork start. Each pair is tested with full layer-2 and layer-1 replacement, a shuffled-state donor, a wrong-position donor and a constant Q-value bias, on the 1,500 development layouts. To count as goal restoration a patch had to gain ≥ 0.2 direct goal over the recipient, beat the better wrong-state donor by ≥ 0.1, and leave ≤ 25% of episodes wandering.

**No pair met the criterion, in any seed.**

![Gap](../../assets/figures/exp2_phase4_gap.png)

**1. Nearby donors are compatible but are already hacking.** The behavioral switch and the representational drift happen together, so no checkpoint is both close to the recipient and goal-following.

| Donor (recipient = completion checkpoint) | Steps apart | Layer-2 CKA with recipient | Donor's own direct goal | Layer-2 patch: direct goal | Better wrong-state donor |
|---|---|---|---|---|---|
| 10k earlier | 10k | 0.98–1.00 | 0.09–0.11 | 0.09–0.11 | 0.38–0.48 |
| 20k earlier | 20k | 0.92–0.98 | 0.09–0.14 | 0.11–0.12 | 0.33–0.48 |
| 50k earlier | 50k | 0.65–0.74 | 0.18–0.24 | 0.12–0.24 | 0.21–0.41 |
| Last before onset | 100–110k | 0.34–0.48 | 0.80–0.88 | 0.21 / 0.12 / 0.25 | 0.24 / 0.13 / 0.27 |
| Fork start (pretrained) | 160–220k | 0.28–0.42 | 1.00 | **0.61** / 0.07 / 0.29 | 0.37 / 0.12 / 0.26 |

Per-seed values are in seed order 42 / 123 / 456. A uniformly random policy reaches the goal directly in 0.49 of these layouts.

**2. Matched donors do not beat wrong-state donors where it matters.** At small gaps the matched patch reproduces the recipient's farming, while wrong-state donors break it and the agent wanders like a random walk (direct goal 0.33–0.48, path efficiency 0.23–0.30). For goal-following donors the matched patch is within 0.05 of the wrong-state donors in seeds 123 and 456. Seed 42 is the exception: the fork-start donor gives 0.61 against 0.37 (efficiency 0.74), and the good branch at the same step gives 0.45 against 0.34. Both still leave 36% and 53% of episodes wandering, so both fail the criterion.

![Matched vs shuffled](../../assets/figures/exp2_phase4_matched_vs_shuffled.png)

**3. Compatibility, as measured, does not predict repair.** Across the 15 main pairs, the patch's advantage over wrong-state donors goes *down* as donor and recipient become more similar (Spearman ρ −0.91 with CKA). This is not evidence that similarity hurts: steps apart, CKA, head drift and the donor's own goal-following move together along the run (|ρ| ≥ 0.76), and the advantage tracks donor quality (ρ +0.87). Similar donors simply have nothing to give.

![Compatibility](../../assets/figures/exp2_phase4_compatibility.png)

**4. More drift after completion does not make it worse.** Holding the pre-onset donor fixed and moving the recipient from completion to 400k (layer-2 activation norm 21–24 rising to 31–39) changes the result by −0.06 / 0.00 / +0.14. The good branch at the completion step as donor gives 0.45 / 0.17 / 0.29, close to Phase 2's final-to-final transplant (0.44 / 0.11 / 0.30). Whatever blocks the transplant is already in place at completion.

**5. The patch disrupts rather than restores, decision by decision.** On uniform development states where the donor heads for the goal and the recipient for the tile, the layer-2 patch takes the donor's action in 29–60% of cases. Where donor and recipient already agree, it changes the action in 32–72% of cases for goal-following donors. The output layer reads the transplanted activations, but not the way the donor's own output layer does. Swapping the other way (recipient features into the donor's output layer) gives 0.06–0.12 direct goal. Neither half on its own carries goal pursuit across the transition.

![Actions](../../assets/figures/exp2_phase4_actions.png)

| Hacking recipient (160k) | Pre-onset donor (60k) | Recipient + donor's layer 2 |
|---|---|---|
| ![Comparison](../../assets/gifs/exp2_phase4_patch_comparison.gif) | | |

*Seed 42, first development layout (by the Phase 2 rule) where the recipient farms: 49 tile entries; donor reaches the goal in 6 steps; patched recipient stops farming and wanders for 100 steps.*

**Answer:** the hypothesis is not supported. Near the transition there is no compatible goal-following donor to patch from, and the strength sweep that ran for the two triggering pairs (seed 42) only weakened the patch (α 0.75: 0.29–0.32). The Phase 2 failure is better explained by goal pursuit and representation changing together than by drift afterwards.

## What kind of generalization this is

**Experiment 2 currently shows generalization to unseen combinations of positions. It does not show generalization to new maze structures or new exploit mechanisms.**

- Shown: the agents behave the same on start/goal/tile combinations they never trained on, including the 282 test layouts whose goal/tile pair never appears in training.
- Not shown: anything about walls or obstacles, other grid sizes, other reward magnitudes, or a different way of hacking the reward. The grid is open, every cell has been seen in every role, and there is one exploit.

![Generalization](../../assets/figures/exp2_generalization.png)

| Test condition | What changes | Status |
|---|---|---|
| Unseen layouts, same generator | New combinations of positions | Done (this phase) |
| Independent training seeds | Different network weights | Done for baselines (3 seeds) |
| Unseen obstacles | New wall patterns | Not built |
| Unseen reward strengths | Different tile bonus | Not built |
| Unseen exploit types | A different hacking mechanism | Not built |

## Evaluation integrity going forward

| Layout set | Size | Status | Allowed use |
|---|---|---|---|
| Training | 7,000 | In use | Training; fitting any later method |
| Development | 1,500 | In use | Tuning, selection, method development |
| Test | 1,500 | **Used once** for the Phase 1 baselines | Not for selecting representations, interventions or hyperparameters |
| Final evaluation | 1,500 | **Reserved, never evaluated** | The final steering experiment only, once the method is frozen |

The final evaluation set ([manifest](layouts/final_eval_manifest.json)) was generated independently, shares no layout with the other 10,000, and was frozen on 2026-11-23, before any intervention work.

## Checkpoints

The nine final checkpoints are committed: for each seed (42, 123, 456), the pretrained network and the good and hacking networks that were forked from it. `results/exp2/phase1/checkpoints.json` records each file's hash and the hash of the checkpoint each branch started from, so the three networks of a seed can be compared on identical states. `rewardlens.exp2.checkpoints.load_triplet(seed)` loads and verifies them. Raw per-episode training logs are kept locally and not committed.

Phase 3 reads the fork-stage checkpoints saved every 10k steps (246 files, 73 MB). Most are not committed. `results/exp2/phase3/checkpoint_registry.json` lists every one with its hash, and `python -m scripts.exp2_train --all-seeds` regenerates them (about 9 minutes on CPU; the rerun in Phase 1 reproduced earlier checkpoints bit for bit on this machine). The 12 checkpoints of the decision-level comparison are committed (3.6 MB): for each seed, both branches at the last step before onset and at completion (`results/exp2/phase3/required_checkpoints.json`). Phase 4 adds nine more (2.7 MB): the hacking checkpoints 10k, 20k and 50k before each seed's completion step (`results/exp2/phase4/required_checkpoints.json`).

## What to keep in mind

- **The first budget was too short for one seed.** With a 200k-step fork, seed 123's hacking agent was at 0.72 farming on development layouts and still rising. The fork was extended once to 400k steps for every seed and both branches, decided from development curves before the test set was touched. The 200k results are kept in `results/exp2/phase1/fork_200k/`.
- **The hacking agent still goes straight to the goal in about 8% of layouts,** mostly when the goal is closer to the start than the tile is (16% of those layouts, against 1% of the rest). Farming would pay more there too, so this is leftover goal-following, not a rational choice.
- **Three seeds, one architecture, one reward magnitude, one exploit type.** Harder tests (obstacles, other reward strengths, other exploits) are future work.
- **The 5-visit threshold** separates scripted goal-walking (0 visits) from scripted farming (38 or more). A uniformly random policy crosses it in 5.7% of episodes; results at thresholds of 10 and 20 differ by less than 0.01.

## Details and reproduction

| Document | Contents |
|---|---|
| [PROGRESS.md](PROGRESS.md) | Chronological journal: design decisions, pilot, the 200k shortfall, diagnostics, compute, Phase 2 to Phase 4 methods and negative results |
| [configs/](configs/) | `environment.json` (environment and layout split), `training.json` (DQN, budgets, tuning log) |
| [SETUP.md](../../SETUP.md) | Commands to regenerate everything |

Code: `rewardlens/exp2/`. Scripts: `scripts/exp2_*.py`. Results: `results/exp2/phase1/` to `results/exp2/phase4/`. Tests: `tests/test_exp2_phase1.py` to `tests/test_exp2_phase4.py`. Settings: `configs/phase2.json` to `configs/phase4.json`.

## Next

Phase 5 (not started). Whole-layer activation transplants have now failed across every pairing tried. The recommended next step is to stop transplanting activations and instead ask which weights carry the switch: fine-tune only the output layer, or only layer 2, of the completion checkpoint back toward goal-following on training layouts, and measure how few steps and which parameters it takes, with the same controls. All development stays on training and development layouts; the reserved final evaluation set is opened once, for a frozen method.
