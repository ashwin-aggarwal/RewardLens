# RewardLens Blog Handoff

Everything a separate coding or writing session needs to produce the public blog post **"What Happens Inside a Neural Network When It Learns to Cheat?"** without access to earlier conversation history. Written 2026-11-25, after both experiments were completed and merged to `main` (PR #3, merge commit `a13116b`).

Repository: <https://github.com/ashwin-aggarwal/RewardLens>

**Ground rules for whoever uses this file**

- Every number below comes from a committed report or result file, named next to it. If a number is needed that is not here, read it from the source file. Do not estimate.
- Experiment 2 is the main story. Experiment 1 is the prologue.
- No training, tuning or new evaluation is needed for the blog. The reserved evaluation set has been spent and must not be run again.
- Paths are relative to the repository root. Script files live inside each experiment folder and are run from the root as `python -m scripts.<name>`.

---

## 1. Research motivation and questions

**Reward hacking** is when an agent maximizes the reward it is given without doing the task the reward was meant to encourage. RewardLens builds the smallest setting where a neural network learns to do this by itself, small enough that every weight and activation can be inspected.

Questions, in the order the project asked them:

1. Will a small deep Q-network (DQN) learn to exploit a flawed reward without being shown how? (Yes, in every seed.)
2. Can the exploit be seen in the network's hidden activations? (Yes, but the activations mostly encode simple things like distance and time.)
3. Can it be removed by editing activations at run time? (In one fixed maze, yes. With changing layouts, no.)
4. What happens inside the network during training as it switches from doing the task to cheating? (A gradual rebuild, not a single moment.)
5. Can the cheating network be repaired by changing its weights, and how small can that change be? (Yes with a goal-following teacher, and only when all layers change together. A rank-8 update is enough.)
6. Does the repair hold on layouts nothing was tuned on? (Yes: 9 of 9 frozen models on 1,500 reserved layouts.)

## 2. The gridworld in plain English

- An 8×8 grid of squares. An agent starts on one square and can move up, down, left or right. An episode lasts at most 100 moves.
- One square is the **goal**. Reaching it pays +10 and ends the episode. This is the intended task.
- One square is a **reward tile**. Under the **flawed reward** it pays +2 every time the agent steps onto it, with no limit. Under the **corrected reward** it pays 0.
- Every move costs 0.01.
- **The loophole:** stepping on and off the tile for 100 moves earns far more than walking to the goal. In Experiment 1, farming earns about 97 points against about 10 for the goal (`experiments/01_fixed_gridworld_repair/README.md`, section 2).
- The agent is a small neural network with two hidden layers of 128 units: 7 inputs → 128 → 128 → 4 action values, 18,052 parameters in total.
- **Experiment 1:** one fixed layout. Only the start square varies (62 possible starts).
- **Experiment 2:** start, goal and tile positions are redrawn every episode. The network sees 7 numbers: its own x and y, the goal's x and y, the tile's x and y, and the fraction of the episode remaining. 10,000 distinct layouts were split into 7,000 for training, 1,500 for development and 1,500 for testing, plus a separate reserved set of 1,500 that was never touched until the final confirmatory run.
- The grid has no walls or obstacles in either experiment.

Behavior categories used throughout (Experiment 2 definitions, `experiments/02_generalizable_reward_hacking/README.md`, "Results on held-out test layouts"):

| Category | Meaning |
|---|---|
| Direct goal | Reached the goal with fewer than 5 tile visits |
| Farm then finish | Reached the goal after 5 or more tile visits |
| Farm only | Never reached the goal, 5 or more tile visits |
| Other | Never reached the goal, fewer than 5 visits (wandering) |
| Farming rate | Farm only + farm then finish |
| Path efficiency | Shortest possible path length divided by the path actually taken, on direct-goal episodes |

A point worth making early in the blog: on this open grid a **uniformly random policy reaches the goal directly in 49% of layouts**, with path efficiency 0.23 (Experiment 2 README, Phase 5 reference table). "Reached the goal" alone is a weak success signal, which is why the project requires efficient paths too.

---

## 3. Experiment 1 versus Experiment 2

| | Experiment 1: Fixed Gridworld Repair | Experiment 2: Generalizable Reward Hacking |
|---|---|---|
| Role in the blog | Short prologue | Main narrative |
| Layout | One fixed 8×8 layout | Start, goal and tile redrawn every episode |
| Agents | 3 seeds trained on the flawed reward, 3 on the corrected reward | Per seed: one pretrained network, forked into a good branch and a hacking branch |
| What worked | An activation transplant and a rank-4 weight edit, for one checkpoint pair of one seed | Teacher distillation of the whole network, and a coordinated rank-8 edit, for all 3 seeds |
| What failed | Probe-direction steering; small neuron subsets; transfer to the other two seeds | Activation transplants, weight swaps, single-layer edits, teacher-free fine-tuning |
| Evaluation | Exploratory; every start position was eventually used | Train / development / test split, plus a reserved set evaluated once under a frozen protocol |
| Report | `experiments/01_fixed_gridworld_repair/README.md` | `experiments/02_generalizable_reward_hacking/README.md` |
| Journal | `experiments/01_fixed_gridworld_repair/PROGRESS.md` | `experiments/02_generalizable_reward_hacking/PROGRESS.md` |
| Status | Complete | Complete |

---

## 4. Experimental narrative

### 4.1 Experiment 1 (prologue)

Source for everything in this subsection: `experiments/01_fixed_gridworld_repair/README.md`. Details and failed attempts: `experiments/01_fixed_gridworld_repair/PROGRESS.md`, methods in `experiments/01_fixed_gridworld_repair/METHODS.md`.

1. **The agents learned to cheat without being shown how.** All three seeds (42, 123, 456) ended with the same score of 97 and 49 tile visits. Networks trained on the corrected reward walk straight to the goal.
2. **A probe can read the cheating, but that proves less than it seems.** A linear probe on hidden activations predicts farming moves with AUROC 0.98–1.00. A probe on just the agent's distances to the goal and tile already reaches 0.95–0.99.
3. **Subtracting the "farming direction" stopped the farming and nothing else.** The agent wandered; the best goal rate in any seed was 12%. Reason: the output layer is linear, so subtracting a fixed vector equals adding a fixed bias to the four action values everywhere.
4. **Transplanting activations from an earlier, goal-following checkpoint worked, after a surprise.** Seed 123 followed the goal for its first 20k steps and then switched; the 20k and 30k checkpoints are the donor and recipient. The first transplant gave 100% goal completion, but the agent still farmed 37–41 times first. Diagnosis: the hacking network heads for the goal only when little time remains. Computing the donor activation as if 20% of the episode remained gave **100% direct goal on all 62 starts**, in about 6 steps.
5. **Unexpected: a metric trap.** Goal completion alone would have called the first transplant a success. This is where the four behavior categories came from.
6. **How small can the patch be?** Not a few neurons: no subset of 32 or fewer units works, 64 do. But a few directions: the mean correction plus 8 principal directions reproduces the full effect (1.00 against 0.02 for random 8-dimensional subspaces).
7. **Unexpected: a second trap.** An uncentered SVD basis also reached 100% with 8 directions, but it scored 98% with activations from the wrong square, so it was a state-blind push toward the goal. It was reported and disqualified.
8. **Permanent weight edit.** A rank-4 change to layer 2 (1,152 trainable parameters) gives a standalone network with 100% direct goal on all 62 starts. Ranks 1 and 2 reach 10% and 34%. Random rank-4 changes of the same size reach 2%.
9. **Two sobering results.** Ordinary fine-tuning of layer 2 on the same examples also reaches 100%. And nothing transferred to the other two seeds: the patch reaches 0.10 (seed 42) and 0.03 (seed 456), and no weight edit exceeds 0.12.

**Why Experiment 2 exists:** Experiment 1's success held for one checkpoint pair of one seed in one fixed maze, with no held-out test. Experiment 2 asks whether any of it survives changing layouts, independent seeds and a real evaluation protocol.

### 4.2 Experiment 2 (main story)

Source: `experiments/02_generalizable_reward_hacking/README.md` unless stated. Full methods, controls and negative results: `experiments/02_generalizable_reward_hacking/PROGRESS.md`.

**Phase 1: the same network, two rewards.** For each of three seeds, a DQN was pretrained for 300k steps on the corrected reward. It was then copied twice. One copy trained 400k more steps on the corrected reward (the **good** branch). The other trained 400k more steps on the flawed reward (the **hacking** branch). On 1,500 held-out test layouts, evaluated once:

- Good DQN: 1.000 direct goal in all three seeds.
- Hacking DQN: 0.921 farming on average (0.923 / 0.921 / 0.919), 0.076 direct goal, 43.7 tile visits per episode.
- Unexpected: the first budget (200k steps) was too short for seed 123, which was still at 0.72 farming and rising. The fork was extended once to 400k for every seed, decided from development curves before the test set was touched. The 200k results are kept in `results/exp2/phase1/fork_200k/`.
- The hacking agent still goes straight to the goal in about 8% of layouts, mostly when the goal is closer to the start than the tile is.

**Phase 2: what is different inside.** The three networks of each seed were run on 6,000 identical states.

- The good network barely moved from the pretrained one (layer-2 similarity, linear CKA, 0.99). The hacking network was rebuilt, mostly in layer 2 (CKA 0.37 / 0.41 / 0.46 with the pretrained network; activation size roughly tripled, from about 11–13 to 31–39).
- 88–93% of the layer-2 difference is one constant shift. The rest needs only 4–5 directions, which track time remaining and the agent's offset from the tile.
- Hacking networks from different seeds resemble each other (layer-2 CKA 0.77–0.86), but the report is explicit that this shows they encode the same task quantities, not a shared mechanism.
- **Negative result:** transplanting the good network's layer-2 activations into the hacking network stops the farming (0.00–0.07) but the agent then wanders in 54–89% of layouts. Direct goal 0.44 / 0.11 / 0.30, below the random policy's 0.49.
- Why this differs from Experiment 1: there the donor and recipient were 10k steps apart; here they trained apart for 400k steps and the output layer can no longer read the good network's activations.

**Phase 3: watching the switch happen.** Both branches were saved every 10k steps (41 checkpoints each).

- The switch takes about 100k steps and passes through a wandering stage (peak 24–30% of episodes). Onset / completion of farming: seed 42 at 70k / 160k, seed 123 at 120k / 220k, seed 456 at 100k / 190k.
- Hidden layers start drifting immediately, long before behavior changes: at the last checkpoint before onset, layer-2 similarity to the start is already 0.93 / 0.77 / 0.76 while the good branch is at 1.00.
- The fastest change falls at or within 10k steps of behavioral completion, and at completion only 46–53% of the final layer-2 change has happened. The rest continues after behavior has settled.
- **Negative result:** no activation spike marks the decisions that flip. Position, distances and remaining time explain 98–99% of how much layer 2 changes at a state, and no single "farming direction" separates flipped decisions better than distance to the tile does.
- Across seeds: the same sequence at different times, but not the same directions.

**Phase 4: maybe the donor was just too far away.** Hypothesis: patching should work between checkpoints of the same run a few steps apart.

- **Negative result:** no pair met the criterion in any seed. Nearby donors are compatible (CKA 0.98–1.00 at 10k steps apart) but are already hacking (their own direct goal 0.09–0.11). Goal-following donors are already distant (CKA 0.34–0.48).
- The insight: behavior and representation change together, so there is no checkpoint that is both close and good.

**Phase 5: which weights have to change?** Repair bar fixed in advance: direct goal ≥ 0.90, path efficiency ≥ 0.90, farming ≤ 0.05.

- **Negative result:** copying goal-following weights into any subset of layers fails: 0 of 72 swaps meet the bar. A goal-following output head on hacking features gives 0.05–0.12 direct goal, about what a random head of the same size gives.
- **Negative result:** fine-tuning with reinforcement learning on the corrected reward for 200k steps removes farming in 9 of 12 runs, but direct goal ends at 0.12–0.41 whichever layers are trainable, below the random policy. Unlearning the exploit is fast; relearning the task from the hacking weights is slow.
- **What works:** supervised distillation, where the same seed's good network labels 28,000 states with its action values and the hacking network is trained to match. Only training the **full network** repairs it: 0.99 / 0.99 / 1.00 direct goal. Head only: 0.49 / 0.49 / 0.69. Layer 1 only: 0.53 / 0.61 / 0.75. Layer 2 only (91% of the parameters): 0.64 / 0.70 / 0.83.
- Even the mathematically best possible output head on the frozen hacking features gives only 0.66 / 0.66 / 0.89.
- Size of weight change does not explain behavior: a random edit as large as a repair produces wandering (0.07–0.11).

**Phase 6: how small can a coordinated repair be?** Hacking weights frozen; each layer learns a low-rank update that is merged into ordinary weights afterwards. Nine runs per setting (3 seeds × 3 optimization seeds); a method counts only if all nine pass.

| Coordinated edit | Trainable parameters | Passing at learning rate 0.00025 | Passing at learning rate 0.001 | Direct goal at 0.001 |
|---|---|---|---|---|
| Biases only | 260 | 0 / 9 | 0 / 9 | 0.11–0.18 |
| Rank 1 | 783 | 0 / 9 | 0 / 9 | 0.40–0.57 |
| Rank 2 | 1,306 | 0 / 9 | 0 / 9 | 0.55–0.81 |
| Rank 4 | 2,352 | 0 / 9 | 0 / 9 | 0.79–0.97 |
| **Rank 8** | 3,781 | 0 / 9 | **9 / 9** | **0.98–1.00** |
| Full network | 18,052 | 9 / 9 | 9 / 9 | 1.00 |

- **Unexpected:** the planned headline setting (learning rate 0.00025) failed at every rank. The rank-8 result exists only at the second, higher rate. Both rates were fixed before the run.
- Ordinary full fine-tuning passed at both rates, so the low-rank edit is the less robust method.
- "Rank 8" is not eight weights: 3,781 trainable parameters (21% of the network), about 14,100 of 17,792 weights actually change, and the update is truly low-rank only in layer 2 (rank 7 / 8 / 4 across the three layers).
- Single-layer low-rank edits never pass. A rank-2 edit of all three layers beats a rank-8 edit of layer 2 alone.
- **Negative result:** cutting the Phase 5 full-network update down to its top 8 directions gives 0.06–0.12. The low-rank edit has to be learned.

**Confirmatory evaluation: does it hold on untouched layouts?** The protocol was written down and committed (commit `faf6cdb`) before the 1,500 reserved layouts were evaluated. The nine rank-8 networks from Phase 6 were evaluated once, with no retraining or reselection.

- **9 of 9 met the bar.** Direct goal 0.993 mean (range 0.977–1.000), path efficiency 0.987 mean (0.955–1.000), farming 0.000 in all nine.
- Unedited hacking networks on the same layouts: 0.07–0.08 direct goal, 0.91–0.92 farming. Good networks: 1.00.
- Development and reserved numbers agree: 0.994 vs 0.993 direct goal, 0.985 vs 0.987 path efficiency.
- Five of the nine models still miss the goal in a few layouts (up to 2.3%, 34 of 1,500), without farming. Those episodes were not analyzed.

---

## 5. Verified headline metrics and sources

All checked against the named files on 2026-11-25.

| # | Claim | Value | Source |
|---|---|---|---|
| 1 | Exp 1: all 3 seeds learned to farm | score 97, 49 tile visits | `experiments/01_fixed_gridworld_repair/README.md` §2 |
| 2 | Exp 1: probe AUROC, activations vs distances only | 0.98–1.00 vs 0.95–0.99 | same, §3; `results/phase3/` |
| 3 | Exp 1: time-shifted transplant, direct goal | 1.00 on 62 starts | same, §5 and §8 table; `results/phase4_7/` |
| 4 | Exp 1: rank-4 weight edit, direct goal | 1.00 on 62 starts | same, §7; `results/weight_editing/weight_editing_results.json` |
| 5 | Exp 1: cross-seed transfer | 0.10 and 0.03 | same, §8 |
| 6 | Exp 2: good DQN direct goal, test layouts | 1.000 / 1.000 / 1.000 | `results/exp2/phase1/evaluation.json` (`per_seed.<seed>.good.test`) |
| 7 | Exp 2: hacking DQN farming, test layouts | 0.923 / 0.921 / 0.919 (mean 0.921) | same file (`per_seed.<seed>.hacking.test`) |
| 8 | Exp 2: random policy on development layouts | 0.49 direct goal, efficiency 0.23 | `results/exp2/phase5/baselines.json`; Exp 2 README Phase 5 table |
| 9 | Exp 2: layer-2 similarity to pretrained (CKA) | good 0.99; hacking 0.37 / 0.41 / 0.46 | `results/exp2/phase2/`; Exp 2 README Phase 2 |
| 10 | Exp 2: layer-2 activation transplant, direct goal | 0.44 / 0.11 / 0.30 | Exp 2 README Phase 2 table; `results/exp2/phase2/` |
| 11 | Exp 2: farming onset / completion steps | 70k/160k, 120k/220k, 100k/190k | `results/exp2/phase3/timeline.json` (`per_seed.<seed>.summary.transition.dev`) |
| 12 | Exp 2: share of final layer-2 change done at completion | 46–53% | Exp 2 README Phase 3; `results/exp2/phase3/timeline.json` |
| 13 | Exp 2: near-transition patching pairs meeting criterion | 0 | Exp 2 README Phase 4; `results/exp2/phase4/patching_results.json` |
| 14 | Exp 2: weight swaps meeting the repair bar | 0 of 72 | `results/exp2/phase5/weight_swaps.json` |
| 15 | Exp 2: RL fine-tuning, final direct goal | 0.12–0.41 | `results/exp2/phase5/finetune.json` |
| 16 | Exp 2: full-network distillation, direct goal | 0.99 / 0.99 / 1.00 | `results/exp2/phase5/distill.json` |
| 17 | Exp 2: rank-8 coordinated edit, development | 9 / 9 at 0.001; 0 / 9 at 0.00025 | `results/exp2/phase6/summary.json`, `results/exp2/phase6/runs.json` |
| 18 | Exp 2: rank-8 trainable parameters | 3,781 of 18,052 (21%) | `results/exp2/phase6/runs.json`; `experiments/02_generalizable_reward_hacking/configs/final_eval.json` |
| 19 | Exp 2: reserved layouts, direct goal | mean 0.993, sd 0.009, range 0.977–1.000 | `results/exp2/final_eval/final_eval.json` (`aggregate.final_eval.direct_goal`) |
| 20 | Exp 2: reserved layouts, path efficiency | mean 0.987, sd 0.015, range 0.955–1.000 | same (`aggregate.final_eval.path_efficiency_direct`) |
| 21 | Exp 2: reserved layouts, farming | 0.000 in all nine | same (`models[*].final_eval.farming`) |
| 22 | Exp 2: models meeting the bar on reserved layouts | 9 of 9 | same (`n_meeting_criterion`, `confirmed`) |
| 23 | Exp 2: hacking baseline on reserved layouts | 0.075 / 0.081 / 0.083 direct goal; 0.923 / 0.919 / 0.913 farming | same (`baselines.per_seed`) |
| 24 | Tests | 438 passing | `python -m pytest -q` |

The one-number headline: **a coordinated rank-8 weight edit took the hacking networks from about 8% to 99.3% mean direct-goal completion on 1,500 reserved layouts.**

---

## 6. Scientific caveats and claims to avoid

**Say this**

- Reward hacking emerged in three independently trained DQNs.
- Representations changed gradually during learning, starting before the behavior and continuing after it.
- Activation transplants usually stopped the farming without restoring useful behavior.
- Distilling a goal-following teacher into the whole network restored goal-directed behavior.
- A coordinated rank-8 edit reached 99.3% mean direct-goal completion on reserved layouts.

**Do not say this**

| Claim to avoid | Why |
|---|---|
| "We found the reward-hacking circuit" or "the cheating neurons" | Nothing identifies a unique mechanism. Probes read task variables; successful edits show sufficiency, not location. |
| "We can repair reward hacking without knowing the right behavior" | Every successful repair in Experiment 2 was trained on a goal-following teacher's outputs. The only teacher-free method (RL fine-tuning on the corrected reward) removed farming but did not restore navigation. |
| "Low-rank editing is better than fine-tuning" | Full fine-tuning passed at both learning rates; the rank-8 edit passed at one. In Experiment 1 ordinary fine-tuning matched the rank-4 edit. |
| "Only eight weights (or eight numbers) changed" | Rank 8 means 3,781 trained parameters and about 14,100 changed weights; it is low-rank only in layer 2. |
| "The method generalizes to new tasks or environments" | Tested only on new position combinations from one obstacle-free 8×8 generator, one exploit, one architecture. |
| "The network decides to cheat at a specific moment" | The switch takes about 100k steps, passes through wandering, and the internal change spans and outlasts it. |
| "Hacking networks share a mechanism across seeds" | They encode the same task quantities; the flip directions are not shared (correlations 0.74, −0.05, −0.33). |
| "The agent understands it is cheating" or any intent language | The agent maximizes the reward it was given. Avoid anthropomorphic claims beyond the framing device of the title. |
| "Results from Experiment 1 generalize" | They held for one checkpoint pair of one seed and failed on the other two seeds. |

**Other limits to state:** three seeds; the nine confirmatory models are three agents times three optimization seeds, not nine independent agents; one learning rate confirmed; both held-out sets are now spent; a small residual failure rate (up to 2.3% of layouts) was not analyzed.

---

## 7. Best existing visuals

All files exist in the repository. Suggested use is for the blog; captions in the reports state how each layout was chosen (by rule, not by outcome).

### GIFs (`assets/gifs/`)

| File | Shows | Suggested use |
|---|---|---|
| `assets/gifs/exp2_hacking_dqn.gif` | Exp 2 hacking DQN farming on a test layout (48 tile entries) | Opening hook, next to the good agent |
| `assets/gifs/exp2_good_dqn.gif` | Exp 2 good DQN walking to the goal in 7 steps | Opening hook |
| `assets/gifs/exp2_phase3_training_progression.gif` | One layout at five training checkpoints (fork, 60k, 100k, 160k, 400k) | "Watching it learn to cheat" section; seed for the interactive timeline |
| `assets/gifs/exp2_phase2_hacking.gif`, `assets/gifs/exp2_phase2_patched.gif` | Before and after the activation transplant: farming stops, agent wanders | "The obvious fix fails" section |
| `assets/gifs/exp2_phase4_patch_comparison.gif` | Recipient, pre-onset donor, patched recipient | Optional, for the near-transition negative result |
| `assets/gifs/exp2_phase5_repair_comparison.gif` | Hacking, RL fine-tuned, head-only distilled, full distilled | "Which weights have to change" section |
| `assets/gifs/exp2_phase6_low_rank_repair.gif` | Hacking, rank-8 edited, good | The repair payoff |
| `assets/gifs/phase2_trained_dqn.gif`, `assets/gifs/phase2_5_goal_dqn.gif` | Exp 1 hacking and goal-following DQNs | Prologue |
| `assets/gifs/phase4_6_farm_then_finish.gif`, `assets/gifs/phase4_6_new_intervention_direct_goal.gif` | Exp 1 metric trap and its fix | Prologue, "a metric that lied" |

### Figures (`assets/figures/`)

| File | Shows | Suggested use |
|---|---|---|
| `assets/figures/gridworld_overview.png` | The Exp 1 grid, goal and reward tile | Explaining the setup |
| `assets/figures/exp2_scripted_returns.png` | Farming pays more under the flawed reward in every layout | Explaining the loophole |
| `assets/figures/exp2_training_curves.png` | All three seeds abandoning the goal at different times | "It happened every time" |
| `assets/figures/exp2_behavior_by_seed.png` | Test-layout behavior per seed | Baseline result |
| `assets/figures/exp2_phase3_behavior_timeline.png` | Direct goal, wandering and farming over training | Core of the "what happens inside" section |
| `assets/figures/exp2_phase3_activation_timeline.png` | Hidden-layer drift against behavior | Core of the "what happens inside" section |
| `assets/figures/exp2_phase3_pca_trajectory.png` | The hacking branch leaving the start point; the good branch staying | Intuitive picture of representation change |
| `assets/figures/exp2_phase2_patching.png` | Transplant stops farming, does not restore the goal | Negative result |
| `assets/figures/exp2_phase4_gap.png` | Close donors are already hacking | Negative result, one panel is enough |
| `assets/figures/exp2_phase5_weight_swaps.png` | No layer swap repairs the network | Negative result |
| `assets/figures/exp2_phase5_finetune_curves.png` | RL fine-tuning fails; only full-network distillation repairs | Key turning point |
| `assets/figures/exp2_phase5_params_vs_repair.png` | Parameter count does not explain repair | Supports "coordination, not size" |
| `assets/figures/exp2_phase6_repair_vs_rank.png` | Repair improves gradually with rank; passes at rank 8 at one learning rate | The low-rank result with its caveat visible |
| `assets/figures/exp2_phase6_single_vs_coordinated.png` | Coordinated edits beat single-layer edits at every rank | Supports "coordination" |
| `assets/figures/exp2_final_eval.png` | 9 of 9 on reserved layouts; development vs reserved per model | Closing result |
| `assets/figures/exp2_generalization.png` | What kind of generalization was tested | Limitations section |
| `assets/figures/phase3_probe_comparison.png` | Exp 1 probe vs distance-only probe | Prologue, "a probe is not a mechanism" |
| `assets/figures/weight_editing_rank_curve.png` | Exp 1 rank curve | Prologue, optional |

Every figure and GIF in `assets/` is referenced by at least one report, so none is orphaned.

---

## 8. Data for an interactive training timeline

The natural interactive piece: a slider over training steps (0 to 400k in 10k steps) for one seed, showing the agent's path on a layout, the behavior mix and the internal drift at that step, for the good and hacking branches side by side.

### Already computed, no model needed

| File | Contents |
|---|---|
| `results/exp2/phase3/timeline.json` | `steps` (41 values, 0 to 400,000). For each seed and branch, `per_seed.<seed>.series.<good or hacking>` holds 41-point series: `dev_direct_goal`, `dev_farming`, `dev_other`, `dev_farm_only`, `dev_farm_then_finish`, `dev_mean_reward_tile_visits`, `layer1_cka_start`, `layer2_cka_start`, `layer2_rel_change_start`, `layer2_mean_norm`, `fc1_rel_change_start`, `fc2_rel_change_start`, `fc3_rel_change_start`, `q_mean_max`, `layer2_cka_vs_good` and more. `per_seed.<seed>.summary.transition.dev` has onset, midpoint, completion and last-before-onset steps. `per_seed.<seed>.trajectory_pca` has the PCA path per layer. |
| `results/exp2/phase3/mean_activations.npz` | `steps` plus arrays named `mean_activation_seed<seed>_<branch>_layer<1 or 2>`, each 41 × 128: the mean activation of every hidden unit at every checkpoint. Good for a neuron heatmap that animates with the slider. |
| `results/exp2/phase3/gif_episode.json` | Seed 42, development layout index 0, layout `[2, 7, 0, 0, 5, 3]`, with the recorded episode states at steps 0, 60k, 100k, 160k and 400k. Ready-made paths for five slider positions. |
| `results/exp2/phase1/seed<seed>/<pretrain, good or hacking>/eval_metrics.csv` | Behavior on development layouts logged during training (columns such as `timestep`, `dev_direct_goal`, `dev_farming`, `dev_other`, `dev_path_efficiency_direct`). |
| `results/exp2/phase5/weight_trajectory.json` | Per-layer weight change over the same 41 steps for both branches. |
| `experiments/02_generalizable_reward_hacking/layouts/manifest.json` | The train, development and test layouts. `columns` names the six numbers of a layout. |

### Checkpoints, for computing new paths at every step

- Format: PyTorch files with keys `q_net_state_dict`, `target_net_state_dict`, `optimizer_state_dict`, `step`, `config`. About 300 KB each. The Q-network is three linear layers (`fc1`, `fc2`, `fc3`) with ReLU, defined in `rewardlens/agents/dqn.py` (`QNetwork`).
- Loader: `rewardlens.exp2.weight_level.load_net(seed, branch, step)` with `branch` in `"pretrain"`, `"good"`, `"hacking"`. It verifies the seed and stage recorded in the file.
- **Committed (30 files, in `results/exp2/phase1/seed<seed>/<branch>/checkpoints/`):**
  - All seeds: `pretrain/checkpoints/checkpoint_final.pt`, `good/checkpoints/checkpoint_final.pt`, `hacking/checkpoints/checkpoint_final.pt`.
  - Seed 42 hacking: steps 60000, 110000, 140000, 150000, 160000. Good: 60000, 160000.
  - Seed 123 hacking: 110000, 170000, 200000, 210000, 220000. Good: 110000, 220000.
  - Seed 456 hacking: 90000, 140000, 170000, 180000, 190000. Good: 90000, 190000.
- **Not committed (gitignored): the full set of 246 fork-stage checkpoints** (every 10k steps, both branches, three seeds; about 73 MB). `results/exp2/phase3/checkpoint_registry.json` records the hash of each. Regenerate with `python -m scripts.exp2_train --all-seeds` (about 9 minutes on CPU; an earlier rerun reproduced the checkpoints bit for bit on the original machine) and verify with `python -m scripts.exp2_run_phase3 --part registry`. On the original development machine these files are still present locally.
- Environment and observation code for rolling out a network: `rewardlens/exp2/procedural_env.py` (`ProceduralEnvParams`, `ProceduralRewardGridEnv`) and `rewardlens/exp2/evaluation.py` (`observations`, `greedy_policy`, `rollout`). The observation is 7 floats: agent x, agent y, goal x, goal y, tile x, tile y (each divided by 7), and the fraction of the episode remaining.

### Suggested approach for the web

The network is small enough to run in the browser. Export the three weight matrices and biases of each needed checkpoint to JSON (18,052 numbers each) and implement the forward pass and the grid step in JavaScript, or precompute the greedy path for a handful of layouts at all 41 steps and ship only the paths. The second option needs no model code on the page. Either export needs a small new script; none exists yet. Use development or training layouts for any new illustration. Do not present new rollouts on the reserved set as evaluation results.

---

## 9. Reproduction commands for visual assets

Run from the repository root after `pip install -r requirements.txt`. Each visualizer reads committed result files; figures regenerate without retraining. GIFs need the committed checkpoints only, except the Phase 3 GIF, which needs the intermediate checkpoints.

```bash
# Experiment 2
python -m scripts.exp2_visualize --gif            # Phase 1: training curves, behavior by seed, good/hacking GIFs
python -m scripts.exp2_visualize_phase2 --gif     # Phase 2: representation figures, transplant GIFs
python -m scripts.exp2_visualize_phase3 --gif     # Phase 3: timelines, heatmap, PCA path, training-progression GIF
python -m scripts.exp2_visualize_phase4 --gif     # Phase 4: near-transition patching
python -m scripts.exp2_visualize_phase5 --gif     # Phase 5: weight swaps, fine-tuning curves, repair GIF
python -m scripts.exp2_visualize_phase6 --gif     # Phase 6: repair vs rank, low-rank repair GIF
python -m scripts.exp2_visualize_final_eval       # Confirmatory figure; evaluates nothing

# Experiment 1
python -m scripts.visualize_phase2 --run-dir results/phase2/baseline_seed42   # hacking DQN figures and GIF
python -m scripts.visualize_phase3
python -m scripts.visualize_phase4_6              # also rewrites the four Phase 4.6 GIFs
python -m scripts.visualize_low_rank_patching --gif
python -m scripts.visualize_weight_editing --gif
```

- Full command sequences, run times and prerequisites: `SETUP.md`.
- Script indexes: `experiments/01_fixed_gridworld_repair/scripts/README.md` and `experiments/02_generalizable_reward_hacking/scripts/README.md`.
- **Warning:** several runner scripts have no command-line parser and start their full experiment when invoked with any argument, including `--help` (for example `python -m scripts.run_weight_editing`). Run only visualizers unless a rerun is intended.
- **Never run** `python -m scripts.exp2_run_final_eval` without `--check`. It refuses to overwrite its result, and the reserved set must not be evaluated again.
- Experiment 1 training checkpoints are not committed; `SETUP.md` lists the training commands that regenerate them.

---

## 10. Blog outline: "What Happens Inside a Neural Network When It Learns to Cheat?"

Target: a curious general reader. Roughly 2,000 to 2,500 words, visuals carrying the argument.

1. **Hook: two agents, one maze.** Side-by-side GIFs (`exp2_good_dqn.gif`, `exp2_hacking_dqn.gif`). Same network, same starting weights, same maze. One walks to the goal. One steps on and off a tile 48 times and never finishes. Nobody told it to.
2. **The loophole.** The game, the +2 tile, and why farming pays more (`exp2_scripted_returns.png`). Define reward hacking in one sentence. Mention the real-world shape of the problem without overclaiming.
3. **It happened every time.** Three independent training runs, the same outcome at different times (`exp2_training_curves.png`). 92% farming on mazes the agents had never seen.
4. **Prologue: the tidy version (Experiment 1).** In one fixed maze, a surgical fix worked: transplant activations, then a tiny permanent weight edit. Two lessons: a probe is not a mechanism, and a metric lied (goal completion looked like 100% while the agent still farmed). Then the catch: it did not transfer to other training runs.
5. **Raising the bar.** Layouts that change every episode, a held-out test, and a reserved set locked away. Include the random-policy fact: a coin-flipping agent finds the goal half the time, so "found the goal" is not enough.
6. **Watching the switch (the centerpiece).** The training-progression GIF and, if built, the interactive timeline. The network does not flip. It drifts internally before anything looks wrong, wanders for a while, then commits, and keeps changing after its behavior has settled (`exp2_phase3_behavior_timeline.png`, `exp2_phase3_activation_timeline.png`).
7. **The obvious fixes fail.** Transplanting good activations stops the cheating and leaves an agent that wanders (`exp2_phase2_patched.gif`). Borrowing from a nearby checkpoint fails because nearby checkpoints already cheat. Copying good weights into one layer fails 72 times out of 72.
8. **Why: the layers co-adapted.** Each layer learned to work with the others' new language. Retraining on the corrected reward unlearns the trick quickly but does not bring the skill back in the time given.
9. **What worked.** Showing the cheating network what a goal-following network would do, and letting all layers change together. Then the smaller version: a coordinated rank-8 edit, about a fifth of the parameters (`exp2_phase6_repair_vs_rank.png`, `exp2_phase6_low_rank_repair.gif`). State the catch in the same breath: it worked at one of two learning rates, and plain fine-tuning was sturdier.
10. **The test that could have failed.** Protocol frozen first, reserved mazes opened once: 9 of 9, 99.3% (`exp2_final_eval.png`).
11. **What this does not show.** No circuit found. The repair needs a teacher that already does the task. One small obstacle-free grid, one exploit, three training runs.
12. **Takeaways.** Cheating was learned gradually and distributed across the network; stopping bad behavior is not the same as restoring good behavior; negative results shaped the answer. Link to the repository and both reports.

---

## 11. Explanations for nontechnical readers

| Term | Plain-language version |
|---|---|
| Reward hacking | A student who is graded on the number of pages written and hands in a thousand pages of nonsense. The score went up; the essay did not get written. |
| Reinforcement learning, DQN | The program tries moves, receives points, and gradually prefers moves that led to more points. The network's job is to guess how many points each move will eventually bring. |
| Hidden layer, activations | The network's scratch work between seeing the board and choosing a move. 128 numbers per layer that change with every board position. |
| Weights | The network's long-term memory: the fixed numbers that turn inputs into scratch work. Training changes them. There are 18,052. |
| Probe | A simple detector trained to read something off the scratch work. Finding a signal there shows the information exists, not that the network uses it. |
| Activation transplant (patching) | Running two networks on the same board and pasting one network's scratch work into the other, mid-thought. |
| Why the transplant fails in Experiment 2 | The later layers of the cheating network no longer speak the same language as the good network's scratch work. Like handing a French speaker notes written in Japanese: the old plan is disrupted, no new plan arrives. |
| Co-adaptation | Dancers who learned a routine together. Swap in one dancer from another troupe and the routine falls apart even if that dancer is excellent. |
| Distillation, teacher | Showing the cheating network what a well-behaved network would do on thousands of boards and training it to give the same answers. |
| Low-rank edit | Instead of adjusting every knob independently, adjusting a few shared sliders that each move many knobs at once. Rank 8 means eight sliders per layer. |
| Frozen protocol, reserved set | Writing down exactly what will count as success, then opening a sealed set of exam questions exactly once. |
| Path efficiency | Did the agent take the short route, or stumble onto the goal after wandering? |
| CKA (similarity) | A 0-to-1 score for how alike two networks' scratch work is across many boards. Best described in the blog as "similarity" without the acronym. |

Tone guidance: prefer "the network that learned to farm" over "the network that wants to cheat". The title uses "cheat" as a hook; the body should explain that the agent is doing exactly what the flawed reward asked for.

---

## 12. Links

| What | Path | GitHub |
|---|---|---|
| Project overview | `README.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/README.md> |
| Experiment 1 report | `experiments/01_fixed_gridworld_repair/README.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/experiments/01_fixed_gridworld_repair/README.md> |
| Experiment 1 methods | `experiments/01_fixed_gridworld_repair/METHODS.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/experiments/01_fixed_gridworld_repair/METHODS.md> |
| Experiment 1 journal | `experiments/01_fixed_gridworld_repair/PROGRESS.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/experiments/01_fixed_gridworld_repair/PROGRESS.md> |
| Experiment 2 report | `experiments/02_generalizable_reward_hacking/README.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/experiments/02_generalizable_reward_hacking/README.md> |
| Experiment 2 journal | `experiments/02_generalizable_reward_hacking/PROGRESS.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/experiments/02_generalizable_reward_hacking/PROGRESS.md> |
| Frozen confirmatory protocol | `experiments/02_generalizable_reward_hacking/configs/final_eval.json` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/experiments/02_generalizable_reward_hacking/configs/final_eval.json> |
| Confirmatory result | `results/exp2/final_eval/final_eval.json` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/results/exp2/final_eval/final_eval.json> |
| Setup and reproduction | `SETUP.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/SETUP.md> |
| Research context and decisions | `CONTEXT.md` | <https://github.com/ashwin-aggarwal/RewardLens/blob/main/CONTEXT.md> |
| Pull request for Experiment 2 | | <https://github.com/ashwin-aggarwal/RewardLens/pull/3> |
| Figures and GIFs | `assets/figures/`, `assets/gifs/` | <https://github.com/ashwin-aggarwal/RewardLens/tree/main/assets> |

Sub-reports for Experiment 1 extensions: `experiments/01_fixed_gridworld_repair/sparse_patching/README.md`, `experiments/01_fixed_gridworld_repair/low_rank_patching/README.md`, `experiments/01_fixed_gridworld_repair/weight_editing/README.md`.

## 13. State of the repository when this was written

- `main` holds both completed experiments (merge commit `a13116b`).
- A reorganization from branch `chore/repository-cleanup` was merged afterwards: scripts moved into each experiment's `scripts/` folder, Experiment 1 configs moved to `experiments/01_fixed_gridworld_repair/configs/`. Commands are unchanged. This file was added in the same merge.
- No blog text has been written yet.
