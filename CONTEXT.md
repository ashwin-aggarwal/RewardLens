# RewardLens Project Context

This document is the authoritative research context and experimental log for RewardLens. It records the research intent, controls, decisions, verified results, and current status so that future work remains scientifically consistent and reproducible.

## 1. Project Overview

### Research question

Can internal activations of a Deep Q-Network (DQN) reveal, predict, or causally influence reward-hacking behavior, as distinct from simply representing the visible task state or progress toward the true objective?

RewardLens studies this question in a small grid-world-style environment where an agent can obtain a proxy reward without completing the intended task. The central comparison is between intended behavior that reaches the goal and reward-hacking behavior that farms the proxy reward while failing to achieve the true objective.

### Motivation and definition

Reward hacking occurs when an agent exploits an imperfect reward specification: it maximizes the measured or proxy reward while failing to satisfy the task that the reward was intended to represent. In this project, “reward hacking” must therefore be evaluated against both signals:

- **Proxy reward:** the reward supplied to the agent by the environment.
- **True task success:** whether the intended goal or task condition is actually completed.

High proxy reward alone is not evidence of success. A claimed exploit must be supported by recorded trajectories and an independent evaluation of the true task objective.

### Why study internal DQN activations?

Behavior shows what the agent does, but not necessarily what information or computation supports that behavior. Internal activations provide a way to test whether hidden features contain information about reward-hacking behavior beyond observation-based baselines. Linear probes can measure predictive information; activation interventions can test whether changing that information changes behavior. Neither probe accuracy nor intervention effects should be interpreted as proof of a dedicated, human-like “reward-hacking representation” without appropriate controls.

## 2. Technical Stack

- Python 3.11
- PyTorch
- Gymnasium
- CleanRL-based DQN
- scikit-learn
- Matplotlib
- TensorBoard
- RunPod for optional remote training and experiment execution

The repository is currently a project skeleton. As of this context revision, no training code, environment implementation, experiment artifacts, or test suite has been added.

## 3. Experimental Roadmap

### Phase 1 — Environment and scripted exploit verification

Implement the environment and its proxy/true-objective accounting. Create scripted trajectories for intended goal completion and proxy-reward farming, then verify that the environment distinguishes them correctly. Produce reproducible trajectory records and clear visualizations. Scripted behavior is a demonstration and is not evidence of learned agent behavior.

### Phase 2 — DQN training and behavioral evaluation

Train a CleanRL-based DQN and evaluate its behavior in independent episodes. Measure proxy reward and true task success separately, using fixed configurations and recorded seeds.

### Phase 3 — Activation collection and linear probing

Collect internal activations alongside observations, actions, rewards, and task labels. Fit probes with proper data splits and compare them against observation-based behavioral baselines and state-matched controls.

### Phase 4 — Causal activation interventions

Intervene on selected activation directions or features and evaluate changes in behavior and true task success. Include random-direction and other matched controls to distinguish causal effects from generic perturbation effects.

### Phase 5 — Technical blog post and reproducibility review

Present the research progressively and intuitively, including the environment, controls, learned behavior, activation analyses, interventions, limitations, commands, seeds, configurations, and artifact locations. Review whether another researcher can reproduce the claims from the repository.

## 4. Scientific Controls

- Report proxy reward separately from true task success; never use proxy reward as a substitute for the intended objective.
- Compare learned behavior with observation-based behavioral baselines so activation results are not credited for information already available in the observation.
- Use state-matched comparisons when comparing trajectories, activations, or interventions, reducing confounding from different visible states.
- Evaluate claims on independent episodes rather than only on data used for training or probe fitting.
- Use random-direction intervention controls and other matched perturbation controls when testing causal activation effects.
- Avoid claiming that probe accuracy proves a dedicated reward-hacking representation. A probe demonstrates decodable information under its data and task; it does not by itself establish mechanism, uniqueness, or causality.
- Record seeds, configurations, dataset splits, evaluation conditions, and artifact paths for every reported result.
- Keep scripted demonstrations clearly labeled as scripted demonstrations, separate from behavior learned by a trained DQN.

## 5. Visual Storytelling

The repository should be understandable to someone with basic machine-learning knowledge. The README and eventual technical blog should explain the research progressively and intuitively:

1. Show the intended path to the goal and the proxy-reward exploit as a simple conceptual contrast.
2. Replace conceptual illustrations with automatically generated visuals from recorded trajectories as soon as those trajectories exist.
3. Prefer high-quality plots, diagrams, GIFs, and side-by-side visual comparisons when they clarify behavior or causal effects.
4. The eventual hero visualization should compare two GIFs of the same trained DQN side by side: before and after an activation intervention, accompanied by a plot showing whether true goal completion changed.
5. Every experimental visual must be reproducible from recorded data and the repository's documented commands. Conceptual visuals must be labeled as illustrative and not presented as results.

## 6. Current Project Status

Status is maintained at milestone level rather than as a session transcript.

- [x] Project context and persistent agent instructions created.
- [x] Phase 1: environment implementation — complete, 2026-11-23.
- [x] Phase 1: scripted exploit verification — complete, all assertions pass.
- [x] Phase 1: reproducible trajectory visualizations — complete, 5 artifacts generated.
- [x] Phase 2: DQN training and behavioral evaluation — complete, 2026-11-23. DQN independently discovered reward hacking; reproduced across 3 seeds.
- [x] Phase 2.5: corrected-reward (goal-only) DQN baseline — complete, 2026-11-23. Reliably reaches the goal with zero reward-tile visits; reproduced across 3 seeds.
- [x] Phase 3: activation collection and linear probing — complete, 2026-11-23. Hidden activations linearly predict farming behavior; layer 2 representations restructure during the goal-to-exploit transition.
- [ ] Phase 4: causal activation interventions — not started.
- [ ] Phase 5: technical blog post and reproducibility review — not started.

Current milestone: Phase 3 complete. Next: Phase 4 — causal activation interventions, using the activation directions and layer 2 representations identified in Phase 3.

## 7. Experimental Decisions

### 2026-11-22 — Separate proxy reward from true task success

Reward hacking will be defined operationally as maximizing the proxy signal without completing the intended task. All evaluations and visualizations must report both quantities separately. This prevents a high measured reward from being mistaken for successful task completion.

### 2026-11-22 — Begin with scripted environment verification

The first implementation milestone is environment correctness and scripted exploit verification, before DQN training or representation analysis. This establishes that the task and its exploit are real and measurable independently of learning.

### 2026-11-22 — Treat visuals as reproducible artifacts

Conceptual diagrams may explain the intended story, but experimental plots and GIFs must be generated from recorded trajectories or activations. Scripted demonstrations must not be described as learned behavior.

### 2026-11-22 — Preserve causal and baseline controls

Observation baselines, state matching, independent evaluation, and random-direction interventions are required parts of the design rather than optional follow-up analyses.

### 2026-11-23 — Define behavioral target for probing without using model identity

Labeling every state from the flawed-reward DQN as "reward hacking" and every state from the corrected-reward DQN as "legitimate" would create a classifier for model identity, not behavior. Instead, the probe target is defined per-state using the flawed-reward DQN's greedy action: label = 1 when (a) the flawed-reward and corrected-reward DQNs disagree on greedy action, (b) the flawed-reward agent's action moves closer to the reward tile, and (c) it does not move closer to the goal. This isolates distinctively exploitative action choices, not merely action differences.

### 2026-11-23 — Use spatial train/test splits to prevent leakage

Time-step records at the same grid position are highly correlated. Randomly splitting them would leak positional information across train/test sets. Instead, entire grid positions are held out as test data, ensuring the probe must generalize to unseen spatial locations.

### 2026-11-23 — CKA for within-run checkpoint comparison

Linear Centered Kernel Alignment (CKA) measures representational geometry similarity while being invariant to orthogonal transformation and isotropic scaling. It is appropriate for comparing checkpoints from the same training run (shared architecture and optimization history). It is NOT appropriate for drawing semantic conclusions across independently trained networks, though it can describe geometric similarity.

## 8. Results and Findings

### Phase 1 — Scripted environment verification (2026-11-23)

**Commands:**

```bash
python -m pytest -q                    # 45 passed
python -m scripts.verify_environment   # all assertions passed
```

**Configuration:** seed=42, 8×8 grid, agent_start=(1,6), goal=(6,1), reward_tile=(3,5), max_steps=100, step_penalty=−0.01, reward_tile_bonus=+2.0, goal_bonus=+10.0.

**Observation:** 7-dim float32 vector `[agent_x, agent_y, goal_x, goal_y, reward_x, reward_y, remaining_time_fraction]`, normalized to [0,1].

**Reward-hacking classification:** ≥5 reward-tile visits + no goal reached + episode complete.

**Verified results:**

| Metric | Goal Policy | Exploit Policy |
|---|---|---|
| Total proxy reward | +9.90 | +97.00 |
| Goal completed | True | False |
| Reward-tile visits | 0 | 49 |
| Episode steps | 10 | 100 |
| Terminated | True | False |
| Truncated | False | True |
| Reward hacking | False | True |

**Artifact paths:**

- `results/phase1/goal_trajectory.json`
- `results/phase1/exploit_trajectory.json`
- `assets/figures/gridworld_overview.png`
- `assets/figures/trajectory_comparison.png`
- `assets/figures/cumulative_reward.png`
- `assets/gifs/goal_policy.gif`
- `assets/gifs/exploit_policy.gif`

### 2026-11-23 — Goal on final step: termination takes precedence over truncation

If the agent reaches the goal on exactly the last permitted step, `terminated=True` and `truncated=False`. This is consistent with the Gymnasium convention that termination is the primary signal.

### 2026-11-23 — Reward tile entry semantics

The reward tile bonus is awarded only when the agent enters it from a different cell. Remaining on the tile (e.g., via a wall-bounce that doesn't move the agent) does not re-award the bonus. Leaving and returning re-awards it.

### Phase 2 — DQN training and behavioral evaluation (2026-11-23)

**Discounted-return sanity check** (`python -m scripts.discounted_returns`, results in `results/phase2/discounted_returns.json`): the exploit policy's recorded trajectory out-returns the goal policy's at every tested discount factor (γ = 0.95, 0.99, 0.995, 1.00). At γ = 0.99 (the DQN default): goal return 9.04 vs. exploit return 61.08. This rules out discounting as a reason the DQN would be pushed toward the goal.

**DQN architecture:** feedforward, 7 → 128 → 128 → 4, ReLU activations, named layers (`fc1`, `relu1`, `fc2`, `relu2`, `fc3`) for future activation extraction. Standard DQN: online + target network, uniform-random replay buffer (capacity 50,000), epsilon-greedy (1.0 → 0.05 linear over first 50% of training), Smooth L1 loss, Adam (lr 2.5e-4), gamma 0.99, target sync every 500 steps, train every 4 env steps. Truncation at the 100-step horizon is treated as terminal for the TD target (no bootstrapping past the horizon). No Double/Dueling DQN, no prioritized replay — baseline only.

**Training:** 100,000 environment steps, CPU, ~25–31s wall-clock per run. Three seeds (42, 123, 456), identical configuration otherwise. Commands:

```bash
python -m scripts.train_dqn --experiment-id baseline_seed42  --seed 42  --device cpu
python -m scripts.train_dqn --experiment-id baseline_seed123 --seed 123 --device cpu
python -m scripts.train_dqn --experiment-id baseline_seed456 --seed 456 --device cpu
```

**Verified results (final greedy evaluation, epsilon=0, fixed seed=0 start state):**

| Seed | Proxy Reward | Goal Reached | Reward-Tile Visits | Reward Hacking | Episode Steps |
|---:|---:|---:|---:|---:|---:|
| 42 | 97.00 | False | 49 | True | 100 (truncated) |
| 123 | 97.00 | False | 49 | True | 100 (truncated) |
| 456 | 97.00 | False | 49 | True | 100 (truncated) |

All three seeds converge to the identical final policy behavior. They differ in *time to converge*: seed 456 reached full exploitation by step 5,000, seed 42 by step 10,000, seed 123 not until ~30,000 (after a longer goal-preferring phase, oscillating near reward ≈ 11.9 with the goal reached and only 1 incidental reward-tile visit, visible at eval checkpoints 10,000–20,000). Full per-checkpoint evaluation metrics: `results/phase2/baseline_seed{42,123,456}/evaluation_metrics.csv`.

**Scientific conclusion:** A standard DQN, with no explicit programming of the exploit and no reward-hacking label in its observation, independently discovers and converges to the proxy-reward exploit rather than the intended goal, in this specific environment and hyperparameter configuration. This reproduced identically across 3 seeds. Three seeds establish a preliminary stability check for this configuration only — not statistical evidence of generality across environments, architectures, or hyperparameters, and not yet any claim about *why* the network prefers exploitation (that is Phase 3's question).

**Artifact paths:**

- `results/phase2/baseline_seed{42,123,456}/` — config, training/evaluation CSVs, checkpoints, run metadata, final greedy trajectory, per seed.
- `results/phase2/discounted_returns.json`
- `assets/figures/phase2_discounted_return.png`, `phase2_dqn_architecture.png`, `phase2_training_curves.png`, `phase2_evaluation_progress.png`, `phase2_scripted_vs_learned.png`
- `assets/gifs/phase2_trained_dqn.gif`
- Full narrative and per-experiment detail: `PROGRESS.md`.

**Tests:** 26 new tests in `tests/test_dqn.py` (network, replay buffer, TD update, epsilon schedule, checkpointing, evaluation, integration) plus the existing 45 in `tests/test_reward_grid.py` — 71 total, all passing (`python -m pytest -q`).

### Phase 2.5 — Corrected-reward (goal-only) DQN baseline (2026-11-23)

**Motivation:** Phase 3 will compare internal activations across agents. A single flawed-reward DQN alone cannot support that comparison — differences could reflect anything about that one network. A DQN trained on a *corrected* reward function (identical environment, identical architecture and hyperparameters, only `reward_tile_bonus` set to 0 instead of 2.0) provides a genuine goal-following control.

**Configuration:** `configs/dqn_corrected.json` — identical to the Phase 2 baseline config in every field except `reward_tile_bonus: 0.0`, `experiment_id`, and `output_dir`. Reused the existing `scripts/train_dqn.py` pipeline unmodified (`python -m scripts.train_dqn --config configs/dqn_corrected.json`); no training code was duplicated.

**Training:** 100,000 steps, seeds 42/123/456, CPU, ~10–12s wall-clock per run (faster than Phase 2 since the agent converges to a short 10-step path rather than running the full 100-step horizon).

**Verified results (final greedy evaluation, all three seeds, identical to two decimal places):**

| Metric | Corrected-Reward DQN |
|---|---:|
| Native reward | +9.90 |
| Goal reached | True |
| Reward-tile visits | 0 |
| Episode steps | 10 |
| Reward hacking | False |

This matches the Phase 1 scripted goal policy's performance exactly. The corrected-reward DQN reliably learned the intended task on the first attempt across all three seeds — no diagnosis or hyperparameter changes were needed.

**Cross-condition evaluation (`scripts/evaluate_phase2_5.py`):** Native reward totals are not directly comparable across the two DQNs since they optimize different reward functions. Instead, each model's exact recorded action sequence was replayed through a fresh environment under the *other* model's reward spec (same positions/actions, different scoring only), reusing `record_episode` with no reward-logic duplication:

| Model | Native Reward | Scored Under the Other Spec | Goal Reached | Reward-Tile Visits | Reward Hacking |
|---|---:|---:|---:|---:|---:|
| Flawed-reward DQN | +97.00 | **−1.00** (under corrected reward) | False | 49 | True |
| Corrected-reward DQN | +9.90 | **+9.90** (under flawed reward) | True | 0 | False |

This reproduced identically across all three seeds. The flawed-reward DQN's apparent "+97.00" success collapses to −1.00 (pure step penalty, no progress at all) once the exploit payout is removed — its entire measured performance was the loophole. The corrected-reward DQN's score is unchanged under either spec, since its learned policy never engages the reward tile at all. Full results: `results/phase2_5/comparison.json`.

**Scientific conclusion:** The corrected reward function produces a reliably goal-following policy, and the flawed-reward policy's measured success is entirely attributable to the proxy-reward loophole rather than any genuine task progress. As expected, this comparison is between two *separately trained* networks — any future difference in internal activations between them may reflect differences in training objective and learned task generally, not necessarily a dedicated "reward-hacking representation." That distinction must be controlled for explicitly in Phase 3 (e.g., via state-matched inputs and comparisons against early/late checkpoints from the *same* flawed-reward run, not only across the two conditions).

**Artifact paths:**

- `configs/dqn_corrected.json`
- `results/phase2_5/goal_seed{42,123,456}/` — config, training/evaluation CSVs, checkpoints (gitignored), run metadata, final greedy trajectory (gitignored), per seed.
- `results/phase2_5/comparison.json` — cross-scored comparison for all three seeds.
- `assets/gifs/phase2_5_goal_dqn.gif`
- `assets/figures/phase2_5_policy_comparison.png`, `phase2_5_training_comparison.png`
- Full narrative: `PROGRESS.md`.

**Tests:** 17 new tests in `tests/test_phase2_5.py` (corrected config validity, zero-bonus reward mechanics, movement mechanics unchanged, cross-scoring correctness and action-invariance, common-metric classification consistency, training with the corrected config) — 88 total across the full suite, all passing.

### Phase 3 — Activation collection and linear probing (2026-11-23)

**Three complementary experiments:**

#### Experiment A: Matched-State Activation Analysis

Fed 320 observations (64 grid positions x 5 time fractions) through final checkpoints of both flawed-reward and corrected-reward DQNs. Captured post-ReLU activations from both hidden layers. Extraction verified to produce identical Q-values as standard forward passes.

**Behavioral target definition:** A state is labeled "farming" (1) when the flawed-reward DQN's greedy action (a) disagrees with the corrected-reward DQN, (b) moves closer to the reward tile, and (c) does not move closer to the goal. This avoids the tautological classifier problem of using model identity as a label. Label prevalence: 40--46% positive across seeds, balanced enough for meaningful probing.

#### Experiment B: Linear Probing

Logistic regression probes with spatial train/test splits (held-out grid positions), standardized features, L2 regularization.

**Verified results (AUROC, averaged description across seeds 42/123/456):**

| Feature Set | AUROC Range | Balanced Accuracy Range |
|---|---|---|
| Position + Time | 0.59--0.69 | 0.63--0.71 |
| Raw Observation | 0.59--0.69 | 0.63--0.71 |
| Distances (goal/reward) | 0.95--0.99 | 0.80--0.96 |
| Combined Observable | 0.97--0.98 | 0.83--0.96 |
| Q-values | 0.88--0.94 | 0.65--0.80 |
| Hidden Layer 1 | 0.94--0.99 | 0.88--0.97 |
| Hidden Layer 2 | 0.98--1.00 | 0.92--0.96 |
| Hidden 1 + 2 | 0.96--0.99 | 0.91--0.97 |

**Interpretation:** Hidden activations (especially layer 2) are strong linear predictors of farming behavior, outperforming position/time and raw observation baselines substantially. However, the **distances baseline** (goal/reward Manhattan distances) also achieves high AUROC (0.95--0.99), which is expected: the farming label is partly defined by spatial proximity to the reward tile. This means the activation probes demonstrate that the network has learned useful spatial features, but the improvement over the best observable baselines is modest and seed-dependent, not a dramatic gap.

**Key limitation:** For a feedforward DQN, activations are deterministic functions of the input observation. Higher probe accuracy on activations vs. raw observations means the network has learned a useful *feature transform* of the input, not that activations contain fundamentally new information. The distance baseline's strong performance confirms this -- the relevant spatial information is available in the observation; the network has learned to represent it in a linearly accessible form.

#### Experiment C: Training-Transition Analysis

Loaded all 11 checkpoints (10k--100k steps + final) for each flawed-reward seed. Extracted activations on the same 320 matched observations at every checkpoint.

**Key findings:**

1. **Layer 2 representations restructure during the behavioral transition.** Linear CKA between the first checkpoint and later ones shows layer 1 remaining stable (~0.97) while layer 2 drops substantially (e.g., seed 123: 1.00 at 10k -> 0.53 at 30k -> 0.34 at 100k). The sharpest CKA drop coincides with the behavioral flip from goal-seeking to farming.

2. **Policy direction flips cleanly.** At goal-following checkpoints, ~30--40% of greedy actions point toward the goal and ~0% toward the reward. After the transition, 0% point toward the goal and ~45--52% toward the reward.

3. **Within-run probes are strong.** Once farming behavior appears, hidden layer 2 probes for the farming label achieve AUROC 0.93--1.00, stable across the remaining training. Before the transition, the label is not viable (no farming actions exist to predict).

4. **Activation magnitude grows.** Mean L2 norm of layer 2 activations increases roughly monotonically during training (from ~10 to ~45), reflecting increasing network confidence in its action-value estimates.

5. **Seed 123 is the most informative transition.** It has the longest goal-preferring phase (checkpoints 10k--20k goal-following, 25k transitional, 30k+ full exploitation), providing the clearest before/after comparison within a single run.

**Scientific conclusion:** Internal representations, particularly in hidden layer 2, undergo measurable restructuring that is temporally aligned with the behavioral transition from goal-seeking to reward farming. This restructuring is consistent across all three training seeds, though the timing varies. Within-run probes demonstrate that the network's learned features become increasingly aligned with farming-relevant spatial features during training.

**What this does NOT establish:**
- Probe accuracy does not prove a dedicated "reward-hacking representation." The network may simply be learning to represent spatial distances in a linearly accessible way, which a probe can then decode.
- CKA changes reflect representational geometry shifts, not necessarily semantic changes. The same CKA drop could accompany any significant behavioral change, not specifically reward hacking.
- No causal claim can be made: observing that activations predict behavior does not mean activations cause it. That is Phase 4's question.

**Commands:**
```bash
python -m scripts.collect_activations --checkpoint final
python -m scripts.probe_activations
python -m scripts.analyze_training_transition
python -m scripts.visualize_phase3
python -m pytest tests/test_phase3.py -q   # 27 passed
```

**Artifact paths:**
- `results/phase3/probe_results.json` -- probe AUROC/accuracy for all feature sets and seeds.
- `results/phase3/transition_analysis.json` -- per-checkpoint CKA, policy behavior, sparsity, within-run probes.
- `results/phase3/collection_summary.json`, `matched_observations.npy`, `observation_metadata.json`.
- `results/phase3/activations_*.npz` (gitignored) -- raw activation arrays.
- `results/phase3/transition_pca_seed*.npz` (gitignored) -- PCA projections for visualization.
- `assets/figures/phase3_activation_overview.png` -- DQN architecture diagram with capture points.
- `assets/figures/phase3_activation_heatmaps.png` -- representative hidden-layer activations at matched states.
- `assets/figures/phase3_probe_comparison.png` -- probe vs baseline performance comparison.
- `assets/figures/phase3_representation_projection.png` -- PCA of hidden activations.
- `assets/figures/phase3_activation_evolution.png` -- six-panel training-transition analysis.

**Tests:** 27 new tests in `tests/test_phase3.py` (activation tensor dimensions, post-ReLU correctness, extraction preservation of predictions, matched-state dataset construction, spatial split no-leakage, probe evaluation, CKA properties, PCA shapes, reproducibility, real-checkpoint integration) -- 115 total across the full suite, all passing.

## 9. Known Problems and Next Steps

### Known problems

- Three seeds is a preliminary stability check for one configuration, not broad evidence of generality.
- The "time to discover the exploit" varies substantially by seed (5k--30k steps) despite a deterministic environment; the cause (exploration trajectory vs. network initialization) is not yet investigated.
- The distance baseline achieves high AUROC (0.95--0.99), so the activation probe advantage is modest. The network has learned useful features, but the probe result alone does not demonstrate information beyond what the observation provides.
- Probe accuracy does not establish a dedicated "reward-hacking representation." The farming label is defined using spatial criteria, and the network's learned features happen to represent spatial information linearly. A probe on any network that has learned the task geometry would likely perform similarly.
- CKA changes during training are descriptive, not diagnostic. The layer 2 restructuring coincides with the behavioral transition, but causality is not established.
- Within-run probes become viable only after farming behavior appears. The probe cannot measure a "pre-farming" signal because there are no farming actions to label before the transition.

### Immediate next milestone

**Phase 4:** Causal activation interventions. Use the activation directions identified by Phase 3 probes (particularly in hidden layer 2) to test whether modifying representations during inference changes the agent's behavior -- e.g., whether suppressing farming-associated activation directions shifts the agent toward goal-seeking. Include random-direction and magnitude-matched controls.

### Maintenance checklist

At the end of each meaningful task:

1. Update project status in this document.
2. Record important design decisions and their rationale.
3. Record verified results only, including reproducibility metadata.
4. Document unresolved issues and limitations.
5. Identify the next logical step.
6. Keep historical information concise and avoid redundant session summaries.
