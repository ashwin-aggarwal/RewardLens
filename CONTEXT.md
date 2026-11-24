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

Intervene on selected activation directions or features and evaluate changes in behavior and true task success. Include random-direction and other matched controls to distinguish causal effects from generic perturbation effects. Phase 4.5 extends this to donor-to-recipient activation patching; Phase 4.6 adds the four-category behavioral metric (direct goal is the success criterion, not goal completion), a time-dependence analysis, and scheduled, stronger, layer-1 and time-shifted patches.

### Phase 5 — Technical blog post and reproducibility review

Present the research progressively and intuitively, including the environment, controls, learned behavior, activation analyses, interventions, limitations, commands, seeds, configurations, and artifact locations. Review whether another researcher can reproduce the claims from the repository.

*Revised 2026-11-23:* Phase 5 became low-rank weight editing (permanent repair). The write-up and reproducibility review are delivered as the Experiment 1 report in `experiments/01_fixed_gridworld_repair/`.

### Experiment structure (2026-11-23)

Phases 1–5 together form **Experiment 1: Fixed Gridworld Reward-Hacking Repair**, now complete. **Experiment 2: Generalizable Reward-Hacking Interventions** lives in `experiments/02_generalizable_reward_hacking/` (report, journal, configs, frozen layout manifest), with code in `rewardlens/exp2/` and scripts in `scripts/exp2_*.py`.

Experiment 2, Phase 1 (behavioral baselines) — complete, 2026-11-23, on `experiment/generalizable-reward-hacking`:
- Procedural 8×8 gridworld: start, goal and reward tile randomized per episode; same 7-number observation, rewards and horizon as Experiment 1; no obstacles.
- 10,000 distinct layouts split by whole layout 7,000 / 1,500 / 1,500 (train / dev / test), frozen in a hashed manifest before training. Training reads only the training split. The test split was evaluated once.
- Scripted check (γ = 0.99): farming beats goal-reaching in every layout under the flawed reward (margin ≥ 41 discounted points); the goal wins in every layout under the corrected reward.
- Per seed (42, 123, 456): pretrain 300k steps on the corrected reward, then fork good (corrected) and hacking (flawed) from the same checkpoint for 400k steps each. Final checkpoints, no selection.
- Test results: good DQN direct goal 1.000 on every seed; hacking DQN farming (farm-only + farm-then-finish) 0.923 / 0.921 / 0.919. Both 0.80 targets met. No gap between training, dev and test layouts, including the 282 test layouts with a goal/tile pair never seen in training.
- Hacking emerged after the fork in every seed, with farming ≥ 0.8 reached at 160k–220k steps. The first fork budget (200k) left seed 123 at 0.72 on dev; the budget was extended once to 400k for all seeds, from dev curves, before any test evaluation. The 200k results are kept.
- Decision: the 5-visit threshold stays primary (it separates scripted policies; a random policy trips it 5.7% of the time), with 10 and 20 reported as sensitivity checks.
- Limitations: unseen combinations from one generator, not new structure; 3 seeds; one reward magnitude and exploit type. The hacking agent still goes directly to the goal in about 8% of layouts, mostly when the goal is closer than the tile.
- Scope of the claim: generalization to unseen combinations of positions on an open grid. Not shown: new maze structures (no obstacles exist) or new exploit mechanisms.
- Finalization (2026-11-23): the nine final checkpoints (pretrain / good / hacking × seeds 42, 123, 456) are committed, with hashes and fork correspondence in `results/exp2/phase1/checkpoints.json` (`rewardlens.exp2.checkpoints.load_triplet`). Intermediate checkpoints and raw per-episode logs are local only.
- Evaluation integrity rule: the 1,500 test layouts were used once for Phase 1 baselines and must not be used to select representations, interventions or hyperparameters. Phase 2 development uses training and development layouts only. A fresh 1,500-layout final evaluation set, disjoint from all others, is frozen in `experiments/02_generalizable_reward_hacking/layouts/final_eval_manifest.json` and reserved for the final steering experiment. It has never been evaluated.

Experiment 2, Phase 2 (representation analysis, preliminary transplantation) — complete, 2026-11-23, uncommitted. Training and development layouts only; details in the Experiment 2 `PROGRESS.md`.
- Activations of all nine networks on 6,000 identical development states (6,000 training-layout states for fitting). Settings fixed in `configs/phase2.json`; one run, no tuning.
- Good stays near pretrained (layer-2 CKA 0.99, activation change 6–7%). Hacking diverges, mostly in layer 2 (CKA with pretrained 0.37–0.46; activation norm about 3× larger; fc2 and fc3 weights moved by about 1.8–2× their norm).
- The layer-2 (hacking − good) difference is 88–93% a constant offset; the state-dependent remainder needs 4–5 components for 90% of its variance, and these track time remaining and the tile's offset from the agent (R² 0.96 from simple state features).
- Across seeds (no raw averaging; CKA and held-out linear maps): hacking networks resemble each other (layer-2 CKA 0.77–0.86) and not the good ones (0.32–0.48). One seed's top-8 difference components are predicted from another's with R² 0.92–0.96, but equally well from the other seed's plain hacking activations, so this is shared encoding of task variables, not a demonstrated shared mechanism.
- Transplant good → hacking at α = 1 on 1,500 development layouts: farming drops to 0–7%, but direct goal reaches only 0.44 / 0.11 / 0.30 (layer 2) and "other" is 54–89%. Random-state donors give 0.27 / 0.13 / 0.19; a constant Q bias changes nothing. State-dependent benefit in seeds 42 and 456 only; none in seed 123. Layer 1 is worse; hacking → good breaks the good network without creating farming. Negative result for repair.
- Decision: similarity results are reported as correlational; the transplant was left untuned by design.
- Next: Phase 3 tracks the emergence of hacking across training checkpoints (observational). Intervention candidates are deferred to Phase 4: patching between fork checkpoints near the transition, patching only the state-dependent components, or a fitted linear translation between activation spaces. Not started.

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
- [x] Phase 4: causal activation interventions — complete, 2026-11-23. Probe-direction edits selectively suppress farming (vs norm-matched random controls) but do not restore goal completion; projection removal has no effect. Verdict: behavioral disruption, not redirection.
- [x] Phase 4.5: goal-directed activation patching — complete, 2026-11-23. Same-run patching (20k donor → 30k recipient) produces 100% goal completion that controls cannot reproduce, but the behavior is a time-triggered "farm-then-dash" (37–41 tile visits, goal reached at step 95–96), not clean goal pursuit. Cross-model patching fails under all three alignment variants.
- [x] Phase 4.6: behavioral reclassification and direct-goal steering — complete, 2026-11-23. Four behavior categories (direct goal / farm-then-finish / farm-only / other) applied retrospectively: the Phase 4.5 patch is 94% farm-then-finish. The time-gating mechanism is confirmed with Q-values and an input clamp. Schedules, α > 1 and layer-1 patches fail or sit inside the shuffled-control range; a time-shifted donor patch (α=1.0, donor sees remaining time 0.2) is direct-goal 1.00 on held-out, fit and all-62 start positions against shuffled-control means of 0.48–0.52.
- [x] Phase 4.7: time controls for the Phase 4.6 intervention — complete, 2026-11-23. Clamping the hacking network's own time input does nothing (farm-only ≥ 0.95); the donor is time-insensitive; the time-only activation difference (12% of the patch by norm) has no effect; the matched-time patch gives direct 0.55 with time-gated farm-then-finish for the rest; donor state representation plus low-time donor features together give 1.00. The donor's head cannot read the recipient's representation (other 0.81). Time alone does not explain Phase 4.6; the effect is an interaction between the donor's state-dependent representation and the clock the recipient's output head reads out of it.
- [~] Extension (branch `experiment/sparse-patching`, Phase 4.8): sparse causal activation patching — complete, 2026-11-23, merged to `main`. Masked replacement of layer-2 units with the Phase 4.7 donor (t = 0.2, α = 1). Finding: the effect is distributed, not localized. No subset of 32 or fewer units restores direct goal-seeking under any of three selection methods; the 64 units with the largest donor-recipient differences (effectively every unit that differs at all on dev states, 68 of 128) reproduce the full patch (direct 1.00 on the 46 non-selection starts and on all 62) against random 64-unit subsets at 0.05 (max 0.30) and shuffled-state donors at 0.50. Single units are necessary (restoring any one of six units collapses the full patch from 1.00 to 0.12 on dev) but not jointly sufficient (the top-64 by ablation importance reach 0.24), and greedy forward selection stalls at 0.12 because no single addition improves direct-goal completion. Details: `experiments/01_fixed_gridworld_repair/sparse_patching/README.md`.

- [~] Extension (branch `experiment/low-rank-patching`): low-rank activation patching — complete, 2026-11-23, merged to `main` (PR #1). The mean delta plus 8 PCA directions reproduce the full patch (direct 1.00 on 46 eval and all 62 starts) against random 8-dimensional subspaces at 0.02, and fail with shuffled-state donors (0.05), so the compact patch is state-specific. SVD k = 8 also reaches 1.00 but is state-agnostic (shuffled 0.98). PC1 (92% of variance) is behaviorally irrelevant; PC2–PC5 are necessary. Its planned Experiment B (permanent repair) became Phase 5. Details: `experiments/01_fixed_gridworld_repair/low_rank_patching/README.md`.

- [~] Phase 5 (branch `experiment/weight-editing`): low-rank weight editing — complete, 2026-11-23, uncommitted.
  - For seed 123, a rank-4 layer-2 weight edit trained by activation matching against the PCA patch turns the hacking DQN into a standalone direct goal-seeker. It reaches 1.00 on all 62 starts with 3/3 training seeds, against 0.02 for norm-matched random rank-4 changes.
  - It offers no behavioral advantage over conventional fine-tuning. Unconstrained layer-2 and whole-network fine-tuning on the same examples also reach 1.00, with equally sized and similarly low-stable-rank changes.
  - Transfer to seeds 42 and 456 fails because the runtime patch itself fails there (0.10, 0.03).
  - Interpretation: behavioral imitation of a patch through a compact edit, not discovery of the reward-hacking mechanism. Details: `experiments/01_fixed_gridworld_repair/weight_editing/README.md`.

Current milestone: Experiment 1 complete and merged to `main` (PR #2). Experiment 2 Phase 1 committed and pushed on `experiment/generalizable-reward-hacking`; Phase 2 complete there, uncommitted.

Documentation layout (revised 2026-11-23, on `experiment/weight-editing`): the root README is a short landing page listing experiments. Experiment 1 lives in `experiments/01_fixed_gridworld_repair/`: `README.md` (research report), `METHODS.md` (shared methods and limitations), `PROGRESS.md` (the full chronological journal, moved from the root, including the archived phase-by-phase README), and one folder per extension (`sparse_patching/`, `low_rank_patching/`, `weight_editing/`) with its report and config. The root `PROGRESS.md` is a short index. `SETUP.md` holds all installation and reproduction commands. Code, scripts, results and assets stay at their shared top-level locations; only the three runner config paths changed. The edited-model checkpoint `results/weight_editing/models/` is gitignored like the training checkpoints and regenerated by `scripts.run_weight_editing`. `requirements.txt` lists `torch` and `scikit-learn`, which the code always required but the file originally omitted.

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

### 2026-11-23 — Interventions are defined per-network, in raw activation coordinates

A probe fitted on standardized activations has its coefficients transformed back to raw coordinates (w_raw = w_std / scaler.scale_) and unit-normalized before use as an intervention direction; the sign convention (+direction = higher predicted farming) is verified against the fitting data. Directions are never transferred between independently trained networks. Direction fitting, strength selection, and final evaluation use disjoint spatial partitions of start states (30 train / 16 dev / 16 test positions, goal and reward tiles excluded).

### 2026-11-23 — Constant-vector edits before a linear readout are global Q biases

Because fc3 is linear, subtracting a constant vector alpha*v at the post-ReLU layer-2 output shifts every state's Q-values by the same constant, -alpha * W3 v. Such an edit is therefore a global action-preference bias, not a state-sensitive representational intervention. The state-sensitive variant is projection removal (h - (h.v)v), which is evaluated separately. This distinction is central to interpreting Phase 4.

### 2026-11-23 — Patching uses a fresh prespecified split and audited alignment

Phase 4.5 uses a new start-state partition (partition_seed=11: 30 fit / 16 dev / 16 test positions) so its test states were never used to select any Phase 4.5 design choice. Cross-model alignment maps are fitted on fit-position states only and audited on test states for how closely they reproduce the *recipient's own* activations: a map that reproduces the recipient cannot inject donor information, and is reported as circular rather than as evidence.

### 2026-11-23 — Goal completion must be read together with tile visits and timing

Phase 4.5 showed that a 100% goal-completion rate can coexist with ~40 reward-tile visits per episode. The project's operational hacking definition (≥5 visits AND goal not reached) does not flag this "farm-then-finish" pattern. All Phase 4.5 reporting therefore pairs goal rate with mean tile visits, episode length, and proxy reward; goal rate alone is not accepted as evidence of restored intended behavior.

### 2026-11-23 — Four behavior categories; direct-goal rate is the steering success metric

Phase 4.6 replaces goal rate as the success criterion with four mutually exclusive categories per completed episode: direct goal (goal reached, < threshold tile visits), farm-then-finish (goal reached, ≥ threshold visits), farm-only (goal not reached, ≥ threshold visits; identical to the original `is_reward_hacking` flag), and other/failure (goal not reached with few visits, or incomplete). The threshold defaults to the environment's exploit threshold (5) and is configurable. Original classifications are preserved alongside the new ones; no environment rewards, checkpoints, or historical result files are altered. Tile visits, steps to goal, path efficiency and proxy reward are reported regardless of category.

### 2026-11-23 — The shuffled-state donor control is the bar for direct-goal claims

At patch strengths α ≥ 1 the recipient's own layer-2 information is removed, and the recipient's output head reading a donor activation for a *random* state produces a state-agnostic drift (about 60% "Right" actions) that reaches the goal corner from roughly half of all start positions without farming. "Direct goal" outcomes at such strengths are therefore compared against the shuffled-state control's mean and maximum, and path efficiency (Manhattan distance / steps) is reported to separate navigation from drift.

### 2026-11-23 — Post-hoc interventions are disclosed and re-checked on untouched positions

Phase 4.6's block D (time-shifted donor) was designed after the time-dependence diagnostics, which were computed on the new test positions. The block's grid was prespecified in `configs/phase4_6.json` before it ran and selection used dev states only, but the test set is not naive for it. The selected intervention and all its controls were therefore re-evaluated on the 30 fit positions (used only as the shuffled-donor pool) and on all 62 valid starts; both are reported with the test result.

### 2026-11-23 — Input-feature interventions are decomposed exactly before being credited

When a patch changes both the source of an activation (donor vs recipient) and the input it is computed from (clamped vs real time), the two changes are separated by exact decomposition: `h_donor(s,τ) − h_rec(s,t) = [h_donor(s,t) − h_rec(s,t)] + [h_donor(s,τ) − h_donor(s,t)]`. Each term is evaluated alone, with its own random, shuffled-state and constant-bias controls, and the recipient's own input clamp is evaluated without any patch. A success that appears only for the sum is reported as an interaction, not attributed to either term. Phase 4.7 applies this to Phase 4.6's result.

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

### Phase 4 — Causal activation interventions (2026-11-23)

**Setup.** Interventions edit the post-ReLU layer-2 activations through a temporary forward hook; weights are never modified, hooks are removed after each evaluation, and `alpha=0` reproduces baseline Q-values exactly (tested). For each flawed-reward seed, a probe direction was fitted in that network's own raw activation space on matched states restricted to 30 train positions (coefficients destandardized, unit-normalized, sign verified). Strengths are relative to the network's typical layer-2 norm (~43-50) and were selected on 16 dev start positions; all reported results below are from 16 disjoint held-out test start positions (one deterministic greedy episode per start state — rates are exhaustive fractions, not stochastic samples). Controls: 20 fixed-seed random unit directions at matched perturbation norms, probe-direction addition, projection removal, and a contrastive mean-difference direction.

**Verified held-out results (per seed, at the dev-selected strength):**

| Seed | Rel. strength | Baseline goal/hack | Probe subtract goal/hack | Probe add goal/hack | Projection removal goal/hack | Random hack (mean±std, n=20) |
|---|---|---|---|---|---|---|
| 42 | 0.05 | 0.00 / 1.00 | 0.00 / **0.00** | 0.12 / 0.00 | 0.00 / 1.00 | 0.92 ± 0.21 |
| 123 | 0.10 | 0.00 / 1.00 | 0.06 / **0.00** | 0.00 / 0.00 | 0.00 / 1.00 | 0.51 ± 0.42 |
| 456 | 0.25 | 0.00 / 1.00 | 0.06 / **0.00** | 0.12 / 0.00 | 0.00 / 1.00 | 0.20 ± 0.32 |

Mean proxy reward under probe subtraction collapses from ~97.5 to ~0 (0.50 / 0.31 / −0.06); episodes still mostly run to the 100-step horizon with the agent wandering (greedy flip rate on matched states 0.36–0.63; ~6–9% of edited activation units pushed below zero — measured, not clipped).

**Outcome classification (project taxonomy): partial effect / behavioral disruption, NOT successful redirection.**

1. **Farming suppression is selective.** At the same perturbation norm where the probe direction drives hacking to 0.00, random directions leave substantial farming (0.92 mean at seed 42's tiny alpha of 2.4). The targeted direction is reliably outside the random-control distribution for farming disruption.
2. **Goal completion is NOT restored.** Goal rates under targeted edits (0.00–0.12) are at or near random-control levels. The agent stops farming and wanders; reward collapses to roughly the step penalty.
3. **Direction symmetry.** Adding the farming direction also abolishes farming (and in seeds 42/456 yields the same small goal rate as subtraction). A semantically meaningful "farming knob" should not act symmetrically; a global action bias in either direction breaking a movement loop does.
4. **Projection removal has no behavioral effect** (hacking stays 1.00 in all seeds). This is the state-dependent edit that actually deletes the per-state component the probe reads. The farming signal that is *decodable* along the probe direction is therefore not *causally necessary* through that direction — a clean negative causal result.
5. **Mechanism.** Subtracting a constant vector before the linear fc3 layer adds a fixed per-action Q bias at every state (verified analytically and in tests; for seed 123: roughly +0.8 Up, +1.4 Right, −2.6 Down, −4.4 Left). Breaking the farming loop therefore does not require state-sensitive representation surgery — a global action-preference tilt suffices, and the probe direction happens to encode an effective tilt at much smaller norms than random directions.
6. **Contrastive direction** (cosine 0.13–0.40 to the probe direction) matches probe-subtraction behavior in seeds 123/456 but has no effect in seed 42 — further evidence that multiple quite different directions can produce the same disruption through the global bias mechanism.

**Fragility control (corrected-reward DQN, seed 123, own activation space):** a farming direction is not identifiable from its own behavior (single-class labels — documented, not faked). Under random perturbations its goal rate degrades from 1.00 to 0.91 (rel 0.05), 0.55 (0.1), 0.25 (0.25), 0.13 (0.5), 0.05 (1.0). Both trained policies are fragile to generic layer-2 noise at moderate relative norms; the targeted effect on farming at rel 0.05–0.1 stands out against this backdrop.

**Before/after transition (seed 123, checkpoints 20k vs 30k, direction fitted at 30k in its own space; no within-checkpoint direction exists at 20k because it produces no farming actions):**

- 20k (goal-preferring; baseline goal 0.94): unaffected by the farming direction at rel ≤ 0.1 (goal stays 0.94, while randoms degrade it to 0.72 mean); collapses to 0.06 at rel ≥ 0.25 — generic disruption, no selective protection of goal behavior beyond small norms.
- 30k (freshly farming; baseline hack 1.00): robust to targeted subtraction up to rel 0.5; only at rel 1.0 does hacking drop (to 0.25, goal 0.06). Contrast with the 100k checkpoint, where rel 0.05–0.1 abolished farming entirely. The long-converged policy is far more sensitive to this direction than the freshly transitioned one — consistent with Phase 3's finding that representations and activation magnitudes keep reorganizing after the behavioral flip.

**Scientific conclusion.** The probe-identified direction has a real, selective, reproducible causal effect on farming behavior — but the effect is behavioral disruption through a state-independent Q bias, not redirection toward the intended task, and the state-dependent content along the direction is causally inert (projection removal changes nothing). We found no evidence of an isolable internal "reward-hacking representation" whose removal restores aligned behavior. This is a negative result for activation steering in this setting and is preserved as such.

**Commands:**
```bash
python -m scripts.run_interventions          # Experiments 1-3
python -m scripts.evaluate_interventions     # Experiment 4 (transition checkpoints)
python -m scripts.visualize_phase4
python -m pytest tests/test_phase4.py -q     # 21 passed
```

**Artifact paths:**
- `configs/phase4_interventions.json` — strengths, partitions, seeds.
- `results/phase4/intervention_results.json` — sweeps, held-out evaluations, random-control distributions, Q-value effects, direction provenance.
- `results/phase4/transition_interventions.json` — before/after checkpoint comparison.
- `assets/figures/phase4_intervention_diagram.png` (conceptual, labeled as such), `phase4_strength_sweep.png`, `phase4_random_controls.png`, `phase4_behavior_comparison.png`, `phase4_q_value_shift.png`.
- `assets/gifs/phase4_original_policy.gif`, `phase4_intervened_policy.gif` — same start state (1,7), same checkpoint, actual recorded actions.

**Tests:** 21 new tests in `tests/test_phase4.py` (hook lifecycle and exception safety, zero-strength equivalence, weight immutability, exact constant-Q-shift semantics, destandardization correctness, sign conventions, random-control reproducibility and norm matching, partition disjointness, rollout determinism, end-to-end mini evaluation) — 136 total across the full suite, all passing.

### Phase 4.5 — Goal-directed activation patching (2026-11-23)

**Question.** Phase 4's constant edits stopped farming but caused wandering. Can *state-dependent* layer-2 activations from a goal-following network, injected as `h_patched(s) = h_hack(s) + α (h_goal(s) − h_hack(s))` and read out by the recipient's own fc3, selectively restore goal completion?

**Setup.** Seed 123 throughout. Fresh partition (seed 11): 30 fit / 16 dev / 16 test start positions. α ∈ {0, 0.25, 0.5, 0.75, 1.0} swept on dev; α selected on dev (max goal rate); everything below is held-out test (one deterministic episode per start state). Controls at the selected α: unmodified recipient; 20 random-direction patches with per-state norm matched to the goal patch; 20 shuffled-state donor patches (donor sees a random fit-pool observation instead of the current one); constant Q-bias (the mean layer-2 delta over dev states, applied uniformly); the donor evaluated alone. Checkpoint behaviors verified on the test states before use (20k: goal 0.75, hack 0.00; 30k: goal 0.06, hack 0.94; 100k: goal 0.00, hack 0.94).

**Experiment A — same-run patching (donor 20k):**

| Recipient | α | Condition | Goal | Hack | Tile visits | Reward | Steps |
|---|---|---|---|---|---|---|---|
| 30k | — | unmodified | 0.06 | 0.94 | 45.8 | 91.2 | 93.9 |
| 30k | 0.75 | **matched goal patch** | **1.00** | **0.00** | **36.9** | 82.9 | 89.8 |
| 30k | 0.75 | constant Q-bias | 0.00 | 1.00 | 48.6 | 96.2 | 100 |
| 30k | 0.75 | random (n=20) | 0.03 ± 0.03 (max 0.06) | 0.97 | 23.2 | — | — |
| 30k | 0.75 | shuffled donor (n=20) | 0.16 ± 0.08 (max 0.31) | 0.67 | 8.4 | — | — |
| — | — | donor 20k alone | 0.75 | 0.00 | 0.2 | 7.7 | 29.9 |
| 100k | 1.0 | matched goal patch | 0.88 | 0.00 | 15.1 | 38.4 | 48.5 |
| 100k | 1.0 | random (n=20) | 0.33 ± 0.37 (max 1.00) | 0.26 | 2.8 | — | — |
| 100k | 1.0 | shuffled donor (n=20) | 0.36 ± 0.10 | 0.00 | 0.1 | — | — |
| 100k | 1.0 | constant Q-bias | 0.00 | 0.88 | 42.6 | 84.1 | 100 |

- **20k → 30k is a genuine, state-dependent, selective effect**: goal 1.00 against random max 0.06, shuffled max 0.31, and constant Q-bias 0.00. Shuffling the donor's input destroys most of the effect, so matching the donor activation to the current state matters. The injected delta is small (mean norm 6.0; donor/recipient cosine 0.99; donor norm 12.5 vs recipient 17.9). 73% of the ΔQ energy on test states is a constant, but that constant is nearly uniform across actions ([−2.3, −2.4, −2.7, −2.5]) and does not change the argmax — which is why the constant control does nothing and the state-dependent remainder carries the effect.
- **But the restored behavior is "farm-then-dash."** In 15 of 16 test episodes the patched agent makes 37–41 reward-tile visits and reaches the goal at step 95–96 of 100 (the exception starts adjacent to the goal and finishes in 2 steps; per-episode data in `results/phase4_5/same_run_30k_matched_test_episodes.json`). Proxy reward under the flawed spec drops from 91 to 83 (fewer visits plus the goal bonus); under the corrected spec it rises from about −1 to about +9. The patch does not make the agent *want* the goal; it appears to make the goal attractive only once little time remains. This evades the project's operational hacking definition and is not clean restoration of intended behavior.
- **20k → 100k (full replacement, α=1.0)** reaches goal 0.88 with 15 visits, but at this perturbation scale (norm 38, recipient norm 50) random directions span 0.00–1.00 goal rate. The matched result is above the random mean but inside the random range: weak evidence only.

**Experiment B — cross-model patching (donor: corrected-reward final; recipient: flawed-reward final; policies agree on 30.3% of matched states):**

| Mapping | Fit R² | Test residual to recipient (unaligned 37.9) | α | Goal | Hack | Visits | Note |
|---|---|---|---|---|---|---|---|
| identity | — | 37.9 | 1.0 | 0.00 | 0.00 | 0.1 | stuck/wandering: 73 no-move steps of 100 |
| ridge, all fit states | 0.933 | 2.1 | 0.75 | 0.06 | 0.56 | 27.4 | **circular**: M(h_goal) ≈ h_hack; patch norm only 2.1 |
| ridge, agreement states (n=39) | 0.878 | 3.8 | 0.75 | 0.00 | 0.00 | 0.3 | wandering; shuffled donors also give hack 0.00 → not state-specific |

Donor alone: goal 1.00. No cross-model variant restored goal completion. The failure modes are distinguishable: identity patching is coordinate/scale incompatibility (donor norm 14 vs recipient 50, cosine 0.90, output head cannot read it); ridge-all is an alignment objective that preserves the recipient rather than the donor (audited, as the design anticipated); ridge-agreement produces small disruptive perturbations without transferable goal content. Whether a *better* alignment could transfer goal information remains open — these results do not show that no transferable signal exists.

**Scientific conclusion.** State-dependent activations from an earlier checkpoint of the *same* training run causally and selectively change the hacking network's behavior in a way no control reproduces — the first intervention in this project with a state-dependent effect. However, the behavior produced is a time-triggered compromise (farm until the horizon nearly expires, then finish), not the donor's direct goal pursuit. Across independently trained networks, no tested alignment transferred goal-directed behavior. Predictive features and even same-run goal activations do not straightforwardly provide selective restoration of intended behavior.

**Commands:**
```bash
python -m scripts.run_phase4_5
python -m scripts.visualize_phase4_5
python -m pytest tests/test_phase4_5.py -q   # 18 passed
```

**Artifact paths:** `configs/phase4_5.json`; `results/phase4_5/patching_results.json`, `same_run_30k_matched_test_episodes.json`; `assets/figures/phase4_5_patching_diagram.png` (conceptual), `phase4_5_strength_sweep.png`, `phase4_5_controls.png`, `phase4_5_behavior_comparison.png`; `assets/gifs/phase4_5_original_hacking.gif`, `phase4_5_goal_donor.gif`, `phase4_5_patched.gif` (same start (1,1), actual recorded actions).

**Tests:** 18 new tests in `tests/test_phase4_5.py` (α=0 identity, α=1 full replacement, interpolation, donor sees the identical observation, post-ReLU capture, weight immutability, hook removal incl. on exception, norm-matched random reproducibility, shuffled-donor reproducibility, constant mode as exact Q shift, ridge mapping recovery, split disjointness, mapping fit excludes test positions, actual-trajectory rollouts, real-checkpoint behavior) — 154 total, all passing.

### Phase 4.6 — Behavioral reclassification and direct-goal steering (2026-11-23)

**Question.** Phase 4.5's 100% goal completion hid ~40 tile visits per episode. Which of the project's results survive a metric that separates "reached the goal" from "pursued the goal"? Is the late dash caused by a time-gated goal preference? And can any modification of the patch make the 30k hacking checkpoint pursue the goal directly?

**Setup.** Seed 123, donor 20k, recipient 30k (as in Phase 4.5). Fresh prespecified partition, seed 13 (30 fit / 16 dev / 16 test; the new test set overlaps the seed-11 split by 2 old-test, 7 old-dev and 7 old-fit positions, unavoidable with 62 positions). Selection on dev by direct-goal rate (tie-break: fewer visits, then less total intervention). Controls under the identical schedule, layer and donor transform: unmodified recipient, donor alone, 20 norm-matched random patches, 20 shuffled-state donor patches, constant Q-bias (mean dev delta). Visit threshold 5.

**Retrospective reclassification** (`results/phase4_6/reclassification.json`; stored trajectories classified directly, others regenerated deterministically with equality asserted against the stored summaries):

| Trajectories | Direct goal | Farm-then-finish | Farm-only | Other |
|---|---|---|---|---|
| Phase 2 flawed DQNs, final (3 seeds, 62 starts) | 0.00–0.02 | 0.00 | 0.97–1.00 | 0.00–0.02 |
| Phase 2.5 corrected DQNs, final (3 seeds, 62 starts) | 1.00 | 0.00 | 0.00 | 0.00 |
| Phase 4 probe-subtract, test (seeds 42/123/456) | 0.00/0.06/0.06 | 0.00 | 0.00 | 1.00/0.94/0.94 |
| Phase 4.5 same-run 20k→30k, α=0.75, test | 0.06 | **0.94** | 0.00 | 0.00 |
| Phase 4.5 same-run 20k→30k, α=1.0, same test | 0.38 | 0.62 | 0.00 | 0.00 |
| Phase 4.5 same-run 20k→100k, α=1.0, test | 0.38 | 0.50 | 0.00 | 0.12 |
| Phase 4.5 cross-model identity / ridge-agreement | 0.00 | 0.00 | 0.00 | 1.00 |
| Phase 4.5 cross-model ridge-all | 0.06 | 0.00 | 0.56 | 0.38 |

Phase 4.5's selection rule (max goal rate, tie → smaller α) picked α=0.75 over α=1.0 although α=1.0 had fewer visits on dev (16.5 vs 36.4) and six times the direct-goal rate on test. Classification of the farm-then-finish episodes is unchanged for thresholds 5–20 (they have 35–41 visits).

**Time dependence** (same 16 test positions × 21 remaining-time values; `results/phase4_6/steering_results.json` → `time_dependence`):
- Unmodified recipient: goal-minus-reward Q gap −0.39 at t=1.0 → −0.20 at t=0.0, never positive; greedy goal-approach fraction 0.69 → 0.56.
- Patched α=0.75: gap +0.09 → +0.30 and goal-approach 0.69 → 1.00, rising monotonically as time runs out. Exact affine decomposition `Q_patched = (1−α)Q_rec + α fc3_rec(h2_donor)` (max error 5e-6) shows the moving part is the recipient's term (mean max-Q 13.6 → 10.8), while the donor term through the recipient's head is goal-preferring at every time (gap +0.34 → +0.52). The donor's own head is time-insensitive (goal-approach 0.94 throughout).
- Input clamp (network sees a fixed remaining time; environment unchanged), patched α=0.75: clamp 1.0 → farm-only 0.94 (no dash ever); 0.5 → farm-only 0.50 / other 0.44; 0.1 → direct 0.25; **0.05 and 0.0 → direct goal 1.00 in ~6 steps, zero visits**. Unmodified recipient at clamp 0.0: farm-only 1.00. Donor: 0.69 at every clamp. Mechanism confirmed: the patch installs a goal preference gated by the time feature.

**Steering blocks** (held-out test, 16 states; full table and dev sweeps in PROGRESS.md):

| Intervention | Direct goal | Farm-then-finish | Other | Visits | Path eff. | Shuffled ctrl (mean / max) | Random ctrl | Q-bias |
|---|---|---|---|---|---|---|---|---|
| Phase 4.5 patch, layer 2, α=0.75 | 0.06 | 0.94 | 0.00 | 36.6 | — | 0.10 / 0.25 | 0.04 | 0.06 |
| A. best schedule = constant, α=1.0 | 0.56 | 0.44 | 0.00 | 16.5 | 0.49 | 0.81 / 0.94 | 0.05 | 0.06 |
| B. α=1.25 | 0.62 | 0.38 | 0.00 | 13.6 | 0.49 | 0.55 / 0.81 | 0.06 | 0.06 |
| C. layer 1, α=2.0 | 0.12 | 0.00 | 0.75 | 1.6 | — | 0.08 / 0.19 | 0.04 | 0.00 |
| D. α=1.0, donor sees t=0.2 | **1.00** | 0.00 | 0.00 | 0.4 | **0.97** | 0.52 / 0.75 | 0.05 | 0.06 |

- A: early-only and decaying schedules let farming resume as soon as the patch ends (dev farm-only 0.56–0.88); late schedules reproduce farm-then-finish. Negative.
- B: α ≥ 1.5 pushes 6–21% of units negative and produces wandering; α=1.25 is inside the shuffled-control range. Not selective.
- C: no effect below α=1.5, wandering at α=2.0. Negative.
- Shuffled-state donors at α ≥ 1 reach the goal from about half of starts by a state-agnostic drift (58–60% Right actions, path efficiency 0.3–0.4, ~28 no-move steps). This, not the unmodified recipient, is the bar.
- D: selected on dev (all α=1.0 variants with donor time ≤ 0.2 gave direct 1.00 in 5.7 steps; α=0.75 variants abolished farming but wandered, path efficiency 0.18). Test: direct 1.00, 6.1 steps, efficiency 0.97, 0 no-move steps, 0 negative units, corrected-spec reward 10.7; margins +0.48 over the shuffled mean and +0.25 over the shuffled maximum. Re-evaluated on the 30 fit positions: 1.00 (efficiency 0.92) vs shuffled 0.49 (max 0.67), random 0.01, donor alone 0.83, Phase 4.5 patch 0.00; on all 62 starts: 1.00 (efficiency 0.96) vs shuffled 0.48 (max 0.61), donor alone 0.76.

**Scientific conclusion.** Direct-goal steering improved from 0.06 (Phase 4.5) to 1.00 on held-out, fit and all-62 positions, and the improvement is state-dependent (shuffled donors reach ~0.5 with inefficient paths). Three qualifications are part of the result: (i) at α=1.0 the recipient's layer-2 activation is fully replaced, so the recipient contributes only its output head; the hacking network's middle layer was not steered, it was substituted; (ii) the donor activation is computed for a counterfactual input (current position, remaining time 0.2), so it is not the donor's activation for the recipient's observation; (iii) one seed, one donor/recipient pair. The mechanistic reading is that the recipient's output head reads a time signal out of layer 2 and is willing to pursue the goal when that signal says the deadline is near; the Phase 4.5 patch supplied that signal only at the real end of the episode, and block D supplies it from step one.

**Commands:**
```bash
python -m scripts.reclassify_trajectories
python -m scripts.run_phase4_6
python -m scripts.visualize_phase4_6
python -m pytest tests/test_phase4_6.py -q   # 29 passed
```

**Artifact paths:** `configs/phase4_6.json`; `results/phase4_6/reclassification.json`, `steering_results.json`; `assets/figures/phase4_6_behavior_categories.png`, `phase4_6_visits_vs_completion.png`, `phase4_6_time_dependent_q.png`, `phase4_6_steering_comparison.png`, `phase4_6_trajectory_comparison.png`; `assets/gifs/phase4_6_original_hacking.gif`, `phase4_6_farm_then_finish.gif`, `phase4_6_new_intervention_direct_goal.gif`, `phase4_6_shuffled_control.gif` (same start (0,4), recorded actions). Code: `rewardlens/analysis/behavior_classification.py`, `steering.py`, `time_dependence.py`.

**Tests (Phase 4.6):** 29 new tests in `tests/test_phase4_6.py` (all four categories, threshold boundaries and configurability, incomplete trajectories, farm-only ≡ original flag, scripted trajectories, aggregate rates and rows, retrospective file consistency, schedule shapes, hook follows the step counter, layer-2 α=1 semantics, layer-1 routing through the recipient's fc2, α > 1 affine extrapolation, weight immutability and hook removal on both layers, norm-matched random under schedules, constant mode at layer 1, shuffled reproducibility, rollout bookkeeping, observation transform touches only the network input, evaluation aggregates, fresh partition disjointness, time grid, action geometry, synthetic time-preference policies, exact affine decomposition, time clamp, real-checkpoint farm-then-finish) — 183 total, all passing.

### Phase 4.7 — Controlling for time in activation patching (2026-11-23)

**Question.** Does Phase 4.6's time-shifted patch (α = 1.0, donor's time input clamped to 0.2) transfer goal-directed representation, or does it succeed by manipulating the agent's representation of remaining time?

**Setup.** Donor 20k, recipient 30k, seed 123. No selection or tuning; all conditions prespecified in `configs/phase4_7.json`. The patch is decomposed exactly into a matched-time term and a time-only term (see decision above). The environment's step counter and truncation are unaffected by any clamp (tested). **No untouched validation states exist**: all 62 start positions were used in Phase 4.6, so results are exhaustive over all 62, with the 16 Phase 4.6 test states reported separately (differences ≤ 0.1 everywhere). Phase 4.6's block D is reproduced exactly by this code path.

**Results (all 62 starts):**

| Condition | Direct | Farm-then-finish | Farm-only | Other | Visits | Path eff. |
|---|---|---|---|---|---|---|
| A unmodified recipient | 0.05 | 0 | 0.95 | 0 | 46.4 | — |
| B recipient time input clamped (0.2; same at 0.1, 0.05, 0.0) | 0.02 | 0 | 0.98 | 0 | 47.9 | — |
| C donor, any time input | 0.76 | 0 | 0 | 0.24 | 0.2 | 1.0 |
| D matched-time patch α = 1.0 | 0.55 | 0.45 | 0 | 0 | 17.1 | 0.65 |
| E time-shifted patch, donor t = 1.0 / 0.5 | 0.34 / 0.66 | 0 | 0.45 / 0.34 | 0.21 / 0 | 22 / 17 | 1.0 |
| **E time-shifted patch, donor t = 0.2** | **1.00** | 0 | 0 | 0 | 0.3 | 0.96 |
| F time-only delta (any t, α = 1.0 or 0.75) | 0.02–0.05 | 0 | ≥ 0.95 | 0 | 46–48 | — |
| G donor head reads recipient layer 2 | 0.00 | 0 | 0.19 | 0.81 | 7.6 | — |
| Controls at E: random / shuffled / Q-bias | 0.04 / 0.48 (max 0.61) / 0.02 | — | — | — | — | — / 0.44 / — |
| Controls at F: random / shuffled / Q-bias | 0.03 / 0.03 / 0.05 | — | — | — | — | — |

**Q-values at identical positions (62 × 11 time inputs):** A and B prefer the reward tile at every time (gap −0.56 / −0.41; B changes 0.08 of greedy actions). C is goal-preferring and flat (0.95). D is time-gated (goal-approach 0.77 → 0.98 as time runs out): the recipient's output head reads a clock out of the donor's features. E is flat at 0.90 with gap +0.36. F tracks A (gap −0.37, 0.11 of actions changed). Magnitudes: matched-time delta 6.10, donor time delta 0.78, full 6.60; the recipient's own time delta has norm 1.44 and cosine 0.85 with the donor's.

**Answers.** (1) Time input alone to the hacking DQN: no effect. (2) The donor behaves identically under every clock. (3) The matched-time patch succeeds partially (0.55) and is time-gated. (4) The transferred information is state-dependent: shuffled-state donors reach 0.48 with path efficiency 0.44 and a one-direction-heavy histogram; the matched patch reaches 1.00 with efficiency 0.96 and a spread histogram. (5) The time-only difference reproduces none of the effect, at either strength and every clamp. (6) The recipient's output head is the reader that makes this work; it gates the donor's goal preference by the donor's clock features, and the donor's own head cannot read the recipient's features at all.

**Scientific conclusion.** Time alone does not explain Phase 4.6. The donor's state-dependent representation is necessary and accounts for the Phase 4.5-style partial success; the counterfactual clock removes a gate in the recipient head's readout of that representation; neither component works alone. This is not evidence of transferring a dedicated goal representation: the patch replaces the recipient's entire layer-2 representation, and its success is an interaction with the recipient head's time gate.

**Commands:**
```bash
python -m scripts.run_phase4_7
python -m scripts.visualize_phase4_7
python -m pytest tests/test_phase4_7.py -q   # 14 passed
```

**Artifact paths:** `configs/phase4_7.json`; `results/phase4_7/time_control_results.json`; `assets/figures/phase4_7_time_sweep.png`, `phase4_7_control_comparison.png`, `phase4_7_q_value_analysis.png`, `phase4_7_trajectory_comparison.png`. No new GIFs (the time-clamped recipient's episode is indistinguishable from the unmodified one; the patch episode is the Phase 4.6 GIF). Code: `make_time_delta_donor` in `rewardlens/analysis/steering.py`; `compare_to_reference` and greedy/Q matrices in `time_dependence.py`.

**Tests (Phase 4.7):** 14 new tests in `tests/test_phase4_7.py` (numpy and tensor clamps touch only the time feature, environment horizon unchanged under the clamp, step counter independent of network input, time-delta formula, exact matched-plus-time decomposition, zero delta when the clamp equals the real time, time-only patch through the hook, shuffled time-delta reproducibility, reference comparison, deterministic evaluation, four-category classification, exact reproduction of the Phase 4.6 intervention, recipient-clamp-still-farms on real checkpoints) — 197 total, all passing.

## 9. Known Problems and Next Steps

### Known problems

- Three seeds is a preliminary stability check for one configuration, not broad evidence of generality.
- The "time to discover the exploit" varies substantially by seed (5k--30k steps) despite a deterministic environment; the cause (exploration trajectory vs. network initialization) is not yet investigated.
- The distance baseline achieves high AUROC (0.95--0.99), so the activation probe advantage is modest. The network has learned useful features, but the probe result alone does not demonstrate information beyond what the observation provides.
- Probe accuracy does not establish a dedicated "reward-hacking representation." The farming label is defined using spatial criteria, and the network's learned features happen to represent spatial information linearly. A probe on any network that has learned the task geometry would likely perform similarly.
- CKA changes during training are descriptive, not diagnostic. The layer 2 restructuring coincides with the behavioral transition, but causality is not established.
- Within-run probes become viable only after farming behavior appears. The probe cannot measure a "pre-farming" signal because there are no farming actions to label before the transition.
- Phase 4's constant-vector edits are equivalent to global per-action Q biases because the readout layer is linear; they cannot distinguish "removed a farming representation" from "tilted all action preferences." The one state-sensitive edit tested (projection removal) had no behavioral effect. More surgical interventions (per-state clamping, layer-1 edits, weight-space analysis) were not explored.
- With 16 held-out start states per condition and a deterministic environment/policy, reported rates are exhaustive over those states but coarse-grained (1/16 resolution); no sampling-based uncertainty applies.
- The dev-selected strength differs per seed (0.05/0.1/0.25); the qualitative pattern (farming suppressed, goal not restored) holds at every strength ≥ the threshold in the dev sweeps, but strength-transfer across seeds was not evaluated.
- Phase 4.5's same-run result is from one seed (123) and one donor/recipient pair with a clear transition; it was not replicated on seeds 42/456 (whose early checkpoints show weaker or no goal-preferring phase). The farm-then-dash mechanism (goal becomes attractive only at low remaining time) is a hypothesis inferred from the timing pattern, not a verified mechanism.
- The operational hacking definition does not flag farm-then-finish behavior; any future metric should penalize tile visits even when the goal is eventually reached.
- Cross-model alignment was limited to ridge regression on 150 (or 39 agreement) states; richer alignment methods were not tried, so "no transferable goal signal" is not established.
- Phase 4.6's time-shifted donor result uses α=1.0 (full replacement of the recipient's layer-2 activation) and a counterfactual donor input (remaining time clamped to 0.2). It demonstrates that the recipient's output head produces direct goal pursuit from such activations; it does not demonstrate steering of the recipient's own representation. The α=0.75 variants, which keep the recipient's representation, abolish farming but wander (path efficiency 0.18).
- Block D was designed after diagnostics on the new test positions. Mitigation: re-evaluation on the 30 fit positions and on all 62 starts (both direct-goal 1.00 against shuffled means of 0.48–0.49). A replication on a genuinely untouched set is impossible in a 62-position grid; replication across seeds is the real test and was not done.
- The four behavior categories are threshold-based and operational. They are insensitive to the threshold over 5–20 for the trajectories here, but "direct goal with 4 visits" would still count as direct.
- With only 62 start positions, every "fresh" split overlaps earlier ones; the Phase 4.6 test set shares 2 positions with the Phase 4.5 test set and 7 with its dev set. After Phase 4.6 no untouched start position remains; Phase 4.7 therefore reports exhaustively over all 62 and performs no selection.
- Phase 4.7 locates the time gate in the recipient head's readout of donor features but does not identify which donor units carry it, and the head asymmetry (recipient head reads donor trunk; donor head cannot read recipient trunk) is observed, not explained.

### Immediate next milestone

Phase 3 (not started): track when hacking emerges across the saved training checkpoints and how the hidden layers change around that transition (observational). Phase 4 then looks for a transplant or edit the hacking network can read. Both are developed on training and development layouts only. The test split is not available for selection, and the reserved final evaluation set is opened only for the final steering experiment after its method is frozen. Keep the pretrained checkpoint as a reference, since it is behaviorally identical to the good agent.

### Maintenance checklist

At the end of each meaningful task:

1. Update project status in this document.
2. Record important design decisions and their rationale.
3. Record verified results only, including reproducibility metadata.
4. Document unresolved issues and limitations.
5. Identify the next logical step.
6. Keep historical information concise and avoid redundant session summaries.
