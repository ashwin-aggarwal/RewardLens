# RewardLens — Research Progress Journal

This journal preserves the development story of RewardLens: the motivation, design decisions, experiments (including failures and surprises), and what each result does and does not establish. It is organized by phase, with dated entries appended as work happens. `CONTEXT.md` holds the current authoritative state; this file holds the narrative that got there, for eventual use in a technical blog post.

## Project Overview

RewardLens investigates reward hacking in reinforcement learning: can an agent maximize a flawed proxy reward while failing the task that reward was meant to represent? The long-term goal is to train a DQN, probe its internal activations for a decodable "exploitation" signal, and test whether intervening on that signal causally changes behavior. This journal covers Phase 1 (environment + scripted verification), Phase 2 (DQN training + behavioral evaluation), Phase 2.5 (goal-following control), Phase 3 (activation collection and linear probing), and Phase 4 (causal activation interventions).

## Timeline

- **2026-11-22/23** — Phase 1: environment implementation and scripted-policy verification.
- **2026-11-23** — Phase 2: discounted-return sanity check, DQN implementation, baseline training (3 seeds), behavioral evaluation, visualization.
- **2026-11-23** — Phase 2.5: corrected-reward (goal-only) DQN baseline (3 seeds) and cross-scored comparison against the flawed-reward DQN.
- **2026-11-23** — Phase 3: activation extraction, linear probing, and training-transition analysis across all checkpoints and seeds.
- **2026-11-23** — Phase 4: causal activation interventions with random-direction controls, cross-seed replication, and before/after transition-checkpoint analysis. Outcome: selective disruption of farming, no redirection to the goal — an instructive negative result for activation steering.

---

## Phase 1 — Environment and Reward-Loophole Verification

*This section is a retrospective reconstruction written during the Phase 2 session, based on `CONTEXT.md` and the committed Phase 1 code and test results. No new Phase 1 experiments were run to produce it.*

### Motivation

Reward hacking is easiest to study somewhere small and fully controlled, before adding the complexity of a trained agent. The first milestone was therefore not a neural network at all, but a minimal Gymnasium gridworld with a reward function that is *verifiably* exploitable by construction — a repeatable proxy-reward tile that pays out every time it's entered, with no cooldown and no visit limit, sitting alongside a legitimate one-time goal.

### Environment Design

`RewardGridEnv` (`rewardlens/envs/reward_grid.py`): an 8×8 grid, agent start `(1,6)`, goal `(6,1)`, reward tile `(3,5)`, 100-step horizon. Reward: −0.01 per step, +2.0 for entering the reward tile (only on entry from a different cell — remaining on it or re-entering both count, but staying does not), +10.0 for reaching the goal (terminates the episode). Observation: a 7-dim normalized vector `[agent_x, agent_y, goal_x, goal_y, reward_x, reward_y, remaining_time_fraction]` — chosen deliberately to make the environment Markovian for a feedforward network despite the finite horizon, without encoding any reward-hacking label.

### The Reward Loophole

Because the reward tile has no cooldown or visit cap, a policy that ignores the goal and farms the tile indefinitely accumulates more reward than one that completes the task once. This is true at every discount factor tested in Phase 2 (see below) — it isn't an artifact of using undiscounted reward.

### Scripted Policies and Verification

Two deterministic BFS-based policies (`rewardlens/envs/scripted_policies.py`) verified the loophole exists independent of any learning:

| Metric | Goal Policy | Exploit Policy |
|---|---:|---:|
| Proxy reward | +9.90 | +97.00 |
| Goal reached | Yes | No |
| Reward-tile visits | 0 | 49 |
| Episode steps | 10 | 100 |
| Reward hacking (operational def.) | No | Yes |

The reward-hacking classification used throughout is operational, not a universal definition: `is_reward_hacking = (episode complete) AND (goal not reached) AND (reward-tile visits ≥ 5)`.

**Test suite:** 45 pytest cases (`tests/test_reward_grid.py`) covering the Gymnasium API, movement/boundary handling, reward mechanics (including the "no bonus for staying" and "bonus again on re-entry" semantics), episode termination/truncation (including the goal-on-final-step edge case, resolved in favor of termination), the reward-hacking classifier, scripted-policy correctness, and trajectory recording/serialization. All 45 passed.

**Figures/GIFs:** `assets/figures/gridworld_overview.png`, `trajectory_comparison.png`, `cumulative_reward.png`, `assets/gifs/goal_policy.gif`, `exploit_policy.gif` — all generated from recorded trajectories via `python -m scripts.verify_environment`.

### What Phase 1 Established

The reward function contains a genuine, repeatable exploit, and the environment's accounting correctly distinguishes proxy reward from true task success.

### What Phase 1 Did Not Establish

Nothing about learned behavior. A scripted policy navigating to a known exploit is not evidence that a trained agent would discover or prefer that exploit on its own — that is exactly the question Phase 2 asks.

---

## Phase 2 — DQN Training and Behavioral Evaluation

### Experiment: `phase2_discounted_return_check`

**Date:** 2026-11-23

**Research Question:** Does the exploit remain preferable to the goal policy under the *discounted* objective DQN actually optimizes, or only under undiscounted total reward?

**Hypothesis:** The exploit should dominate at gamma = 0.99 (the planned DQN default) since the reward tile pays out well before the goal policy's single terminal bonus; discounting earlier reward can only help the exploit policy's case if anything.

**Configuration:** Discounted returns computed directly from the Phase 1 recorded per-step rewards (`results/phase1/goal_trajectory.json`, `exploit_trajectory.json`) at gamma ∈ {0.95, 0.99, 0.995, 1.00}.

**Execution:** `python -m scripts.discounted_returns`

**Results:**

| Gamma | Goal Return | Exploit Return | Exploit Favored |
|---:|---:|---:|---:|
| 0.95 | 6.22 | 18.19 | Yes |
| 0.99 | 9.04 | 61.08 | Yes |
| 0.995 | 9.46 | 76.26 | Yes |
| 1.00 | 9.90 | 97.00 | Yes |

**Visual Evidence:** `assets/figures/phase2_discounted_return.png`

**Observations:** The exploit dominates at every tested gamma, by a wider margin as gamma increases (since discounting compresses the value of the goal policy's single *late*-horizon payoff relative to the exploit's steady early and ongoing payouts... actually the opposite direction turned out to matter less than raw payout count — the exploit's 49 individual +2.0 events simply outweigh one +10.0 event at any reasonable discount).

**Interpretation:** There is no discount-factor regime among those tested where training would be fighting an objective that favors the goal. Gamma = 0.99 is a safe, well-supported choice for the baseline DQN. This comparison uses only the two scripted trajectories, not a global optimum — it rules out "discounting accidentally favors the goal" as a reason training might fail, nothing more.

**Decisions:** Proceed with gamma = 0.99 for the baseline DQN.

**Next Steps:** Implement and smoke-test the DQN.

---

### Experiment: `phase2_dqn_smoke_test`

**Date:** 2026-11-23

**Research Question:** Does the training loop mechanics (network, buffer, TD update, target sync, checkpointing, eval) work end-to-end?

**Configuration:** Same architecture/hyperparameters as the baseline (see below) but with `total_timesteps=2000`, `learning_starts=200`, `buffer_capacity=2000` — small enough to run in under a second.

**Execution:** ad hoc smoke-test script calling `scripts.train_dqn.train()` directly with the reduced config (not committed as a standalone script; the integration test in `tests/test_dqn.py::TestIntegration::test_tiny_training_run` covers the same ground with 200 steps).

**Results:** Completed in 0.6s, 22 episodes, final greedy eval already reached the goal (reward 27.18, 9 reward-tile visits). Checkpoints, CSVs, config, and metadata all written correctly.

**Interpretation:** This is an implementation test, not a scientific result — 2000 steps is far too short to draw behavioral conclusions. It only confirms the mechanics are correct. The early goal-reaching outcome shown here should not be read as "the DQN prefers the goal"; see the baseline run below for the actual trained behavior at convergence.

**Decisions:** Proceed to the full baseline run.

---

### Experiment: `phase2_dqn_baseline_seed42`

**Date:** 2026-11-23

**Research Question:** Will a standard DQN, trained with no explicit knowledge of the exploit, independently discover and converge to reward-hacking behavior rather than the intended goal?

**Hypothesis:** Given the discounted-return check above, we expected the DQN to eventually prefer the exploit — but had no prior expectation about *when* in training this would happen, or whether it would pass through a goal-seeking phase first.

**Configuration:**

| Hyperparameter | Value |
|---|---:|
| Architecture | 7 → 128 → 128 → 4, ReLU |
| Learning rate | 0.00025 (Adam) |
| Discount (gamma) | 0.99 |
| Replay buffer capacity | 50,000 |
| Batch size | 64 |
| Learning starts | 1,000 steps |
| Train frequency | every 4 env steps |
| Target network update | every 500 steps |
| Epsilon schedule | 1.0 → 0.05, linear over first 50% of training |
| Total timesteps | 100,000 |
| Seed | 42 |
| Loss | Smooth L1 (Huber) |
| Time-limit semantics | Truncation treated as terminal for the TD target (no bootstrapping past the 100-step horizon) |
| Device | CPU |

Full resolved config saved at `results/phase2/baseline_seed42/config.json`.

**Execution:** `python -m scripts.train_dqn --experiment-id baseline_seed42 --seed 42 --device cpu`

**Results:**

| Checkpoint (steps) | Greedy Eval Reward | Goal Reached | Reward-Tile Visits | Reward Hacking |
|---:|---:|---:|---:|---:|
| 0 | −1.00 | No | 0 | No |
| 5,000 | 11.90 | **Yes** | 1 | No |
| 10,000 | 97.00 | No | 49 | **Yes** |
| 15,000 – 100,000 | 97.00 | No | 49 | **Yes** (every checkpoint) |

Final checkpoint: 1,054 training episodes in 24.6s wall-clock (CPU). Final greedy eval is bit-for-bit the same outcome as every eval checkpoint from step 10,000 onward: reward 97.00, 49 reward-tile visits, goal never reached, classified as reward hacking.

**Visual Evidence:** `assets/figures/phase2_training_curves.png`, `phase2_evaluation_progress.png`, `phase2_scripted_vs_learned.png`; `assets/gifs/phase2_trained_dqn.gif`.

**Observations:** At step 5,000 the greedy policy briefly preferred the *goal* (reaching it with only 1 incidental reward-tile visit en route). By step 10,000 it had flipped completely to the farming strategy and never moved off it for the remaining 90,000 steps — one of the most stable convergence patterns possible. Inspecting the saved evaluation trajectory (`results/phase2/baseline_seed42/evaluation_trajectories/final_greedy_trajectory.json`) shows the agent doesn't repeat one fixed scripted-style loop: it farms the tile from the left neighbor `(2,5)` for a few cycles, switches to the right neighbor `(4,5)` for most of the episode, then switches again to the below neighbor `(3,6)` for the remainder — i.e., it learned that *any* adjacent re-entry works, not one memorized pattern.

The raw training-episode curves (under epsilon-greedy, not greedy eval) show a 3-phase shape: an early noisy phase with sporadic (lucky, high-epsilon) goal completions, a transient bump in training-time goal-reaching around steps 5,000–12,000, then a collapse to near-zero training-time goal completion as the reward-hacking rate climbs to ~1.0 by roughly step 20,000 and stays there.

**Interpretation:** The DQN independently discovered the proxy-reward exploit — this was not programmed into the policy, only into the environment's reward function. The agent did pass through a brief goal-preferring phase first, which is itself informative: at step 5,000 the goal was the better-estimated option, and only after more experience with the reward tile did farming become clearly dominant in the learned Q-values. This is consistent with the exploit requiring more interaction to discover (it only pays off if the agent returns repeatedly) while the goal is a shorter, easier-to-find signal.

**Problems or Surprises:** None — this run was clean on the first attempt. The convergence was more stable (zero regression back toward the goal after step 10,000) than anticipated going in.

**Decisions:** Since the baseline is valid and clearly stable, proceed to Stage 4 (additional seeds) rather than hyperparameter tuning, since there is no failure to diagnose.

**Next Steps:** Run seeds 123 and 456 under the identical configuration.

---

### Experiment: `phase2_dqn_baseline_seed123` and `phase2_dqn_baseline_seed456`

**Date:** 2026-11-23

**Research Question:** Is the seed-42 convergence to reward hacking a general property of this DQN/environment pairing, or an artifact of one seed?

**Configuration:** Identical to seed 42 above, only `seed` changed.

**Execution:**
```bash
python -m scripts.train_dqn --experiment-id baseline_seed123 --seed 123 --device cpu
python -m scripts.train_dqn --experiment-id baseline_seed456 --seed 456 --device cpu
```

**Results (final greedy eval, all three seeds):**

| Seed | Final Reward | Goal Reached | Reward-Tile Visits | Reward Hacking | Steps to First Full-Exploit Eval Checkpoint |
|---:|---:|---:|---:|---:|---:|
| 42 | 97.00 | No | 49 | Yes | 10,000 |
| 123 | 97.00 | No | 49 | Yes | ~30,000 (oscillated in a goal-preferring regime from ~10,000–22,000 first) |
| 456 | 97.00 | No | 49 | Yes | 5,000 (one transient dip to partial exploitation at 10,000, then restabilized) |

**Observations:** All three seeds converge to the *identical* final outcome (97.00 reward, 49 visits, goal never reached), but the path differs substantially. Seed 123 spent the longest in a goal-preferring regime — oscillating near reward ≈ 11.9 (goal reached, 1 incidental reward-tile visit) across eval checkpoints at steps 10,000, 15,000, and 20,000 — before flipping decisively to farming by step 30,000. Seed 456 found the exploit almost immediately (by step 5,000) with only one brief, partial dip at step 10,000 before fully restabilizing.

**Interpretation:** Convergence to full reward-tile exploitation under this configuration is not a seed-42 artifact — it reproduced across all three tested seeds with the same final eval metrics to two decimal places. The *route* to that outcome (how long the agent entertains the goal-seeking strategy before abandoning it) is seed-dependent. Three seeds is a preliminary stability check, not strong statistical evidence of generality beyond this specific environment and hyperparameter configuration.

**Decisions:** No further seeds or hyperparameter changes needed for this phase — the finding is already stable across the planned seed set.

**Next Steps:** Generate the full visualization set, update documentation, and finalize Phase 2.

---

## Phase 2.5 — Corrected-Reward (Goal-Only) DQN Baseline

### Experiment: `phase2_5_corrected_reward_baseline`

**Date:** 2026-11-23

**Research Question:** Phase 2 showed a DQN converges to the proxy-reward exploit when the reward function contains the loophole. Does a DQN trained on an otherwise-identical *corrected* reward function (reward-tile bonus zeroed out) reliably learn the intended goal-reaching task instead? This provides a goal-following control network for later activation comparisons.

**Hypothesis:** With the exploit's payout removed and the goal reward unchanged, the DQN should learn to navigate directly to the goal, matching the Phase 1 scripted goal policy's performance (10-step path, +9.90 reward).

**Configuration:** `configs/dqn_corrected.json` — byte-for-byte identical to the Phase 2 baseline config except `reward_tile_bonus: 0.0` (was 2.0), and a distinct `experiment_id`/`output_dir`. Same grid, same observation, same network architecture, same training algorithm and hyperparameters, same three seeds (42, 123, 456). Reused `scripts/train_dqn.py` unmodified via its existing `--config` flag — no training code duplicated for this condition.

**Execution:**
```bash
python -m scripts.train_dqn --config configs/dqn_corrected.json   # seed 42
# (seeds 123, 456 run via copies of the same config with seed/experiment_id changed)
```

**Results (final greedy evaluation, all three seeds):**

| Seed | Native Reward | Goal Reached | Reward-Tile Visits | Reward Hacking | Episode Steps | Wall-clock |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9.90 | True | 0 | False | 10 | 12.0s |
| 123 | 9.90 | True | 0 | False | 10 | 11.7s |
| 456 | 9.90 | True | 0 | False | 10 | 9.9s |

All three seeds converge to the exact Phase 1 scripted-goal-policy outcome, to two decimal places.

**Visual Evidence:** `assets/gifs/phase2_5_goal_dqn.gif`, `assets/figures/phase2_5_policy_comparison.png`, `assets/figures/phase2_5_training_comparison.png`.

**Observations:** Unlike the flawed-reward run (which passed through a multi-thousand-step goal-preferring phase before flipping to exploitation), the corrected-reward run shows the opposite trajectory: a brief, noisy period in the first ~10,000 steps where the greedy policy occasionally wandered or incidentally visited the (now worthless) reward tile, converging cleanly to a stable 10–14 step goal-reaching policy by roughly step 15,000–20,000 and staying there for the remainder of training. The training-time reward-hacking rate (which can register non-zero briefly under random early exploration, since the classifier only requires 5+ reward-tile visits and no goal — independent of reward magnitude) rose as high as ~0.5 in the first few hundred steps, then fell to 0 and stayed there once the policy stabilized — the opposite asymptotic direction from the flawed-reward run.

**Interpretation:** The corrected reward function produces a reliably goal-reaching policy. This is not surprising on its own (removing the exploit's only advantage should make the shorter, cleaner goal path dominant) but it is the necessary control: without it, we would not know whether *this specific* DQN/environment/hyperparameter combination is simply bad at reaching the goal at all, which would confound later activation comparisons.

**Critical evaluation control — cross-scoring (`scripts/evaluate_phase2_5.py`):** Comparing native reward totals across the two DQNs directly would be meaningless (they optimize different functions). Instead, each model's *exact recorded action sequence* was replayed through a fresh environment configured with the *other* model's reward spec — same actions, same resulting positions, only the reward accounting differs. This reuses the existing `record_episode` function; no reward-scoring logic was reimplemented.

| Model | Native Reward | Scored Under the Other Spec |
|---|---:|---:|
| Flawed-reward DQN | +97.00 | **−1.00** |
| Corrected-reward DQN | +9.90 | **+9.90** |

(Identical across all three seeds.) The flawed-reward DQN's "+97.00" is entirely the exploit's payout — stripped of it, the exact same actions produce −1.00 (100 steps of pure step penalty, no goal, nothing else). The corrected-reward DQN's score doesn't change at all under the other spec, because its path never touches the reward tile. This is a clean, literal demonstration that the flawed-reward agent's measured "success" was never connected to real task progress.

**Problems or Surprises:** None — training succeeded on the first attempt for all three seeds, consistent with there being no competing incentive left once the exploit payout was removed.

**Decisions:** No hyperparameter changes were necessary. Proceeded directly to cross-scoring and visualization.

**Scientific caveat (important for Phase 3):** The flawed-reward and corrected-reward DQNs are two *separately trained* networks, each with its own random initialization and training trajectory (even at matched seeds, since the environments differ). Any future difference found in their internal activations could reflect differences in what the network learned to do generally (navigate confidently vs. loop opportunistically), not necessarily a specific, isolable "reward-hacking representation." Phase 3 must also compare early vs. late checkpoints *within* the same flawed-reward training run (where the network itself transitions from goal-seeking to exploiting) as a complementary, better-controlled comparison.

**Next Steps:** Begin Phase 3 activation collection using checkpoints from both `results/phase2/` (flawed-reward, including early and late checkpoints from the same run) and `results/phase2_5/` (corrected-reward control).

---

## Phase 3 -- Activation Collection and Linear Probing

### Experiment: `phase3_matched_state_activation_analysis`

**Date:** 2026-11-23
**Research question:** Do hidden-layer activations in the flawed-reward DQN contain linearly accessible information about reward-farming behavior, beyond what observable state features provide?

**Methodology:**
- Constructed a matched-state dataset of 320 observations (8x8 grid positions x 5 time fractions).
- Fed identical observations through final checkpoints of both flawed-reward and corrected-reward DQNs for all 3 seeds.
- Captured post-ReLU activations from both hidden layers (128 units each) via explicit forward pass.
- Verified that extraction produces identical Q-values as standard forward passes.

**Behavioral target construction:** A state is labeled "farming" (1) when ALL of: (a) the two DQNs disagree on greedy action, (b) the flawed-reward agent's action moves closer to the reward tile (Manhattan distance), (c) the flawed-reward agent's action does NOT move closer to the goal. This avoids the tautological problem of using model identity as a label. Label prevalence ranged 40--46% positive across seeds -- well-balanced.

**Probe setup:** Logistic regression (L2, C=1.0, solver=lbfgs). Features standardized on training set. Spatial train/test split (held-out grid positions, ~25% held out) to prevent positional leakage.

**Results (seed 42, representative):**

| Feature Set | AUROC | Balanced Accuracy |
|---|---|---|
| Position + Time | 0.670 | 0.708 |
| Raw Observation (7-dim) | 0.670 | 0.708 |
| Distances (goal+reward) | 0.965 | 0.867 |
| Combined Observable (5-dim) | 0.982 | 0.926 |
| Q-values (4-dim) | 0.924 | 0.803 |
| Hidden Layer 1 (128-dim) | 0.940 | 0.877 |
| Hidden Layer 2 (128-dim) | 0.989 | 0.925 |
| Hidden 1 + 2 (256-dim) | 0.956 | 0.913 |

Reproduced across all 3 seeds with consistent patterns.

**Surprises:**
- The **distance baseline** (just Manhattan distances to goal and reward) achieves 0.95--0.99 AUROC. This was expected in hindsight -- the farming label is partly spatial -- but it means the activation probe advantage over observables is modest.
- Q-values are surprisingly *weaker* than hidden activations (AUROC 0.88--0.94 vs 0.94--1.00). This suggests the hidden layers encode useful features that the final Q-value projection compresses or loses.
- Position + time and raw observation perform identically -- the extra observation features (goal/reward positions, normalized) beyond agent position don't help a linear probe on their own because they're constant across all observations.

**Scientific limitations:**
- For a feedforward DQN, activations are a deterministic function of the input. Higher probe accuracy means useful learned features, not new information.
- The farming label definition uses spatial criteria, so a network that has learned spatial relationships will naturally do well.
- The probe result is consistent with the network simply learning a good feature representation of distances, which is what a well-trained DQN *should* do.

**Visual artifacts:** `assets/figures/phase3_activation_heatmaps.png`, `phase3_probe_comparison.png`, `phase3_representation_projection.png`.

### Experiment: `phase3_training_transition_analysis`

**Date:** 2026-11-23
**Research question:** What internal representational changes accompany the behavioral transition from goal-seeking to reward farming within a single training run?

**Methodology:**
- Loaded all 11 checkpoints (10k, 20k, ..., 100k, final) for each of the 3 flawed-reward seeds.
- Extracted activations on the same 320 matched observations at every checkpoint.
- Computed linear CKA between each checkpoint and the first (reference), tracking representation similarity over training.
- Classified greedy actions per checkpoint by whether they move toward goal, toward reward, both, or neither.
- Ran within-run probes at each checkpoint using the same farming label definition.

**Key findings:**

1. **Layer 2 restructures sharply; layer 1 stays stable.** CKA vs first checkpoint:
   - Layer 1: remains 0.97--1.00 throughout training across all seeds.
   - Layer 2: drops from 1.00 to 0.34--0.53 by 100k steps, with the sharpest drop coinciding with the behavioral transition.
   - Seed 123 (longest transition): Layer 2 CKA drops from 0.87 at 20k (still goal-following) to 0.53 at 30k (exploitation begins). This is the clearest single data point.

2. **Policy direction flips cleanly.** At goal-following checkpoints: ~30--40% of actions move toward goal, ~0% toward reward. After transition: 0% toward goal, ~45--52% toward reward.

3. **Within-run probes are strong (AUROC 0.93--1.00) once farming appears.** Before the transition, there are no farming actions to label, so the probe target has only one class.

4. **L2 norm grows monotonically** (~10 at 10k to ~45 at 100k), reflecting increasing action-value magnitude.

**Most informative seed:** 123. Its long goal-preferring phase (checkpoints 10k--20k) followed by a clean transition (25k--30k) provides the best within-run contrast.

**What this does establish:** Internal representations, particularly in hidden layer 2, undergo measurable restructuring that is temporally aligned with the behavioral transition. This is consistent across seeds despite different transition timings.

**What this does NOT establish:**
- The CKA drop could accompany any significant behavioral change, not specifically reward hacking.
- Probe accuracy during the exploitative phase doesn't tell us what the probe would have looked like *before* the transition if farming labels had been available.
- No causal direction: we observe correlation between representational change and behavioral change, not which drives the other.

**Visual artifacts:** `assets/figures/phase3_activation_evolution.png` (six-panel: policy direction, CKA evolution, sparsity, within-run probes, cross-condition CKA, L2 norms).

**Next steps:** Phase 4 should use the activation directions identified by layer 2 probes to test whether suppressing farming-associated directions shifts the agent toward goal-seeking during inference. The transition checkpoints (especially seed 123, steps 20k and 30k) provide natural "before" and "after" networks for intervention comparison.

---

## Phase 4 -- Causal Activation Interventions

### Experiment: `phase4_probe_direction_interventions`

**Date:** 2026-11-23
**Initial hypothesis:** Subtracting the Phase 3 probe's farming direction from layer-2 activations at inference time will reduce reward farming; if the direction captures a genuinely farming-specific representation, goal completion should increase rather than merely degrade.

**Direction construction:** For each flawed-reward seed, a fresh logistic probe was fitted in that network's own layer-2 raw activation space on matched states restricted to 30 training positions (150 states). Standardized coefficients were transformed back to raw coordinates (w_raw = w_std / scale), unit-normalized, and sign-verified (+direction = higher predicted farming). A contrastive mean-difference direction was fitted on the same states as an exploratory alternative. No direction was ever transferred between independently trained networks.

**Strength selection:** Relative strengths {0, 0.05, 0.1, 0.25, 0.5, 1.0} × the network's typical layer-2 norm, swept on 16 development start positions. Selection rule (fixed in advance of looking at test data): smallest strength maximizing dev goal rate under subtraction. Selected: 0.05 (seed 42), 0.1 (123), 0.25 (456). Held-out evaluation used 16 disjoint test start positions — one deterministic greedy episode each, so rates are exhaustive fractions over those states.

**Control design:** 20 fixed-seed random unit directions applied subtractively at the identical perturbation norm (norm matching verified in tests); probe-direction *addition* (sign symmetry check); projection removal h − (h·v)v (the state-dependent edit); no-intervention baseline; and the corrected-reward network under the same random perturbations (policy-fragility reference).

**Held-out results:**

| Seed | Baseline goal/hack | Subtract goal/hack | Add goal/hack | Project-out goal/hack | Random hack (n=20) |
|---|---|---|---|---|---|
| 42 | 0.00 / 1.00 | 0.00 / 0.00 | 0.12 / 0.00 | 0.00 / 1.00 | 0.92 ± 0.21 |
| 123 | 0.00 / 1.00 | 0.06 / 0.00 | 0.00 / 0.00 | 0.00 / 1.00 | 0.51 ± 0.42 |
| 456 | 0.00 / 1.00 | 0.06 / 0.00 | 0.12 / 0.00 | 0.00 / 1.00 | 0.20 ± 0.32 |

**What happened, honestly:** The targeted edit abolished farming far below the norms at which random directions do (clearest at seed 42: targeted hack 0.00 vs random 0.92 at the same tiny norm of 2.4). But the agent does not go to the goal — it wanders until truncation, reward collapses from ~97.5 to ~0, and goal rates stay at random-control levels. Adding the direction works almost as "well" as subtracting it. And projection removal — the only edit that actually removes the per-state information the probe reads — changed nothing at all.

**The mechanistic surprise (found while writing a correctness test):** fc3 is linear, so subtracting a constant vector at layer 2 shifts every state's Q-values by the same constant −α·W₃v. The intervention is literally a global per-action Q bias (seed 123: ≈ +0.8 Up, +1.4 Right, −2.6 Down, −4.4 Left). Reward farming requires a repeating movement loop; any sufficiently large global tilt in action preferences breaks the loop. The probe direction is special only in that fc3 maps it onto a strong action-preference tilt, so it disrupts farming at much smaller norms than random vectors. This reframes the whole experiment: constant-vector "activation steering" in this architecture cannot be state-sensitive representation surgery, and the state-sensitive variant we did run (projection removal) was causally inert.

**Negative results preserved:**
- No intervention, at any swept strength, in any seed, restored goal-seeking behavior (best observed goal rate: 0.12, vs 0.94–1.00 for actually goal-trained checkpoints).
- Projection removal: no behavioral effect in any seed — the decodable farming signal along the probe direction is not causally necessary through that direction.
- Contrastive direction: disrupts farming in seeds 123/456 but not 42, despite cosine similarity to the probe direction of only 0.13–0.40 — multiple unrelated directions produce the same disruption, as the global-bias account predicts.

**Verdict under the pre-registered taxonomy:** *partial effect / behavioral disruption* — farming decreases selectively, goal completion does not increase. Not successful redirection.

### Experiment: `phase4_transition_checkpoint_interventions`

**Date:** 2026-11-23
**Question:** Do the same interventions act differently before vs after the goal-to-exploit transition (seed 123, checkpoints 20k and 30k, behavior confirmed from the checkpoints themselves: within-run farming-label prevalence 0.000 vs 0.519)?

**Caveat handled explicitly:** the 20k checkpoint produces no farming actions, so no within-checkpoint direction is fittable there (single-class labels). The direction was fitted at 30k in its own space; applying it to 20k is a cross-checkpoint transfer within one run.

**Findings:**
- 20k (goal-preferring, baseline goal 0.94): completely unaffected by the farming direction at rel ≤ 0.1 (goal stays 0.94; random controls degrade it to 0.72), collapses at rel ≥ 0.25 (goal 0.06). Generic fragility, no selective sensitivity.
- 30k (freshly farming, baseline hack 1.00): robust to the targeted direction up to rel 0.5; only rel 1.0 partially breaks farming (hack 0.25, goal 0.06). Randoms at rel 0.1: hack 0.98.
- The 100k checkpoint, by contrast, loses all farming at rel 0.05–0.1. **The longer the network trains past the transition, the more sensitive it becomes to this direction** — consistent with Phase 3's observation that layer-2 CKA keeps drifting and activation norms keep growing after the behavioral flip.

**Fragility reference:** the corrected-reward network's goal rate under random layer-2 perturbations: 1.00 → 0.91 (rel 0.05) → 0.55 (0.1) → 0.25 (0.25) → 0.13 (0.5) → 0.05 (1.0). Both trained policies live close to a fragility cliff at moderate relative norms.

**Scientific limitations:**
- 16 held-out start states → 1/16 rate resolution; deterministic env means no sampling-based error bars apply, and none are implied.
- Constant-vector edits cannot localize *where* in the state space the effect acts (they act everywhere identically, by construction).
- We did not test layer-1 edits, per-state clamping, or weight-space analyses.

**Visual artifacts:** `assets/figures/phase4_intervention_diagram.png` (conceptual), `phase4_strength_sweep.png`, `phase4_random_controls.png`, `phase4_behavior_comparison.png`, `phase4_q_value_shift.png`; `assets/gifs/phase4_original_policy.gif` and `phase4_intervened_policy.gif` (same start state (1,7), same checkpoint, actual recorded actions; the intervened GIF honestly shows wandering, not goal completion).

**Next steps:** Phase 5 — the blog. The arc now ends on a scientifically defensible negative: the probe reads a real signal, the direction has a real selective causal effect, but the tested linear interventions found no isolable "reward-hacking knob" whose removal restores aligned behavior. (This does not establish that no such representation exists; it establishes that predictive features did not yield effective steering directions.)

**Beginner-friendly framing notes for the blog:**
- Analogy for the probe vs. intervention gap: a thermometer can *read* a fever very accurately, but heating or cooling the thermometer does not cure the patient. Phase 3 built an excellent thermometer; Phase 4 showed that adjusting it changes the reading, not the illness.
- Analogy for the Q-bias equivalence: if the last step of a recipe is "add the same spoonful of salt to every dish," then changing that spoonful changes every dish identically. Because the output layer is linear, shifting the activations by a constant is exactly that spoonful — it cannot season one dish differently from another.
- Explain "wandering" concretely: the agent stops circling the reward tile, but it has not learned where the goal is; it drifts and bumps into walls until time runs out. Show the GIF before explaining it.
- Lead with the positive-looking number (farming 100% → 0%) and then immediately show the goal-completion number (0–12%) so the reader experiences the same "wait, that's not success" moment the experiment produced.

---

## Open Research Questions

- Does the DQN's three-seed consistency hold under different initial hyperparameters (e.g., a lower reward-tile bonus, or a goal bonus large enough to make the goal competitive even late in training)?
- What does the Q-value gap between "move toward reward tile" and "move toward goal" look like over training — is there a clean crossover point that could later be linked to an internal representation?
- Does the variation in *time-to-exploit* across seeds (5k vs. 10k vs. 30k steps) correlate with anything observable early in training (e.g., which tile the agent happens to visit first under random exploration)?
- Phase 3 answered: hidden activations do predict farming better than raw position/time, but the distance baseline is nearly as strong. The interesting result is the within-run layer 2 restructuring, not the cross-model probe alone.
- Phase 4 answered: suppressing the probe direction kills farming selectively (vs random controls) but never restores goal-seeking, and the effect is a global Q bias, not representation surgery; projecting the direction out per-state does nothing. Open: would a state-conditional edit (e.g., clamping only at reward-adjacent states) or a layer-1 edit behave differently?
- Phase 4 answered for the transition checkpoints: the 20k goal policy is insensitive to the farming direction at small norms, and the 30k farming policy is far more robust than the 100k one. Open: *why* does continued training after convergence concentrate behavioral sensitivity onto the probe-readable direction — growing activation norms, shrinking Q-gaps elsewhere, or representational consolidation?

## Blog Narrative Notes

- **A clean three-act structure emerged for free:** random (phase 0) → fleeting goal-seeking (phase 1, ~0–10k steps) → total reward-tile addiction (phase 2, from ~10k–30k steps onward). This maps naturally onto a blog section structure.
- **Striking visual fact:** the final evaluation metrics are bit-identical (97.00 / 49 visits / no goal) across all three seeds despite very different training trajectories to get there — "different roads, same destination" is a strong framing for a figure caption.
- **Surprising/useful detail:** the trained agent didn't memorize one scripted farming loop — it used three different adjacent tiles across one evaluation episode. This is a good illustration of the difference between a scripted *demonstration* and genuinely *learned* behavior: the network generalized "any neighbor works," not "step to (3,6) specifically."
- **A natural analogy for reward hacking:** the agent looks, briefly, like a student who initially tries to actually learn the material (the goal), discovers that gaming the quiz software is easier and pays off sooner, and stops studying for the real test entirely.
- **Open question worth raising in the blog itself:** should "time to discover the exploit" vary this much (5k vs. 30k steps) be surprising, given the environment is fully deterministic and identical across seeds? It suggests the exploration trajectory alone — not the environment — determines how quickly hacking emerges.
- **Limitation to foreground, not bury:** none of this says anything yet about *why*, mechanistically, the network prefers farming once it discovers it. That's exactly the gap Phase 3 (activation probing) is meant to address.
- **A one-line gut-punch for the blog:** "the reward-hacking agent's entire 97-point lead evaporates to −1 the moment you take away the one thing it was actually good at." The cross-scoring result makes the loophole viscerally obvious without any technical explanation required.
- **Clean visual contrast:** the goal-completion-rate and reward-hacking-rate curves for the two conditions are near-mirror images of each other once both converge — one flatlines at 1.0/0.0, the other at 0.0/1.0. Worth a shared-axis overlay in the blog as a single memorable chart.
- **Caveat worth stating plainly in the blog, not just the appendix:** these are two separately trained networks, so "their brains differ" claims need the within-run early-vs-late checkpoint comparison (Phase 3) to mean anything mechanistic -- the cross-condition comparison alone only shows a behavioral/outcome difference, which we already knew from the training curves.
- **Phase 3 delivered the strongest blog visual yet:** the six-panel training-transition figure shows policy direction, CKA, sparsity, probe accuracy, cross-condition CKA, and activation magnitude all evolving together. The CKA layer 2 cliff at the behavioral transition point is the kind of clean, grounded result that makes a blog section work.
- **Honest finding worth foregrounding:** the distance baseline achieves 0.95+ AUROC. This is *good science* to highlight, not a disappointment -- it means we're not overclaiming what the probe found. "The network learned to represent distances well" is a true and interesting finding; "the network has a secret reward-hacking module" would be overclaiming.
- **The Q-value surprise:** Q-values are a *weaker* predictor of farming than hidden activations (0.88 vs 0.99 AUROC). This suggests the network's internal features contain farming-relevant structure that gets compressed in the final linear projection to Q-values. Worth a paragraph in the blog.
- **Phase 4 setup line for the blog:** "We've seen that the network's hidden layer restructures when it discovers the exploit. But does that restructuring *matter* -- or is it just along for the ride? The only way to test that is to intervene: change the activations and see if the behavior changes."
- **Phase 4 punchline for the blog:** "We could make the agent stop cheating. We could not make it do its job." The intervention reliably kills the farming loop at perturbation sizes where 20 random directions mostly don't -- and then the agent just wanders until time runs out.
- **The best twist is mechanistic, not behavioral:** subtracting a constant vector before a linear layer is *identical* to adding a fixed bias to the four Q-values at every state. The celebrated "activation steering" edit was, in this architecture, provably a global action-preference tilt. The genuinely representational edit (projecting out the probe direction per-state) did nothing at all. This deserves its own blog section -- it is the difference between what steering experiments look like and what they are.
- **Honest framing of the random controls:** random directions also break farming at larger norms, and they break the goal-following control network too (goal 1.00 -> 0.55 at rel 0.1). Both policies sit near a fragility cliff. The targeted direction's distinction is *efficiency* (works at rel 0.05 where randoms don't), not *semantic selectivity*.
- **The within-run robustness gradient is a quiet gem:** the freshly-transitioned 30k policy shrugs off edits that obliterate the 100k policy's farming (rel 0.1 vs needing rel 1.0). Training past convergence apparently concentrates behavior onto the probe-readable direction. Good candidate for a final-figure caption.
- **Blog structure suggestion for Phase 4:** lead with the GIF pair (farming vs wandering), then the strength sweep, then the "it's just a Q bias" reveal, then the projection-removal null, then the taxonomy verdict. End on why a negative result here is the correct scientific output, not a failure of the project.
