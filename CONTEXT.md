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
- [ ] Phase 2: DQN training and behavioral evaluation — not started.
- [ ] Phase 3: activation collection and linear probing — not started.
- [ ] Phase 4: causal activation interventions — not started.
- [ ] Phase 5: technical blog post and reproducibility review — not started.

Current milestone: Phase 1 complete. Next: Phase 2 — DQN training.

## 7. Experimental Decisions

### 2026-11-22 — Separate proxy reward from true task success

Reward hacking will be defined operationally as maximizing the proxy signal without completing the intended task. All evaluations and visualizations must report both quantities separately. This prevents a high measured reward from being mistaken for successful task completion.

### 2026-11-22 — Begin with scripted environment verification

The first implementation milestone is environment correctness and scripted exploit verification, before DQN training or representation analysis. This establishes that the task and its exploit are real and measurable independently of learning.

### 2026-11-22 — Treat visuals as reproducible artifacts

Conceptual diagrams may explain the intended story, but experimental plots and GIFs must be generated from recorded trajectories or activations. Scripted demonstrations must not be described as learned behavior.

### 2026-11-22 — Preserve causal and baseline controls

Observation baselines, state matching, independent evaluation, and random-direction interventions are required parts of the design rather than optional follow-up analyses.

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

## 9. Known Problems and Next Steps

### Known problems

- No empirical claim about reward hacking or DQN representations can currently be made — only scripted demonstrations exist.
- Alternative agent strategies (collect reward then reach goal, stand near reward tile, wander) have been considered in the design review but are not tested with trained agents yet.

### Immediate next milestone

**Phase 2:** Train a CleanRL-based DQN agent on RewardGridEnv and evaluate whether it independently discovers the exploit. Record proxy reward, true task success, and behavioral trajectories for comparison with scripted baselines.

### Maintenance checklist

At the end of each meaningful task:

1. Update project status in this document.
2. Record important design decisions and their rationale.
3. Record verified results only, including reproducibility metadata.
4. Document unresolved issues and limitations.
5. Identify the next logical step.
6. Keep historical information concise and avoid redundant session summaries.
