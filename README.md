# RewardLens

**Can we find reward hacking inside a neural network, and change it there?** RewardLens trains DQN agents that learn to exploit a flawed reward, then tests whether their hidden activations reveal that behavior and whether editing those activations changes it.

## The problem

Reward hacking happens when an agent maximizes the reward it is given without doing the task that reward was meant to encourage. Our 8×8 gridworld has a goal worth +10 that ends the episode, and a reward tile worth +2 every time it is entered. Farming that tile pays about ten times more than finishing the task.

| Trained on the flawed reward | Trained on the corrected reward |
|---|---|
| ![Reward-hacking DQN](assets/gifs/phase2_trained_dqn.gif) | ![Goal-following DQN](assets/gifs/phase2_5_goal_dqn.gif) |
| Farms the tile 49 times and never finishes | Walks to the goal in 10 steps |

## What we found

**1. The agents learned to hack on their own.** All three training seeds discovered the exploit without being shown it. Seed 123 followed the goal for its first 20k steps, then switched to farming.

**2. Activations predict hacking, but removing that signal does not restore the task.** A linear probe on the hidden layer predicts farming moves with AUROC 0.98–1.00. A simple distance baseline already reaches 0.95–0.99, so the probe adds only a little. Subtracting the probe's direction stopped the farming but left the agent wandering, with goal completion at 0–12%.

![Probe comparison](assets/figures/phase3_probe_comparison.png)
![Intervention behavior](assets/figures/phase4_behavior_comparison.png)

**3. Transplanting state-dependent activations restores the task, and eight directions are enough.**
- Copying the hidden layer from an earlier, goal-following checkpoint of the same training run made the hacking network reach the goal. At first it still farmed until the deadline before finishing.
- Controls traced this to a time gate. It disappears when the donor's activations are computed as if little time remains.
- That patch spans half the layer's units. It is still compact in direction space: the average correction plus 8 principal directions reproduces it.

![Low-rank success curve](assets/figures/low_rank_patching_success_vs_rank.png)

## Main result

| Condition (seed 123, 30k-step hacking checkpoint) | Direct-goal completion |
|---|---|
| Unmodified reward-hacking DQN | 5% (62 starts) |
| Low-rank PCA patch: mean + 8 directions | **100%** (all 62 starts) |
| Random 8-dimensional subspaces (20 plain, 20 norm-matched) | 2% on average, best 13% (46 starts) |
| Same patch, with donor activations from the wrong square | 5% on average (46 starts) |

"Direct goal" means the agent reaches the goal with fewer than 5 reward-tile visits.

The PCA patch depends on the current state. An uncentered SVD basis with 8 directions also reached 100%, but it worked just as well with activations from the wrong square (98%), so it is a state-agnostic push toward the goal and does not count as the result.

**This is an exploratory result.** It comes from one donor/recipient checkpoint pair from a single training seed, and every start position had been analyzed in earlier phases. Only powers of two were tested, so the true minimum rank lies somewhere from 5 to 8. The patch still needs the donor network at inference. The direction that explains 92% of the variance turned out not to matter for behavior. Success with a few directions does not show that they "represent" goal-seeking. Full limitations are in [CONTEXT.md](CONTEXT.md) and the experiment reports.

## Explore the project

- [SETUP.md](SETUP.md): installation, tests, and reproduction commands for every experiment
- [PROGRESS.md](PROGRESS.md): chronological research journal, including the previous phase-by-phase README
- [CONTEXT.md](CONTEXT.md): current scientific state, design decisions, and limitations
- [Sparse patching report](experiments/sparse_patching/README.md): no small subset of units suffices
- [Low-rank patching report](experiments/low_rank_patching/README.md): eight directions do
