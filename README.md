# RewardLens

**Can we find reward hacking inside a neural network, and fix it there?**

Reward hacking happens when an agent maximizes the reward it is given without doing the task that reward was meant to encourage. RewardLens trains small reinforcement-learning agents that learn to exploit a flawed reward, then asks whether their hidden activations reveal the exploit and whether editing activations or weights repairs it.

| Trained on the flawed reward | Trained on the corrected reward |
|---|---|
| ![Reward-hacking DQN](assets/gifs/phase2_trained_dqn.gif) | ![Goal-following DQN](assets/gifs/phase2_5_goal_dqn.gif) |

## Experiments

| # | Experiment | Status |
|---|---|---|
| 1 | [Fixed Gridworld Reward-Hacking Repair](experiments/01_fixed_gridworld_repair/README.md) | Complete |
| 2 | [Generalizable Reward-Hacking Interventions](experiments/02_generalizable_reward_hacking/README.md) | Phases 1–2 complete; no working repair yet |

**Experiment 1.** In one fixed 8×8 gridworld, transplanting a goal-following checkpoint's activations restored direct goal pursuit in a hacking DQN, and a rank-4 weight edit made that repair permanent. Ordinary fine-tuning worked as well, and neither repair transferred to other training seeds.

**Experiment 2 (in progress).** Start, goal and reward tile move every episode. From one shared pretrained checkpoint, a good DQN reaches the goal directly in 100% of unseen layouts and a hacking DQN farms in 92%, across 3 seeds. Their internals differ in a similar, low-dimensional way across seeds, but a first activation transplant only stops the farming and does not restore goal-seeking.

## More

- [SETUP.md](SETUP.md): installation, tests and reproduction commands
- [CONTEXT.md](CONTEXT.md): current research state and design decisions
- [Experiment 1 progress journal](experiments/01_fixed_gridworld_repair/PROGRESS.md)
- [Experiment 2 progress journal](experiments/02_generalizable_reward_hacking/PROGRESS.md)
- [Project progress index](PROGRESS.md)
