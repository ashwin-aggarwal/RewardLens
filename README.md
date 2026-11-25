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
| 2 | [Generalizable Reward-Hacking Interventions](experiments/02_generalizable_reward_hacking/README.md) | Phases 1–6 and the confirmatory evaluation complete; a rank-8 edit across all layers repairs the agents, confirmed on reserved layouts |

**Experiment 1.** In one fixed 8×8 gridworld, transplanting a goal-following checkpoint's activations restored direct goal pursuit in a hacking DQN, and a rank-4 weight edit made that repair permanent. Ordinary fine-tuning worked as well, and neither repair transferred to other training seeds.

**Experiment 2 (research complete, write-up pending).** Start, goal and reward tile move every episode. From one shared pretrained checkpoint, a good DQN reaches the goal directly in 100% of unseen layouts and a hacking DQN farms in 92%, across 3 seeds. A first activation transplant only stops the farming and does not restore goal-seeking. Tracking training checkpoints shows the hidden layers drifting before behavior flips and still changing long after, and patching between nearby checkpoints of the same run does not help: the nearby ones are already hacking. Copying goal-following weights into any subset of layers fails too. Retraining toward a goal-following teacher repairs the network only when all three layers are updated together, and a rank-8 update across the layers (a fifth of the parameters) is enough. Evaluated once on 1,500 reserved layouts, all nine frozen rank-8 models met the repair bar (0.98–1.00 direct goal, no farming). The repair is distilled from a goal-following teacher; a repair without one was not found.

## More

- [SETUP.md](SETUP.md): installation, tests and reproduction commands
- [CONTEXT.md](CONTEXT.md): current research state and design decisions
- [Experiment 1 progress journal](experiments/01_fixed_gridworld_repair/PROGRESS.md)
- [Experiment 2 progress journal](experiments/02_generalizable_reward_hacking/PROGRESS.md)
- [Project progress index](PROGRESS.md)
