# RewardLens

**Investigating Reward Hacking in Reinforcement Learning Through Internal Neural Activations**

---

## The Problem

What happens when an AI learns to maximize the score rather than accomplish the task?

In reinforcement learning, agents learn by maximizing a reward signal. But reward functions are written by humans, and humans make mistakes. When a reward function contains a loophole — a way to accumulate reward without completing the intended task — agents can learn to exploit it. This is called **reward hacking**.

RewardLens investigates this phenomenon from the inside out: not just *whether* agents exploit reward loopholes, but *what their internal representations look like* when they do, and *whether modifying those representations changes their behavior*.

## The Environment

We use a deliberately simple 8×8 gridworld with two ways to earn reward:

![Gridworld Overview](assets/figures/gridworld_overview.png)

- **Goal tile** (★): reaching it awards +10 and ends the episode — the intended task.
- **Reward tile** ($): entering it awards +2 each time — but the agent can leave and come back for more.
- **Step penalty**: every action costs −0.01, discouraging idle behavior.

The reward tile is repeatable with no cooldown. This is the loophole.

## Two Behaviors, One Reward Function

We verify the loophole with two scripted policies (not trained agents — these are handcrafted demonstrations):

| | Goal Policy | Exploit Policy |
|---|---|---|
| **Strategy** | Shortest path to the goal | Farm the reward tile repeatedly |
| **GIF** | ![Goal](assets/gifs/goal_policy.gif) | ![Exploit](assets/gifs/exploit_policy.gif) |
| **Proxy reward** | +9.90 | +97.00 |
| **Goal reached** | Yes | No |
| **Steps** | 10 | 100 |

The exploit policy earns **~10× more reward** while completely failing the task.

## The Reward Loophole

![Trajectory Comparison](assets/figures/trajectory_comparison.png)

The side-by-side view makes the problem clear: same reward function, opposite outcomes.

![Cumulative Reward](assets/figures/cumulative_reward.png)

The cumulative reward chart shows the exploit policy's steady reward accumulation dwarfing the goal policy's one-time completion bonus. Under the proxy reward function alone, the exploiter is the "better" agent.

## What We Want to Investigate

This environment is the foundation for three experiments:

1. **Train a DQN agent** and determine whether it independently discovers the exploit.
2. **Probe its internal activations** to test whether reward-hacking behavior is predictable from hidden-layer features beyond what the observation alone provides.
3. **Intervene on selected activations** during inference to test whether modifying internal representations causally changes the agent's behavior.

These experiments are planned but not yet implemented. The scripted demonstrations above verify the environment, not a trained agent.

## Current Results (Phase 1)

| Metric | Goal Policy | Exploit Policy |
|---|---|---|
| Total proxy reward | +9.90 | +97.00 |
| Goal completed | True | False |
| Reward-tile visits | 0 | 49 |
| Episode steps | 10 | 100 |
| Reward hacking | False | True |

These are verified experimental results from scripted policies, not learned behavior.

## Reproduce the Experiment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test suite
python -m pytest -q

# Run environment verification and generate all visuals
python -m scripts.verify_environment
```

Generated artifacts:
- `assets/figures/gridworld_overview.png` — environment layout
- `assets/gifs/goal_policy.gif` — goal policy animation
- `assets/gifs/exploit_policy.gif` — exploit policy animation
- `assets/figures/trajectory_comparison.png` — side-by-side paths
- `assets/figures/cumulative_reward.png` — reward accumulation chart
- `results/phase1/` — machine-readable trajectory data

## Research Roadmap

- [x] **Phase 1**: Environment implementation, scripted exploit verification, visualizations
- [ ] **Phase 2**: DQN training and behavioral evaluation
- [ ] **Phase 3**: Activation collection and linear probing
- [ ] **Phase 4**: Causal activation interventions
- [ ] **Phase 5**: Technical blog post and reproducibility review

## Limitations

- The scripted policies demonstrate that the reward loophole *exists*, not that an RL agent will *discover* it. That is a Phase 2 question.
- The reward-hacking classification uses an operational threshold (≥5 reward-tile visits without reaching the goal). This is a research definition, not a universal detection method.
- The environment is deliberately minimal. Real-world reward hacking involves far more complex dynamics.
- Internal activation analysis requires a trained neural network. No claims about DQN representations can be made until Phase 3.

## Repository Structure

```
RewardLens/
├── rewardlens/
│   ├── envs/
│   │   ├── reward_grid.py        # Gymnasium environment
│   │   ├── scripted_policies.py  # Goal and exploit policies
│   │   └── trajectory.py         # Trajectory recording
│   └── visualization/
│       └── gridworld_viz.py      # Figures and GIF generation
├── scripts/
│   └── verify_environment.py     # Phase 1 verification
├── tests/
│   └── test_reward_grid.py       # 45 tests
├── assets/                       # Generated figures and GIFs
├── results/                      # Machine-readable trajectory data
├── CONTEXT.md                    # Research log
└── AGENTS.md                     # Agent instructions
```
