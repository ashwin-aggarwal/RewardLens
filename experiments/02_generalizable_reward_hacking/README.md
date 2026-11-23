# Experiment 2: Generalizable Reward-Hacking Interventions

**Status: planned. Nothing here has been implemented or run, and there are no results.**

## Research question

Do the detection and repair methods from [Experiment 1](../01_fixed_gridworld_repair/README.md) work beyond the single checkpoint pair they were built on? Experiment 1 found a working repair for one seed in one fixed layout, and it failed on the other two seeds. Experiment 2 asks whether reward hacking, and its repair, can be made to generalize.

## Intended scope

- **Randomized environments.** Goal and reward-tile positions (and possibly grid size or obstacles) vary per episode, so a network cannot memorize one layout and must represent where the goal and the exploit are.
- **More seeds, prespecified rules.** Enough training seeds that per-seed failures can be counted, with donor selection, ranks and thresholds fixed before evaluation.
- **Untouched test sets.** Held-out layouts and seeds that no selection step has seen, so results can be confirmatory rather than exploratory.
- **Transfer evaluation.** Methods fitted on some layouts, seeds or checkpoints and tested on others. Conventional fine-tuning stays as a baseline throughout.

## Planned milestones

1. Randomized-layout environment, with a scripted check that the exploit still pays more than the goal in every layout.
2. Train hacking and goal-following agents on the flawed and corrected rewards across seeds; confirm hacking emerges and record when.
3. Detection: probes against layout-aware baselines, tested on held-out layouts.
4. Repair: activation patching and weight editing, fitted on training layouts and seeds, evaluated on held-out ones, against fine-tuning and random controls.
5. Report, including all negative results.

The milestones may change once the environment exists; changes will be recorded here with their reasons.
