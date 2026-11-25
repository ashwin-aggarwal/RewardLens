# Experiment 1 scripts

Run every script from the repository root as a module, for example `python -m scripts.visualize_phase2`. The top-level `scripts` package adds this folder to its search path, so module names did not change when the files moved here. Full command sequences, run times and prerequisites are in [SETUP.md](../../../SETUP.md).

Listed in the order the experiment runs them.

| Script | Purpose |
|---|---|
| [`verify_environment`](verify_environment.py) | Phase 1 verification: run scripted policies, validate assertions, generate visuals |
| [`train_dqn`](train_dqn.py) | Phase 2: Train a DQN on RewardGridEnv and evaluate periodically |
| [`visualize_phase2`](visualize_phase2.py) | Generate all Phase 2 visualizations from recorded experiment data |
| [`discounted_returns`](discounted_returns.py) | Pre-training sanity check: compare scripted policies under discounted returns |
| [`evaluate_phase2_5`](evaluate_phase2_5.py) | Phase 2.5: compare the flawed-reward DQN and the corrected-reward (goal-only) DQN |
| [`visualize_phase2_5`](visualize_phase2_5.py) | Generate Phase 2.5 visualizations: flawed-reward vs. corrected-reward DQN |
| [`collect_activations`](collect_activations.py) | Collect activations from trained DQN checkpoints on a matched-state dataset |
| [`probe_activations`](probe_activations.py) | Experiment B: Linear probing of hidden activations for reward-farming behavior |
| [`analyze_training_transition`](analyze_training_transition.py) | Experiment C: Analyze representational changes during the goal-to-exploit transition |
| [`visualize_phase3`](visualize_phase3.py) | Generate Phase 3 visualizations from collected activation and probing data |
| [`run_interventions`](run_interventions.py) | Phase 4 Experiments 1-3: intervention strength sweeps, random controls, cross-seed |
| [`evaluate_interventions`](evaluate_interventions.py) | Phase 4 Experiment 4: interventions before vs after the goal-to-exploit transition |
| [`visualize_phase4`](visualize_phase4.py) | Generate Phase 4 visualizations from recorded intervention results |
| [`run_phase4_5`](run_phase4_5.py) | Phase 4.5: goal-directed activation patching |
| [`visualize_phase4_5`](visualize_phase4_5.py) | Generate Phase 4.5 visualizations from recorded patching results |
| [`run_phase4_6`](run_phase4_6.py) | Phase 4.6: behavioral reclassification, time-dependence analysis, and |
| [`reclassify_trajectories`](reclassify_trajectories.py) | Phase 4.6: retrospective behavioral reclassification of Phase 1-4.5 trajectories |
| [`visualize_phase4_6`](visualize_phase4_6.py) | Generate Phase 4.6 visualizations from recorded results |
| [`run_phase4_7`](run_phase4_7.py) | Phase 4.7: controlling for time in activation patching |
| [`visualize_phase4_7`](visualize_phase4_7.py) | Generate Phase 4.7 visualizations from results/phase4_7/time_control_results.json |
| [`run_sparse_patching`](run_sparse_patching.py) | Phase 4.8 (experimental extension): sparse causal activation patching |
| [`visualize_sparse_patching`](visualize_sparse_patching.py) | Generate sparse-patching (Phase 4.8) visualizations from recorded results |
| [`run_low_rank_patching`](run_low_rank_patching.py) | Low-rank activation patching (experimental extension) |
| [`visualize_low_rank_patching`](visualize_low_rank_patching.py) | Generate low-rank patching visualizations from recorded results |
| [`run_weight_editing`](run_weight_editing.py) | Phase 5 (experimental extension): low-rank weight editing |
| [`visualize_weight_editing`](visualize_weight_editing.py) | Generate Phase 5 weight-editing visualizations from recorded results |
