# Experiment 2 scripts

Run every script from the repository root as a module, for example `python -m scripts.exp2_train`. The top-level `scripts` package adds this folder to its search path, so module names did not change when the files moved here. Full command sequences, run times and prerequisites are in [SETUP.md](../../../SETUP.md).

Listed in the order the experiment runs them.

| Script | Purpose |
|---|---|
| [`exp2_make_layouts`](exp2_make_layouts.py) | Experiment 2: generate the layout manifest and freeze the train/dev/test split |
| [`exp2_scripted_returns`](exp2_scripted_returns.py) | Experiment 2: scripted goal-reaching vs reward-farming returns across layouts |
| [`exp2_train`](exp2_train.py) | Experiment 2, Phase 1: navigation pretraining and the good/hacking fork |
| [`exp2_evaluate`](exp2_evaluate.py) | Experiment 2, Phase 1: evaluate the pretrained, good and hacking DQNs |
| [`exp2_visualize`](exp2_visualize.py) | Experiment 2, Phase 1: figures and GIFs from recorded results |
| [`exp2_run_phase2`](exp2_run_phase2.py) | Experiment 2, Phase 2: representation analysis and preliminary activation transplantation |
| [`exp2_visualize_phase2`](exp2_visualize_phase2.py) | Experiment 2, Phase 2: figures and GIFs from recorded results |
| [`exp2_run_phase3`](exp2_run_phase3.py) | Experiment 2, Phase 3: when reward hacking emerges and how the hidden layers change around it |
| [`exp2_visualize_phase3`](exp2_visualize_phase3.py) | Experiment 2, Phase 3: figures and the training-progression GIF, from recorded results |
| [`exp2_run_phase4`](exp2_run_phase4.py) | Experiment 2, Phase 4: activation patching between nearby checkpoints of one hacking run |
| [`exp2_visualize_phase4`](exp2_visualize_phase4.py) | Experiment 2, Phase 4: figures and a comparison GIF, from recorded results |
| [`exp2_run_phase5`](exp2_run_phase5.py) | Experiment 2, Phase 5: where is reward hacking encoded? Weight-level interventions |
| [`exp2_visualize_phase5`](exp2_visualize_phase5.py) | Experiment 2, Phase 5 figures and GIF |
| [`exp2_run_phase6`](exp2_run_phase6.py) | Experiment 2, Phase 6: coordinated low-rank weight editing of the hacking networks |
| [`exp2_visualize_phase6`](exp2_visualize_phase6.py) | Experiment 2, Phase 6 figures and GIF |
| [`exp2_visualize_final_eval`](exp2_visualize_final_eval.py) | Experiment 2 confirmatory evaluation figure. Reads the recorded result; evaluates nothing |

`exp2_run_final_eval` (the confirmatory evaluation) is the one script that stays in the top-level [`scripts/`](../../../scripts/) folder, because the frozen protocol records that path. It refuses to run a second time.
