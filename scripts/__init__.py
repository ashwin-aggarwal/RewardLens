"""The `scripts` package: one entry point per experiment phase.

The script files live with their experiment:

    experiments/01_fixed_gridworld_repair/scripts/         Experiment 1 (train_dqn, run_phase4_6, ...)
    experiments/02_generalizable_reward_hacking/scripts/   Experiment 2 (exp2_train, exp2_run_phase5, ...)

Both folders are added to this package's search path, so every script keeps its module name and is run
from the repository root exactly as before, for example `python -m scripts.exp2_run_phase5`.
`exp2_run_final_eval.py` stays in this folder because the frozen confirmatory protocol records this path.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
__path__ += [str(_ROOT / "experiments" / name / "scripts")
             for name in ("01_fixed_gridworld_repair", "02_generalizable_reward_hacking")]
