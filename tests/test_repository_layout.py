"""Repository layout: the `scripts` package redirect, unique script names and safe command-line handling."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rewardlens.cli import script_args

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIRS = [ROOT / "experiments/01_fixed_gridworld_repair/scripts",
               ROOT / "experiments/02_generalizable_reward_hacking/scripts",
               ROOT / "scripts"]
# Experiment 1 runners that take no options; before they had a parser, `--help` started the full run.
NO_OPTION_RUNNERS = ["run_interventions", "run_phase4_5", "run_phase4_6", "run_phase4_7", "run_sparse_patching",
                     "run_low_rank_patching", "run_weight_editing"]


def script_files() -> list[Path]:
    return sorted(p for d in SCRIPT_DIRS for p in d.glob("*.py") if p.name != "__init__.py")


class TestScriptsPackage:
    def test_script_names_are_unique_across_experiments(self):
        """All three folders share the `scripts` namespace, so a repeated filename would shadow another script."""
        seen: dict[str, Path] = {}
        for path in script_files():
            assert path.stem not in seen, f"{path} collides with {seen[path.stem]}"
            seen[path.stem] = path
        assert len(seen) >= 43

    def test_every_script_resolves_to_its_own_file(self):
        importlib.invalidate_caches()
        for path in script_files():
            spec = importlib.util.find_spec(f"scripts.{path.stem}")
            assert spec is not None and Path(spec.origin).resolve() == path.resolve(), path

    def test_experiment_script_folders_are_not_packages(self):
        # An __init__.py there would give the files a second import name.
        for d in SCRIPT_DIRS[:2]:
            assert not (d / "__init__.py").exists()

    def test_frozen_evaluation_script_is_where_the_protocol_says(self):
        protocol = json.load(open(ROOT / "experiments/02_generalizable_reward_hacking/configs/final_eval.json"))
        assert protocol["evaluation"]["script"] == "scripts/exp2_run_final_eval.py"
        assert (ROOT / protocol["evaluation"]["script"]).exists()


class TestCommandLine:
    def test_every_entry_point_parses_its_arguments(self):
        """A script with a __main__ block must build a parser, so --help or a typo cannot start a run."""
        for path in script_files():
            src = path.read_text()
            if "__main__" not in src:
                continue
            assert "argparse" in src or "script_args(" in src, f"{path} runs without parsing its arguments"

    def test_main_block_parses_before_it_runs(self):
        for name in NO_OPTION_RUNNERS:
            path = SCRIPT_DIRS[0] / f"{name}.py"
            block = ast.parse(path.read_text()).body[-1]
            assert isinstance(block, ast.If)
            calls = [n.func.id for stmt in block.body for n in ast.walk(stmt) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            assert calls[:2] == ["script_args", "main"], (name, calls)

    def test_script_args(self):
        assert script_args("doc", argv=[]).__dict__ == {}
        assert script_args("doc", "--gif", argv=["--gif"]).gif is True
        assert script_args("doc", "--gif", argv=[]).gif is False
        for bad in (["--help"], ["--unknown"], ["extra"]):
            with pytest.raises(SystemExit):
                script_args("doc", argv=bad)

    @pytest.mark.parametrize("name", NO_OPTION_RUNNERS)
    def test_help_exits_without_running(self, name):
        # A full run takes minutes; the timeout fails the test if --help ever starts one again.
        out = subprocess.run([sys.executable, "-m", f"scripts.{name}", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0 and out.stdout.lstrip().startswith("usage:")
        bad = subprocess.run([sys.executable, "-m", f"scripts.{name}", "--no-such-option"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert bad.returncode == 2
