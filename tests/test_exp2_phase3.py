"""Experiment 2, Phase 3: checkpoint trajectories, transition detection, decision labels and controlled statistics."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.exp2.emergence import (
    AGREE, EQUIVALENT, OTHER_DISAGREE, TILE_OVER_GOAL, action_effects, checkpoint_file, checkpoint_record,
    classify_decisions, cohens_d, detect_transition, rollout_states, subsample_states, sustained_crossing,
    trajectory_pca, trajectory_steps, verify_trajectory, weights_sha256, within_stratum_auroc,
)
from rewardlens.exp2.evaluation import greedy_policy, rollout
from rewardlens.exp2.layouts import generate_layouts
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import forward_all, linear_cka, relative_change, sample_matched_states

PARAMS = ProceduralEnvParams()
RESULTS = Path("results/exp2/phase3")
EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def _records(steps, hashes=None):
    hashes = hashes or [f"h{s}" for s in steps]
    return [{"step": s, "weights_sha256": h} for s, h in zip(steps, hashes)]


class TestCheckpointTrajectory:
    def test_steps_are_ordered_and_include_both_ends(self):
        steps = trajectory_steps(10_000, 400_000)
        assert steps[0] == 0 and steps[-1] == 400_000 and len(steps) == 41
        assert steps == sorted(set(steps))

    def test_verify_accepts_a_consistent_trajectory(self):
        verify_trajectory(_records([0, 10, 20]), "h0", "h20")

    @pytest.mark.parametrize("records, start, final", [
        (_records([0, 20, 10]), "h0", "h10"),                  # out of order
        (_records([0, 10, 20]), "other", "h20"),               # wrong fork start
        (_records([0, 10, 20]), "h0", "other"),                # wrong final checkpoint
        (_records([10, 20]), "h10", "h20"),                    # does not start at step 0
        (_records([0, 10, 20], ["h0", "same", "same"]), "h0", "same"),  # duplicated weights
    ])
    def test_verify_rejects_broken_trajectories(self, records, start, final):
        with pytest.raises(ValueError):
            verify_trajectory(records, start, final)

    def test_weight_hash_is_deterministic_and_sensitive(self):
        torch.manual_seed(0)
        net = QNetwork()
        first = weights_sha256(net.state_dict())
        assert first == weights_sha256(net.state_dict())
        with torch.no_grad():
            net.fc3.bias[0] += 1e-3
        assert first != weights_sha256(net.state_dict())

    def test_unknown_branch_is_rejected(self):
        with pytest.raises(ValueError):
            checkpoint_file(42, "pretrain", 0)


class TestRecordedRegistry:
    @pytest.fixture
    def registry(self):
        path = RESULTS / "checkpoint_registry.json"
        if not path.exists():
            pytest.skip("Phase 3 registry not present")
        return json.load(open(path))

    def test_every_trajectory_is_ordered_and_shares_its_seeds_fork_start(self, registry):
        for seed, entry in registry["seeds"].items():
            for branch in ("good", "hacking"):
                recs = entry[branch]
                assert [r["step"] for r in recs] == trajectory_steps(10_000, 400_000)
                assert recs[0]["weights_sha256"] == entry["fork_start_weights_sha256"]
                assert all(Path(r["path"]).name == f"checkpoint_{r['step']}.pt" for r in recs)
            assert entry["good"][1]["weights_sha256"] != entry["hacking"][1]["weights_sha256"]

    def test_required_checkpoints_exist_and_match_their_hashes(self, registry):
        req = json.load(open(RESULTS / "required_checkpoints.json"))
        assert len(req["files"]) == 12
        for rec in req["files"]:
            path = Path(rec["path"])
            assert path.exists(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["file_sha256"]
            fresh = checkpoint_record(rec["seed"], rec["branch"], rec["step"], registry["checkpoint_dir"])
            assert fresh["weights_sha256"] == rec["weights_sha256"]

    def test_required_checkpoints_are_the_decision_pairs(self, registry):
        req = json.load(open(RESULTS / "required_checkpoints.json"))
        tl = json.load(open(RESULTS / "timeline.json"))
        for seed in ("42", "123", "456"):
            tr = tl["per_seed"][seed]["summary"]["transition"]["dev"]
            steps = {r["step"] for r in req["files"] if r["seed"] == int(seed)}
            assert steps == {tr["last_before_onset"], tr["completion"]}


class TestMatchedObservations:
    def test_same_seed_gives_identical_states_and_activations(self):
        layouts = generate_layouts(60, seed=5)
        a = sample_matched_states(layouts, 4, 202, PARAMS)
        b = sample_matched_states(layouts, 4, 202, PARAMS)
        assert np.array_equal(a["obs"], b["obs"])
        torch.manual_seed(1)
        net = QNetwork()
        x, y = forward_all(net, a["obs"]), forward_all(net, b["obs"])
        assert all(np.array_equal(x[k], y[k]) for k in ("layer1", "layer2", "q"))
        assert x["layer1"].shape == (240, 128) and x["layer2"].shape == (240, 128)

    def test_phase3_uses_the_phase2_matched_states(self):
        p2 = json.load(open(EXP_DIR / "configs/phase2.json"))["states"]
        p3 = json.load(open(EXP_DIR / "configs/phase3.json"))["states"]
        for key in ("n_fit_layouts", "states_per_layout", "fit_state_seed", "dev_state_seed"):
            assert p2[key] == p3[key]

    def test_rollout_states_cover_every_step_the_policy_takes(self):
        layouts = generate_layouts(40, seed=3)
        torch.manual_seed(2)
        net = QNetwork()
        states = rollout_states(net, layouts, PARAMS)
        res = rollout(greedy_policy(net), layouts, PARAMS)
        assert len(states["steps"]) == int(res["steps"].sum())
        first = states["steps"] == 0
        assert np.array_equal(states["pos"][first], np.array([l.start for l in layouts]))
        assert states["obs"].shape == (len(states["steps"]), 7)
        sub = subsample_states(states, 50, seed=0)
        assert len(sub["steps"]) == 50 and np.array_equal(sub["obs"], subsample_states(states, 50, seed=0)["obs"])


class TestTransitionDetection:
    def test_first_step_from_which_the_rate_stays_above(self):
        steps = [0, 10, 20, 30, 40, 50]
        assert sustained_crossing(steps, [0, 0.2, 0.05, 0.3, 0.6, 0.9], 0.1) == 30   # the early blip does not count
        assert sustained_crossing(steps, [0, 0.2, 0.05, 0.3, 0.6, 0.9], 0.5) == 40
        assert sustained_crossing(steps, [0, 0, 0, 0, 0, 0.05], 0.1) is None
        assert sustained_crossing(steps, [0.9] * 6, 0.5) == 0
        assert sustained_crossing(steps, [0, 0, 0, 0, 0.9, 0.2], 0.5) is None        # falls back at the end

    def test_detect_transition_reports_the_last_step_before_onset(self):
        out = detect_transition([0, 10, 20, 30, 40], [0, 0.05, 0.2, 0.6, 0.9], {"onset": 0.1, "midpoint": 0.5, "completion": 0.8})
        assert out == {"onset": 20, "midpoint": 30, "completion": 40, "last_before_onset": 10}

    def test_recorded_transitions_follow_from_the_recorded_curves(self):
        path = RESULTS / "timeline.json"
        if not path.exists():
            pytest.skip("Phase 3 timeline not present")
        tl = json.load(open(path))
        for seed, block in tl["per_seed"].items():
            farming = block["series"]["hacking"]["dev_farming"]
            tr = detect_transition(tl["steps"], farming, tl["transition_thresholds"])
            assert tr == block["summary"]["transition"]["dev"]
            assert tr["onset"] < tr["midpoint"] < tr["completion"]
            assert max(block["series"]["good"]["dev_farming"]) == 0.0
            assert block["summary"]["max_abs_diff_vs_phase1_log"] < 1e-9


class TestDecisionLabels:
    #                 start    goal     tile
    LAY = np.array([[0, 0, 3, 3, 5, 3]] * 5 + [[0, 0, 3, 3, 3, 5]])
    POS = np.array([[3, 5], [3, 5], [3, 5], [3, 5], [5, 3], [1, 3]])

    def test_action_effects_respect_walls(self):
        eff = action_effects(np.array([[0, 0]]), np.array([[0, 0, 0, 3, 3, 0]]), np.array([LEFT]), PARAMS)
        assert not eff["toward_goal"][0] and not eff["toward_tile"][0]
        eff = action_effects(np.array([[0, 0]]), np.array([[0, 0, 0, 3, 3, 0]]), np.array([RIGHT]), PARAMS)
        assert eff["toward_tile"][0] and not eff["toward_goal"][0]

    def test_labels(self):
        # Agent at (3,5), goal straight up at (3,3), tile up-right at (5,3).
        ref = np.array([UP, UP, UP, UP, LEFT, RIGHT])
        new = np.array([UP, RIGHT, DOWN, LEFT, DOWN, UP])
        labels = classify_decisions(ref, new, self.POS, self.LAY, PARAMS)
        assert labels[0] == AGREE
        assert labels[1] == TILE_OVER_GOAL        # right: closer to the tile, not to the goal
        assert labels[2] == OTHER_DISAGREE        # down: away from both
        assert labels[3] == OTHER_DISAGREE        # left: away from the tile, not closer to the goal
        assert labels[4] == TILE_OVER_GOAL        # standing on the tile and stepping off it instead of toward the goal
        assert labels[5] == OTHER_DISAGREE        # at (1,3): right approaches the goal, up approaches neither

    def test_two_shortest_paths_are_not_counted_as_hacking(self):
        lay = np.array([[0, 0, 3, 3, 7, 7]])
        labels = classify_decisions(np.array([RIGHT]), np.array([DOWN]), np.array([[0, 0]]), lay, PARAMS)
        assert labels[0] == EQUIVALENT

    def test_recorded_labels_cover_every_state(self):
        path = RESULTS / "decision_analysis.json"
        if not path.exists():
            pytest.skip("Phase 3 decision analysis not present")
        dec = json.load(open(path))
        for block in dec["per_seed"].values():
            assert block["goal_following_checkpoint_step"] < block["hacking_checkpoint_step"]
            assert set(block["sampling"]) == {"uniform", "rollout_goal_following", "rollout_hacking"}
            for res in block["sampling"].values():
                frac = res["label_fraction"]
                assert abs(sum(frac.values()) - 1) < 1e-9
                assert abs(res["disagreement_rate"] - (1 - frac["agree"])) < 1e-9


class TestMetrics:
    def test_cka_and_relative_change_of_identical_sets(self):
        x = np.random.RandomState(0).randn(200, 16)
        assert linear_cka(x, x) == pytest.approx(1.0)
        assert linear_cka(x, 3 * x) == pytest.approx(1.0)
        assert relative_change(x, x) == 0.0
        assert relative_change(x, 2 * x) == pytest.approx(1.0)

    def test_stratified_auroc_removes_what_the_strata_explain(self):
        rng = np.random.RandomState(0)
        strata = rng.randint(0, 4, 4000)
        y = rng.rand(4000) < 0.1 + 0.2 * strata
        confounded = strata + 0.01 * rng.randn(4000)       # tracks the stratum, nothing else
        res = within_stratum_auroc(confounded, y, strata[:, None])
        assert res["n_strata"] == 4 and abs(res["auroc"] - 0.5) < 0.05
        informative = y + 0.1 * rng.randn(4000)
        assert within_stratum_auroc(informative, y, strata[:, None])["auroc"] > 0.99

    def test_stratified_auroc_skips_strata_without_both_classes(self):
        y = np.array([True] * 10 + [False] * 10)
        strata = np.array([0] * 10 + [1] * 10)[:, None]
        assert within_stratum_auroc(np.arange(20.0), y, strata)["auroc"] is None

    def test_cohens_d(self):
        rng = np.random.RandomState(1)
        assert cohens_d(rng.randn(5000) + 1, rng.randn(5000)) == pytest.approx(1.0, abs=0.08)
        assert cohens_d(np.array([1.0]), np.array([1.0, 2.0])) is None

    def test_trajectory_pca_of_points_on_a_line(self):
        direction = np.random.RandomState(2).randn(500)
        rows = np.outer(np.arange(10.0), direction)
        coords, ratio = trajectory_pca(rows, 2)
        assert ratio[0] == pytest.approx(1.0) and ratio[1] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.diff(coords[:, 0]) > 0) or np.all(np.diff(coords[:, 0]) < 0)


class TestEvaluationIntegrity:
    SOURCES = ("rewardlens/exp2/emergence.py", "experiments/02_generalizable_reward_hacking/scripts/exp2_run_phase3.py", "experiments/02_generalizable_reward_hacking/scripts/exp2_visualize_phase3.py")

    def test_phase3_code_never_touches_test_or_final_eval_layouts(self):
        for path in self.SOURCES:
            src = Path(path).read_text()
            assert "final_eval" not in src and "load_layout_set" not in src, path
            assert not re.search(r"""\[\s*["']test["']\s*\]""", src), path

    def test_setup_only_exposes_training_and_development_layouts(self):
        from rewardlens.exp2.layouts import load_manifest
        from scripts.exp2_run_phase3 import load_setup
        setup = load_setup()
        splits, _ = load_manifest(EXP_DIR / "layouts/manifest.json")
        assert set(setup["layouts"]) == {"fit", "dev"} and set(setup["states"]) == {"fit", "dev"}
        assert setup["layouts"]["fit"] == splits["train"][:1500] and setup["layouts"]["dev"] == splits["dev"]
        test = {l.as_tuple() for l in splits["test"]}
        used = {l.as_tuple() for v in setup["layouts"].values() for l in v}
        assert not test & used
        assert not {l.as_tuple() for l in setup["layouts"]["fit"]} & {l.as_tuple() for l in setup["layouts"]["dev"]}

    def test_results_record_only_train_and_dev(self):
        for name in ("checkpoint_registry.json", "timeline.json", "decision_analysis.json", "required_checkpoints.json"):
            path = RESULTS / name
            if not path.exists():
                pytest.skip("Phase 3 results not present")
            assert json.load(open(path))["splits_used"] == ["train", "dev"]

    def test_protected_manifests_are_unchanged_from_the_committed_versions(self):
        for name in ("manifest.json", "final_eval_manifest.json"):
            path = EXP_DIR / "layouts" / name
            try:
                committed = subprocess.run(["git", "rev-parse", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout.strip()
                current = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                pytest.skip("git not available")
            assert committed == current, name
