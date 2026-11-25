"""Experiment 2, Phase 4: near-transition checkpoint pairs, patching correctness, controls and evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.exp2.evaluation import rollout, summarize
from rewardlens.exp2.layouts import generate_layouts
from rewardlens.exp2.near_transition import (
    action_diagnostics, compatibility, meets_success, restoration_fraction, select_pairs, spearman, sweep_triggered,
)
from rewardlens.exp2.patching import donor_activation, make_policy, mean_q_shift, patched_q
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import sample_matched_states

PARAMS = ProceduralEnvParams()
RESULTS = Path("results/exp2/phase4")
EXP_DIR = Path("experiments/02_generalizable_reward_hacking")
STEPS = list(range(0, 400_001, 10_000))
TRANSITION = {"onset": 70_000, "midpoint": 100_000, "completion": 160_000, "last_before_onset": 60_000}


def _nets(seed_a=0, seed_b=1):
    torch.manual_seed(seed_a)
    a = QNetwork()
    torch.manual_seed(seed_b)
    b = QNetwork()
    return a.eval(), b.eval()


def _behavior(dg, other=0.0):
    return {"direct_goal": dg, "other": other}


def _control(dg):
    return {"direct_goal": {"mean": dg}}


class TestPairSelection:
    def test_rule_gives_the_documented_pairs(self):
        pairs = select_pairs(TRANSITION, STEPS, [10_000, 20_000, 50_000], True, True,
                             ("pre_onset_to_final", "good_same_step_to_completion"))
        got = [(p["name"], p["donor_branch"], p["donor_step"], p["recipient_step"]) for p in pairs]
        assert got == [("minus_10k", "hacking", 150_000, 160_000), ("minus_20k", "hacking", 140_000, 160_000),
                       ("minus_50k", "hacking", 110_000, 160_000), ("pre_onset", "hacking", 60_000, 160_000),
                       ("fork_start", "hacking", 0, 160_000), ("pre_onset_to_final", "hacking", 60_000, 400_000),
                       ("good_same_step", "good", 160_000, 160_000)]
        main = [p for p in pairs if p["kind"] == "main"]
        assert all(p["recipient_step"] == 160_000 and p["recipient_branch"] == "hacking" for p in main)
        assert [p["gap"] for p in main] == sorted(p["gap"] for p in main)

    def test_rule_does_not_depend_on_anything_but_the_transition(self):
        a = select_pairs(TRANSITION, STEPS, [50_000, 10_000, 20_000])
        b = select_pairs(TRANSITION, STEPS, [10_000, 20_000, 50_000])
        assert a == b

    def test_gaps_that_reach_before_the_fork_are_skipped(self):
        early = {"onset": 20_000, "completion": 30_000, "last_before_onset": 10_000}
        names = [p["name"] for p in select_pairs(early, STEPS, [10_000, 50_000], False, False)]
        assert names == ["minus_10k"]

    def test_missing_checkpoint_is_an_error(self):
        with pytest.raises(ValueError):
            select_pairs({**TRANSITION, "completion": 165_000}, STEPS, [10_000])

    def test_duplicate_pairs_are_an_error(self):
        # A pre-onset checkpoint 10k before completion duplicates the 10k-gap pair.
        with pytest.raises(ValueError):
            select_pairs({**TRANSITION, "last_before_onset": 150_000}, STEPS, [10_000])

    def test_recorded_pairs_follow_the_rule(self):
        path = RESULTS / "patching_results.json"
        if not path.exists():
            pytest.skip("Phase 4 results not present")
        res = json.load(open(path))
        cfg = json.load(open(EXP_DIR / "configs/phase4.json"))["pairs"]
        tl = json.load(open("results/exp2/phase3/timeline.json"))
        for seed, block in res["per_seed"].items():
            tr = tl["per_seed"][seed]["summary"]["transition"]["dev"]
            want = select_pairs(tr, tl["steps"], cfg["gaps_before_completion"], cfg["include_last_before_onset"],
                                cfg["include_fork_start"], tuple(cfg["reference_pairs"]))
            got = [{k: p[k] for k in want[0]} for p in block["pairs"]]
            assert got == want


class TestRequiredCheckpoints:
    def test_listed_checkpoints_match_their_hashes(self):
        import hashlib
        path = RESULTS / "required_checkpoints.json"
        if not path.exists():
            pytest.skip("Phase 4 results not present")
        from rewardlens.exp2.emergence import load_trajectory_net, weights_sha256
        for rec in json.load(open(path))["files"]:
            p = Path(rec["path"])
            if not p.exists():
                pytest.skip(f"{p} not present; regenerate with scripts.exp2_train")
            assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["file_sha256"]
            assert weights_sha256(load_trajectory_net(p)[0].state_dict()) == rec["weights_sha256"]

    def test_step_0_and_400k_load_the_committed_checkpoints_with_identical_weights(self):
        from rewardlens.exp2.emergence import checkpoint_file, load_trajectory_net, weights_sha256
        from scripts.exp2_run_phase4 import checkpoint_for
        assert checkpoint_for(42, "hacking", 0, "results/exp2/phase1").name == "checkpoint_final.pt"
        assert "pretrain" in str(checkpoint_for(42, "hacking", 0, "results/exp2/phase1"))
        assert checkpoint_for(42, "hacking", 150_000, "results/exp2/phase1").name == "checkpoint_150000.pt"
        traj = checkpoint_file(42, "hacking", 400_000, "results/exp2/phase1")
        if not traj.exists():
            pytest.skip("trajectory checkpoints not present")
        a = weights_sha256(load_trajectory_net(checkpoint_for(42, "hacking", 400_000, "results/exp2/phase1"))[0].state_dict())
        assert a == weights_sha256(load_trajectory_net(traj)[0].state_dict())


class TestInterventions:
    def test_full_layer2_replacement_feeds_donor_activation_to_recipient_head(self):
        donor, rec = _nets()
        obs = torch.rand(64, 7)
        with torch.no_grad():
            q = patched_q(rec, obs, donor_activation(donor, obs, 2), 2, 1.0)
            expected = rec.fc3(torch.relu(donor.fc2(torch.relu(donor.fc1(obs)))))
        assert torch.allclose(q, expected, atol=1e-6)

    def test_full_layer1_replacement_runs_recipient_layers_2_and_3(self):
        donor, rec = _nets()
        obs = torch.rand(64, 7)
        with torch.no_grad():
            q = patched_q(rec, obs, donor_activation(donor, obs, 1), 1, 1.0)
            expected = rec.fc3(torch.relu(rec.fc2(torch.relu(donor.fc1(obs)))))
        assert torch.allclose(q, expected, atol=1e-6)

    @pytest.mark.parametrize("layer", [1, 2])
    def test_alpha_zero_and_self_patch_are_identity(self, layer):
        donor, rec = _nets()
        obs = torch.rand(64, 7)
        with torch.no_grad():
            assert torch.allclose(patched_q(rec, obs, donor_activation(donor, obs, layer), layer, 0.0), rec(obs), atol=1e-6)
            assert torch.allclose(patched_q(rec, obs, donor_activation(rec, obs, layer), layer, 1.0), rec(obs), atol=1e-6)

    def test_shuffled_donor_uses_pool_states_not_the_current_one(self):
        donor, rec = _nets()
        obs = np.random.RandomState(0).rand(50, 7).astype(np.float32)
        pool = np.random.RandomState(1).rand(200, 7).astype(np.float32)
        matched = make_policy(rec, donor, 2, "matched")(obs)
        shuffled = [make_policy(rec, donor, 2, "shuffled", 1.0, pool, seed=s)(obs) for s in range(3)]
        assert any((s != matched).any() for s in shuffled)
        again = make_policy(rec, donor, 2, "shuffled", 1.0, pool, seed=0)(obs)
        assert np.array_equal(again, shuffled[0])

    def test_q_bias_adds_a_constant_to_every_state(self):
        donor, rec = _nets()
        fit = np.random.RandomState(0).rand(300, 7).astype(np.float32)
        bias = mean_q_shift(rec, donor, fit, 2)
        with torch.no_grad():
            t = torch.from_numpy(fit)
            direct = (patched_q(rec, t, donor_activation(donor, t, 2), 2) - rec(t)).mean(0).numpy()
            assert np.allclose(bias, direct, atol=1e-5)
            obs = np.random.RandomState(2).rand(40, 7).astype(np.float32)
            expected = (rec(torch.from_numpy(obs)) + torch.from_numpy(bias)).argmax(1).numpy()
        assert np.array_equal(make_policy(rec, donor, 2, "q_bias", q_bias=bias)(obs), expected)
        # A zero bias is the unmodified recipient.
        assert np.array_equal(make_policy(rec, donor, 2, "q_bias", q_bias=np.zeros(4))(obs), make_policy(rec)(obs))


class TestDiagnostics:
    def test_compatibility_of_a_network_with_itself(self):
        a, _ = _nets()
        obs = np.random.RandomState(0).rand(300, 7).astype(np.float32)
        c = compatibility(a, a, obs)
        assert c["layer2"]["cka"] == pytest.approx(1.0) and c["layer2"]["norm_ratio"] == pytest.approx(1.0)
        assert c["output_head_drift"] == 0.0 and c["greedy_action_agreement"] == 1.0

    def test_action_diagnostics_of_a_self_patch(self):
        a, _ = _nets()
        layouts = generate_layouts(60, seed=4)
        states = sample_matched_states(layouts, 2, 0, PARAMS)
        d = action_diagnostics(a, a, states, 2, PARAMS)
        assert d["patched_action_changed"] == 0.0 and d["patched_matches_donor"] == 1.0
        assert d["label_fraction"]["agree"] == 1.0 and d["at_flip_states"]["n"] == 0
        assert d["mean_abs_q_change"] == pytest.approx(0.0, abs=1e-6)

    def test_action_diagnostics_of_a_full_swap_follow_the_donor(self):
        donor, rec = _nets()
        layouts = generate_layouts(60, seed=4)
        states = sample_matched_states(layouts, 2, 0, PARAMS)
        # Layer-2 patch of a recipient whose head equals the donor's reproduces the donor exactly.
        rec.fc3.load_state_dict(donor.fc3.state_dict())
        d = action_diagnostics(rec, donor, states, 2, PARAMS)
        assert d["patched_matches_donor"] == 1.0
        if d["at_flip_states"]["n"]:
            assert d["at_flip_states"]["restored_to_donor_action"] == 1.0


class TestEvaluation:
    def test_restoration_fraction(self):
        assert restoration_fraction(0.5, 0.1, 0.9) == pytest.approx(0.5)
        assert restoration_fraction(0.9, 0.1, 0.9) == pytest.approx(1.0)
        assert restoration_fraction(0.5, 0.1, 0.2) is None          # donor has no real advantage

    def test_success_requires_goal_restoration_not_just_less_farming(self):
        crit = {"min_gain_over_recipient": 0.2, "min_gain_over_best_state_control": 0.1, "max_other": 0.25}
        rec = _behavior(0.1)
        good = meets_success(_behavior(0.7, other=0.1), rec, [_control(0.3), _control(0.4)], crit)
        assert good["success"] and good["best_state_control_direct_goal"] == 0.4
        disrupt = meets_success(_behavior(0.45, other=0.5), rec, [_control(0.2), _control(0.3)], crit)
        assert not disrupt["success"] and not disrupt["checks"]["other_at_most"]
        control_as_good = meets_success(_behavior(0.7, other=0.1), rec, [_control(0.65), _control(0.3)], crit)
        assert not control_as_good["success"] and not control_as_good["checks"]["gain_over_best_state_control"]
        small = meets_success(_behavior(0.25, other=0.0), rec, [_control(0.1), _control(0.1)], crit)
        assert not small["checks"]["gain_over_recipient"]

    def test_sweep_trigger(self):
        assert sweep_triggered(_behavior(0.5), _behavior(0.1), [_control(0.3), _control(0.35)])
        assert not sweep_triggered(_behavior(0.5), _behavior(0.1), [_control(0.45), _control(0.3)])
        assert not sweep_triggered(_behavior(0.15), _behavior(0.1), [_control(0.0), _control(0.0)])

    def test_spearman(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
        assert spearman([1, 2], [1, 2]) is None
        assert spearman([1, 1, 1], [1, 2, 3]) is None

    def test_categories_partition_every_episode(self):
        _, rec = _nets()
        layouts = generate_layouts(80, seed=6)
        s = summarize(rollout(make_policy(rec), layouts, PARAMS), PARAMS.exploit_visit_threshold)
        assert s["direct_goal"] + s["farm_then_finish"] + s["farm_only"] + s["other"] == pytest.approx(1.0)
        assert s["farming"] == pytest.approx(s["farm_only"] + s["farm_then_finish"])

    def test_recorded_results_are_internally_consistent(self):
        path = RESULTS / "patching_results.json"
        if not path.exists():
            pytest.skip("Phase 4 results not present")
        res = json.load(open(path))
        crit = json.load(open(EXP_DIR / "configs/phase4.json"))["success_criterion"]
        assert res["splits_used"] == ["train", "dev"] and res["n_dev_layouts"] == 1500
        for block in res["per_seed"].values():
            for p in block["pairs"]:
                for layer in ("layer1", "layer2"):
                    m = p["layers"][layer]["matched"]
                    assert m["direct_goal"] + m["farm_then_finish"] + m["farm_only"] + m["other"] == pytest.approx(1.0)
                    assert p["layers"][layer]["shuffled"]["n_draws"] == 5
                l2 = p["layers"]["layer2"]
                again = meets_success(l2["matched"], p["recipient"], [l2["shuffled"], l2["wrong_position"]], crit)
                assert again == p["evaluation"]["success"]
                assert ("alpha_sweep_layer2" in p) == p["evaluation"]["sweep_triggered"]


class TestEvaluationIntegrity:
    SOURCES = ("rewardlens/exp2/near_transition.py", "experiments/02_generalizable_reward_hacking/scripts/exp2_run_phase4.py", "experiments/02_generalizable_reward_hacking/scripts/exp2_visualize_phase4.py")

    def test_phase4_code_never_touches_test_or_final_eval_layouts(self):
        for path in self.SOURCES:
            if not Path(path).exists():
                continue
            src = Path(path).read_text()
            assert "final_eval" not in src and "load_layout_set" not in src, path
            assert not re.search(r"""\[\s*["']test["']\s*\]""", src), path

    def test_setup_only_exposes_training_and_development_layouts(self):
        from rewardlens.exp2.layouts import load_manifest
        from scripts.exp2_run_phase4 import load_setup
        setup = load_setup()
        splits, _ = load_manifest(EXP_DIR / "layouts/manifest.json")
        assert set(setup["layouts"]) == {"fit", "dev"}
        assert setup["layouts"]["fit"] == splits["train"][:1500] and setup["layouts"]["dev"] == splits["dev"]
        assert not {l.as_tuple() for l in splits["test"]} & {l.as_tuple() for v in setup["layouts"].values() for l in v}
