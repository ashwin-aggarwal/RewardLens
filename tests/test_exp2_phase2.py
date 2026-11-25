"""Experiment 2, Phase 2: matched states, representation comparisons and activation transplantation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.exp2.evaluation import greedy_policy, rollout, summarize
from rewardlens.exp2.layouts import generate_layouts
from rewardlens.exp2.patching import donor_activation, make_policy, mean_q_shift, patched_q
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.representations import (
    DifferencePCA, forward_all, held_out_r2, linear_cka, relative_change, sample_matched_states, state_features,
    top_pc_scores,
)

PARAMS = ProceduralEnvParams()
RESULTS = Path("results/exp2/phase2")


@pytest.fixture(scope="module")
def layouts():
    return generate_layouts(120, seed=9)


@pytest.fixture(scope="module")
def nets():
    out = []
    for s in (0, 1):
        torch.manual_seed(s)
        out.append(QNetwork().eval())
    return out


class TestMatchedStates:
    def test_deterministic_valid_and_policy_independent(self, layouts):
        a = sample_matched_states(layouts, 3, 5, PARAMS)
        b = sample_matched_states(layouts, 3, 5, PARAMS)
        np.testing.assert_array_equal(a["obs"], b["obs"])
        assert a["obs"].shape == (360, 7) and a["obs"].min() >= 0 and a["obs"].max() <= 1
        assert not (a["pos"] == a["layouts"][:, 2:4]).all(axis=1).any()      # never on the goal
        assert (a["obs"][:, 6] > 0).all()                                    # time left is never zero
        assert not np.array_equal(a["obs"], sample_matched_states(layouts, 3, 6, PARAMS)["obs"])
        np.testing.assert_array_equal(a["layout_index"][:6], [0, 0, 0, 1, 1, 1])

    def test_features(self, layouts):
        st = sample_matched_states(layouts, 2, 1, PARAMS)
        f, names = state_features(st, PARAMS)
        assert f.shape == (240, len(names))
        i = names.index
        np.testing.assert_allclose(f[:, i("goal_distance")], np.abs(f[:, i("goal_dx")]) + np.abs(f[:, i("goal_dy")]))
        np.testing.assert_allclose(f[:, i("time_left")], st["obs"][:, 6], rtol=1e-6)


class TestRepresentations:
    def test_forward_all_matches_network(self, layouts, nets):
        obs = sample_matched_states(layouts, 2, 1, PARAMS)["obs"]
        acts = forward_all(nets[0], obs)
        assert acts["layer1"].shape == (240, 128) and acts["layer2"].shape == (240, 128)
        assert acts["layer1"].min() >= 0 and acts["layer2"].min() >= 0
        np.testing.assert_allclose(acts["q"], nets[0](torch.from_numpy(obs)).detach().numpy(), atol=1e-5)

    def test_cka_ignores_neuron_order_and_rotation(self):
        rng = np.random.RandomState(0)
        x = rng.randn(300, 20)
        rot, _ = np.linalg.qr(rng.randn(20, 20))
        assert linear_cka(x, x[:, rng.permutation(20)]) == pytest.approx(1.0)
        assert linear_cka(x, 3.0 * x @ rot) == pytest.approx(1.0)
        assert linear_cka(x, rng.randn(300, 20)) < 0.3

    def test_relative_change(self):
        x = np.ones((10, 4))
        assert relative_change(x, x) == 0.0
        assert relative_change(x, 2 * x) == pytest.approx(1.0)

    def test_difference_pca_recovers_planted_structure(self):
        rng = np.random.RandomState(1)
        basis, _ = np.linalg.qr(rng.randn(30, 3))
        offset = 5.0 * rng.randn(30)

        def make(n):
            return offset + rng.randn(n, 3) @ (basis.T * np.array([3.0, 2.0, 1.0])[:, None]) + 0.01 * rng.randn(n, 30)

        pca = DifferencePCA(make(2000))
        s = pca.summary(make(1000), top_k=3, threshold=0.9)
        assert s["cumulative_top_k"] > 0.99 and s["n_components_for_threshold"] <= 3
        assert 0.5 < s["constant_offset_share"] < 1.0
        assert pca.scores(make(5), 3).shape == (5, 3)
        assert s["explained_variance_ratio"][0] > s["explained_variance_ratio"][1] > s["explained_variance_ratio"][2]

    def test_held_out_r2_finds_shared_structure_despite_misalignment(self):
        rng = np.random.RandomState(2)
        z_fit, z_eval = rng.randn(800, 4), rng.randn(400, 4)
        a, b = rng.randn(4, 4), rng.randn(4, 6)          # two different, misaligned linear read-outs of z
        assert held_out_r2(z_fit @ a, z_fit @ b, z_eval @ a, z_eval @ b) > 0.99
        perm = rng.permutation(800)
        assert abs(held_out_r2((z_fit @ a)[perm], z_fit @ b, z_eval @ a, z_eval @ b)) < 0.1
        assert held_out_r2(rng.randn(800, 4), z_fit @ b, rng.randn(400, 4), z_eval @ b) < 0.1

    def test_top_pc_scores_shapes(self):
        rng = np.random.RandomState(3)
        f, e = top_pc_scores(rng.randn(200, 12), rng.randn(50, 12), 5)
        assert f.shape == (200, 5) and e.shape == (50, 5)


class TestPatching:
    def test_alpha_zero_and_self_patch_change_nothing(self, layouts, nets):
        obs = torch.from_numpy(sample_matched_states(layouts, 2, 1, PARAMS)["obs"])
        rec, donor = nets
        with torch.no_grad():
            base = rec(obs)
            for layer in (1, 2):
                assert torch.allclose(patched_q(rec, obs, donor_activation(donor, obs, layer), layer, alpha=0.0), base, atol=1e-5)
                assert torch.allclose(patched_q(rec, obs, donor_activation(rec, obs, layer), layer, alpha=1.0), base, atol=1e-5)

    def test_full_layer2_patch_is_recipient_head_on_donor_trunk(self, layouts, nets):
        obs = torch.from_numpy(sample_matched_states(layouts, 2, 1, PARAMS)["obs"])
        rec, donor = nets
        with torch.no_grad():
            expected = rec.fc3(torch.relu(donor.fc2(torch.relu(donor.fc1(obs)))))
            assert torch.allclose(patched_q(rec, obs, donor_activation(donor, obs, 2), 2), expected, atol=1e-5)
            expected1 = rec.fc3(torch.relu(rec.fc2(torch.relu(donor.fc1(obs)))))
            assert torch.allclose(patched_q(rec, obs, donor_activation(donor, obs, 1), 1), expected1, atol=1e-5)
        with pytest.raises(ValueError):
            patched_q(rec, obs, donor_activation(donor, obs, 2), 3)

    def test_unmodified_policy_equals_greedy(self, layouts, nets):
        obs = sample_matched_states(layouts, 2, 1, PARAMS)["obs"]
        np.testing.assert_array_equal(make_policy(nets[0])(obs), greedy_policy(nets[0])(obs))

    def test_matched_patch_with_identical_donor_is_a_no_op(self, layouts, nets):
        a = summarize(rollout(make_policy(nets[0]), layouts, PARAMS), 5)
        b = summarize(rollout(make_policy(nets[0], nets[0], 2, "matched"), layouts, PARAMS), 5)
        assert a == b

    def test_q_bias_is_state_independent(self, layouts, nets):
        fit = sample_matched_states(layouts, 2, 1, PARAMS)["obs"]
        bias = mean_q_shift(nets[0], nets[1], fit, 2)
        assert bias.shape == (4,)
        obs = sample_matched_states(layouts, 1, 2, PARAMS)["obs"]
        with torch.no_grad():
            expected = (nets[0](torch.from_numpy(obs)) + torch.from_numpy(bias)).argmax(dim=1).numpy()
        np.testing.assert_array_equal(make_policy(nets[0], nets[1], 2, "q_bias", q_bias=bias)(obs), expected)
        # A bias equal for all actions never changes the greedy action.
        flat = make_policy(nets[0], nets[1], 2, "q_bias", q_bias=np.full(4, 3.0, dtype=np.float32))(obs)
        np.testing.assert_array_equal(flat, greedy_policy(nets[0])(obs))

    def test_controls_are_seeded_and_differ_from_matched(self, layouts, nets):
        pool = sample_matched_states(layouts, 2, 1, PARAMS)["obs"]
        obs = sample_matched_states(layouts, 2, 3, PARAMS)["obs"]
        matched = make_policy(nets[0], nets[1], 2, "matched")(obs)
        for mode in ("shuffled", "wrong_position"):
            a = make_policy(nets[0], nets[1], 2, mode, pool_obs=pool, seed=4)(obs)
            b = make_policy(nets[0], nets[1], 2, mode, pool_obs=pool, seed=4)(obs)
            np.testing.assert_array_equal(a, b)
            assert (a != matched).any()
        with pytest.raises(ValueError):
            make_policy(nets[0], nets[1], 2, "mystery")


class TestEvaluationIntegrity:
    SOURCES = ("rewardlens/exp2/representations.py", "rewardlens/exp2/patching.py", "experiments/02_generalizable_reward_hacking/scripts/exp2_run_phase2.py",
               "experiments/02_generalizable_reward_hacking/scripts/exp2_visualize_phase2.py")

    def test_phase2_code_never_touches_test_or_final_eval_layouts(self):
        for path in self.SOURCES:
            src = Path(path).read_text()
            assert "final_eval" not in src and "load_layout_set" not in src, path
            assert not re.search(r"""\[\s*["']test["']\s*\]""", src), path

    def test_setup_only_exposes_train_and_dev(self):
        from scripts.exp2_run_phase2 import load_setup
        if not Path("results/exp2/phase1/checkpoints.json").exists():
            pytest.skip("checkpoints not present")
        from rewardlens.exp2.layouts import load_manifest
        setup = load_setup()
        splits, _ = load_manifest("experiments/02_generalizable_reward_hacking/layouts/manifest.json")
        test = {l.as_tuple() for l in splits["test"]}
        assert set(setup["layouts"]) == {"fit", "dev"}
        assert not test & {l.as_tuple() for v in setup["layouts"].values() for l in v}
        assert setup["layouts"]["dev"] == splits["dev"] and setup["layouts"]["fit"] == splits["train"][:1500]
        assert set(setup["nets"]) == {42, 123, 456}

    def test_recorded_results_cover_all_seeds_and_reproduce(self):
        if not (RESULTS / "patching_results.json").exists():
            pytest.skip("phase 2 results not present")
        from scripts.exp2_run_phase2 import GAMMA, load_setup
        pat = json.load(open(RESULTS / "patching_results.json"))
        rep = json.load(open(RESULTS / "representation_analysis.json"))
        assert set(pat["per_seed"]) == set(rep["within_seed"]) == {"42", "123", "456"}
        assert pat["splits_used"] == rep["splits_used"] == ["train", "dev"]
        setup = load_setup()
        trip = setup["nets"][42]
        s = summarize(rollout(make_policy(trip["hacking"], trip["good"], 2, "matched"), setup["layouts"]["dev"],
                              setup["params"], GAMMA), 5)
        rec = pat["per_seed"]["42"]["good_to_hacking"]["layer2"]["matched"]
        for k in ("direct_goal", "farming", "other", "mean_reward_tile_visits"):
            assert s[k] == pytest.approx(rec[k])
