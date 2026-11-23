"""Tests for Phase 4.5: goal-directed activation patching."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork, load_q_network_for_eval
from rewardlens.analysis.activation_extraction import build_matched_state_dataset
from rewardlens.analysis.activation_intervention import partition_start_positions
from rewardlens.analysis.activation_patching import (
    PatchStats,
    donor_layer2,
    evaluate_patching,
    fit_linear_mapping,
    identity_mapping,
    layer2_patching,
    make_matched_donor,
    make_shuffled_donor,
    patched_rollout,
    q_change_analysis,
)
from rewardlens.envs.reward_grid import RewardGridConfig


@pytest.fixture
def nets():
    torch.manual_seed(0)
    recipient = QNetwork()
    torch.manual_seed(1)
    donor = QNetwork()
    return recipient, donor


@pytest.fixture
def obs():
    return torch.as_tensor(np.random.RandomState(0).rand(12, 7).astype(np.float32))


class TestPatchSemantics:
    def test_alpha_zero_is_recipient(self, nets, obs):
        rec, don = nets
        with torch.no_grad():
            q0 = rec(obs).clone()
        with layer2_patching(rec, 0.0, make_matched_donor(don), "matched"), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, q0)

    def test_alpha_one_is_full_replacement(self, nets, obs):
        rec, don = nets
        with torch.no_grad():
            expected = rec.fc3(donor_layer2(don, obs))
        with layer2_patching(rec, 1.0, make_matched_donor(don), "matched"), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, expected, atol=1e-5, rtol=1e-5)

    def test_intermediate_alpha_interpolates(self, nets, obs):
        rec, don = nets
        with torch.no_grad():
            h_r = rec.relu2(rec.fc2(rec.relu1(rec.fc1(obs))))
            h_d = donor_layer2(don, obs)
            expected = rec.fc3(h_r + 0.5 * (h_d - h_r))
        with layer2_patching(rec, 0.5, make_matched_donor(don), "matched"), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, expected, atol=1e-5, rtol=1e-5)

    def test_donor_sees_identical_observation(self, nets, obs):
        rec, don = nets
        seen = {}
        orig = make_matched_donor(don)

        def spy(o):
            seen["obs"] = o.clone()
            return orig(o)

        with layer2_patching(rec, 1.0, spy, "matched"), torch.no_grad():
            rec(obs)
        torch.testing.assert_close(seen["obs"], obs)

    def test_post_relu_capture_nonnegative(self, nets, obs):
        _, don = nets
        with torch.no_grad():
            h = donor_layer2(don, obs)
        assert (h >= 0).all()

    def test_weights_unchanged_and_hooks_removed(self, nets, obs):
        rec, don = nets
        before = copy.deepcopy(rec.state_dict())
        with layer2_patching(rec, 1.0, make_matched_donor(don), "matched"), torch.no_grad():
            rec(obs)
        for k, v in rec.state_dict().items():
            torch.testing.assert_close(v, before[k])
        assert len(rec.relu2._forward_hooks) == 0
        assert len(rec.fc1._forward_pre_hooks) == 0
        with torch.no_grad():
            torch.testing.assert_close(rec(obs), rec(obs))

    def test_hooks_removed_after_exception(self, nets, obs):
        rec, don = nets
        with pytest.raises(RuntimeError):
            with layer2_patching(rec, 1.0, make_matched_donor(don), "matched"):
                raise RuntimeError("boom")
        assert len(rec.relu2._forward_hooks) == 0
        assert len(rec.fc1._forward_pre_hooks) == 0


class TestControls:
    def test_random_patch_norm_matched_and_reproducible(self, nets, obs):
        rec, don = nets
        fn = make_matched_donor(don)
        s_m, s_r1, s_r2 = PatchStats(), PatchStats(), PatchStats()
        with layer2_patching(rec, 0.5, fn, "matched", stats=s_m), torch.no_grad():
            rec(obs)
        with layer2_patching(rec, 0.5, fn, "random", random_seed=3, stats=s_r1), torch.no_grad():
            q1 = rec(obs).clone()
        with layer2_patching(rec, 0.5, fn, "random", random_seed=3, stats=s_r2), torch.no_grad():
            q2 = rec(obs)
        assert s_m.summary()["mean_applied_norm"] == pytest.approx(
            s_r1.summary()["mean_applied_norm"], rel=1e-4)
        torch.testing.assert_close(q1, q2)

    def test_shuffled_donor_reproducible_and_differs_from_matched(self, nets, obs):
        rec, don = nets
        pool = np.random.RandomState(5).rand(50, 7).astype(np.float32)
        with torch.no_grad():
            a = make_shuffled_donor(don, pool, seed=1)(obs)
            b = make_shuffled_donor(don, pool, seed=1)(obs)
            m = make_matched_donor(don)(obs)
        torch.testing.assert_close(a, b)
        assert not torch.allclose(a, m)

    def test_constant_mode_is_state_independent_q_shift(self, nets, obs):
        rec, _ = nets
        delta = np.random.RandomState(2).randn(128).astype(np.float32)
        with torch.no_grad():
            q0 = rec(obs).clone()
            expected = (0.7 * torch.as_tensor(delta)) @ rec.fc3.weight.T
        with layer2_patching(rec, 0.7, None, "constant", constant_delta=delta), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q - q0, expected.expand_as(q0), atol=1e-5, rtol=1e-5)

    def test_q_change_analysis_constant_fraction(self, nets):
        rec, _ = nets
        obs_np = np.random.RandomState(0).rand(30, 7).astype(np.float32)
        delta = np.random.RandomState(2).randn(128).astype(np.float32)
        r = q_change_analysis(rec, obs_np, 0.5, None, "constant", constant_delta=delta)
        assert r["constant_fraction_of_dq_energy"] == pytest.approx(1.0, abs=1e-4)


class TestMappingAndSplits:
    def test_ridge_mapping_fits_linear_relation(self):
        rng = np.random.RandomState(0)
        hd = np.abs(rng.randn(300, 128)).astype(np.float32)
        W = rng.randn(128, 128).astype(np.float32) * 0.1
        hr = hd @ W.T + 0.5
        M = fit_linear_mapping(hd, hr, ridge_alpha=1e-3)
        assert M.fit_r2 > 0.99
        np.testing.assert_allclose(hd[:5] @ M.weight.T + M.bias, hr[:5], atol=0.2)

    def test_identity_mapping_is_identity(self, nets, obs):
        _, don = nets
        with torch.no_grad():
            h = donor_layer2(don, obs)
            torch.testing.assert_close(identity_mapping().apply(h), h)

    def test_partition_fresh_seed_disjoint(self):
        parts = partition_start_positions(RewardGridConfig(), n_dev=16, n_test=16, seed=11)
        s = [set(parts[k]) for k in ("train", "dev", "test")]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
        assert len(s[0] | s[1] | s[2]) == 62

    def test_mapping_fit_excludes_test_positions(self):
        """Mapping is fitted only on train-position states; test positions never enter."""
        from scripts.run_interventions import _position_mask
        obs = build_matched_state_dataset()
        parts = partition_start_positions(RewardGridConfig(), seed=11)
        fit_mask = _position_mask(obs, parts["train"])
        test_mask = _position_mask(obs, parts["test"])
        assert not (fit_mask & test_mask).any()


class TestRollouts:
    def test_rollout_uses_actual_trajectory(self, nets):
        rec, don = nets
        cfg = RewardGridConfig(agent_start=(0, 0), max_steps=25)
        r = patched_rollout(rec, cfg, 0.5, make_matched_donor(don), "matched")
        assert r["episode_steps"] == len(r["actions"]) <= 25
        assert r["patch"]["n_calls"] == r["episode_steps"]
        assert r["patch"]["mean_applied_norm"] >= 0.0

    def test_evaluate_patching_aggregates(self, nets):
        rec, don = nets
        out = evaluate_patching(rec, RewardGridConfig(max_steps=15), [(0, 0), (4, 4), (7, 7)],
                                0.5, make_matched_donor(don), "matched")
        assert out["n_start_states"] == 3
        assert 0.0 <= out["goal_rate"] <= 1.0
        assert "actions" not in out["episodes"][0]

    def test_real_checkpoints_load_and_behave_as_documented(self):
        p20 = Path("results/phase2/baseline_seed123/checkpoints/checkpoint_20000.pt")
        p30 = Path("results/phase2/baseline_seed123/checkpoints/checkpoint_30000.pt")
        if not (p20.exists() and p30.exists()):
            pytest.skip("Real checkpoints not available")
        donor, _ = load_q_network_for_eval(p20)
        recipient, _ = load_q_network_for_eval(p30)
        d = evaluate_patching(donor, RewardGridConfig(), [(1, 6)], 0.0)
        r = evaluate_patching(recipient, RewardGridConfig(), [(1, 6)], 0.0)
        assert d["goal_rate"] == 1.0
        assert r["hacking_rate"] == 1.0
