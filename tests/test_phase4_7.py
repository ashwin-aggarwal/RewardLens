"""Tests for Phase 4.7: time controls for activation patching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork, load_q_network_for_eval
from rewardlens.analysis.behavior_classification import CATEGORIES, DIRECT_GOAL, FARM_ONLY, classify_rollout
from rewardlens.analysis.steering import (
    PatchController,
    clamp_time_tensor,
    donor_activation,
    evaluate_steering,
    make_matched_donor_layer,
    make_time_delta_donor,
    scheduled_patching,
    schedule_constant,
    steered_rollout,
)
from rewardlens.analysis.time_dependence import clamp_time, compare_to_reference, goal_preference_by_time
from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv

CK = Path("results/phase2/baseline_seed123/checkpoints")


@pytest.fixture
def nets():
    torch.manual_seed(0)
    recipient = QNetwork()
    torch.manual_seed(1)
    donor = QNetwork()
    return recipient, donor


@pytest.fixture
def obs():
    return torch.as_tensor(np.random.RandomState(0).rand(10, 7).astype(np.float32))


class TestTimeManipulation:
    def test_numpy_clamp_only_touches_time_feature(self):
        o = np.random.RandomState(1).rand(7).astype(np.float32)
        c = clamp_time(0.2)(o)
        assert c[6] == pytest.approx(0.2)
        np.testing.assert_array_equal(c[:6], o[:6])
        assert o[6] != 0.2  # original untouched

    def test_tensor_clamp_only_touches_time_feature(self, obs):
        c = clamp_time_tensor(0.2)(obs)
        assert torch.all(c[:, 6] == 0.2)
        torch.testing.assert_close(c[:, :6], obs[:, :6])
        assert not torch.all(obs[:, 6] == 0.2)

    def test_environment_horizon_unchanged_under_clamp(self, nets):
        """The clamp changes only what the network sees; the env still truncates at max_steps
        and its own observation still reports the real remaining time."""
        rec, _ = nets
        cfg = RewardGridConfig(agent_start=(0, 0), max_steps=12)
        seen_env_time = []

        def spy(o):
            seen_env_time.append(float(o[6]))
            return clamp_time(0.2)(o)
        r = steered_rollout(rec, cfg, obs_transform=spy)
        assert r["episode_steps"] <= 12
        assert r["episode_complete"]
        # env time decreases linearly regardless of the clamp
        expected = [max(0.0, (12 - k) / 12) for k in range(len(seen_env_time))]
        np.testing.assert_allclose(seen_env_time, expected, atol=1e-6)

    def test_env_step_counter_independent_of_network_input(self):
        env = RewardGridEnv(RewardGridConfig(agent_start=(0, 0), max_steps=5))
        env.reset(seed=0)
        done = False
        n = 0
        while not done:
            obs, _, term, trunc, info = env.step(1)
            n += 1
            done = term or trunc
        assert n == 5 and info["episode_steps"] == 5 and obs[6] == 0.0


class TestTimeOnlyDelta:
    def test_time_delta_formula(self, nets, obs):
        rec, don = nets
        fn = make_time_delta_donor(rec, don, 2, 0.2)
        with torch.no_grad():
            got = fn(obs)
            h_rec = rec.relu2(rec.fc2(rec.relu1(rec.fc1(obs))))
            expected = h_rec + donor_activation(don, clamp_time_tensor(0.2)(obs), 2) - donor_activation(don, obs, 2)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_decomposition_matched_plus_time_equals_full(self, nets, obs):
        rec, don = nets
        with torch.no_grad():
            h_rec = rec.relu2(rec.fc2(rec.relu1(rec.fc1(obs))))
            full = make_matched_donor_layer(don, 2, clamp_time_tensor(0.2))(obs) - h_rec
            matched = make_matched_donor_layer(don, 2)(obs) - h_rec
            time_only = make_time_delta_donor(rec, don, 2, 0.2)(obs) - h_rec
        torch.testing.assert_close(full, matched + time_only, atol=1e-5, rtol=1e-5)

    def test_time_delta_zero_when_clamp_equals_real_time(self, nets, obs):
        rec, don = nets
        o = obs.clone()
        o[:, 6] = 0.2
        with torch.no_grad():
            got = make_time_delta_donor(rec, don, 2, 0.2)(o)
            h_rec = rec.relu2(rec.fc2(rec.relu1(rec.fc1(o))))
        torch.testing.assert_close(got, h_rec, atol=1e-6, rtol=1e-6)

    def test_time_only_patch_through_hook(self, nets, obs):
        rec, don = nets
        fn = make_time_delta_donor(rec, don, 2, 0.2)
        with torch.no_grad():
            expected = rec.fc3(fn(obs))
        with scheduled_patching(rec, fn, schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, expected, atol=1e-5, rtol=1e-5)

    def test_shuffled_time_delta_reproducible_and_differs(self, nets, obs):
        rec, don = nets
        pool = np.random.RandomState(3).rand(30, 7).astype(np.float32)
        with torch.no_grad():
            a = make_time_delta_donor(rec, don, 2, 0.2, pool, 1)(obs)
            b = make_time_delta_donor(rec, don, 2, 0.2, pool, 1)(obs)
            m = make_time_delta_donor(rec, don, 2, 0.2)(obs)
        torch.testing.assert_close(a, b)
        assert not torch.allclose(a, m)


class TestQAnalysis:
    def test_compare_to_reference(self):
        positions = [(1, 6), (4, 4)]
        tf = [1.0, 0.0]

        def const_q(vals):
            return lambda o: torch.tensor(vals, dtype=torch.float32).repeat(o.shape[0], 1)
        ref = goal_preference_by_time(const_q([1, 0, 0, 0]), positions, tf)
        other = goal_preference_by_time(const_q([0, 1, 0, 0]), positions, tf)
        same = goal_preference_by_time(const_q([2, 0, 0, 0]), positions, tf)
        assert compare_to_reference(other, ref)["greedy_flip_rate"] == [1.0, 1.0]
        assert compare_to_reference(same, ref)["greedy_flip_rate"] == [0.0, 0.0]
        assert compare_to_reference(other, ref)["dominant_action_fraction"] == [1.0, 1.0]
        assert compare_to_reference(other, ref)["action_fractions"][0] == [0.0, 1.0, 0.0, 0.0]
        assert np.asarray(ref["q_matrix"]).shape == (2, 2, 4)


class TestDeterminismAndClassification:
    def test_evaluation_deterministic(self, nets):
        rec, don = nets
        fn = make_time_delta_donor(rec, don, 2, 0.2)
        cfg = RewardGridConfig(max_steps=20)
        a = evaluate_steering(rec, cfg, [(0, 0), (5, 5)], fn, schedule_constant(1.0), 2, "matched")
        b = evaluate_steering(rec, cfg, [(0, 0), (5, 5)], fn, schedule_constant(1.0), 2, "matched")
        assert a == b

    def test_four_categories_on_conditions(self, nets):
        rec, _ = nets
        r = evaluate_steering(rec, RewardGridConfig(max_steps=10), [(0, 0), (6, 2), (3, 4)], obs_transform=clamp_time(0.2))
        assert sum(r[f"{c}_rate"] for c in CATEGORIES) == pytest.approx(1.0)
        for ep in r["episodes"]:
            assert ep["category"] == classify_rollout(ep)


class TestRealCheckpoints:
    @pytest.fixture
    def real(self):
        p20, p30 = CK / "checkpoint_20000.pt", CK / "checkpoint_30000.pt"
        if not (p20.exists() and p30.exists()):
            pytest.skip("Real checkpoints not available")
        return load_q_network_for_eval(p30)[0], load_q_network_for_eval(p20)[0]

    def test_phase4_6_intervention_reproduced_exactly(self, real):
        p = Path("results/phase4_6/steering_results.json")
        if not p.exists():
            pytest.skip("Phase 4.6 results not available")
        rec, don = real
        d = json.load(open(p))
        ref = d["blocks"]["D_time_shifted_donor"]["test"]["matched"]
        positions = [tuple(x) for x in d["partition"]["test"]]
        assert d["blocks"]["D_time_shifted_donor"]["selected_donor_time_clamp"] == 0.2
        fn = make_matched_donor_layer(don, 2, clamp_time_tensor(0.2))
        r = evaluate_steering(rec, RewardGridConfig(), positions, fn, schedule_constant(1.0), 2, "matched")
        for k in ("direct_goal_rate", "mean_reward_tile_visits", "mean_episode_steps", "mean_applied_norm"):
            assert r[k] == pytest.approx(ref[k], abs=1e-6)

    def test_recipient_clamp_alone_still_farms_and_donor_direct(self, real):
        rec, don = real
        r = steered_rollout(rec, RewardGridConfig(agent_start=(0, 4)), obs_transform=clamp_time(0.2))
        assert r["category"] == FARM_ONLY
        d = steered_rollout(don, RewardGridConfig(agent_start=(0, 4)))
        assert d["category"] == DIRECT_GOAL
