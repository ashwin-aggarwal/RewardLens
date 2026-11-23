"""Tests for Phase 4.6: behavioral reclassification, time dependence, scheduled steering."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork, load_q_network_for_eval
from rewardlens.analysis.activation_intervention import partition_start_positions
from rewardlens.analysis.behavior_classification import (
    CATEGORIES,
    DIRECT_GOAL,
    FARM_ONLY,
    FARM_THEN_FINISH,
    OTHER,
    category_rates,
    classify_episode,
    classify_rollout,
    classify_trajectory,
    goal_step,
    threshold_sensitivity,
    visits_vs_completion,
)
from rewardlens.analysis.steering import (
    PatchController,
    SteeringStats,
    donor_activation,
    evaluate_steering,
    make_matched_donor_layer,
    make_schedule,
    make_shuffled_donor_layer,
    mean_layer_delta,
    schedule_constant,
    schedule_early_decay,
    schedule_early_window,
    schedule_late_window,
    scheduled_patching,
    steered_rollout,
)
from rewardlens.analysis.time_dependence import (
    action_geometry,
    clamp_time,
    earliest_goal_preferring_time,
    goal_preference_by_time,
    observations_for_positions,
    patched_q_components,
)
from rewardlens.envs.reward_grid import RewardGridConfig, RewardGridEnv
from rewardlens.envs.scripted_policies import exploit_policy_actions, goal_policy_actions
from rewardlens.envs.trajectory import record_episode


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


# ---------------------------------------------------------------------------
# Behavior categories
# ---------------------------------------------------------------------------

class TestClassification:
    def test_four_categories(self):
        assert classify_episode(True, 0) == DIRECT_GOAL
        assert classify_episode(True, 40) == FARM_THEN_FINISH
        assert classify_episode(False, 40, episode_complete=True) == FARM_ONLY
        assert classify_episode(False, 0, episode_complete=True) == OTHER

    def test_threshold_boundary(self):
        assert classify_episode(True, 4, threshold=5) == DIRECT_GOAL
        assert classify_episode(True, 5, threshold=5) == FARM_THEN_FINISH
        assert classify_episode(False, 4, True, threshold=5) == OTHER
        assert classify_episode(False, 5, True, threshold=5) == FARM_ONLY
        # Threshold is configurable
        assert classify_episode(True, 5, threshold=10) == DIRECT_GOAL
        assert classify_episode(True, 10, threshold=10) == FARM_THEN_FINISH

    def test_incomplete_trajectory_is_other_even_when_farming(self):
        assert classify_episode(False, 40, episode_complete=False) == OTHER

    def test_farm_only_matches_original_hacking_flag(self):
        for goal in (True, False):
            for visits in (0, 4, 5, 49):
                original = (not goal) and visits >= 5
                assert (classify_episode(goal, visits, True) == FARM_ONLY) == original

    def test_scripted_trajectories_classify_as_expected(self):
        cfg = RewardGridConfig()
        env = RewardGridEnv(cfg)
        goal_tr = record_episode(env, goal_policy_actions(cfg), "goal", "g", seed=0)
        exploit_tr = record_episode(env, exploit_policy_actions(cfg), "exploit", "e", seed=0)
        assert classify_trajectory(goal_tr) == DIRECT_GOAL
        assert classify_trajectory(exploit_tr) == FARM_ONLY
        assert goal_step(goal_tr) == goal_tr.total_steps
        assert goal_step(exploit_tr) is None

    def test_category_rates_and_rows(self):
        eps = [
            {"goal_reached": True, "reward_tile_visits": 0, "episode_steps": 10, "proxy_reward": 9.9, "start_pos": [0, 0]},
            {"goal_reached": True, "reward_tile_visits": 40, "episode_steps": 95, "proxy_reward": 80.0, "start_pos": [1, 1]},
            {"goal_reached": False, "reward_tile_visits": 49, "episode_steps": 100, "is_reward_hacking": True,
             "proxy_reward": 97.0, "start_pos": [2, 2]},
            {"goal_reached": False, "reward_tile_visits": 0, "episode_steps": 100, "proxy_reward": -1.0, "start_pos": [3, 3]},
        ]
        r = category_rates(eps)
        assert r["direct_goal_rate"] == 0.25 and r["farm_then_finish_rate"] == 0.25
        assert r["farm_only_rate"] == 0.25 and r["other_rate"] == 0.25
        assert r["goal_rate"] == 0.5 and r["original_hacking_rate"] == 0.25
        assert r["mean_steps_to_goal"] == pytest.approx(52.5)
        assert r["mean_steps_to_goal_direct"] == 10
        assert sum(r[f"{c}_rate"] for c in CATEGORIES) == pytest.approx(1.0)
        rows = visits_vs_completion(eps)
        assert [x["goal_step"] for x in rows] == [10, 95, None, None]
        assert [x["category"] for x in rows] == [DIRECT_GOAL, FARM_THEN_FINISH, FARM_ONLY, OTHER]
        sens = threshold_sensitivity(eps, [1, 50])
        assert sens["50"]["direct_goal_rate"] == 0.5 and sens["1"]["direct_goal_rate"] == 0.25

    def test_empty_and_rollout_classification(self):
        assert category_rates([])["n_episodes"] == 0
        assert classify_rollout({"goal_reached": True, "reward_tile_visits": 7, "episode_steps": 50}) == FARM_THEN_FINISH


# ---------------------------------------------------------------------------
# Retrospective analysis (uses stored results when present)
# ---------------------------------------------------------------------------

class TestRetrospective:
    def test_phase4_5_stored_episodes_are_farm_then_finish(self):
        p = Path("results/phase4_5/same_run_30k_matched_test_episodes.json")
        if not p.exists():
            pytest.skip("Phase 4.5 per-episode file not available")
        import json
        eps = json.load(open(p))["episodes"]
        r = category_rates(eps)
        assert r["goal_rate"] == 1.0
        assert r["original_hacking_rate"] == 0.0
        assert r["farm_then_finish_rate"] > 0.9
        assert r["direct_goal_rate"] < 0.1

    def test_reclassification_preserves_original_flags(self):
        p = Path("results/phase4_6/reclassification.json")
        if not p.exists():
            pytest.skip("reclassification.json not generated")
        import json
        d = json.load(open(p))
        for row in d["stored_trajectories"]["rows"]:
            assert (row["category"] == FARM_ONLY) == row["original_is_reward_hacking"]
        blk = d["phase4_5"]["same_run"]["30000"]
        assert blk["matched"]["stored_goal_rate"] == blk["matched"]["goal_rate"]


# ---------------------------------------------------------------------------
# Schedules and hook correctness
# ---------------------------------------------------------------------------

class TestSchedules:
    def test_schedule_shapes(self):
        assert [schedule_constant(0.5)(s) for s in (0, 50, 99)] == [0.5, 0.5, 0.5]
        ew = schedule_early_window(1.0, 10)
        assert ew(9) == 1.0 and ew(10) == 0.0
        ed = schedule_early_decay(1.0, 20)
        assert ed(0) == 1.0 and ed(10) == pytest.approx(0.5) and ed(20) == 0.0 and ed(50) == 0.0
        lw = schedule_late_window(1.0, 80)
        assert lw(79) == 0.0 and lw(80) == 1.0
        assert make_schedule("early_window", 0.75, n_steps=5)(4) == 0.75
        with pytest.raises(ValueError):
            make_schedule("bogus", 1.0)

    def test_hook_follows_controller_step(self, nets, obs):
        rec, don = nets
        ctrl = PatchController()
        stats = SteeringStats()
        with torch.no_grad():
            q0 = rec(obs).clone()
        with scheduled_patching(rec, make_matched_donor_layer(don, 2), schedule_early_window(1.0, 1), ctrl, 2,
                                "matched", stats=stats), torch.no_grad():
            ctrl.step = 0
            q_on = rec(obs).clone()
            ctrl.step = 1
            q_off = rec(obs).clone()
        assert not torch.allclose(q_on, q0)
        torch.testing.assert_close(q_off, q0)
        assert stats.alphas == [1.0, 0.0]

    def test_layer2_alpha_one_matches_phase4_5_semantics(self, nets, obs):
        rec, don = nets
        with torch.no_grad():
            expected = rec.fc3(donor_activation(don, obs, 2))
        with scheduled_patching(rec, make_matched_donor_layer(don, 2), schedule_constant(1.0), PatchController(), 2), \
                torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, expected, atol=1e-5, rtol=1e-5)

    def test_layer1_patch_routes_through_recipient_fc2(self, nets, obs):
        rec, don = nets
        with torch.no_grad():
            h1_don = donor_activation(don, obs, 1)
            expected = rec.fc3(rec.relu2(rec.fc2(h1_don)))
        with scheduled_patching(rec, make_matched_donor_layer(don, 1), schedule_constant(1.0), PatchController(), 1), \
                torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, expected, atol=1e-5, rtol=1e-5)

    def test_alpha_above_one_is_affine_extrapolation(self, nets, obs):
        rec, don = nets
        a = 2.0
        with torch.no_grad():
            q_rec = rec(obs)
            q_don_via = rec.fc3(donor_activation(don, obs, 2))
        with scheduled_patching(rec, make_matched_donor_layer(don, 2), schedule_constant(a), PatchController(), 2), \
                torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, (1 - a) * q_rec + a * q_don_via, atol=1e-4, rtol=1e-4)

    def test_weights_unchanged_and_hooks_removed(self, nets, obs):
        rec, don = nets
        before = copy.deepcopy(rec.state_dict())
        for layer in (1, 2):
            with scheduled_patching(rec, make_matched_donor_layer(don, layer), schedule_constant(1.5),
                                    PatchController(), layer), torch.no_grad():
                rec(obs)
        for k, v in rec.state_dict().items():
            torch.testing.assert_close(v, before[k])
        assert len(rec.relu1._forward_hooks) == 0 and len(rec.relu2._forward_hooks) == 0
        assert len(rec.fc1._forward_pre_hooks) == 0
        with pytest.raises(RuntimeError):
            with scheduled_patching(rec, make_matched_donor_layer(don, 2), schedule_constant(1.0), PatchController(), 2):
                raise RuntimeError("boom")
        assert len(rec.relu2._forward_hooks) == 0 and len(rec.fc1._forward_pre_hooks) == 0

    def test_random_control_norm_matched_under_schedule(self, nets, obs):
        rec, don = nets
        fn = make_matched_donor_layer(don, 2)
        sm, sr = SteeringStats(), SteeringStats()
        sched = schedule_early_decay(1.0, 4)
        for mode, st in (("matched", sm), ("random", sr)):
            ctrl = PatchController()
            with scheduled_patching(rec, fn, sched, ctrl, 2, mode, random_seed=1, stats=st), torch.no_grad():
                for s in range(3):
                    ctrl.step = s
                    rec(obs)
        assert sm.alphas == sr.alphas
        np.testing.assert_allclose(sm.applied_norms, sr.applied_norms, rtol=1e-4)

    def test_constant_mode_layer1_is_state_independent_h1_shift(self, nets, obs):
        rec, _ = nets
        delta = np.random.RandomState(2).randn(128).astype(np.float32)
        with torch.no_grad():
            h1 = rec.relu1(rec.fc1(obs))
            expected = rec.fc3(rec.relu2(rec.fc2(h1 + 0.5 * torch.as_tensor(delta))))
        with scheduled_patching(rec, None, schedule_constant(0.5), PatchController(), 1, "constant",
                                constant_delta=delta), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, expected, atol=1e-5, rtol=1e-5)

    def test_shuffled_donor_layer_reproducible(self, nets, obs):
        _, don = nets
        pool = np.random.RandomState(5).rand(40, 7).astype(np.float32)
        with torch.no_grad():
            a = make_shuffled_donor_layer(don, pool, 1, 1)(obs)
            b = make_shuffled_donor_layer(don, pool, 1, 1)(obs)
        torch.testing.assert_close(a, b)
        assert a.shape == (12, 128)


# ---------------------------------------------------------------------------
# Rollouts, evaluation, splits
# ---------------------------------------------------------------------------

class TestRollouts:
    def test_rollout_records_schedule_and_category(self, nets):
        rec, don = nets
        cfg = RewardGridConfig(agent_start=(0, 0), max_steps=25)
        r = steered_rollout(rec, cfg, make_matched_donor_layer(don, 2), schedule_early_window(1.0, 5), 2, "matched")
        assert r["episode_steps"] == len(r["actions"]) <= 25
        assert r["patch"]["n_calls"] == r["episode_steps"]
        assert r["patch"]["n_active_steps"] == min(5, r["episode_steps"])
        assert r["category"] in CATEGORIES
        assert len(r["positions"]) == r["episode_steps"] + 1
        assert len(r["visits_trace"]) == r["episode_steps"] + 1

    def test_obs_transform_only_changes_network_input(self, nets):
        rec, _ = nets
        cfg = RewardGridConfig(agent_start=(0, 0), max_steps=10)
        seen = []

        def spy(o):
            seen.append(float(o[6]))
            return clamp_time(0.0)(o)
        r = steered_rollout(rec, cfg, obs_transform=spy)
        assert len(seen) == r["episode_steps"]
        assert seen[0] == 1.0  # the real env time is untouched; the clamp happens after
        assert r["episode_steps"] <= 10

    def test_evaluate_steering_aggregates(self, nets):
        rec, don = nets
        out = evaluate_steering(rec, RewardGridConfig(max_steps=15), [(0, 0), (4, 4), (7, 7)],
                                make_matched_donor_layer(don, 2), schedule_constant(0.5), 2, "matched")
        assert out["n_start_states"] == 3
        assert sum(out[f"{c}_rate"] for c in CATEGORIES) == pytest.approx(1.0)
        assert "actions" not in out["episodes"][0]
        assert out["mean_active_steps"] > 0

    def test_mean_layer_delta_shape(self, nets):
        rec, don = nets
        o = np.random.RandomState(0).rand(20, 7).astype(np.float32)
        assert mean_layer_delta(rec, make_matched_donor_layer(don, 1), o, 1).shape == (128,)

    def test_fresh_partition_disjoint_and_differs_from_previous(self):
        new = partition_start_positions(RewardGridConfig(), 16, 16, seed=13)
        old = partition_start_positions(RewardGridConfig(), 16, 16, seed=11)
        s = [set(new[k]) for k in ("train", "dev", "test")]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
        assert len(s[0] | s[1] | s[2]) == 62
        assert set(new["test"]) != set(old["test"])
        assert new == partition_start_positions(RewardGridConfig(), 16, 16, seed=13)  # reproducible


# ---------------------------------------------------------------------------
# Time dependence
# ---------------------------------------------------------------------------

class TestTimeDependence:
    def test_observations_grid(self):
        obs, pidx, t = observations_for_positions([(1, 1), (6, 6)], [1.0, 0.5, 0.0])
        assert obs.shape == (6, 7)
        assert list(pidx) == [0, 0, 0, 1, 1, 1]
        np.testing.assert_allclose(obs[:, 6], [1, 0.5, 0, 1, 0.5, 0])
        np.testing.assert_allclose(obs[0, :2], [1 / 7, 1 / 7])

    def test_action_geometry(self):
        g = action_geometry((1, 6))  # goal (6,1): right/up closer; reward (3,5): right/up closer too
        assert g[1]["goal_closer"] and g[0]["goal_closer"]
        assert not g[3]["goal_closer"] and not g[2]["goal_closer"]
        g2 = action_geometry((0, 0))
        assert g2[0]["no_move"] and g2[3]["no_move"]

    def test_goal_preference_by_time_with_synthetic_policies(self):
        positions = [(1, 6), (4, 4), (0, 2)]
        tf = [1.0, 0.5, 0.0]
        geoms = [action_geometry(p) for p in positions]

        def make_fn(prefer_goal_when_t_below):
            def fn(obs):
                q = torch.zeros(obs.shape[0], 4)
                for k in range(obs.shape[0]):
                    pos_i = k // len(tf)
                    g = geoms[pos_i]
                    prefer_goal = float(obs[k, 6]) < prefer_goal_when_t_below
                    for a in range(4):
                        if prefer_goal and g[a]["goal_closer"]:
                            q[k, a] = 1.0
                        if not prefer_goal and g[a]["reward_closer"] and not g[a]["goal_closer"]:
                            q[k, a] = 1.0
                return q
            return fn
        late = goal_preference_by_time(make_fn(0.3), positions, tf)
        assert late["goal_approach_fraction"][0] < 1.0 and late["goal_approach_fraction"][-1] == 1.0
        assert late["mean_goal_minus_reward_q"][-1] > 0
        assert earliest_goal_preferring_time(late) == 0.0
        always = goal_preference_by_time(make_fn(2.0), positions, tf)
        assert always["goal_approach_fraction"] == [1.0, 1.0, 1.0]
        assert earliest_goal_preferring_time(always) == 1.0

    def test_affine_decomposition_exact(self, nets, obs):
        rec, don = nets
        for a in (0.75, 1.0, 1.5):
            c = patched_q_components(rec, don, obs, a)
            torch.testing.assert_close(c["q_patched"], c["q_affine_reconstruction"], atol=1e-4, rtol=1e-4)
        c0 = patched_q_components(rec, don, obs, 0.0)
        torch.testing.assert_close(c0["q_patched"], c0["q_recipient"])

    def test_clamp_time(self):
        o = np.arange(7, dtype=np.float32)
        c = clamp_time(0.25)(o)
        assert c[6] == 0.25 and o[6] == 6.0
        np.testing.assert_array_equal(c[:6], o[:6])


class TestRealCheckpoints:
    def test_patch_reclassifies_as_farm_then_finish(self):
        p20 = Path("results/phase2/baseline_seed123/checkpoints/checkpoint_20000.pt")
        p30 = Path("results/phase2/baseline_seed123/checkpoints/checkpoint_30000.pt")
        if not (p20.exists() and p30.exists()):
            pytest.skip("Real checkpoints not available")
        donor, _ = load_q_network_for_eval(p20)
        recipient, _ = load_q_network_for_eval(p30)
        r = steered_rollout(recipient, RewardGridConfig(agent_start=(1, 1)), make_matched_donor_layer(donor, 2),
                            schedule_constant(0.75), 2, "matched")
        assert r["goal_reached"] and r["category"] == FARM_THEN_FINISH
        d = steered_rollout(donor, RewardGridConfig(agent_start=(1, 1)))
        assert d["category"] == DIRECT_GOAL
