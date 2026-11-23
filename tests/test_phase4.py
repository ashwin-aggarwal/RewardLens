"""Tests for Phase 4: activation intervention correctness and evaluation plumbing."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork
from rewardlens.analysis.activation_intervention import (
    InterventionStats,
    evaluate_on_start_states,
    fit_contrastive_direction,
    fit_probe_direction,
    greedy_rollout_with_intervention,
    layer2_intervention,
    partition_start_positions,
    random_unit_directions,
)
from rewardlens.envs.reward_grid import RewardGridConfig


@pytest.fixture
def net():
    torch.manual_seed(0)
    return QNetwork(obs_dim=7, n_actions=4, hidden_dim=128)


@pytest.fixture
def obs_batch():
    rng = np.random.RandomState(0)
    return torch.as_tensor(rng.rand(16, 7).astype(np.float32))


@pytest.fixture
def unit_direction():
    rng = np.random.RandomState(1)
    v = rng.randn(128).astype(np.float32)
    return v / np.linalg.norm(v)


class TestHookLifecycle:
    def test_hook_removed_after_context(self, net, obs_batch, unit_direction):
        with torch.no_grad():
            q_before = net(obs_batch).clone()
        with layer2_intervention(net, unit_direction, 5.0, "subtract"):
            pass
        with torch.no_grad():
            q_after = net(obs_batch)
        torch.testing.assert_close(q_before, q_after)

    def test_hook_removed_after_exception(self, net, obs_batch, unit_direction):
        with pytest.raises(RuntimeError):
            with layer2_intervention(net, unit_direction, 5.0, "subtract"):
                raise RuntimeError("boom")
        assert len(net.relu2._forward_hooks) == 0

    def test_zero_strength_is_baseline(self, net, obs_batch, unit_direction):
        with torch.no_grad():
            q_base = net(obs_batch).clone()
        with layer2_intervention(net, unit_direction, 0.0, "subtract"), torch.no_grad():
            q_zero = net(obs_batch)
        torch.testing.assert_close(q_base, q_zero)

    def test_none_mode_is_baseline(self, net, obs_batch, unit_direction):
        with torch.no_grad():
            q_base = net(obs_batch).clone()
        with layer2_intervention(net, unit_direction, 5.0, "none"), torch.no_grad():
            q_none = net(obs_batch)
        torch.testing.assert_close(q_base, q_none)

    def test_weights_unchanged_by_intervention(self, net, obs_batch, unit_direction):
        params_before = copy.deepcopy(net.state_dict())
        with layer2_intervention(net, unit_direction, 10.0, "subtract"), torch.no_grad():
            net(obs_batch)
        for k, v in net.state_dict().items():
            torch.testing.assert_close(v, params_before[k])

    def test_only_final_layer_input_changes(self, net, obs_batch, unit_direction):
        """The edit at relu2 shifts Q-values by exactly fc3(alpha*v) difference."""
        alpha = 3.0
        with torch.no_grad():
            q_base = net(obs_batch).clone()
            expected_shift = -(alpha * torch.as_tensor(unit_direction)) @ net.fc3.weight.T
        with layer2_intervention(net, unit_direction, alpha, "subtract"), torch.no_grad():
            q_edit = net(obs_batch)
        torch.testing.assert_close(q_edit - q_base, expected_shift.expand_as(q_base), atol=1e-5, rtol=1e-5)


class TestEditSemantics:
    def test_subtract_vs_add_are_opposite(self, net, obs_batch, unit_direction):
        with torch.no_grad():
            q_base = net(obs_batch).clone()
        with layer2_intervention(net, unit_direction, 2.0, "subtract"), torch.no_grad():
            q_sub = net(obs_batch)
        with layer2_intervention(net, unit_direction, 2.0, "add"), torch.no_grad():
            q_add = net(obs_batch)
        torch.testing.assert_close(q_sub - q_base, -(q_add - q_base), atol=1e-5, rtol=1e-5)

    def test_project_out_removes_component(self, net, unit_direction):
        v = torch.as_tensor(unit_direction)
        captured = {}

        def capture(module, inputs, output):
            captured["h"] = output.detach().clone()

        handle = net.relu2.register_forward_hook(capture)
        rng = np.random.RandomState(2)
        x = torch.as_tensor(rng.rand(8, 7).astype(np.float32))
        with torch.no_grad():
            net(x)
        handle.remove()
        h = captured["h"]
        h_proj = h - (h @ v).unsqueeze(-1) * v
        assert torch.allclose(h_proj @ v, torch.zeros(8), atol=1e-5)

    def test_stats_track_applied_norm(self, net, obs_batch, unit_direction):
        stats = InterventionStats()
        with layer2_intervention(net, unit_direction, 4.0, "subtract", stats), torch.no_grad():
            net(obs_batch)
        assert stats.n_calls == 1
        assert stats.summary()["mean_applied_norm"] == pytest.approx(4.0, abs=1e-4)

    def test_edits_can_go_negative_and_are_measured(self, net, obs_batch, unit_direction):
        stats = InterventionStats()
        with layer2_intervention(net, unit_direction, 100.0, "subtract", stats), torch.no_grad():
            net(obs_batch)
        assert stats.summary()["mean_negative_fraction"] > 0.0


class TestDirections:
    def test_probe_direction_unit_norm_and_sign(self):
        rng = np.random.RandomState(3)
        h = np.abs(rng.randn(200, 128)).astype(np.float32)
        labels = (h[:, 0] > h[:, 0].mean()).astype(np.int64)
        d = fit_probe_direction(h, labels, "ckpt")
        assert np.linalg.norm(d.vector) == pytest.approx(1.0, abs=1e-5)
        scores = h @ d.vector
        assert scores[labels == 1].mean() > scores[labels == 0].mean()
        assert d.sign_verified

    def test_probe_destandardization_preserves_ranking(self):
        """Raw-space direction must induce the same ordering as the standardized probe."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(4)
        h = np.abs(rng.randn(300, 128)).astype(np.float32) * rng.rand(128) * 5
        labels = ((h[:, 3] - h[:, 7]) > 0).astype(np.int64)
        d = fit_probe_direction(h, labels, "ckpt")

        sc = StandardScaler().fit(h)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(sc.transform(h), labels)
        std_scores = sc.transform(h) @ clf.coef_[0]
        raw_scores = h @ d.vector
        corr = np.corrcoef(std_scores, raw_scores)[0, 1]
        assert abs(corr) > 0.9999

    def test_contrastive_direction_sign(self):
        rng = np.random.RandomState(5)
        h = np.abs(rng.randn(100, 128)).astype(np.float32)
        labels = np.array([0] * 50 + [1] * 50)
        h[labels == 1, 10] += 3.0
        d = fit_contrastive_direction(h, labels, "ckpt")
        assert np.linalg.norm(d.vector) == pytest.approx(1.0, abs=1e-5)
        assert d.vector[10] > 0

    def test_random_directions_reproducible_and_unit(self):
        a = random_unit_directions(128, 20, seed=0)
        b = random_unit_directions(128, 20, seed=0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
        c = random_unit_directions(128, 20, seed=1)
        assert not np.allclose(a, c)

    def test_random_norm_matching(self, net, obs_batch):
        """Random controls at the same alpha apply the same perturbation norm."""
        alpha = 6.0
        v_t = random_unit_directions(128, 1, seed=0)[0]
        v_r = random_unit_directions(128, 1, seed=9)[0]
        st, sr = InterventionStats(), InterventionStats()
        with layer2_intervention(net, v_t, alpha, "subtract", st), torch.no_grad():
            net(obs_batch)
        with layer2_intervention(net, v_r, alpha, "subtract", sr), torch.no_grad():
            net(obs_batch)
        assert st.summary()["mean_applied_norm"] == pytest.approx(sr.summary()["mean_applied_norm"], abs=1e-4)


class TestPartitionAndRollout:
    def test_partition_disjoint_and_excludes_special_tiles(self):
        cfg = RewardGridConfig()
        parts = partition_start_positions(cfg, n_dev=16, n_test=16, seed=7)
        train, dev, test = set(parts["train"]), set(parts["dev"]), set(parts["test"])
        assert not (train & dev) and not (train & test) and not (dev & test)
        all_pos = train | dev | test
        assert tuple(cfg.goal_pos) not in all_pos
        assert tuple(cfg.reward_pos) not in all_pos
        assert len(all_pos) == 62

    def test_partition_reproducible(self):
        cfg = RewardGridConfig()
        p1 = partition_start_positions(cfg, seed=7)
        p2 = partition_start_positions(cfg, seed=7)
        assert p1 == p2

    def test_rollout_metrics_shape(self, net):
        cfg = RewardGridConfig(agent_start=(0, 0), max_steps=20)
        r = greedy_rollout_with_intervention(net, cfg)
        assert r["episode_steps"] == len(r["actions"]) <= 20
        assert isinstance(r["goal_reached"], bool)
        assert r["intervention"]["mode"] == "none"

    def test_rollout_deterministic(self, net, unit_direction):
        cfg = RewardGridConfig(agent_start=(2, 2), max_steps=30)
        r1 = greedy_rollout_with_intervention(net, cfg, unit_direction, 3.0, "subtract")
        r2 = greedy_rollout_with_intervention(net, cfg, unit_direction, 3.0, "subtract")
        assert r1["actions"] == r2["actions"]
        assert r1["proxy_reward"] == r2["proxy_reward"]

    def test_evaluate_on_start_states_aggregation(self, net):
        cfg = RewardGridConfig(max_steps=15)
        out = evaluate_on_start_states(net, cfg, [(0, 0), (5, 5), (2, 7)])
        assert out["n_start_states"] == 3
        assert 0.0 <= out["goal_rate"] <= 1.0
        assert len(out["episodes"]) == 3

    def test_end_to_end_small_intervention_eval(self, net):
        """Full mini pipeline: fit direction on synthetic labels, evaluate on states."""
        rng = np.random.RandomState(6)
        h = np.abs(rng.randn(100, 128)).astype(np.float32)
        labels = (h[:, 0] > h[:, 0].mean()).astype(np.int64)
        d = fit_probe_direction(h, labels, "test")
        cfg = RewardGridConfig(max_steps=15)
        base = evaluate_on_start_states(net, cfg, [(0, 0), (4, 4)])
        edit = evaluate_on_start_states(net, cfg, [(0, 0), (4, 4)], d.vector, 5.0, "subtract")
        assert edit["mean_applied_norm"] == pytest.approx(5.0, abs=1e-3)
        assert base["n_start_states"] == edit["n_start_states"]
