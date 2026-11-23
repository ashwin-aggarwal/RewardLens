"""Tests for the Phase 5 low-rank weight-editing extension."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rewardlens.agents.dqn import QNetwork, load_q_network_for_eval
from rewardlens.analysis.low_rank_patching import collect_deltas, fit_basis, make_low_rank_donor
from rewardlens.analysis.steering import clamp_time_tensor, evaluate_steering
from rewardlens.analysis.weight_editing import (
    EditedNet,
    TrainConfig,
    matching_errors,
    patch_targets,
    random_lowrank_model,
    realizability_floor,
    train_edit,
    trajectory_observations,
    weight_change_stats,
)
from rewardlens.envs.reward_grid import RewardGridConfig

CK = Path("results/phase2/baseline_seed123/checkpoints")


@pytest.fixture
def nets():
    torch.manual_seed(0)
    rec = QNetwork()
    torch.manual_seed(1)
    don = QNetwork()
    return rec, don


@pytest.fixture
def targets(nets):
    rec, don = nets
    obs = np.random.RandomState(0).rand(120, 7).astype(np.float32)
    basis = fit_basis(collect_deltas(rec, don, obs, clamp_time_tensor(0.2)), "pca")
    fn = make_low_rank_donor(rec, don, basis, 8, clamp_time_tensor(0.2))
    return patch_targets(rec, fn, obs)


class TestParameterization:
    @pytest.mark.parametrize("rank", [1, 2, 4, 8, 16])
    def test_lowrank_param_count_and_zero_init(self, nets, rank):
        rec, _ = nets
        net = EditedNet(rec, "lowrank", rank, edit_bias=True, seed=0)
        assert net.n_trainable() == 2 * 128 * rank + 128
        assert EditedNet(rec, "lowrank", rank, edit_bias=False).n_trainable() == 2 * 128 * rank
        o = torch.rand(5, 7)
        torch.testing.assert_close(net(o), rec(o))  # B = 0 so the edit starts at exactly zero

    def test_baseline_param_counts(self, nets):
        rec, _ = nets
        assert EditedNet(rec, "l2_full", 0).n_trainable() == 128 * 128 + 128
        assert EditedNet(rec, "head", 0).n_trainable() == 4 * 128 + 4
        assert EditedNet(rec, "qbias", 0).n_trainable() == 4
        assert EditedNet(rec, "full", 0).n_trainable() == sum(p.numel() for p in rec.parameters())
        with pytest.raises(ValueError):
            EditedNet(rec, "lowrank", 0)
        with pytest.raises(ValueError):
            EditedNet(rec, "bogus", 1)

    def test_base_weights_frozen_and_original_untouched(self, nets, targets):
        rec, _ = nets
        before = copy.deepcopy(rec.state_dict())
        net, _ = train_edit(rec, targets, "lowrank", "act", 4, True, TrainConfig(steps=30, seed=0))
        for k, v in rec.state_dict().items():
            torch.testing.assert_close(v, before[k])
        for k, v in net.base.state_dict().items():
            torch.testing.assert_close(v, before[k])
        assert all(not p.requires_grad for p in net.base.parameters())


class TestMerge:
    @pytest.mark.parametrize("method,objective,rank", [("lowrank", "act", 2), ("lowrank", "q", 4), ("l2_full", "q", 0),
                                                        ("head", "q", 0), ("qbias", "q", 0), ("full", "ce", 0)])
    def test_merged_model_is_standalone_and_equivalent(self, nets, targets, method, objective, rank):
        rec, _ = nets
        net, _ = train_edit(rec, targets, method, objective, rank, True, TrainConfig(steps=40, seed=1))
        merged = net.merge()
        assert type(merged) is QNetwork
        assert len(merged.relu2._forward_hooks) == 0 and len(merged.fc1._forward_pre_hooks) == 0
        o = torch.rand(9, 7)
        with torch.no_grad():
            torch.testing.assert_close(merged(o), net(o), atol=1e-5, rtol=1e-5)

    def test_lowrank_merge_has_rank_at_most_r_and_only_layer2_changes(self, nets, targets):
        rec, _ = nets
        net, _ = train_edit(rec, targets, "lowrank", "act", 2, False, TrainConfig(steps=60, seed=0))
        stats = weight_change_stats(rec, net.merge())
        assert stats["fc2"]["numerical_rank"] <= 2 and stats["fc2"]["weight_delta_fro"] > 0
        assert stats["fc1"]["weight_delta_fro"] == 0 and stats["fc3"]["weight_delta_fro"] == 0
        assert stats["fc2"]["bias_delta_norm"] == 0

    def test_saved_merged_model_reloads(self, nets, targets, tmp_path):
        rec, _ = nets
        merged = train_edit(rec, targets, "lowrank", "q", 1, True, TrainConfig(steps=20, seed=0))[0].merge()
        p = tmp_path / "m.pt"
        torch.save({"q_net_state_dict": merged.state_dict(), "config": {"obs_dim": 7, "n_actions": 4, "hidden_dim": 128}}, p)
        re, _ = load_q_network_for_eval(p)
        o = torch.rand(4, 7)
        with torch.no_grad():
            torch.testing.assert_close(re(o), merged(o))


class TestTraining:
    def test_deterministic(self, nets, targets):
        rec, _ = nets
        a = train_edit(rec, targets, "lowrank", "act", 4, True, TrainConfig(steps=50, seed=3))[0].merge()
        b = train_edit(rec, targets, "lowrank", "act", 4, True, TrainConfig(steps=50, seed=3))[0].merge()
        for (ka, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items()):
            torch.testing.assert_close(va, vb)

    def test_training_reduces_objective(self, nets, targets):
        rec, _ = nets
        _, log = train_edit(rec, targets, "lowrank", "act", 4, True, TrainConfig(steps=300, seed=0))
        assert log[-1]["fit"] < log[0]["fit"]

    def test_invalid_objective_combination(self, nets, targets):
        rec, _ = nets
        with pytest.raises(ValueError):
            train_edit(rec, targets, "head", "act", 0)
        with pytest.raises(ValueError):
            train_edit(rec, targets, "lowrank", "nope", 1)


class TestTargetsAndMetrics:
    def test_targets_match_patch_definition(self, nets, targets):
        rec, _ = nets
        with torch.no_grad():
            torch.testing.assert_close(targets["q"], rec.fc3(targets["h2"]))
            torch.testing.assert_close(targets["h_rec"], F.relu(rec.fc2(F.relu(rec.fc1(targets["obs"])))))
        assert torch.equal(targets["a"], targets["q"].argmax(1))

    def test_realizability_floor_is_lower_bound(self, nets, targets):
        rec, _ = nets
        fl = realizability_floor(targets)
        for steps in (0, 200):
            m = train_edit(rec, targets, "l2_full", "act", 0, True, TrainConfig(steps=steps, seed=0))[0].merge() if steps else rec
            assert matching_errors(m, targets)["act_mse"] >= fl["mse_floor"] - 1e-9

    def test_recipient_matching_zero_only_for_target_itself(self, nets, targets):
        rec, _ = nets
        e = matching_errors(rec, targets)
        assert e["act_relative_error"] == pytest.approx(1.0, rel=1e-4)

    def test_random_lowrank_norm_and_rank(self, nets):
        rec, _ = nets
        m = random_lowrank_model(rec, 3, 2.5, seed=0)
        s = weight_change_stats(rec, m)
        assert s["fc2"]["weight_delta_fro"] == pytest.approx(2.5, rel=1e-4)
        assert s["fc2"]["numerical_rank"] == 3
        m2 = random_lowrank_model(rec, 3, 2.5, seed=0)
        torch.testing.assert_close(m.fc2.weight, m2.fc2.weight)

    def test_trajectory_observations(self):
        o = trajectory_observations([(0, 0), (1, 0), (2, 0)], max_steps=100)
        assert o.shape == (2, 7)
        np.testing.assert_allclose(o[:, 6], [1.0, 0.99])
        np.testing.assert_allclose(o[1, :2], [1 / 7, 0.0])


class TestRealCheckpoints:
    def test_pair_rule_and_standalone_evaluation(self):
        if not (CK / "checkpoint_20000.pt").exists() or not Path("results/phase4_6/reclassification.json").exists():
            pytest.skip("checkpoints or reclassification results not available")
        from scripts.run_weight_editing import pair_by_rule
        assert pair_by_rule(123) == (20000, 30000)
        rec, _ = load_q_network_for_eval(CK / "checkpoint_30000.pt")
        # A zero edit merged is exactly the original recipient, evaluated without hooks.
        merged = EditedNet(rec, "lowrank", 2).merge()
        a = evaluate_steering(merged, RewardGridConfig(), [(0, 4), (1, 1)])
        b = evaluate_steering(rec, RewardGridConfig(), [(0, 4), (1, 1)])
        assert a["direct_goal_rate"] == b["direct_goal_rate"] and a["mean_episode_steps"] == b["mean_episode_steps"]
