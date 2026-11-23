"""Tests for the sparse causal activation patching extension (Phase 4.8)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork, load_q_network_for_eval
from rewardlens.analysis.activation_intervention import partition_start_positions
from rewardlens.analysis.behavior_classification import CATEGORIES, classify_rollout
from rewardlens.analysis.sparse_patching import (
    DIM,
    ablation_importance,
    evaluate_mask,
    greedy_actions_under_mask,
    greedy_selection,
    h2_hookfree,
    make_mask,
    make_masked_donor,
    masked_delta,
    random_subsets,
    rank_by_activation_difference,
)
from rewardlens.analysis.steering import (
    PatchController,
    clamp_time_tensor,
    make_matched_donor_layer,
    scheduled_patching,
    schedule_constant,
)
from rewardlens.envs.reward_grid import RewardGridConfig

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


class TestMasks:
    def test_make_mask_sizes_and_validation(self):
        assert make_mask([]).sum() == 0
        m = make_mask([3, 5, 127])
        assert m.sum() == 3 and m[3] == m[5] == m[127] == 1 and m.dtype == np.float32
        with pytest.raises(ValueError):
            make_mask([128])
        with pytest.raises(ValueError):
            make_mask([1, 1])

    def test_random_subsets_reproducible_distinct_sizes(self):
        a = random_subsets(8, 5, seed=0)
        b = random_subsets(8, 5, seed=0)
        c = random_subsets(8, 5, seed=1)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert all(len(s) == 8 and len(set(s)) == 8 for s in a)
        assert not all(np.array_equal(x, y) for x, y in zip(a, c))
        assert all(len(s) == 128 for s in random_subsets(128, 2, 0))

    def test_h2_hookfree_matches_module_forward(self, nets, obs):
        rec, _ = nets
        with torch.no_grad():
            expected = rec.relu2(rec.fc2(rec.relu1(rec.fc1(obs))))
            torch.testing.assert_close(h2_hookfree(rec, obs), expected)


class TestPatchSemantics:
    def test_k0_reproduces_recipient(self, nets, obs):
        rec, don = nets
        with torch.no_grad():
            q0 = rec(obs).clone()
        fn = make_masked_donor(rec, don, make_mask([]), clamp_time_tensor(0.2))
        with scheduled_patching(rec, fn, schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, q0)

    def test_k128_reproduces_full_patch(self, nets, obs):
        rec, don = nets
        full_fn = make_matched_donor_layer(don, 2, clamp_time_tensor(0.2))
        with scheduled_patching(rec, full_fn, schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            q_full = rec(obs).clone()
        fn = make_masked_donor(rec, don, np.ones(DIM, dtype=np.float32), clamp_time_tensor(0.2))
        with scheduled_patching(rec, fn, schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            q = rec(obs)
        torch.testing.assert_close(q, q_full, atol=1e-5, rtol=1e-5)

    def test_partial_mask_only_touches_selected_units(self, nets, obs):
        rec, don = nets
        units = [2, 7, 40]
        fn = make_masked_donor(rec, don, make_mask(units), clamp_time_tensor(0.2))
        with torch.no_grad():
            target = fn(obs)
            h_rec = h2_hookfree(rec, obs)
            h_don = don.relu2(don.fc2(don.relu1(don.fc1(clamp_time_tensor(0.2)(obs)))))
        keep = [i for i in range(DIM) if i not in units]
        torch.testing.assert_close(target[:, keep], h_rec[:, keep])
        torch.testing.assert_close(target[:, units], h_don[:, units])
        d = masked_delta(rec, don, make_mask(units), obs.numpy(), clamp_time_tensor(0.2))
        assert d.shape == (10, DIM) and np.all(d[:, keep] == 0)

    def test_weights_unchanged_and_hooks_removed(self, nets, obs):
        rec, don = nets
        before = copy.deepcopy(rec.state_dict())
        dbefore = copy.deepcopy(don.state_dict())
        fn = make_masked_donor(rec, don, make_mask([1, 2, 3]), clamp_time_tensor(0.2))
        with scheduled_patching(rec, fn, schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            rec(obs)
        for k, v in rec.state_dict().items():
            torch.testing.assert_close(v, before[k])
        for k, v in don.state_dict().items():
            torch.testing.assert_close(v, dbefore[k])
        assert len(rec.relu2._forward_hooks) == 0 and len(rec.fc1._forward_pre_hooks) == 0

    def test_shuffled_pool_donor_differs_and_reproducible(self, nets, obs):
        rec, don = nets
        pool = np.random.RandomState(3).rand(30, 7).astype(np.float32)
        m = make_mask([0, 1, 2, 3])
        with torch.no_grad():
            a = make_masked_donor(rec, don, m, clamp_time_tensor(0.2), pool, 1)(obs)
            b = make_masked_donor(rec, don, m, clamp_time_tensor(0.2), pool, 1)(obs)
            c = make_masked_donor(rec, don, m, clamp_time_tensor(0.2))(obs)
        torch.testing.assert_close(a, b)
        assert not torch.allclose(a, c)


class TestSelection:
    def test_activation_difference_ranking(self, nets):
        rec, don = nets
        o = np.random.RandomState(0).rand(20, 7).astype(np.float32)
        A = rank_by_activation_difference(rec, don, o, clamp_time_tensor(0.2))
        assert len(A["order"]) == DIM and len(set(A["order"])) == DIM
        s = A["score"]
        assert all(s[A["order"][i]] >= s[A["order"][i + 1]] for i in range(DIM - 1))
        A2 = rank_by_activation_difference(rec, don, o, clamp_time_tensor(0.2))
        assert A2["order"] == A["order"]

    def test_ablation_importance_structure(self, nets):
        rec, don = nets
        o = np.random.RandomState(0).rand(8, 7).astype(np.float32)
        B = ablation_importance(rec, don, [(0, 0), (4, 4)], o, clamp_time_tensor(0.2),
                                RewardGridConfig(max_steps=8), group_size=32)
        assert B["n_evaluations"] == 4 and len(B["order"]) == DIM and len(set(B["order"])) == DIM
        assert all(r["direct_goal_drop"] >= -1.0 for r in B["rows"])

    def test_greedy_selection_budget_and_reproducibility(self, nets):
        rec, don = nets
        cfg = RewardGridConfig(max_steps=8)
        C1 = greedy_selection(rec, don, [(0, 0), (4, 4)], clamp_time_tensor(0.2), [0, 1, 2, 3], max_k=2,
                              eval_budget=5, env_cfg=cfg, stop_when_perfect=False)
        C2 = greedy_selection(rec, don, [(0, 0), (4, 4)], clamp_time_tensor(0.2), [0, 1, 2, 3], max_k=2,
                              eval_budget=5, env_cfg=cfg, stop_when_perfect=False)
        assert C1["n_evaluations"] <= 5 and C1["order"] == C2["order"]
        assert len(C1["order"]) <= 2 and len(set(C1["order"])) == len(C1["order"])

    def test_selection_and_evaluation_positions_disjoint(self):
        parts = partition_start_positions(RewardGridConfig(), 16, 16, seed=13)
        dev, ev = set(parts["dev"]), set(parts["test"]) | set(parts["train"])
        assert not (dev & ev) and len(dev) == 16 and len(ev) == 46


class TestMetrics:
    def test_evaluate_mask_metrics(self, nets):
        rec, don = nets
        r = evaluate_mask(rec, don, [(0, 0), (6, 2), (3, 4)], make_mask([1, 2]), clamp_time_tensor(0.2),
                          RewardGridConfig(max_steps=12), keep_traces=True)
        assert r["n_patched_units"] == 2
        assert sum(r[f"{c}_rate"] for c in CATEGORIES) == pytest.approx(1.0)
        for ep in r["episodes"]:
            assert ep["category"] == classify_rollout(ep)
        assert r["mean_applied_norm"] >= 0.0

    def test_greedy_actions_under_mask_shape(self, nets):
        rec, don = nets
        o = np.random.RandomState(0).rand(12, 7).astype(np.float32)
        a = greedy_actions_under_mask(rec, don, make_mask([5]), o, clamp_time_tensor(0.2))
        assert a.shape == (12,) and a.min() >= 0 and a.max() <= 3


class TestRealCheckpoints:
    def test_full_mask_reproduces_phase4_7_result(self):
        p20, p30 = CK / "checkpoint_20000.pt", CK / "checkpoint_30000.pt"
        res = Path("results/phase4_7/time_control_results.json")
        if not (p20.exists() and p30.exists() and res.exists()):
            pytest.skip("checkpoints or Phase 4.7 results not available")
        rec, don = load_q_network_for_eval(p30)[0], load_q_network_for_eval(p20)[0]
        d = json.load(open(res))
        ref = d["all_62"]["E_time_shifted"]["a=1.0,t=0.2"]
        positions = [tuple(p) for p in d["position_sets"]["all_62"]]
        r = evaluate_mask(rec, don, positions, np.ones(DIM, dtype=np.float32), clamp_time_tensor(0.2))
        assert r["direct_goal_rate"] == ref["direct_goal_rate"] == 1.0
        assert r["mean_episode_steps"] == pytest.approx(ref["mean_episode_steps"], abs=1e-6)
