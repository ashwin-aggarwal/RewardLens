"""Tests for the low-rank activation patching extension."""

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
from rewardlens.analysis.low_rank_patching import (
    DIM,
    collect_deltas,
    evaluate_donor_fn,
    fit_basis,
    make_low_rank_donor,
    make_subspace_donor,
    random_orthonormal_bases,
    reconstruction_error,
)
from rewardlens.analysis.sparse_patching import h2_hookfree
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
def obs_np():
    return np.random.RandomState(0).rand(300, 7).astype(np.float32)


@pytest.fixture
def bases(nets, obs_np):
    rec, don = nets
    d = collect_deltas(rec, don, obs_np, clamp_time_tensor(0.2))
    return d, fit_basis(d, "pca"), fit_basis(d, "svd")


class TestBasis:
    def test_deltas_match_definition(self, nets, obs_np):
        rec, don = nets
        d = collect_deltas(rec, don, obs_np, clamp_time_tensor(0.2))
        ot = torch.as_tensor(obs_np)
        with torch.no_grad():
            expected = don.relu2(don.fc2(don.relu1(don.fc1(clamp_time_tensor(0.2)(ot))))) - h2_hookfree(rec, ot)
        np.testing.assert_allclose(d, expected.numpy(), atol=1e-6)

    def test_orthonormal_and_variance_ordered(self, bases):
        _, pca, svd = bases
        for b in (pca, svd):
            U = b.components
            np.testing.assert_allclose(U @ U.T, np.eye(DIM), atol=1e-4)
            assert np.all(np.diff(b.explained_variance_ratio) <= 1e-7)
            assert b.explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-5)
        assert pca.mean is not None and svd.mean is None

    def test_reconstruction_error_monotone_and_exact_at_full_rank(self, bases):
        d, pca, svd = bases
        for b in (pca, svd):
            errs = [reconstruction_error(b, d, k)["relative_error"] for k in (0, 1, 4, 16, 64, 128)]
            assert all(errs[i] >= errs[i + 1] - 1e-6 for i in range(len(errs) - 1))
            assert errs[-1] < 1e-5
        assert reconstruction_error(svd, d, 0)["mean_projection_norm"] == 0.0

    def test_random_bases_orthonormal_reproducible(self):
        a = random_orthonormal_bases(8, 3, seed=0)
        b = random_orthonormal_bases(8, 3, seed=0)
        c = random_orthonormal_bases(8, 3, seed=1)
        for R in a:
            assert R.shape == (8, DIM)
            np.testing.assert_allclose(R @ R.T, np.eye(8), atol=1e-5)
        assert all(np.allclose(x, y) for x, y in zip(a, b))
        assert not np.allclose(a[0], c[0])


class TestPatchSemantics:
    def test_svd_rank0_is_recipient_and_pca_rank0_is_mean(self, nets, bases):
        rec, don = nets
        _, pca, svd = bases
        o = torch.as_tensor(np.random.RandomState(5).rand(6, 7).astype(np.float32))
        with torch.no_grad():
            q0 = rec(o).clone()
            h_rec = h2_hookfree(rec, o)
        with scheduled_patching(rec, make_low_rank_donor(rec, don, svd, 0, clamp_time_tensor(0.2)),
                                schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            torch.testing.assert_close(rec(o), q0)
        with torch.no_grad():
            t = make_low_rank_donor(rec, don, pca, 0, clamp_time_tensor(0.2))(o)
        torch.testing.assert_close(t - h_rec, torch.as_tensor(pca.mean).expand_as(h_rec), atol=1e-5, rtol=1e-5)

    def test_rank128_reproduces_full_patch(self, nets, bases):
        rec, don = nets
        _, pca, svd = bases
        o = torch.as_tensor(np.random.RandomState(5).rand(6, 7).astype(np.float32))
        with scheduled_patching(rec, make_matched_donor_layer(don, 2, clamp_time_tensor(0.2)),
                                schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            q_full = rec(o).clone()
        for b in (pca, svd):
            with scheduled_patching(rec, make_low_rank_donor(rec, don, b, DIM, clamp_time_tensor(0.2)),
                                    schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
                torch.testing.assert_close(rec(o), q_full, atol=1e-4, rtol=1e-4)

    def test_projection_lies_in_subspace(self, nets, bases):
        rec, don = nets
        _, pca, _ = bases
        o = torch.as_tensor(np.random.RandomState(5).rand(6, 7).astype(np.float32))
        with torch.no_grad():
            t = make_low_rank_donor(rec, don, pca, 4, clamp_time_tensor(0.2))(o) - h2_hookfree(rec, o)
        c = (t - torch.as_tensor(pca.mean)).numpy()
        U = pca.components[:4]
        np.testing.assert_allclose(c, (c @ U.T) @ U, atol=1e-4)

    def test_norm_matched_random_subspace(self, nets, bases):
        rec, don = nets
        _, pca, _ = bases
        o = torch.as_tensor(np.random.RandomState(5).rand(6, 7).astype(np.float32))
        R = random_orthonormal_bases(4, 1, 0)[0]
        with torch.no_grad():
            h_rec = h2_hookfree(rec, o)
            t_lr = make_low_rank_donor(rec, don, pca, 4, clamp_time_tensor(0.2))(o) - h_rec
            t_rs = make_subspace_donor(rec, don, R, pca.mean, clamp_time_tensor(0.2), pca, 4)(o) - h_rec
        mu = torch.as_tensor(pca.mean)
        torch.testing.assert_close(torch.linalg.norm(t_lr - mu, dim=1), torch.linalg.norm(t_rs - mu, dim=1), atol=1e-3, rtol=1e-3)

    def test_weights_unchanged_and_hooks_removed(self, nets, bases):
        rec, don = nets
        _, pca, _ = bases
        before = copy.deepcopy(rec.state_dict())
        o = torch.as_tensor(np.random.RandomState(5).rand(6, 7).astype(np.float32))
        with scheduled_patching(rec, make_low_rank_donor(rec, don, pca, 8, clamp_time_tensor(0.2)),
                                schedule_constant(1.0), PatchController(), 2, "matched"), torch.no_grad():
            rec(o)
        for k, v in rec.state_dict().items():
            torch.testing.assert_close(v, before[k])
        assert len(rec.relu2._forward_hooks) == 0 and len(rec.fc1._forward_pre_hooks) == 0


class TestEvaluation:
    def test_evaluation_deterministic_and_classified(self, nets, bases):
        rec, don = nets
        _, pca, _ = bases
        fn = make_low_rank_donor(rec, don, pca, 4, clamp_time_tensor(0.2))
        a = evaluate_donor_fn(rec, [(0, 0), (5, 5)], fn, RewardGridConfig(max_steps=15), keep_traces=True)
        b = evaluate_donor_fn(rec, [(0, 0), (5, 5)], fn, RewardGridConfig(max_steps=15), keep_traces=True)
        assert a == b
        assert sum(a[f"{c}_rate"] for c in CATEGORIES) == pytest.approx(1.0)
        for ep in a["episodes"]:
            assert ep["category"] == classify_rollout(ep)

    def test_fit_and_eval_positions_disjoint(self):
        parts = partition_start_positions(RewardGridConfig(), 16, 16, seed=13)
        assert not (set(parts["dev"]) & (set(parts["test"]) | set(parts["train"])))


class TestRealCheckpoints:
    def test_rank128_matches_phase4_7_on_real_nets(self):
        p20, p30 = CK / "checkpoint_20000.pt", CK / "checkpoint_30000.pt"
        res = Path("results/phase4_7/time_control_results.json")
        if not (p20.exists() and p30.exists() and res.exists()):
            pytest.skip("checkpoints or Phase 4.7 results not available")
        rec, don = load_q_network_for_eval(p30)[0], load_q_network_for_eval(p20)[0]
        d = json.load(open(res))
        ref = d["all_62"]["E_time_shifted"]["a=1.0,t=0.2"]
        positions = [tuple(p) for p in d["position_sets"]["all_62"]]
        obs = np.random.RandomState(0).rand(200, 7).astype(np.float32)
        b = fit_basis(collect_deltas(rec, don, obs, clamp_time_tensor(0.2)), "pca")
        r = evaluate_donor_fn(rec, positions, make_low_rank_donor(rec, don, b, DIM, clamp_time_tensor(0.2)))
        assert r["direct_goal_rate"] == ref["direct_goal_rate"] == 1.0
        assert r["mean_episode_steps"] == pytest.approx(ref["mean_episode_steps"], abs=1e-6)
