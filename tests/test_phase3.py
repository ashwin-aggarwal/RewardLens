"""Tests for Phase 3: activation extraction, probing, and representation analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import QNetwork, DQNConfig, save_checkpoint
from rewardlens.analysis.activation_extraction import (
    ActivationRecord,
    build_matched_state_dataset,
    compute_state_metadata,
    extract_activations,
    verify_extraction_preserves_predictions,
)
from rewardlens.analysis.probing import (
    fit_probe,
    build_observable_baselines,
    spatial_train_test_split,
    check_label_quality,
)
from rewardlens.analysis.representation_analysis import (
    activation_statistics,
    centered_kernel_alignment,
    cosine_similarity_matrix,
    fit_pca_projection,
    project_with_fitted_pca,
    track_representation_evolution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a minimal checkpoint for testing."""
    config = DQNConfig(seed=42)
    net = QNetwork(obs_dim=7, n_actions=4, hidden_dim=128)
    target = QNetwork(obs_dim=7, n_actions=4, hidden_dim=128)
    optimizer = torch.optim.Adam(net.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, net, target, optimizer, step=1000, config=config)
    return path, net


@pytest.fixture
def small_observations():
    return build_matched_state_dataset(
        grid_width=4, grid_height=4,
        goal_pos=(3, 0), reward_pos=(1, 2),
        time_fractions=(0.0, 0.5, 1.0),
    )


# ---------------------------------------------------------------------------
# Activation Extraction Tests
# ---------------------------------------------------------------------------

class TestActivationExtraction:
    def test_matched_state_dataset_shape(self):
        obs = build_matched_state_dataset()
        assert obs.shape == (8 * 8 * 5, 7)
        assert obs.dtype == np.float32

    def test_matched_state_dataset_normalized(self):
        obs = build_matched_state_dataset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_small_dataset_shape(self, small_observations):
        assert small_observations.shape == (4 * 4 * 3, 7)

    def test_extract_activation_dimensions(self, dummy_checkpoint, small_observations):
        path, _ = dummy_checkpoint
        rec = extract_activations(path, small_observations, seed=42, condition="test", step=1000)
        assert rec.hidden1.shape == (len(small_observations), 128)
        assert rec.hidden2.shape == (len(small_observations), 128)
        assert rec.q_values.shape == (len(small_observations), 4)
        assert rec.greedy_actions.shape == (len(small_observations),)

    def test_post_relu_activations_non_negative(self, dummy_checkpoint, small_observations):
        path, _ = dummy_checkpoint
        rec = extract_activations(path, small_observations, seed=42, condition="test", step=1000)
        assert (rec.hidden1 >= 0).all(), "Hidden1 should be post-ReLU (non-negative)"
        assert (rec.hidden2 >= 0).all(), "Hidden2 should be post-ReLU (non-negative)"

    def test_extraction_preserves_predictions(self, dummy_checkpoint, small_observations):
        path, _ = dummy_checkpoint
        assert verify_extraction_preserves_predictions(path, small_observations)

    def test_extraction_does_not_alter_model(self, dummy_checkpoint, small_observations):
        path, net = dummy_checkpoint
        obs_t = torch.as_tensor(small_observations, dtype=torch.float32)
        with torch.no_grad():
            q_before = net(obs_t).numpy().copy()

        extract_activations(path, small_observations, seed=42, condition="test", step=1000)

        with torch.no_grad():
            q_after = net(obs_t).numpy()
        np.testing.assert_array_equal(q_before, q_after)

    def test_greedy_actions_match_argmax(self, dummy_checkpoint, small_observations):
        path, _ = dummy_checkpoint
        rec = extract_activations(path, small_observations, seed=42, condition="test", step=1000)
        expected = rec.q_values.argmax(axis=1)
        np.testing.assert_array_equal(rec.greedy_actions, expected)

    def test_matched_states_identical_across_models(self):
        """The same observation set is used for both models."""
        obs1 = build_matched_state_dataset()
        obs2 = build_matched_state_dataset()
        np.testing.assert_array_equal(obs1, obs2)


class TestStateMetadata:
    def test_metadata_keys(self):
        obs = build_matched_state_dataset()
        meta = compute_state_metadata(obs)
        expected = {"agent_x", "agent_y", "goal_x", "goal_y",
                    "reward_x", "reward_y", "time_remaining",
                    "dist_to_goal", "dist_to_reward"}
        assert set(meta.keys()) == expected

    def test_distances_non_negative(self):
        obs = build_matched_state_dataset()
        meta = compute_state_metadata(obs)
        assert (meta["dist_to_goal"] >= 0).all()
        assert (meta["dist_to_reward"] >= 0).all()


# ---------------------------------------------------------------------------
# Probing Tests
# ---------------------------------------------------------------------------

class TestProbing:
    def test_fit_probe_basic(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 10).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.int64)
        X_test = rng.randn(50, 10).astype(np.float32)
        y_test = (X_test[:, 0] > 0).astype(np.int64)

        result = fit_probe(X_train, y_train, X_test, y_test, name="test")
        assert result.auroc is not None
        assert 0.0 <= result.balanced_accuracy <= 1.0
        assert result.confusion.shape == (2, 2)
        assert result.n_train == 100
        assert result.n_test == 50

    def test_probe_single_class_test(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5).astype(np.float32)
        y_train = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        X_test = rng.randn(20, 5).astype(np.float32)
        y_test = np.zeros(20, dtype=np.int64)

        result = fit_probe(X_train, y_train, X_test, y_test, name="single_class")
        assert result.auroc is None

    def test_observable_baselines_shapes(self):
        obs = build_matched_state_dataset()
        meta = compute_state_metadata(obs)
        baselines = build_observable_baselines(meta)
        n = len(obs)
        assert baselines["position_time"].shape == (n, 3)
        assert baselines["distances"].shape == (n, 2)
        assert baselines["combined_observable"].shape == (n, 5)

    def test_spatial_split_no_position_leakage(self):
        obs = build_matched_state_dataset()
        labels = np.random.RandomState(42).randint(0, 2, len(obs))
        X_tr, y_tr, X_te, y_te = spatial_train_test_split(obs, labels)

        # Extract positions from train and test
        ax_tr = (X_tr[:, 0] * 8).round().astype(int)
        ay_tr = (X_tr[:, 1] * 8).round().astype(int)
        ax_te = (X_te[:, 0] * 8).round().astype(int)
        ay_te = (X_te[:, 1] * 8).round().astype(int)
        train_pos = set(zip(ax_tr.tolist(), ay_tr.tolist()))
        test_pos = set(zip(ax_te.tolist(), ay_te.tolist()))
        assert train_pos.isdisjoint(test_pos), "Train and test positions must not overlap"

    def test_check_label_quality(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        info = check_label_quality(labels)
        assert info["n_classes"] == 2
        assert info["positive_count"] == 5
        assert info["viable"]

    def test_check_label_quality_single_class(self):
        labels = np.zeros(20, dtype=np.int64)
        info = check_label_quality(labels)
        assert not info["viable"]


# ---------------------------------------------------------------------------
# Representation Analysis Tests
# ---------------------------------------------------------------------------

class TestRepresentationAnalysis:
    def test_activation_statistics(self, dummy_checkpoint, small_observations):
        path, _ = dummy_checkpoint
        rec = extract_activations(path, small_observations, seed=42, condition="test", step=1000)
        stats = activation_statistics(rec)
        assert "hidden1" in stats
        assert "hidden2" in stats
        for layer in ["hidden1", "hidden2"]:
            assert "mean" in stats[layer]
            assert "sparsity" in stats[layer]
            assert 0.0 <= stats[layer]["sparsity"] <= 1.0
            assert 0.0 <= stats[layer]["active_fraction"] <= 1.0

    def test_cka_self_similarity(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10).astype(np.float32)
        cka = centered_kernel_alignment(X, X)
        assert abs(cka - 1.0) < 1e-5

    def test_cka_range(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10).astype(np.float32)
        Y = rng.randn(50, 10).astype(np.float32)
        cka = centered_kernel_alignment(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_cosine_similarity_shape(self):
        A = np.random.randn(20, 5).astype(np.float32)
        B = np.random.randn(20, 5).astype(np.float32)
        sim = cosine_similarity_matrix(A, B)
        assert sim.shape == (20,)

    def test_pca_projection_shape(self):
        X = np.random.randn(100, 128).astype(np.float32)
        proj, pca = fit_pca_projection(X, n_components=2)
        assert proj.shape == (100, 2)
        assert len(pca.explained_variance_ratio_) == 2

    def test_project_with_fitted_pca(self):
        X = np.random.randn(100, 128).astype(np.float32)
        _, pca = fit_pca_projection(X, n_components=2)
        Y = np.random.randn(50, 128).astype(np.float32)
        proj = project_with_fitted_pca(Y, pca)
        assert proj.shape == (50, 2)

    def test_track_evolution(self, dummy_checkpoint, small_observations):
        path, _ = dummy_checkpoint
        rec = extract_activations(path, small_observations, seed=42, condition="test", step=1000)
        # Simulate a 3-checkpoint sequence by using the same record
        records = [rec, rec, rec]
        evo = track_representation_evolution(records)
        assert len(evo["steps"]) == 3
        assert all(c == pytest.approx(1.0, abs=1e-4) for c in evo["cka_h1_vs_first"])

    def test_reproducible_extraction(self, dummy_checkpoint, small_observations):
        path, _ = dummy_checkpoint
        rec1 = extract_activations(path, small_observations, seed=42, condition="test", step=1000)
        rec2 = extract_activations(path, small_observations, seed=42, condition="test", step=1000)
        np.testing.assert_array_equal(rec1.hidden1, rec2.hidden1)
        np.testing.assert_array_equal(rec1.q_values, rec2.q_values)


# ---------------------------------------------------------------------------
# Integration / End-to-End
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_end_to_end_small_probe(self):
        """Full pipeline: build dataset, extract, label, split, probe."""
        rng = np.random.RandomState(42)
        obs = build_matched_state_dataset(
            grid_width=4, grid_height=4,
            goal_pos=(3, 0), reward_pos=(1, 2),
            time_fractions=(0.0, 0.5, 1.0),
        )
        # Synthetic activations
        n = len(obs)
        hidden = rng.randn(n, 32).astype(np.float32)
        labels = (hidden[:, 0] > 0.0).astype(np.int64)

        info = check_label_quality(labels)
        assert info["viable"]

        X_tr, y_tr, X_te, y_te = spatial_train_test_split(
            obs, labels, grid_width=4, grid_height=4,
        )
        if check_label_quality(y_tr)["viable"] and check_label_quality(y_te)["viable"]:
            result = fit_probe(
                hidden[:len(X_tr)], y_tr[:len(X_tr)],
                hidden[len(X_tr):len(X_tr) + len(X_te)], y_te[:len(X_te)],
                name="integration",
            )
            assert 0.0 <= result.balanced_accuracy <= 1.0

    def test_checkpoint_loading_on_real_data(self):
        """Verify we can load a real checkpoint if it exists."""
        ckpt = Path("results/phase2/baseline_seed42/checkpoints/checkpoint_final.pt")
        if not ckpt.exists():
            pytest.skip("Real checkpoints not available")
        obs = build_matched_state_dataset()
        assert verify_extraction_preserves_predictions(ckpt, obs)
        rec = extract_activations(ckpt, obs, seed=42, condition="flawed", step=100000)
        assert rec.hidden1.shape == (320, 128)
        assert (rec.hidden1 >= 0).all()
        assert (rec.hidden2 >= 0).all()
