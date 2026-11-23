"""Tests for DQN agent: network, replay buffer, training, evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rewardlens.agents.dqn import (
    DQNConfig,
    QNetwork,
    ReplayBuffer,
    linear_epsilon,
    save_checkpoint,
    load_checkpoint,
    load_q_network_for_eval,
)
from rewardlens.envs.reward_grid import RewardGridEnv, RewardGridConfig


# ============================================================
# Network
# ============================================================

class TestQNetwork:
    def test_forward_shape(self) -> None:
        net = QNetwork(obs_dim=7, n_actions=4, hidden_dim=128)
        x = torch.randn(1, 7)
        out = net(x)
        assert out.shape == (1, 4)

    def test_batch_forward(self) -> None:
        net = QNetwork()
        x = torch.randn(32, 7)
        out = net(x)
        assert out.shape == (32, 4)

    def test_output_finite(self) -> None:
        net = QNetwork()
        x = torch.randn(8, 7)
        out = net(x)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self) -> None:
        net = QNetwork()
        x = torch.randn(4, 7)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_named_layers(self) -> None:
        net = QNetwork()
        names = {name for name, _ in net.named_modules()}
        assert "fc1" in names
        assert "fc2" in names
        assert "fc3" in names
        assert "relu1" in names
        assert "relu2" in names


# ============================================================
# Replay Buffer
# ============================================================

class TestReplayBuffer:
    def test_add_and_len(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_dim=7)
        assert len(buf) == 0
        obs = np.zeros(7, dtype=np.float32)
        buf.add(obs, 0, 1.0, obs, False)
        assert len(buf) == 1

    def test_capacity_respected(self) -> None:
        buf = ReplayBuffer(capacity=10, obs_dim=7)
        obs = np.zeros(7, dtype=np.float32)
        for _ in range(20):
            buf.add(obs, 0, 1.0, obs, False)
        assert len(buf) == 10

    def test_sample_shapes(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_dim=7, seed=42)
        obs = np.zeros(7, dtype=np.float32)
        for _ in range(50):
            buf.add(obs, 1, 0.5, obs, False)

        batch = buf.sample(16)
        assert batch["obs"].shape == (16, 7)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)
        assert batch["next_obs"].shape == (16, 7)
        assert batch["dones"].shape == (16,)

    def test_sample_dtypes(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_dim=7, seed=42)
        obs = np.zeros(7, dtype=np.float32)
        for _ in range(10):
            buf.add(obs, 2, -0.01, obs, True)
        batch = buf.sample(4)
        assert batch["obs"].dtype == np.float32
        assert batch["actions"].dtype == np.int64
        assert batch["rewards"].dtype == np.float32
        assert batch["dones"].dtype == np.float32

    def test_deterministic_sampling(self) -> None:
        buf1 = ReplayBuffer(capacity=100, obs_dim=7, seed=99)
        buf2 = ReplayBuffer(capacity=100, obs_dim=7, seed=99)
        for i in range(30):
            obs = np.full(7, float(i), dtype=np.float32)
            buf1.add(obs, i % 4, float(i), obs, False)
            buf2.add(obs, i % 4, float(i), obs, False)
        s1 = buf1.sample(8)
        s2 = buf2.sample(8)
        np.testing.assert_array_equal(s1["rewards"], s2["rewards"])


# ============================================================
# TD Update
# ============================================================

class TestTDUpdate:
    def test_bellman_target_terminal(self) -> None:
        """Terminal transitions should not bootstrap."""
        target_net = QNetwork()
        target_net.eval()
        obs = torch.randn(1, 7)
        reward = torch.tensor([5.0])
        done = torch.tensor([1.0])
        gamma = 0.99

        with torch.no_grad():
            next_q = target_net(obs).max(dim=1).values
            target = reward + gamma * (1.0 - done) * next_q

        assert target.item() == pytest.approx(5.0)

    def test_bellman_target_nonterminal(self) -> None:
        """Non-terminal transitions should bootstrap."""
        target_net = QNetwork()
        target_net.eval()
        obs = torch.randn(1, 7)
        reward = torch.tensor([1.0])
        done = torch.tensor([0.0])
        gamma = 0.99

        with torch.no_grad():
            next_q = target_net(obs).max(dim=1).values
            target = reward + gamma * (1.0 - done) * next_q

        assert target.item() != pytest.approx(1.0)

    def test_target_net_no_gradients(self) -> None:
        target_net = QNetwork()
        target_net.eval()
        x = torch.randn(4, 7)
        with torch.no_grad():
            out = target_net(x)
        assert not out.requires_grad

    def test_online_weights_change(self) -> None:
        q_net = QNetwork()
        optimizer = torch.optim.Adam(q_net.parameters(), lr=0.01)
        target_net = QNetwork()
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()

        old_params = {k: v.clone() for k, v in q_net.named_parameters()}

        obs = torch.randn(8, 7)
        actions = torch.randint(0, 4, (8,))
        rewards = torch.randn(8)
        next_obs = torch.randn(8, 7)
        dones = torch.zeros(8)

        with torch.no_grad():
            next_q = target_net(next_obs).max(dim=1).values
            targets = rewards + 0.99 * (1 - dones) * next_q

        current_q = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(current_q, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = False
        for k, v in q_net.named_parameters():
            if not torch.equal(v, old_params[k]):
                changed = True
                break
        assert changed


# ============================================================
# Epsilon Schedule
# ============================================================

class TestEpsilon:
    def test_epsilon_start(self) -> None:
        assert linear_epsilon(0, 100000, 1.0, 0.05, 0.5) == pytest.approx(1.0)

    def test_epsilon_end(self) -> None:
        assert linear_epsilon(100000, 100000, 1.0, 0.05, 0.5) == pytest.approx(0.05)

    def test_epsilon_midpoint(self) -> None:
        eps = linear_epsilon(25000, 100000, 1.0, 0.05, 0.5)
        assert 0.05 < eps < 1.0

    def test_epsilon_after_decay(self) -> None:
        eps = linear_epsilon(60000, 100000, 1.0, 0.05, 0.5)
        assert eps == pytest.approx(0.05)

    def test_greedy_at_zero(self) -> None:
        assert linear_epsilon(0, 100, 0.0, 0.0, 0.5) == pytest.approx(0.0)


# ============================================================
# Checkpoints
# ============================================================

class TestCheckpoints:
    def test_save_and_load(self) -> None:
        q_net = QNetwork()
        target_net = QNetwork()
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters())
        config = DQNConfig()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_ckpt.pt"
            save_checkpoint(path, q_net, target_net, optimizer, 1000, config)
            ckpt = load_checkpoint(path)

            assert ckpt["step"] == 1000
            assert "q_net_state_dict" in ckpt
            assert "config" in ckpt

    def test_load_for_eval(self) -> None:
        q_net = QNetwork()
        target_net = QNetwork()
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters())
        config = DQNConfig()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_ckpt.pt"
            save_checkpoint(path, q_net, target_net, optimizer, 500, config)
            loaded_net, ckpt = load_q_network_for_eval(path)

            x = torch.randn(1, 7)
            with torch.no_grad():
                orig = q_net(x)
                loaded = loaded_net(x)
            torch.testing.assert_close(orig, loaded)

    def test_checkpoint_inference_equivalence(self) -> None:
        q_net = QNetwork()
        target_net = QNetwork()
        optimizer = torch.optim.Adam(q_net.parameters())
        config = DQNConfig()

        x = torch.randn(5, 7)
        with torch.no_grad():
            original_output = q_net(x).clone()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, q_net, target_net, optimizer, 0, config)
            loaded_net, _ = load_q_network_for_eval(path)
            with torch.no_grad():
                loaded_output = loaded_net(x)
            torch.testing.assert_close(original_output, loaded_output)


# ============================================================
# Evaluation
# ============================================================

class TestEvaluation:
    def test_greedy_eval_does_not_modify_weights(self) -> None:
        from scripts.train_dqn import greedy_eval

        config = DQNConfig()
        q_net = QNetwork()
        device = torch.device("cpu")

        old_params = {k: v.clone() for k, v in q_net.state_dict().items()}
        greedy_eval(q_net, config, device, seed=0)
        for k, v in q_net.state_dict().items():
            torch.testing.assert_close(v, old_params[k])

    def test_greedy_eval_returns_metrics(self) -> None:
        from scripts.train_dqn import greedy_eval

        config = DQNConfig()
        q_net = QNetwork()
        device = torch.device("cpu")

        result = greedy_eval(q_net, config, device, seed=0)
        assert "proxy_reward" in result
        assert "goal_reached" in result
        assert "reward_tile_visits" in result
        assert "is_reward_hacking" in result
        assert "episode_steps" in result
        assert isinstance(result["goal_reached"], bool)


# ============================================================
# Integration
# ============================================================

class TestIntegration:
    def test_tiny_training_run(self) -> None:
        """A short training run completes without errors."""
        from scripts.train_dqn import train

        with tempfile.TemporaryDirectory() as tmp:
            config = DQNConfig(
                total_timesteps=200,
                learning_starts=50,
                train_frequency=4,
                target_update_frequency=50,
                eval_interval=100,
                checkpoint_interval=100,
                buffer_capacity=500,
                batch_size=16,
                output_dir=tmp,
                experiment_id="test_tiny",
                device="cpu",
            )
            train(config)

            run_dir = Path(tmp) / "test_tiny"
            assert (run_dir / "config.json").exists()
            assert (run_dir / "training_metrics.csv").exists()
            assert (run_dir / "evaluation_metrics.csv").exists()
            assert (run_dir / "run_metadata.json").exists()
            assert (run_dir / "checkpoints" / "checkpoint_final.pt").exists()

    def test_phase1_tests_still_pass(self) -> None:
        """Sanity check that Phase 1 environment API is unchanged."""
        env = RewardGridEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (7,)
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
