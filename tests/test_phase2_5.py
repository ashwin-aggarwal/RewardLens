"""Tests for Phase 2.5: corrected-reward (goal-only) DQN configuration and evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rewardlens.agents.dqn import DQNConfig
from rewardlens.envs.reward_grid import RewardGridEnv, RewardGridConfig
from rewardlens.envs.trajectory import record_episode
from rewardlens.envs.scripted_policies import goal_policy_actions, exploit_policy_actions
from scripts.evaluate_phase2_5 import cross_score, common_metrics, is_reward_hacking, _config_from_dict


CORRECTED_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dqn_corrected.json"


class TestCorrectedConfig:
    def test_corrected_config_loads(self) -> None:
        config = DQNConfig.from_json(CORRECTED_CONFIG_PATH)
        assert config.reward_tile_bonus == 0.0

    def test_corrected_config_goal_reward_unchanged(self) -> None:
        config = DQNConfig.from_json(CORRECTED_CONFIG_PATH)
        assert config.goal_bonus == 10.0

    def test_corrected_config_step_penalty_unchanged(self) -> None:
        config = DQNConfig.from_json(CORRECTED_CONFIG_PATH)
        assert config.step_penalty == -0.01

    def test_corrected_config_same_grid_as_flawed(self) -> None:
        config = DQNConfig.from_json(CORRECTED_CONFIG_PATH)
        default = DQNConfig()
        assert config.agent_start == default.agent_start
        assert config.goal_pos == default.goal_pos
        assert config.reward_pos == default.reward_pos
        assert config.max_steps == default.max_steps

    def test_corrected_config_same_architecture_and_hparams(self) -> None:
        config = DQNConfig.from_json(CORRECTED_CONFIG_PATH)
        default = DQNConfig()
        assert config.hidden_dim == default.hidden_dim
        assert config.learning_rate == default.learning_rate
        assert config.gamma == default.gamma
        assert config.epsilon_start == default.epsilon_start
        assert config.epsilon_end == default.epsilon_end


class TestZeroRewardTileBonus:
    def test_entering_reward_tile_gives_no_bonus(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6), reward_tile_bonus=0.0)
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(0)  # move up into reward tile at (3,5)
        assert reward == pytest.approx(-0.01)

    def test_repeated_entry_still_no_bonus(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6), reward_tile_bonus=0.0)
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(0)  # enter
        env.step(2)  # leave
        _, reward, _, _, _ = env.step(0)  # re-enter
        assert reward == pytest.approx(-0.01)

    def test_goal_reward_unaffected_by_zero_bonus(self) -> None:
        config = RewardGridConfig(agent_start=(5, 1), reward_tile_bonus=0.0)
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, reward, terminated, _, _ = env.step(1)  # move right into goal
        assert reward == pytest.approx(10.0 - 0.01)
        assert terminated

    def test_movement_mechanics_unchanged(self) -> None:
        """Boundary handling and movement deltas are identical regardless of reward_tile_bonus."""
        flawed_config = RewardGridConfig(agent_start=(0, 0), reward_tile_bonus=2.0)
        corrected_config = RewardGridConfig(agent_start=(0, 0), reward_tile_bonus=0.0)
        flawed_env = RewardGridEnv(config=flawed_config)
        corrected_env = RewardGridEnv(config=corrected_config)
        flawed_env.reset(seed=0)
        corrected_env.reset(seed=0)
        for action in [1, 1, 2, 2, 0, 3]:
            fo, _, _, _, finfo = flawed_env.step(action)
            co, _, _, _, cinfo = corrected_env.step(action)
            assert finfo["agent_pos"] == cinfo["agent_pos"]


class TestCrossScoring:
    def test_exploit_trajectory_scores_near_zero_under_corrected_reward(self) -> None:
        """The flawed-reward exploit path should earn almost nothing once the tile bonus is removed."""
        flawed_config = RewardGridConfig()
        env = RewardGridEnv(config=flawed_config)
        actions = exploit_policy_actions(flawed_config)
        traj = record_episode(env, actions, "Exploit", "test_exploit", seed=42)

        cross_traj = cross_score(traj, alt_reward_tile_bonus=0.0, label="cross-scored")
        # Only step penalties remain: no goal, no reward-tile bonus.
        assert cross_traj.total_reward == pytest.approx(-1.0, abs=1e-6)
        assert not cross_traj.goal_reached

    def test_goal_trajectory_unaffected_by_cross_scoring(self) -> None:
        """A trajectory that never visits the reward tile scores identically under either spec."""
        flawed_config = RewardGridConfig()
        env = RewardGridEnv(config=flawed_config)
        actions = goal_policy_actions(flawed_config)
        traj = record_episode(env, actions, "Goal", "test_goal", seed=42)

        cross_traj = cross_score(traj, alt_reward_tile_bonus=0.0, label="cross-scored")
        assert cross_traj.total_reward == pytest.approx(traj.total_reward, abs=1e-6)

    def test_cross_scoring_does_not_change_actions(self) -> None:
        flawed_config = RewardGridConfig()
        env = RewardGridEnv(config=flawed_config)
        actions = exploit_policy_actions(flawed_config)
        traj = record_episode(env, actions, "Exploit", "test_exploit", seed=42)

        cross_traj = cross_score(traj, alt_reward_tile_bonus=0.0, label="cross-scored")
        original_actions = [s.action for s in traj.steps[1:]]
        cross_actions = [s.action for s in cross_traj.steps[1:]]
        assert original_actions == cross_actions

    def test_config_from_dict_overrides_bonus(self) -> None:
        base = RewardGridConfig().to_dict()
        alt = _config_from_dict(base, reward_tile_bonus=0.0)
        assert alt.reward_tile_bonus == 0.0
        assert alt.goal_bonus == base["goal_bonus"]


class TestCommonMetricsAndClassification:
    def test_common_metrics_fields_present(self) -> None:
        config = RewardGridConfig()
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        traj = record_episode(env, actions, "Goal", "test", seed=42)
        metrics = common_metrics(traj)
        assert set(metrics.keys()) == {
            "goal_reached", "episode_steps", "reward_tile_visits", "terminated", "truncated",
        }

    def test_classification_consistent_across_reward_specs(self) -> None:
        """Reward-hacking classification depends only on visits/goal, not reward magnitude."""
        config = RewardGridConfig()
        env = RewardGridEnv(config=config)
        actions = exploit_policy_actions(config)
        traj = record_episode(env, actions, "Exploit", "test", seed=42)

        cross_traj = cross_score(traj, alt_reward_tile_bonus=0.0, label="cross-scored")
        # Same positions/visits regardless of reward spec -> same classification.
        assert is_reward_hacking(traj) == is_reward_hacking(cross_traj)

    def test_goal_policy_never_classified_as_hacking(self) -> None:
        config = RewardGridConfig()
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        traj = record_episode(env, actions, "Goal", "test", seed=42)
        assert not is_reward_hacking(traj)


class TestTrainingWithCorrectedConfig:
    def test_tiny_training_run_with_zero_bonus(self) -> None:
        from scripts.train_dqn import train

        with tempfile.TemporaryDirectory() as tmp:
            config = DQNConfig(
                reward_tile_bonus=0.0,
                total_timesteps=200,
                learning_starts=50,
                train_frequency=4,
                target_update_frequency=50,
                eval_interval=100,
                checkpoint_interval=100,
                buffer_capacity=500,
                batch_size=16,
                output_dir=tmp,
                experiment_id="test_corrected_tiny",
                device="cpu",
            )
            train(config)
            run_dir = Path(tmp) / "test_corrected_tiny"
            assert (run_dir / "config.json").exists()
            assert (run_dir / "training_metrics.csv").exists()
            saved_config = DQNConfig.from_json(run_dir / "config.json")
            assert saved_config.reward_tile_bonus == 0.0
