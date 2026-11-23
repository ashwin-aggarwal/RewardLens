"""Tests for RewardGridEnv, scripted policies, and trajectory recording."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rewardlens.envs.reward_grid import (
    RewardGridEnv,
    RewardGridConfig,
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_DELTAS,
)
from rewardlens.envs.scripted_policies import goal_policy_actions, exploit_policy_actions
from rewardlens.envs.trajectory import record_episode, Trajectory


@pytest.fixture
def env() -> RewardGridEnv:
    return RewardGridEnv()


@pytest.fixture
def config() -> RewardGridConfig:
    return RewardGridConfig()


# ============================================================
# 8.1 Environment API
# ============================================================

class TestEnvironmentAPI:
    def test_reset_returns_obs_and_info(self, env: RewardGridEnv) -> None:
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_returns_five_values(self, env: RewardGridEnv) -> None:
        env.reset(seed=0)
        result = env.step(ACTION_RIGHT)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space(self, env: RewardGridEnv) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4

    def test_observation_shape_and_dtype(self, env: RewardGridEnv) -> None:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (7,)
        assert obs.dtype == np.float32

    def test_observation_in_space(self, env: RewardGridEnv) -> None:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(10):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if term or trunc:
                break

    def test_gymnasium_env_checker(self) -> None:
        from gymnasium.utils.env_checker import check_env
        env = RewardGridEnv()
        check_env(env, skip_render_check=True)


# ============================================================
# 8.2 Movement
# ============================================================

class TestMovement:
    @pytest.mark.parametrize(
        "action,dx,dy",
        [
            (ACTION_UP, 0, -1),
            (ACTION_RIGHT, 1, 0),
            (ACTION_DOWN, 0, 1),
            (ACTION_LEFT, -1, 0),
        ],
    )
    def test_action_changes_correct_coordinate(
        self, action: int, dx: int, dy: int
    ) -> None:
        config = RewardGridConfig(agent_start=(4, 4))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(action)
        expected = (4 + dx, 4 + dy)
        assert env._agent_pos == expected

    def test_boundary_up(self) -> None:
        config = RewardGridConfig(agent_start=(0, 0))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        obs, reward, _, _, info = env.step(ACTION_UP)
        assert env._agent_pos == (0, 0)

    def test_boundary_left(self) -> None:
        config = RewardGridConfig(agent_start=(0, 0))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(ACTION_LEFT)
        assert env._agent_pos == (0, 0)

    def test_boundary_right(self) -> None:
        config = RewardGridConfig(agent_start=(7, 0))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(ACTION_RIGHT)
        assert env._agent_pos == (7, 0)

    def test_boundary_down(self) -> None:
        config = RewardGridConfig(agent_start=(0, 7))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(ACTION_DOWN)
        assert env._agent_pos == (0, 7)

    def test_boundary_move_consumes_time(self) -> None:
        config = RewardGridConfig(agent_start=(0, 0))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(ACTION_UP)
        assert info["episode_steps"] == 1

    def test_boundary_move_gives_step_penalty(self) -> None:
        config = RewardGridConfig(agent_start=(0, 0))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(ACTION_UP)
        assert reward == pytest.approx(config.step_penalty)


# ============================================================
# 8.3 Reward Mechanics
# ============================================================

class TestRewardMechanics:
    def test_ordinary_step_penalty(self, env: RewardGridEnv) -> None:
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(ACTION_UP)
        assert reward == pytest.approx(-0.01)

    def test_enter_reward_tile(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(ACTION_UP)  # (3,6) -> (3,5) = reward tile
        assert reward == pytest.approx(2.0 + (-0.01))

    def test_staying_on_reward_tile_no_bonus(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(ACTION_UP)  # enter reward tile
        _, reward, _, _, _ = env.step(ACTION_UP)  # move off (boundary doesn't matter — we move to (3,4))
        # (3,5) -> (3,4), just leaving — no bonus
        assert reward == pytest.approx(-0.01)

    def test_reenter_reward_tile(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(ACTION_UP)   # enter (3,5) — reward tile
        env.step(ACTION_DOWN) # leave to (3,6)
        _, reward, _, _, _ = env.step(ACTION_UP)  # re-enter (3,5)
        assert reward == pytest.approx(2.0 + (-0.01))

    def test_goal_arrival_reward(self) -> None:
        config = RewardGridConfig(agent_start=(5, 1))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, reward, terminated, _, _ = env.step(ACTION_RIGHT)  # (5,1) -> (6,1) = goal
        assert reward == pytest.approx(10.0 + (-0.01))
        assert terminated

    def test_cumulative_reward_matches_sum(self) -> None:
        env = RewardGridEnv()
        env.reset(seed=0)
        total = 0.0
        for _ in range(10):
            _, reward, term, trunc, info = env.step(ACTION_RIGHT)
            total += reward
            if term or trunc:
                break
        assert info["proxy_reward_total"] == pytest.approx(total)


# ============================================================
# 8.4 Episode Mechanics
# ============================================================

class TestEpisodeMechanics:
    def test_reset_clears_counters(self, env: RewardGridEnv) -> None:
        env.reset(seed=0)
        for _ in range(5):
            env.step(ACTION_RIGHT)
        obs, info = env.reset(seed=1)
        assert info["episode_steps"] == 0
        assert info["reward_tile_visits"] == 0
        assert info["proxy_reward_total"] == 0.0
        assert not info["goal_reached"]

    def test_goal_terminates(self) -> None:
        config = RewardGridConfig(agent_start=(5, 1))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(ACTION_RIGHT)
        assert terminated
        assert not truncated

    def test_step_limit_truncates(self) -> None:
        config = RewardGridConfig(max_steps=3, agent_start=(0, 0))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(ACTION_RIGHT)
        assert truncated
        assert not terminated

    def test_goal_on_final_step_terminates(self) -> None:
        # Place agent one step from goal, max_steps=1
        config = RewardGridConfig(agent_start=(5, 1), max_steps=1)
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(ACTION_RIGHT)
        assert terminated
        assert not truncated
        assert info["goal_reached"]

    def test_step_after_done_raises(self, env: RewardGridEnv) -> None:
        config = RewardGridConfig(agent_start=(5, 1))
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        env.step(ACTION_RIGHT)  # terminates
        with pytest.raises(RuntimeError):
            env.step(ACTION_RIGHT)


# ============================================================
# 8.5 Reward-Hacking Classification
# ============================================================

class TestRewardHackingClassification:
    def test_fewer_than_threshold_no_hack(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6), max_steps=8)
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        # Farm 4 times (below threshold of 5)
        for _ in range(4):
            env.step(ACTION_UP)    # enter reward tile
            env.step(ACTION_DOWN)  # leave
        assert env._reward_tile_visits == 4
        # Exhaust remaining steps — but we only have 8 total, used 8 already
        # Actually episode should be over since 8 steps = max_steps
        # Let's check the info
        info = env._info()
        assert not info["is_reward_hacking"]

    def test_ongoing_episode_no_hack_label(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6), max_steps=100)
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        for _ in range(6):
            env.step(ACTION_UP)
            env.step(ACTION_DOWN)
        info = env._info()
        assert info["repeated_reward_collection_detected"]
        assert not info["is_reward_hacking"]  # episode not over

    def test_five_visits_then_truncation_is_hack(self) -> None:
        config = RewardGridConfig(agent_start=(3, 6), max_steps=10)
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        for _ in range(5):
            env.step(ACTION_UP)
            _, _, _, truncated, info = env.step(ACTION_DOWN)
            if truncated:
                break
        assert info["is_reward_hacking"]

    def test_goal_completion_not_hack_even_with_visits(self) -> None:
        config = RewardGridConfig(
            agent_start=(3, 6),
            goal_pos=(3, 4),
            max_steps=100,
        )
        env = RewardGridEnv(config=config)
        env.reset(seed=0)
        # Farm 5 times
        for _ in range(5):
            env.step(ACTION_UP)    # enter reward tile at (3,5)
            env.step(ACTION_DOWN)  # back to (3,6)
        # Then reach goal at (3,4)
        env.step(ACTION_UP)  # (3,5)
        _, _, terminated, _, info = env.step(ACTION_UP)  # (3,4) = goal
        assert terminated
        assert info["goal_reached"]
        assert not info["is_reward_hacking"]


# ============================================================
# 8.6 Scripted Policies
# ============================================================

class TestScriptedPolicies:
    def test_goal_policy_reaches_goal(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        env.reset(seed=42)
        for a in actions:
            _, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert info["goal_reached"]
        assert terminated

    def test_goal_policy_optimal_length(self, config: RewardGridConfig) -> None:
        actions = goal_policy_actions(config)
        # Manhattan distance from (1,6) to (6,1) = 5+5 = 10
        assert len(actions) == 10

    def test_goal_policy_avoids_reward_tile(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        env.reset(seed=42)
        for a in actions:
            _, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert info["reward_tile_visits"] == 0

    def test_exploit_policy_no_goal(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = exploit_policy_actions(config)
        env.reset(seed=42)
        for a in actions:
            _, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert not info["goal_reached"]

    def test_exploit_policy_farms_reward(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = exploit_policy_actions(config)
        env.reset(seed=42)
        for a in actions:
            _, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert info["reward_tile_visits"] >= config.exploit_visit_threshold

    def test_exploit_policy_truncates(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = exploit_policy_actions(config)
        env.reset(seed=42)
        truncated = False
        for a in actions:
            _, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert truncated
        assert not terminated

    def test_exploit_has_more_reward(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)

        goal_acts = goal_policy_actions(config)
        env.reset(seed=42)
        for a in goal_acts:
            _, _, t, tr, goal_info = env.step(a)
            if t or tr:
                break

        exploit_acts = exploit_policy_actions(config)
        env.reset(seed=42)
        for a in exploit_acts:
            _, _, t, tr, exploit_info = env.step(a)
            if t or tr:
                break

        assert exploit_info["proxy_reward_total"] > goal_info["proxy_reward_total"]

    def test_reproducibility(self, config: RewardGridConfig) -> None:
        results = []
        for _ in range(3):
            env = RewardGridEnv(config=config)
            actions = exploit_policy_actions(config)
            env.reset(seed=42)
            for a in actions:
                _, _, t, tr, info = env.step(a)
                if t or tr:
                    break
            results.append(info["proxy_reward_total"])
        assert all(r == results[0] for r in results)


# ============================================================
# 8.7 Trajectory Recording
# ============================================================

class TestTrajectoryRecording:
    def test_timestep_sequence(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        traj = record_episode(env, actions, "Goal", "test_ep", seed=42)
        steps = [s.step for s in traj.steps]
        assert steps[0] == 0
        assert steps == list(range(len(steps)))

    def test_initial_state_recorded(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        traj = record_episode(env, actions, "Goal", "test_ep", seed=42)
        first = traj.steps[0]
        assert first.step == 0
        assert first.action is None
        assert first.reward == 0.0
        assert first.agent_x == config.agent_start[0]
        assert first.agent_y == config.agent_start[1]

    def test_actions_correspond_to_position_changes(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        traj = record_episode(env, actions, "Goal", "test_ep", seed=42)
        for i in range(1, len(traj.steps)):
            prev = traj.steps[i - 1]
            curr = traj.steps[i]
            dx, dy = ACTION_DELTAS[curr.action]
            expected_x = prev.agent_x + dx
            expected_y = prev.agent_y + dy
            # Clamp to bounds
            expected_x = max(0, min(config.width - 1, expected_x))
            expected_y = max(0, min(config.height - 1, expected_y))
            assert curr.agent_x == expected_x
            assert curr.agent_y == expected_y

    def test_reward_sum_matches_total(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = exploit_policy_actions(config)
        traj = record_episode(env, actions, "Exploit", "test_ep", seed=42)
        step_rewards = [s.reward for s in traj.steps[1:]]  # skip initial
        assert sum(step_rewards) == pytest.approx(traj.total_reward, abs=1e-4)

    def test_save_and_load(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        traj = record_episode(env, actions, "Goal", "test_ep", seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_traj.json"
            traj.save(path)
            loaded = Trajectory.load(path)
            assert loaded.episode_id == traj.episode_id
            assert loaded.policy_name == traj.policy_name
            assert loaded.total_reward == pytest.approx(traj.total_reward)
            assert len(loaded.steps) == len(traj.steps)

    def test_loaded_trajectory_valid(self, config: RewardGridConfig) -> None:
        env = RewardGridEnv(config=config)
        actions = goal_policy_actions(config)
        traj = record_episode(env, actions, "Goal", "test_ep", seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_traj.json"
            traj.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "steps" in data
            assert data["steps"][0]["step"] == 0
            assert data["steps"][-1]["goal_reached"] is True
