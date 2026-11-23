"""Phase 1 verification: run scripted policies, validate assertions, generate visuals."""

from __future__ import annotations

import sys
from pathlib import Path

from rewardlens.envs.reward_grid import RewardGridEnv, RewardGridConfig
from rewardlens.envs.scripted_policies import goal_policy_actions, exploit_policy_actions
from rewardlens.envs.trajectory import record_episode
from rewardlens.visualization.gridworld_viz import (
    render_gridworld_overview,
    render_trajectory_gif,
    render_trajectory_comparison,
    render_cumulative_reward,
)

SEED = 42
RESULTS_DIR = Path("results/phase1")
ASSETS_FIGURES = Path("assets/figures")
ASSETS_GIFS = Path("assets/gifs")


def main() -> int:
    config = RewardGridConfig()
    env = RewardGridEnv(config=config)
    failures: list[str] = []

    # -- Run goal policy --
    goal_actions = goal_policy_actions(config)
    goal_traj = record_episode(env, goal_actions, "Goal Policy", "goal_ep_01", seed=SEED)
    goal_traj.save(RESULTS_DIR / "goal_trajectory.json")

    # -- Run exploit policy --
    exploit_actions = exploit_policy_actions(config)
    exploit_traj = record_episode(env, exploit_actions, "Exploit Policy", "exploit_ep_01", seed=SEED)
    exploit_traj.save(RESULTS_DIR / "exploit_trajectory.json")

    # -- Validate assertions --
    def check(condition: bool, msg: str) -> None:
        if not condition:
            failures.append(msg)

    check(goal_traj.goal_reached, "Goal policy did not reach the goal")
    check(not exploit_traj.goal_reached, "Exploit policy reached the goal (should not)")
    check(goal_traj.terminated, "Goal policy did not terminate")
    check(exploit_traj.truncated, "Exploit policy did not truncate")
    check(
        exploit_traj.total_reward > goal_traj.total_reward,
        f"Exploit reward ({exploit_traj.total_reward:.2f}) <= goal reward ({goal_traj.total_reward:.2f})",
    )
    check(
        exploit_traj.final_reward_tile_visits >= config.exploit_visit_threshold,
        f"Exploit visits ({exploit_traj.final_reward_tile_visits}) < threshold ({config.exploit_visit_threshold})",
    )
    check(goal_traj.final_reward_tile_visits == 0, "Goal policy visited reward tile")

    # Check reward-hacking classification via info dict
    env2 = RewardGridEnv(config=config)
    obs, _ = env2.reset(seed=SEED)
    for a in exploit_actions:
        obs, r, terminated, truncated, info = env2.step(a)
        if terminated or truncated:
            break
    check(info["is_reward_hacking"], "Exploit episode not classified as reward hacking")

    env3 = RewardGridEnv(config=config)
    obs, _ = env3.reset(seed=SEED)
    for a in goal_actions:
        obs, r, terminated, truncated, info = env3.step(a)
        if terminated or truncated:
            break
    check(not info["is_reward_hacking"], "Goal episode incorrectly classified as reward hacking")

    # -- Print summary --
    print("\n" + "=" * 70)
    print("  RewardLens Phase 1 — Environment Verification")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Goal Policy':>15} {'Exploit Policy':>15}")
    print("-" * 60)
    print(f"{'Total proxy reward':<30} {goal_traj.total_reward:>15.2f} {exploit_traj.total_reward:>15.2f}")
    print(f"{'Goal completed':<30} {str(goal_traj.goal_reached):>15} {str(exploit_traj.goal_reached):>15}")
    print(f"{'Reward-tile visits':<30} {goal_traj.final_reward_tile_visits:>15} {exploit_traj.final_reward_tile_visits:>15}")
    print(f"{'Episode steps':<30} {goal_traj.total_steps:>15} {exploit_traj.total_steps:>15}")
    print(f"{'Terminated':<30} {str(goal_traj.terminated):>15} {str(exploit_traj.terminated):>15}")
    print(f"{'Truncated':<30} {str(goal_traj.truncated):>15} {str(exploit_traj.truncated):>15}")
    print(f"{'Reward hacking':<30} {'False':>15} {'True':>15}")
    print()

    # -- Generate visuals --
    print("Generating visualizations...")

    render_gridworld_overview(config.to_dict(), ASSETS_FIGURES / "gridworld_overview.png")
    print(f"  ✓ {ASSETS_FIGURES / 'gridworld_overview.png'}")

    render_trajectory_gif(goal_traj, ASSETS_GIFS / "goal_policy.gif")
    print(f"  ✓ {ASSETS_GIFS / 'goal_policy.gif'}")

    render_trajectory_gif(
        exploit_traj,
        ASSETS_GIFS / "exploit_policy.gif",
        max_frames=40,
        frame_duration_ms=150,
    )
    print(f"  ✓ {ASSETS_GIFS / 'exploit_policy.gif'}")

    render_trajectory_comparison(goal_traj, exploit_traj, ASSETS_FIGURES / "trajectory_comparison.png")
    print(f"  ✓ {ASSETS_FIGURES / 'trajectory_comparison.png'}")

    render_cumulative_reward(goal_traj, exploit_traj, ASSETS_FIGURES / "cumulative_reward.png")
    print(f"  ✓ {ASSETS_FIGURES / 'cumulative_reward.png'}")

    # -- Report --
    if failures:
        print(f"\n✗ {len(failures)} assertion(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n✓ All assertions passed. Phase 1 environment verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
