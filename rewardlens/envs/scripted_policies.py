"""Deterministic scripted policies for environment validation."""

from __future__ import annotations

from collections import deque

from rewardlens.envs.reward_grid import (
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_DOWN,
    ACTION_LEFT,
    RewardGridConfig,
)


def _direction(src: tuple[int, int], dst: tuple[int, int]) -> int:
    """Return the action that moves one step from src toward dst (single axis)."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx > 0:
        return ACTION_RIGHT
    if dx < 0:
        return ACTION_LEFT
    if dy > 0:
        return ACTION_DOWN
    return ACTION_UP


def _bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
    avoid: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """BFS shortest path on a grid, optionally avoiding cells."""
    avoid = avoid or set()
    if start == goal:
        return [start]

    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    visited = {start}
    queue: deque[list[tuple[int, int]]] = deque([[start]])

    while queue:
        path = queue.popleft()
        cx, cy = path[-1]
        for dx, dy in deltas:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if (nx, ny) in avoid and (nx, ny) != goal:
                    continue
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                if (nx, ny) == goal:
                    return new_path
                queue.append(new_path)

    raise ValueError(f"No path from {start} to {goal}")


def goal_policy_actions(config: RewardGridConfig | None = None) -> list[int]:
    """Return the action sequence for the shortest path to the goal, avoiding the reward tile."""
    config = config or RewardGridConfig()
    path = _bfs_path(
        config.agent_start,
        config.goal_pos,
        config.width,
        config.height,
        avoid={config.reward_pos},
    )
    actions = []
    for i in range(len(path) - 1):
        actions.append(_direction(path[i], path[i + 1]))
    return actions


def exploit_policy_actions(config: RewardGridConfig | None = None) -> list[int]:
    """Return the action sequence for the exploit policy: navigate to reward tile, then farm it."""
    config = config or RewardGridConfig()

    path_to_reward = _bfs_path(
        config.agent_start,
        config.reward_pos,
        config.width,
        config.height,
    )
    actions: list[int] = []
    for i in range(len(path_to_reward) - 1):
        actions.append(_direction(path_to_reward[i], path_to_reward[i + 1]))

    # Adjacent cell for farming: one step down from the reward tile
    adj = (config.reward_pos[0], config.reward_pos[1] + 1)
    if adj[1] >= config.height:
        adj = (config.reward_pos[0], config.reward_pos[1] - 1)

    move_to_adj = _direction(config.reward_pos, adj)
    move_to_reward = _direction(adj, config.reward_pos)

    remaining = config.max_steps - len(actions)
    for i in range(remaining):
        if i % 2 == 0:
            actions.append(move_to_adj)
        else:
            actions.append(move_to_reward)

    return actions
