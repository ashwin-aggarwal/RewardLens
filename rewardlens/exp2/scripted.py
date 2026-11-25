"""Scripted reference policies on arbitrary layouts (demonstrations, not learned behavior)."""

from __future__ import annotations

import numpy as np

from rewardlens.envs.scripted_policies import _bfs_path, _direction
from rewardlens.exp2.layouts import Layout, neighbors
from rewardlens.exp2.procedural_env import ProceduralEnvParams


def _actions_along(path) -> list[int]:
    return [_direction(path[i], path[i + 1]) for i in range(len(path) - 1)]


def goal_actions(layout: Layout, params: ProceduralEnvParams) -> list[int]:
    """Shortest path to the goal that never steps on the reward tile."""
    return _actions_along(_bfs_path(layout.start, layout.goal, params.width, params.height, avoid={layout.reward}))


def _farm_cell(layout: Layout, params: ProceduralEnvParams):
    return next(n for n in neighbors(layout.reward, params.width, params.height) if n != layout.goal)


def farm_actions(layout: Layout, params: ProceduralEnvParams) -> list[int]:
    """Walk to the reward tile (avoiding the goal), then step off and on until time runs out."""
    actions = _actions_along(_bfs_path(layout.start, layout.reward, params.width, params.height, avoid={layout.goal}))
    adj = _farm_cell(layout, params)
    out, back = _direction(layout.reward, adj), _direction(adj, layout.reward)
    i = 0
    while len(actions) < params.max_steps:
        actions.append(out if i % 2 == 0 else back)
        i += 1
    return actions[:params.max_steps]


def farm_then_finish_actions(layout: Layout, params: ProceduralEnvParams) -> list[int]:
    """Farm as long as possible, then leave at the last moment that still ends on the goal."""
    farm = farm_actions(layout, params)
    deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    def dist(p):
        return abs(p[0] - layout.goal[0]) + abs(p[1] - layout.goal[1])

    pos = layout.start
    actions: list[int] = []
    for t, a in enumerate(farm):
        dx, dy = deltas[a]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not (0 <= nxt[0] < params.width and 0 <= nxt[1] < params.height):
            nxt = pos
        # Taking the farming step must still leave enough time to walk to the goal.
        if dist(nxt) > params.max_steps - (t + 1):
            return actions + _actions_along(_bfs_path(pos, layout.goal, params.width, params.height))
        actions.append(a)
        pos = nxt
    return actions


def play_actions(layout: Layout, actions: list[int], params: ProceduralEnvParams, gamma: float) -> dict:
    """Replay an action list and score it under both reward specifications."""
    pos, visits, goal = layout.start, 0, False
    deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    ret = {"flawed": 0.0, "corrected": 0.0}
    disc = {"flawed": 0.0, "corrected": 0.0}
    steps = 0
    for a in actions[:params.max_steps]:
        dx, dy = deltas[int(a)]
        nx, ny = pos[0] + dx, pos[1] + dy
        new = (nx, ny) if 0 <= nx < params.width and 0 <= ny < params.height else pos
        entered = new == layout.reward and pos != layout.reward
        pos = new
        visits += int(entered)
        goal = pos == layout.goal
        for mode in ret:
            r = params.step_penalty + (params.tile_bonus(mode) if entered else 0.0) + (params.goal_bonus if goal else 0.0)
            ret[mode] += r
            disc[mode] += (gamma ** steps) * r
        steps += 1
        if goal:
            break
    return {"goal_reached": goal, "reward_tile_visits": visits, "steps": steps,
            "return_flawed": ret["flawed"], "return_corrected": ret["corrected"],
            "discounted_flawed": disc["flawed"], "discounted_corrected": disc["corrected"]}
