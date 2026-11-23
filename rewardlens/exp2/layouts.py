"""Procedural layouts for Experiment 2 and their frozen train/dev/test split.

A *layout* is one complete configuration: agent start, goal position and
reward-tile position on an open grid (no obstacles). Splits are made over
whole layouts, never over observations or episodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

Pos = tuple[int, int]
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Layout:
    start: Pos
    goal: Pos
    reward: Pos

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (*self.start, *self.goal, *self.reward)

    @classmethod
    def from_tuple(cls, t: Iterable[int]) -> "Layout":
        a = [int(v) for v in t]
        return cls((a[0], a[1]), (a[2], a[3]), (a[4], a[5]))

    def manhattan_to_goal(self) -> int:
        return abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])


def neighbors(pos: Pos, width: int, height: int) -> list[Pos]:
    x, y = pos
    out = []
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            out.append((nx, ny))
    return out


def is_valid_layout(layout: Layout, width: int = 8, height: int = 8) -> bool:
    """A layout is valid when both behaviors are mechanically possible.

    - start, goal and reward tile are three different in-bounds cells;
    - the goal can be reached without stepping on the reward tile, and the tile
      without stepping on the goal (on an open grid this only fails for
      degenerate sizes, but it is checked rather than assumed);
    - the tile has a neighbor that is not the goal, so it can be left and
      re-entered repeatedly without ending the episode.
    """
    cells = (layout.start, layout.goal, layout.reward)
    if len(set(cells)) != 3:
        return False
    if any(not (0 <= x < width and 0 <= y < height) for x, y in cells):
        return False
    if not any(n != layout.goal for n in neighbors(layout.reward, width, height)):
        return False
    return (_reachable(layout.start, layout.goal, {layout.reward}, width, height)
            and _reachable(layout.start, layout.reward, {layout.goal}, width, height))


def _reachable(src: Pos, dst: Pos, avoid: set[Pos], width: int, height: int) -> bool:
    seen, stack = {src}, [src]
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        for n in neighbors(cur, width, height):
            if n not in seen and (n == dst or n not in avoid):
                seen.add(n)
                stack.append(n)
    return False


def generate_layouts(n: int, seed: int, width: int = 8, height: int = 8,
                     exclude: Iterable[Layout] = ()) -> list[Layout]:
    """Draw `n` distinct valid layouts, deterministically from `seed`.

    Layouts in `exclude` are never returned, so a new set can be made disjoint
    from every existing one.
    """
    n_cells = width * height
    max_layouts = n_cells * (n_cells - 1) * (n_cells - 2)
    if n > max_layouts:
        raise ValueError(f"requested {n} layouts but only {max_layouts} exist")
    rng = np.random.RandomState(seed)
    seen: set[tuple] = {l.as_tuple() for l in exclude}
    if n > max_layouts - len(seen):
        raise ValueError("not enough layouts left after exclusions")
    out: list[Layout] = []
    while len(out) < n:
        cells = rng.choice(n_cells, size=3, replace=False)
        layout = Layout(*[(int(c) % width, int(c) // width) for c in cells])
        key = layout.as_tuple()
        if key in seen or not is_valid_layout(layout, width, height):
            continue
        seen.add(key)
        out.append(layout)
    return out


def split_layouts(layouts: list[Layout], fractions: tuple[float, float, float],
                  seed: int) -> dict[str, list[Layout]]:
    """Shuffle with `seed` and cut into train/dev/test. Every layout lands in exactly one split."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    order = np.random.RandomState(seed).permutation(len(layouts))
    n_train = int(round(fractions[0] * len(layouts)))
    n_dev = int(round(fractions[1] * len(layouts)))
    idx = {"train": order[:n_train], "dev": order[n_train:n_train + n_dev], "test": order[n_train + n_dev:]}
    return {k: [layouts[i] for i in v] for k, v in idx.items()}


def layouts_to_array(layouts: list[Layout]) -> np.ndarray:
    """(N, 6) int array: start_x, start_y, goal_x, goal_y, reward_x, reward_y."""
    return np.array([l.as_tuple() for l in layouts], dtype=np.int64).reshape(-1, 6)


def manifest_hash(splits: dict[str, list[Layout]]) -> str:
    h = hashlib.sha256()
    for name in SPLITS:
        h.update(name.encode())
        h.update(layouts_to_array(splits[name]).tobytes())
    return h.hexdigest()


def save_manifest(path: str | Path, splits: dict[str, list[Layout]], meta: dict) -> dict:
    manifest = {
        **meta,
        "columns": ["start_x", "start_y", "goal_x", "goal_y", "reward_x", "reward_y"],
        "counts": {k: len(splits[k]) for k in SPLITS},
        "sha256": manifest_hash(splits),
        "splits": {k: [list(l.as_tuple()) for l in splits[k]] for k in SPLITS},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    return manifest


def load_manifest(path: str | Path, verify: bool = True) -> tuple[dict[str, list[Layout]], dict]:
    with open(path) as f:
        manifest = json.load(f)
    splits = {k: [Layout.from_tuple(t) for t in manifest["splits"][k]] for k in SPLITS}
    if verify:
        if manifest_hash(splits) != manifest["sha256"]:
            raise ValueError("layout manifest does not match its recorded hash")
        assert_disjoint(splits)
    return splits, {k: v for k, v in manifest.items() if k != "splits"}


def assert_disjoint(splits: dict[str, list[Layout]]) -> None:
    sets = {k: {l.as_tuple() for l in v} for k, v in splits.items()}
    for k, v in splits.items():
        if len(sets[k]) != len(v):
            raise ValueError(f"duplicate layouts inside split {k}")
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"splits {a} and {b} share layouts")


def unseen_pair_mask(eval_layouts: list[Layout], train_layouts: list[Layout]) -> np.ndarray:
    """True where the (goal, reward tile) pair never occurs in the training split.

    Most held-out layouts share their goal/tile pair with some training layout
    and differ only in the start. This mask picks out the stricter subset.
    """
    seen = {(l.goal, l.reward) for l in train_layouts}
    return np.array([(l.goal, l.reward) not in seen for l in eval_layouts], dtype=bool)


def layout_set_hash(layouts: list[Layout]) -> str:
    return hashlib.sha256(layouts_to_array(layouts).tobytes()).hexdigest()


def save_layout_set(path: str | Path, layouts: list[Layout], meta: dict) -> dict:
    """Save one standalone layout set (for example a reserved evaluation set)."""
    manifest = {**meta, "columns": ["start_x", "start_y", "goal_x", "goal_y", "reward_x", "reward_y"],
                "count": len(layouts), "sha256": layout_set_hash(layouts),
                "layouts": [list(l.as_tuple()) for l in layouts]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    return manifest


def load_layout_set(path: str | Path) -> tuple[list[Layout], dict]:
    with open(path) as f:
        manifest = json.load(f)
    layouts = [Layout.from_tuple(t) for t in manifest["layouts"]]
    if layout_set_hash(layouts) != manifest["sha256"]:
        raise ValueError("layout set does not match its recorded hash")
    if len({l.as_tuple() for l in layouts}) != len(layouts):
        raise ValueError("duplicate layouts in layout set")
    return layouts, {k: v for k, v in manifest.items() if k != "layouts"}
