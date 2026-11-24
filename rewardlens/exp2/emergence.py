"""Experiment 2, Phase 3: tracking how reward hacking emerges during training.

Helpers for the fork-stage checkpoint trajectories, behavioral transition
detection, decision-level comparison of two checkpoints, and controlled
statistics. Everything here is observational: these measures describe how
representations change, not what causes the behavior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from rewardlens.agents.dqn import QNetwork, load_checkpoint
from rewardlens.exp2.evaluation import _DX, _DY, greedy_policy, observations, rollout
from rewardlens.exp2.layouts import Layout, layouts_to_array
from rewardlens.exp2.procedural_env import ProceduralEnvParams

BRANCHES = ("good", "hacking")
REWARD_MODE = {"good": "corrected", "hacking": "flawed"}
AGREE, EQUIVALENT, TILE_OVER_GOAL, OTHER_DISAGREE = "agree", "equivalent_route", "tile_over_goal", "other_disagreement"
DECISION_LABELS = (AGREE, EQUIVALENT, TILE_OVER_GOAL, OTHER_DISAGREE)


# --- checkpoint trajectories -------------------------------------------------

def trajectory_steps(interval: int, total: int) -> list[int]:
    return list(range(0, total + 1, interval))


def checkpoint_file(seed: int, branch: str, step: int, checkpoint_dir: str | Path = "results/exp2/phase1") -> Path:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    return Path(checkpoint_dir) / f"seed{seed}" / branch / "checkpoints" / f"checkpoint_{step}.pt"


def weights_sha256(state_dict: dict) -> str:
    """Hash of the Q-network weights only, independent of optimizer state and file layout."""
    h = hashlib.sha256()
    for name in sorted(state_dict):
        h.update(name.encode())
        h.update(state_dict[name].detach().cpu().numpy().astype(np.float32).tobytes())
    return h.hexdigest()


def load_trajectory_net(path: str | Path) -> tuple[QNetwork, dict]:
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    net = QNetwork(obs_dim=cfg.get("obs_dim", 7), n_actions=cfg.get("n_actions", 4), hidden_dim=cfg.get("hidden_dim", 128))
    net.load_state_dict(ckpt["q_net_state_dict"])
    net.eval()
    return net, ckpt


def checkpoint_record(seed: int, branch: str, step: int, checkpoint_dir: str | Path) -> dict:
    """Metadata of one trajectory checkpoint. Raises if the file's own metadata disagrees with its name."""
    path = checkpoint_file(seed, branch, step, checkpoint_dir)
    net, ckpt = load_trajectory_net(path)
    cfg = ckpt["config"]
    if int(ckpt["step"]) != step:
        raise ValueError(f"{path}: file says step {ckpt['step']}, name says {step}")
    if cfg["reward_mode"] != REWARD_MODE[branch]:
        raise ValueError(f"{path}: reward mode {cfg['reward_mode']} does not match branch {branch}")
    if int(cfg["seed"]) != seed:
        raise ValueError(f"{path}: seed {cfg['seed']} does not match {seed}")
    return {"path": str(path), "step": step, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "weights_sha256": weights_sha256(net.state_dict()), "reward_mode": cfg["reward_mode"]}


def verify_trajectory(records: list[dict], start_weights_sha256: str, final_weights_sha256: str) -> None:
    """Steps strictly increase, the first checkpoint is the fork start and the last is the committed final one."""
    steps = [r["step"] for r in records]
    if steps != sorted(set(steps)):
        raise ValueError("checkpoint steps are not strictly increasing")
    if records[0]["step"] != 0 or records[0]["weights_sha256"] != start_weights_sha256:
        raise ValueError("trajectory does not start from the seed's pretrained checkpoint")
    if records[-1]["weights_sha256"] != final_weights_sha256:
        raise ValueError("last trajectory checkpoint differs from the committed final checkpoint")
    hashes = [r["weights_sha256"] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("two checkpoints of one trajectory have identical weights")


# --- behavioral transition ---------------------------------------------------

def sustained_crossing(steps: list[int], values: list[float], threshold: float) -> int | None:
    """First step from which `values` stays at or above `threshold` until the end; None if it never does."""
    v = np.asarray(values, dtype=float)
    below = np.flatnonzero(v < threshold)
    if len(below) == 0:
        return int(steps[0])
    first = below[-1] + 1
    return int(steps[first]) if first < len(steps) else None


def detect_transition(steps: list[int], farming: list[float], thresholds: dict[str, float]) -> dict:
    """Onset, midpoint and completion steps of a farming-rate curve, plus the last step before onset."""
    out = {name: sustained_crossing(steps, farming, thr) for name, thr in thresholds.items()}
    onset = out.get("onset")
    out["last_before_onset"] = None if onset in (None, steps[0]) else int(steps[steps.index(onset) - 1])
    return out


# --- decisions ---------------------------------------------------------------

def action_effects(pos: np.ndarray, lay: np.ndarray, actions: np.ndarray, params: ProceduralEnvParams) -> dict:
    """Whether each action brings the agent closer to the goal and to the reward tile."""
    new = pos + np.stack([_DX[actions], _DY[actions]], axis=1)
    inside = (new[:, 0] >= 0) & (new[:, 0] < params.width) & (new[:, 1] >= 0) & (new[:, 1] < params.height)
    new = np.where(inside[:, None], new, pos)

    def dist(p, target):
        return np.abs(p - target).sum(axis=1)

    return {"toward_goal": dist(new, lay[:, 2:4]) < dist(pos, lay[:, 2:4]),
            "toward_tile": dist(new, lay[:, 4:6]) < dist(pos, lay[:, 4:6]),
            "on_tile": (pos == lay[:, 4:6]).all(axis=1)}


def classify_decisions(a_ref: np.ndarray, a_new: np.ndarray, pos: np.ndarray, lay: np.ndarray,
                       params: ProceduralEnvParams) -> np.ndarray:
    """Label each state by how the new checkpoint's greedy action relates to the reference checkpoint's.

    agree               same action
    equivalent_route    different actions that both bring the agent closer to the goal
    tile_over_goal      the reference action approaches the goal; the new one does not, and instead
                        approaches the reward tile (or steps off it, which is how farming re-enters)
    other_disagreement  any other difference
    """
    ref, new = action_effects(pos, lay, a_ref, params), action_effects(pos, lay, a_new, params)
    labels = np.full(len(a_ref), OTHER_DISAGREE, dtype=object)
    differ = a_ref != a_new
    labels[~differ] = AGREE
    labels[differ & ref["toward_goal"] & new["toward_goal"]] = EQUIVALENT
    labels[differ & ref["toward_goal"] & ~new["toward_goal"] & (new["toward_tile"] | new["on_tile"])] = TILE_OVER_GOAL
    return labels


def rollout_states(net: torch.nn.Module, layouts: list[Layout], params: ProceduralEnvParams) -> dict:
    """Every state the greedy policy acts in, one episode per layout."""
    lay = layouts_to_array(layouts)
    actions = rollout(greedy_policy(net), lay, params, record_actions=True)["actions"]
    pos = lay[:, 0:2].copy()
    chunks = {"pos": [], "steps": [], "layout_index": []}
    for t in range(params.max_steps):
        a = actions[:, t]
        idx = np.flatnonzero(a >= 0)
        if len(idx) == 0:
            break
        chunks["pos"].append(pos[idx].copy())
        chunks["steps"].append(np.full(len(idx), t))
        chunks["layout_index"].append(idx)
        new = pos[idx] + np.stack([_DX[a[idx]], _DY[a[idx]]], axis=1)
        inside = (new[:, 0] >= 0) & (new[:, 0] < params.width) & (new[:, 1] >= 0) & (new[:, 1] < params.height)
        pos[idx] = np.where(inside[:, None], new, pos[idx])
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    out["layouts"] = lay[out["layout_index"]]
    out["obs"] = observations(out["pos"], out["layouts"], out["steps"], params)
    return out


def subsample_states(states: dict, n: int, seed: int) -> dict:
    total = len(states["steps"])
    if total <= n:
        return states
    keep = np.sort(np.random.RandomState(seed).choice(total, n, replace=False))
    return {k: v[keep] for k, v in states.items()}


# --- controlled statistics ---------------------------------------------------

def safe_auroc(y: np.ndarray, score: np.ndarray, min_count: int = 1) -> float | None:
    y = np.asarray(y, dtype=bool)
    if y.sum() < min_count or (~y).sum() < min_count:
        return None
    return float(roc_auc_score(y, score))


def within_stratum_auroc(score: np.ndarray, y: np.ndarray, strata: np.ndarray, min_per_class: int = 5) -> dict:
    """AUROC computed inside groups of states that share the stratum key, then averaged by group size.

    If a score only separates the classes because of the stratifying variables
    (for example distance to the reward tile), this falls to 0.5.
    """
    y = np.asarray(y, dtype=bool)
    _, inv = np.unique(strata, axis=0, return_inverse=True)
    inv = inv.ravel()
    total, weight, used = 0.0, 0, 0
    for g in range(inv.max() + 1):
        m = inv == g
        if y[m].sum() >= min_per_class and (~y[m]).sum() >= min_per_class:
            total += roc_auc_score(y[m], score[m]) * m.sum()
            weight += int(m.sum())
            used += 1
    return {"auroc": total / weight if weight else None, "n_strata": used, "n_states": weight}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else None


def trajectory_pca(rows: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """PCA of a few very long vectors (one per checkpoint), through their Gram matrix."""
    x = rows - rows.mean(axis=0)
    gram = x @ x.T
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1][:k]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    return vecs * np.sqrt(vals), vals / np.trace(gram)
