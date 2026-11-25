"""Experiment 2, Phase 5: weight-level interventions and weight-drift analysis.

Layer swaps between same-seed checkpoints, norm-matched random edits, fine-tuning
with frozen layers, and per-layer weight-change measures along the training run.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rewardlens.agents.dqn import QNetwork, ReplayBuffer, linear_epsilon, load_checkpoint
from rewardlens.analysis.behavior_classification import DIRECT_GOAL
from rewardlens.exp2.checkpoints import checkpoint_path
from rewardlens.exp2.emergence import action_effects, checkpoint_file, weights_sha256
from rewardlens.exp2.evaluation import categories, greedy_policy, rollout, summarize
from rewardlens.exp2.layouts import Layout, layouts_to_array
from rewardlens.exp2.procedural_env import ProceduralEnvParams, ProceduralRewardGridEnv
from rewardlens.exp2.scripted import goal_actions, play_actions
from rewardlens.exp2.training import EVAL_KEYS, seed_everything

LAYERS = ("fc1", "fc2", "fc3")


# --- checkpoints -------------------------------------------------------------

def resolve_step(name: str | int, transition: dict, final_step: int) -> int:
    """Turn a named point of the run ("start", "final", "completion", ...) into a training step."""
    if isinstance(name, int):
        return name
    if name == "start":
        return 0
    if name == "final":
        return final_step
    return int(transition[name])


def load_net(seed: int, branch: str, step: int, checkpoint_dir: str | Path = "results/exp2/phase1",
             final_step: int = 400_000) -> QNetwork:
    """One network of one seed, checked against the seed and stage recorded in its file.

    `branch` is "pretrain", "good" or "hacking". Step 0 of either fork branch is the
    committed pretrained checkpoint and the last step is the committed final checkpoint
    (identical weights to the trajectory files, see Phase 3's checkpoint registry).
    """
    if branch == "pretrain" or step == 0:
        path, stage = checkpoint_path(seed, "pretrain", checkpoint_dir), "pretrain"
    elif step == final_step:
        path, stage = checkpoint_path(seed, branch, checkpoint_dir), branch
    else:
        path, stage = checkpoint_file(seed, branch, step, checkpoint_dir), branch
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    if int(cfg["seed"]) != int(seed) or cfg["stage"] != stage:
        raise ValueError(f"{path} holds seed {cfg['seed']} stage {cfg['stage']}, expected seed {seed} stage {stage}")
    if stage != "pretrain" and int(ckpt["step"]) != int(step):
        raise ValueError(f"{path} holds step {ckpt['step']}, expected {step}")
    net = QNetwork(cfg.get("obs_dim", 7), cfg.get("n_actions", 4), cfg.get("hidden_dim", 128))
    net.load_state_dict(ckpt["q_net_state_dict"])
    net.eval()
    return net


# --- weights -----------------------------------------------------------------

def layer_vector(net_or_state, layer: str) -> np.ndarray:
    """Weight and bias of one layer as a single flat vector."""
    sd = net_or_state.state_dict() if hasattr(net_or_state, "state_dict") else net_or_state
    return np.concatenate([sd[f"{layer}.weight"].detach().numpy().ravel(), sd[f"{layer}.bias"].detach().numpy().ravel()]).astype(np.float64)


def count_parameters(layers) -> int:
    probe = QNetwork()
    return int(sum(p.numel() for name, p in probe.named_parameters() if name.split(".")[0] in layers))


def weight_difference(reference, other, layers=LAYERS) -> dict:
    """Per layer: size of (other - reference), relative to the reference norm, and their cosine similarity."""
    out = {}
    for layer in layers:
        a, b = layer_vector(reference, layer), layer_vector(other, layer)
        na, nb, nd = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(b - a)
        out[layer] = {"l2": float(nd), "relative": float(nd / na), "cosine": float(a @ b / (na * nb)),
                      "reference_norm": float(na), "other_norm": float(nb), "n_parameters": int(len(a)),
                      "n_changed": int((a != b).sum())}
    return out


def swap_layers(recipient: torch.nn.Module, donor: torch.nn.Module, layers) -> QNetwork:
    """A copy of the recipient with the donor's weight and bias in the named layers."""
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise ValueError(f"unknown layers {sorted(unknown)}")
    new = copy.deepcopy(recipient)
    sd, donor_sd = new.state_dict(), donor.state_dict()
    for key in sd:
        if key.split(".")[0] in layers:
            sd[key] = donor_sd[key].clone()
    new.load_state_dict(sd)
    new.eval()
    return new


def random_edit(recipient: torch.nn.Module, target: torch.nn.Module, layers, seed: int) -> QNetwork:
    """Control edit: change the same tensors by the same norm as (target - recipient), in a random direction."""
    rng = np.random.RandomState(seed)
    new = copy.deepcopy(recipient)
    sd, target_sd = new.state_dict(), target.state_dict()
    for key in sorted(sd):
        if key.split(".")[0] not in layers:
            continue
        norm = float((target_sd[key] - sd[key]).norm())
        noise = torch.from_numpy(rng.standard_normal(tuple(sd[key].shape)).astype(np.float32))
        sd[key] = sd[key] + noise * (norm / float(noise.norm())) if norm > 0 else sd[key]
    new.load_state_dict(sd)
    new.eval()
    return new


# --- behavior and decisions --------------------------------------------------

def behavior(policy, layouts, params: ProceduralEnvParams, gamma: float = 0.99, mask: np.ndarray | None = None) -> dict:
    s = summarize(rollout(policy, layouts, params, gamma), params.exploit_visit_threshold, mask)
    return {k: s[k] for k in EVAL_KEYS}


def scripted_shortest_path(layouts: list[Layout], params: ProceduralEnvParams, gamma: float = 0.99) -> dict:
    """The scripted shortest path that avoids the reward tile, scored like a learned policy."""
    eps = [play_actions(l, goal_actions(l, params), params, gamma) for l in layouts]
    lay = layouts_to_array(layouts)
    result = {k: np.array([e[k] for e in eps]) for k in eps[0]}
    result["manhattan"] = np.abs(lay[:, 0] - lay[:, 2]) + np.abs(lay[:, 1] - lay[:, 3])
    s = summarize(result, params.exploit_visit_threshold)
    return {k: s[k] for k in EVAL_KEYS}


def meets_repair(b: dict, criterion: dict) -> bool:
    """Efficient goal navigation, not just the end of farming."""
    eff = b["path_efficiency_direct"]
    return bool(b["direct_goal"] >= criterion["min_direct_goal"] and eff is not None
                and eff >= criterion["min_path_efficiency"] and b["farming"] <= criterion["max_farming"])


def decision_diagnostics(net: torch.nn.Module, reference: torch.nn.Module, good: torch.nn.Module, states: dict,
                         params: ProceduralEnvParams) -> dict:
    """Greedy actions and Q-values of `net` on fixed states, against the unedited network and the good one."""
    obs = torch.from_numpy(np.asarray(states["obs"], dtype=np.float32))
    with torch.no_grad():
        q, q_ref, q_good = net(obs).numpy(), reference(obs).numpy(), good(obs).numpy()
    a = q.argmax(1)
    eff = action_effects(states["pos"], states["layouts"], a, params)
    top2 = np.sort(q, axis=1)
    return {"agree_with_good": float((a == q_good.argmax(1)).mean()),
            "action_changed": float((a != q_ref.argmax(1)).mean()),
            "toward_goal": float(eff["toward_goal"].mean()),
            "tile_not_goal": float((~eff["toward_goal"] & (eff["toward_tile"] | eff["on_tile"])).mean()),
            "mean_max_q": float(q.max(1).mean()), "mean_q_margin": float((top2[:, -1] - top2[:, -2]).mean()),
            "mean_abs_q_change": float(np.abs(q - q_ref).mean())}


def conflict_masks(states: dict, params: ProceduralEnvParams) -> tuple[np.ndarray, np.ndarray]:
    """Per state and action: (approaches the tile but not the goal, approaches the goal but not the tile)."""
    n = len(states["pos"])
    tile, goal = np.zeros((n, 4), bool), np.zeros((n, 4), bool)
    for a in range(4):
        eff = action_effects(states["pos"], states["layouts"], np.full(n, a), params)
        tile[:, a] = eff["toward_tile"] & ~eff["toward_goal"]
        goal[:, a] = eff["toward_goal"] & ~eff["toward_tile"]
    return tile, goal


def q_preference(net: torch.nn.Module, states: dict, masks: tuple[np.ndarray, np.ndarray]) -> dict:
    """Best tile-ward Q minus best goal-ward Q, at states that offer both kinds of action."""
    tile, goal = masks
    conflict = tile.any(1) & goal.any(1)
    with torch.no_grad():
        q = net(torch.from_numpy(np.asarray(states["obs"], dtype=np.float32))).numpy()[conflict]
    q_tile = np.where(tile[conflict], q, -np.inf).max(1)
    q_goal = np.where(goal[conflict], q, -np.inf).max(1)
    return {"n_conflict_states": int(conflict.sum()), "mean_tile_minus_goal": float((q_tile - q_goal).mean()),
            "fraction_prefer_tile": float((q_tile > q_goal).mean())}


# --- constrained fine-tuning -------------------------------------------------

def set_trainable(net: torch.nn.Module, layers) -> list[torch.nn.Parameter]:
    """Freeze everything outside `layers`; return the parameters left trainable."""
    trainable = []
    for name, p in net.named_parameters():
        p.requires_grad = name.split(".")[0] in layers
        if p.requires_grad:
            trainable.append(p)
    if not trainable:
        raise ValueError("no trainable parameters")
    return trainable


def finetune(init_state: dict, trainable_layers, train_layouts: list[Layout], dev_layouts: np.ndarray,
             params: ProceduralEnvParams, hp: dict, seed: int, reward_mode: str = "corrected",
             select_mask: np.ndarray | None = None) -> dict:
    """DQN fine-tuning from `init_state` with only `trainable_layers` updated.

    The loop is the Phase 1 training loop (same replay, targets, loss and exploration
    stream) with the optimizer restricted to the trainable parameters. Returns the
    evaluation curve on `dev_layouts` and a weight snapshot at every evaluation.
    """
    torch.set_num_threads(1)
    seed_everything(seed)
    q_net, target_net = QNetwork(), QNetwork()
    q_net.load_state_dict(init_state)
    target_net.load_state_dict(init_state)
    target_net.eval()
    optimizer = torch.optim.Adam(set_trainable(q_net, trainable_layers), lr=hp["learning_rate"])
    replay = ReplayBuffer(hp["buffer_capacity"], 7, seed=seed)
    action_rng = np.random.RandomState(seed + 1)
    env = ProceduralRewardGridEnv(train_layouts, reward_mode, params)
    obs, _ = env.reset(seed=seed)
    gamma, total = hp["gamma"], hp["total_timesteps"]
    select = np.ones(len(dev_layouts), bool) if select_mask is None else select_mask
    evals, snapshots = [], []

    def evaluate(step: int) -> None:
        q_net.eval()
        result = rollout(greedy_policy(q_net), dev_layouts, params, gamma)
        q_net.train()
        thr = params.exploit_visit_threshold
        row = {"step": step}
        for name, mask in (("dev", None), ("dev_select", select), ("dev_report", ~select)):
            if mask is not None and not mask.any():
                continue
            s = summarize(result, thr, mask)
            row[name] = {k: s[k] for k in EVAL_KEYS}
        evals.append(row)
        snapshots.append({k: v.detach().clone() for k, v in q_net.state_dict().items()})

    evaluate(0)
    start = time.time()
    for step in range(1, total + 1):
        epsilon = linear_epsilon(step, total, hp["epsilon_start"], hp["epsilon_end"], hp["epsilon_fraction"])
        if action_rng.random_sample() < epsilon:
            action = int(action_rng.randint(4))
        else:
            with torch.no_grad():
                action = int(q_net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()
        if step >= hp["learning_starts"] and step % hp["train_frequency"] == 0:
            batch = replay.sample(hp["batch_size"])
            with torch.no_grad():
                targets = torch.from_numpy(batch["rewards"]) + gamma * (1.0 - torch.from_numpy(batch["dones"])) \
                    * target_net(torch.from_numpy(batch["next_obs"])).max(dim=1).values
            current = q_net(torch.from_numpy(batch["obs"])).gather(1, torch.from_numpy(batch["actions"]).unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(current, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % hp["target_update_frequency"] == 0:
            target_net.load_state_dict(q_net.state_dict())
        if step % hp["eval_interval"] == 0:
            evaluate(step)
    return {"evals": evals, "snapshots": snapshots, "train_seconds": round(time.time() - start, 1),
            "n_trainable": int(sum(p.numel() for p in q_net.parameters() if p.requires_grad))}


def select_checkpoint(evals: list[dict], split: str = "dev_select", metric: str = "mean_return_corrected") -> int:
    """Index of the evaluation with the highest metric on the selection layouts; earliest on ties."""
    values = [e[split][metric] for e in evals]
    return int(np.argmax(values))


def first_sustained(flags: list[bool]) -> int | None:
    """Index from which every flag is true, or None."""
    idx = None
    for i, f in enumerate(flags):
        if f and idx is None:
            idx = i
        elif not f:
            idx = None
    return idx


def frozen_unchanged(before: dict, after: dict, trainable_layers) -> bool:
    return all(torch.equal(before[k], after[k]) for k in before if k.split(".")[0] not in trainable_layers)


# --- supervised distillation (labelled separately from RL fine-tuning) -------

def distill(init_state: dict, teacher: torch.nn.Module, trainable_layers, train_obs: np.ndarray, dev_layouts: np.ndarray,
            params: ProceduralEnvParams, hp: dict, seed: int, select_mask: np.ndarray | None = None) -> dict:
    """Regress the student's Q-values onto the teacher's on fixed training-layout states.

    Only `trainable_layers` are updated. Minibatches are drawn with a generator seeded by
    `seed`, so every condition of one seed sees the same batches.
    """
    torch.set_num_threads(1)
    seed_everything(seed)
    student = QNetwork()
    student.load_state_dict(init_state)
    optimizer = torch.optim.Adam(set_trainable(student, trainable_layers), lr=hp["learning_rate"])
    obs = torch.from_numpy(np.asarray(train_obs, dtype=np.float32))
    with torch.no_grad():
        target = teacher(obs)
    rng = np.random.RandomState(seed)
    select = np.ones(len(dev_layouts), bool) if select_mask is None else select_mask
    evals, snapshots = [], []

    def evaluate(update: int) -> None:
        student.eval()
        result = rollout(greedy_policy(student), dev_layouts, params, hp.get("gamma", 0.99))
        with torch.no_grad():
            loss = float(F.smooth_l1_loss(student(obs), target))
        student.train()
        row = {"step": update, "train_loss": loss}
        for name, mask in (("dev", None), ("dev_select", select), ("dev_report", ~select)):
            if mask is not None and not mask.any():
                continue
            s = summarize(result, params.exploit_visit_threshold, mask)
            row[name] = {k: s[k] for k in EVAL_KEYS}
        evals.append(row)
        snapshots.append({k: v.detach().clone() for k, v in student.state_dict().items()})

    evaluate(0)
    start = time.time()
    for update in range(1, hp["n_updates"] + 1):
        idx = torch.from_numpy(rng.randint(0, len(obs), hp["batch_size"]))
        loss = F.smooth_l1_loss(student(obs[idx]), target[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % hp["eval_interval_updates"] == 0:
            evaluate(update)
    return {"evals": evals, "snapshots": snapshots, "train_seconds": round(time.time() - start, 1),
            "n_trainable": int(sum(p.numel() for p in student.parameters() if p.requires_grad))}


def closed_form_head(student: torch.nn.Module, teacher: torch.nn.Module, train_obs: np.ndarray) -> QNetwork:
    """Least-squares output layer on the student's frozen layer-2 activations, fitted to the teacher's Q-values."""
    obs = torch.from_numpy(np.asarray(train_obs, dtype=np.float32))
    with torch.no_grad():
        h = student.relu2(student.fc2(student.relu1(student.fc1(obs)))).numpy().astype(np.float64)
        y = teacher(obs).numpy().astype(np.float64)
    x = np.concatenate([h, np.ones((len(h), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    new = copy.deepcopy(student)
    sd = new.state_dict()
    sd["fc3.weight"] = torch.from_numpy(coef[:-1].T.astype(np.float32))
    sd["fc3.bias"] = torch.from_numpy(coef[-1].astype(np.float32))
    new.load_state_dict(sd)
    new.eval()
    return new


__all__ = ["LAYERS", "DIRECT_GOAL", "categories", "weights_sha256", "load_net", "resolve_step", "layer_vector",
           "count_parameters", "weight_difference", "swap_layers", "random_edit", "behavior", "scripted_shortest_path",
           "meets_repair", "decision_diagnostics", "conflict_masks", "q_preference", "set_trainable", "finetune",
           "select_checkpoint", "first_sustained", "frozen_unchanged", "distill", "closed_form_head"]
