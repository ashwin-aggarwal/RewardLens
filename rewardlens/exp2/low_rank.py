"""Experiment 2, Phase 6: coordinated low-rank weight edits.

A low-rank update `A @ B.T` is learned for each edited linear layer on top of
frozen hacking-network weights, then merged into ordinary weight matrices. The
merged network is a plain `QNetwork`: no adapter, donor network or hook remains.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from rewardlens.agents.dqn import QNetwork
from rewardlens.exp2.evaluation import greedy_policy, rollout, summarize
from rewardlens.exp2.procedural_env import ProceduralEnvParams
from rewardlens.exp2.training import EVAL_KEYS
from rewardlens.exp2.weight_level import LAYERS

SHAPES = {"fc1": (128, 7), "fc2": (128, 128), "fc3": (4, 128)}   # (out, in)


def layer_rank(layer: str, rank: int) -> int:
    """Rank actually used in a layer: the nominal rank, capped at the matrix's smaller side."""
    return int(min(rank, *SHAPES[layer]))


def parameter_counts(layers, rank: int, train_bias: bool = True) -> dict:
    """Trainable parameters of a low-rank edit, factors and bias reported separately."""
    per_layer = {}
    for layer in layers:
        out_dim, in_dim = SHAPES[layer]
        r = layer_rank(layer, rank)
        per_layer[layer] = {"rank": r, "factor_parameters": r * (out_dim + in_dim), "bias_parameters": out_dim if train_bias else 0,
                            "weights_in_matrix": out_dim * in_dim}
    factors = sum(v["factor_parameters"] for v in per_layer.values())
    bias = sum(v["bias_parameters"] for v in per_layer.values())
    return {"per_layer": per_layer, "factor_parameters": factors, "bias_parameters": bias, "trainable_parameters": factors + bias,
            "weights_in_edited_matrices": sum(v["weights_in_matrix"] for v in per_layer.values())}


def merge(base_state: dict, factors: dict) -> dict:
    """Ordinary state dict with every low-rank and bias update added to the base weights.

    `factors` maps a layer name to {"A": (out, r), "B": (in, r), "bias": (out,)}; A/B may be absent (rank 0).
    """
    merged = {k: v.detach().clone() for k, v in base_state.items()}
    for layer, f in factors.items():
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}")
        if f.get("A") is not None and f["A"].shape[1] > 0:
            merged[f"{layer}.weight"] = merged[f"{layer}.weight"] + f["A"].detach() @ f["B"].detach().T
        if f.get("bias") is not None:
            merged[f"{layer}.bias"] = merged[f"{layer}.bias"] + f["bias"].detach()
    return merged


def merged_network(base_state: dict, factors: dict) -> QNetwork:
    net = QNetwork()
    net.load_state_dict(merge(base_state, factors))
    net.eval()
    return net


class LowRankStudent(torch.nn.Module):
    """Frozen base network plus trainable low-rank and bias updates on the named layers."""

    def __init__(self, base_state: dict, layers, rank: int, seed: int, train_bias: bool = True) -> None:
        super().__init__()
        unknown = set(layers) - set(LAYERS)
        if unknown:
            raise ValueError(f"unknown layers {sorted(unknown)}")
        self.layers, self.rank = tuple(layers), int(rank)
        for key, value in base_state.items():
            self.register_buffer("base_" + key.replace(".", "_"), value.detach().clone())
        gen = torch.Generator().manual_seed(seed)
        self.A, self.B, self.bias = torch.nn.ParameterDict(), torch.nn.ParameterDict(), torch.nn.ParameterDict()
        for layer in self.layers:
            out_dim, in_dim = SHAPES[layer]
            r = layer_rank(layer, rank)
            if r > 0:
                self.A[layer] = torch.nn.Parameter(torch.zeros(out_dim, r))
                self.B[layer] = torch.nn.Parameter(torch.randn(in_dim, r, generator=gen) / in_dim ** 0.5)
            if train_bias:
                self.bias[layer] = torch.nn.Parameter(torch.zeros(out_dim))
        if not list(self.parameters()):
            raise ValueError("no trainable parameters")

    def base_state(self) -> dict:
        return {f"{l}.{p}": getattr(self, f"base_{l}_{p}") for l in LAYERS for p in ("weight", "bias")}

    def factors(self) -> dict:
        return {l: {"A": self.A[l].detach().clone() if l in self.A else None, "B": self.B[l].detach().clone() if l in self.B else None,
                    "bias": self.bias[l].detach().clone() if l in self.bias else None} for l in self.layers}

    def _linear(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        w, b = getattr(self, f"base_{layer}_weight"), getattr(self, f"base_{layer}_bias")
        if layer in self.A:
            w = w + self.A[layer] @ self.B[layer].T
        if layer in self.bias:
            b = b + self.bias[layer]
        return F.linear(x, w, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(torch.relu(self._linear(torch.relu(self._linear(x, "fc1")), "fc2")), "fc3")

    def merged_state(self) -> dict:
        return merge(self.base_state(), self.factors())


def distill_low_rank(base_state: dict, teacher: torch.nn.Module, layers, rank: int, train_obs: np.ndarray,
                     dev_layouts: np.ndarray, params: ProceduralEnvParams, hp: dict, init_seed: int, batch_seed: int,
                     select_mask: np.ndarray | None = None, train_bias: bool = True) -> dict:
    """Phase 5's distillation loop with the student restricted to low-rank and bias updates.

    Same loss, batches (for the same `batch_seed`), optimizer and evaluation as
    `weight_level.distill`. Every evaluation and snapshot uses the merged plain network.
    """
    torch.set_num_threads(1)
    torch.manual_seed(batch_seed)
    student = LowRankStudent(base_state, layers, rank, init_seed, train_bias)
    optimizer = torch.optim.Adam(student.parameters(), lr=hp["learning_rate"])
    obs = torch.from_numpy(np.asarray(train_obs, dtype=np.float32))
    with torch.no_grad():
        target = teacher(obs)
    rng = np.random.RandomState(batch_seed)
    select = np.ones(len(dev_layouts), bool) if select_mask is None else select_mask
    evals, snapshots, factor_snapshots = [], [], []

    def evaluate(update: int) -> None:
        state = student.merged_state()
        net = QNetwork()
        net.load_state_dict(state)
        net.eval()
        result = rollout(greedy_policy(net), dev_layouts, params, hp.get("gamma", 0.99))
        with torch.no_grad():
            loss = float(F.smooth_l1_loss(net(obs), target))
        row = {"step": update, "train_loss": loss}
        for name, mask in (("dev", None), ("dev_select", select), ("dev_report", ~select)):
            if mask is not None and not mask.any():
                continue
            s = summarize(result, params.exploit_visit_threshold, mask)
            row[name] = {k: s[k] for k in EVAL_KEYS}
        evals.append(row)
        snapshots.append(state)
        factor_snapshots.append(student.factors())

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
    return {"evals": evals, "snapshots": snapshots, "factors": factor_snapshots, "train_seconds": round(time.time() - start, 1),
            "n_trainable": int(sum(p.numel() for p in student.parameters())), "base_unchanged":
            all(torch.equal(student.base_state()[k], base_state[k]) for k in base_state)}


def effective_rank(matrix: np.ndarray, rel_tol: float = 1e-5) -> int:
    """Number of singular values above `rel_tol` times the largest; 0 for a zero matrix."""
    s = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return int((s > rel_tol * s[0]).sum()) if s[0] > 0 else 0


def update_summary(base_state: dict, new_state: dict) -> dict:
    """What an edit did to each layer: size, rank and how many entries changed."""
    out, sq_delta, sq_base = {}, 0.0, 0.0
    for layer in LAYERS:
        w0, w1 = (s[f"{layer}.weight"].detach().numpy().astype(np.float64) for s in (base_state, new_state))
        b0, b1 = (s[f"{layer}.bias"].detach().numpy().astype(np.float64) for s in (base_state, new_state))
        dw, db = w1 - w0, b1 - b0
        sv = np.linalg.svd(dw, compute_uv=False)
        energy = np.cumsum(sv ** 2) / max(float((sv ** 2).sum()), 1e-300)
        out[layer] = {"weight_relative_norm": float(np.linalg.norm(dw) / np.linalg.norm(w0)),
                      "bias_relative_norm": float(np.linalg.norm(db) / np.linalg.norm(b0)),
                      "relative_norm": float(np.sqrt((dw ** 2).sum() + (db ** 2).sum()) / np.sqrt((w0 ** 2).sum() + (b0 ** 2).sum())),
                      "effective_rank": effective_rank(dw), "rank_for_90pct_energy": int(np.searchsorted(energy, 0.9) + 1) if sv[0] > 0 else 0,
                      "weights_changed": int((dw != 0).sum()), "weights_total": int(dw.size),
                      "biases_changed": int((db != 0).sum()), "top_singular_values": [float(v) for v in sv[:8]]}
        sq_delta += float((dw ** 2).sum() + (db ** 2).sum())
        sq_base += float((w0 ** 2).sum() + (b0 ** 2).sum())
    out["network"] = {"relative_norm": float(np.sqrt(sq_delta / sq_base)),
                      "weights_changed": int(sum(out[l]["weights_changed"] for l in LAYERS)),
                      "biases_changed": int(sum(out[l]["biases_changed"] for l in LAYERS)),
                      "parameters_total": int(sum(out[l]["weights_total"] for l in LAYERS) + sum(SHAPES[l][0] for l in LAYERS))}
    return out


def random_low_rank_edit(base_state: dict, target_state: dict, layers, rank: int, seed: int) -> QNetwork:
    """Control: a random update of the same rank and the same per-layer norm as (target - base)."""
    rng = np.random.RandomState(seed)
    new = {k: v.detach().clone() for k, v in base_state.items()}
    for layer in layers:
        out_dim, in_dim = SHAPES[layer]
        r = layer_rank(layer, rank)
        dw_norm = float((target_state[f"{layer}.weight"] - base_state[f"{layer}.weight"]).norm())
        if r > 0 and dw_norm > 0:
            noise = torch.from_numpy((rng.standard_normal((out_dim, r)) @ rng.standard_normal((in_dim, r)).T).astype(np.float32))
            new[f"{layer}.weight"] = new[f"{layer}.weight"] + noise * (dw_norm / float(noise.norm()))
        db_norm = float((target_state[f"{layer}.bias"] - base_state[f"{layer}.bias"]).norm())
        if db_norm > 0:
            noise = torch.from_numpy(rng.standard_normal(out_dim).astype(np.float32))
            new[f"{layer}.bias"] = new[f"{layer}.bias"] + noise * (db_norm / float(noise.norm()))
    net = QNetwork()
    net.load_state_dict(new)
    net.eval()
    return net


def truncate_update(base_state: dict, target_state: dict, rank: int) -> QNetwork:
    """The update (target - base) with each weight matrix cut to its top `rank` singular directions; bias updates kept."""
    new = {k: v.detach().clone() for k, v in target_state.items()}
    for layer in LAYERS:
        dw = (target_state[f"{layer}.weight"] - base_state[f"{layer}.weight"]).numpy().astype(np.float64)
        u, s, vt = np.linalg.svd(dw, full_matrices=False)
        r = layer_rank(layer, rank)
        low = (u[:, :r] * s[:r]) @ vt[:r]
        new[f"{layer}.weight"] = base_state[f"{layer}.weight"] + torch.from_numpy(low.astype(np.float32))
    net = QNetwork()
    net.load_state_dict(new)
    net.eval()
    return net
