"""Phase 5 (experimental extension): low-rank weight editing.

Goal: fold the donor-dependent low-rank activation patch (mean + 8 centered
PCA directions, donor time clamped to 0.2) into the hacking DQN's own weights,
so that the edited network is an ordinary standalone QNetwork.

The donor and the PCA patch are used OFFLINE only, to produce supervision:
    h2*(s) = h_rec(s) + P_8(delta(s))          (target layer-2 activation)
    Q*(s)  = fc3_rec(h2*(s))                    (target Q-values)
    a*(s)  = argmax Q*(s)                       (target greedy action)

Edits are parameterized as additive deltas on frozen base weights:
    lowrank  : dW2 = A @ B.T (rank r), optional db2
    l2_full  : dW2, db2 (unconstrained layer-2 fine-tuning)
    head     : dW3, db3 (output-head-only fine-tuning)
    qbias    : db3 (constant Q-value bias)
    full     : deltas on every layer (ordinary supervised fine-tuning)
Objectives: "act" (layer-2 activation MSE), "q" (Q-value MSE), "ce" (cross-
entropy on target actions, i.e. policy distillation). Every objective adds
lambda * sum ||delta||_F^2. `merge` returns a plain QNetwork.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from rewardlens.agents.dqn import QNetwork

METHODS = ("lowrank", "l2_full", "head", "qbias", "full")
OBJECTIVES = ("act", "q", "ce")


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def patch_targets(recipient: QNetwork, donor_fn, obs: np.ndarray) -> dict:
    """Offline targets from a donor-dependent patch function (h_rec -> target h2)."""
    ot = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        h1 = F.relu(recipient.fc1(ot))
        h_rec = F.relu(recipient.fc2(h1))
        h_t = donor_fn(ot)
        q_t = recipient.fc3(h_t)
    return {"obs": ot, "h_rec": h_rec, "h2": h_t, "q": q_t, "a": q_t.argmax(1)}


def realizability_floor(targets: dict) -> dict:
    """Any layer-2 edit with layer 1 frozen outputs relu(.) >= 0, so negative
    target entries cannot be matched: mean(min(h2*, 0)^2) is an exact lower
    bound on the activation MSE of every layer-2-only edit."""
    h = targets["h2"]
    neg = torch.clamp(h, max=0.0)
    d = targets["h2"] - targets["h_rec"]
    return {"negative_fraction": float((h < 0).float().mean()),
            "mse_floor": float((neg ** 2).mean()),
            "relative_floor": float((neg ** 2).sum() / (d ** 2).sum()) if float((d ** 2).sum()) > 0 else 0.0}


# ---------------------------------------------------------------------------
# Editable network
# ---------------------------------------------------------------------------

class EditedNet(torch.nn.Module):
    """Frozen base QNetwork plus trainable additive deltas."""

    def __init__(self, base: QNetwork, method: str, rank: int = 0, edit_bias: bool = True, seed: int = 0):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.method, self.rank, self.edit_bias = method, rank, edit_bias
        g = torch.Generator().manual_seed(seed)
        out2, in2 = self.base.fc2.weight.shape
        P = torch.nn.Parameter
        z = lambda *s: P(torch.zeros(*s))
        self.deltas = torch.nn.ParameterDict()
        if method == "lowrank":
            if rank < 1:
                raise ValueError("rank must be >= 1")
            # LoRA-style init: A small random, B zero, so the edit starts at exactly 0.
            self.deltas["A"] = P(torch.randn(out2, rank, generator=g) * 0.01)
            self.deltas["B"] = z(in2, rank)
            if edit_bias:
                self.deltas["b2"] = z(out2)
        elif method == "l2_full":
            self.deltas["W2"] = z(out2, in2)
            if edit_bias:
                self.deltas["b2"] = z(out2)
        elif method == "head":
            self.deltas["W3"] = z(*self.base.fc3.weight.shape)
            self.deltas["b3"] = z(self.base.fc3.bias.shape[0])
        elif method == "qbias":
            self.deltas["b3"] = z(self.base.fc3.bias.shape[0])
        elif method == "full":
            for name, mod in (("1", self.base.fc1), ("2", self.base.fc2), ("3", self.base.fc3)):
                self.deltas["W" + name] = z(*mod.weight.shape)
                self.deltas["b" + name] = z(mod.bias.shape[0])

    def delta(self, key: str) -> torch.Tensor | None:
        if key == "W2" and self.method == "lowrank":
            return self.deltas["A"] @ self.deltas["B"].T
        return self.deltas[key] if key in self.deltas else None

    def _w(self, layer: int):
        mod = {1: self.base.fc1, 2: self.base.fc2, 3: self.base.fc3}[layer]
        dw, db = self.delta(f"W{layer}"), self.delta(f"b{layer}")
        return (mod.weight + dw if dw is not None else mod.weight,
                mod.bias + db if db is not None else mod.bias)

    def forward_all(self, obs: torch.Tensor):
        W1, b1 = self._w(1)
        W2, b2 = self._w(2)
        W3, b3 = self._w(3)
        h1 = F.relu(F.linear(obs, W1, b1))
        h2 = F.relu(F.linear(h1, W2, b2))
        return h2, F.linear(h2, W3, b3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward_all(obs)[1]

    def reg(self) -> torch.Tensor:
        total = torch.zeros(())
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            d = self.delta(k)
            if d is not None:
                total = total + (d ** 2).sum()
        return total

    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.deltas.values()))

    def merge(self) -> QNetwork:
        """Standalone QNetwork with the deltas folded into its weights."""
        net = copy.deepcopy(self.base)
        with torch.no_grad():
            for layer, mod in ((1, net.fc1), (2, net.fc2), (3, net.fc3)):
                dw, db = self.delta(f"W{layer}"), self.delta(f"b{layer}")
                if dw is not None:
                    mod.weight.add_(dw)
                if db is not None:
                    mod.bias.add_(db)
        for p in net.parameters():
            p.requires_grad_(True)
        net.eval()
        return net


def weight_change_stats(base: QNetwork, edited: QNetwork) -> dict:
    """Frobenius norms, numerical rank and stable rank of the per-layer weight change."""
    out = {}
    total_sq = 0.0
    n_changed = 0
    for name in ("fc1", "fc2", "fc3"):
        W0 = getattr(base, name).weight.detach()
        W1 = getattr(edited, name).weight.detach()
        b0 = getattr(base, name).bias.detach()
        b1 = getattr(edited, name).bias.detach()
        dW, db = (W1 - W0).double(), (b1 - b0).double()
        s = torch.linalg.svdvals(dW) if dW.abs().max() > 0 else torch.zeros(1, dtype=torch.float64)
        fro = float(torch.linalg.norm(dW))
        num_rank = int((s > 1e-6 * max(float(s.max()), 1e-12)).sum()) if fro > 1e-9 else 0
        out[name] = {"weight_delta_fro": fro, "bias_delta_norm": float(torch.linalg.norm(db)),
                     "relative_weight_change": fro / float(torch.linalg.norm(W0.double())),
                     "numerical_rank": num_rank,
                     "stable_rank": float((s ** 2).sum() / (s.max() ** 2)) if fro > 1e-9 else 0.0}
        total_sq += fro ** 2 + float(torch.linalg.norm(db)) ** 2
        n_changed += int((dW.abs() > 1e-8).sum()) + int((db.abs() > 1e-8).sum())
    out["total_delta_norm"] = total_sq ** 0.5
    out["n_parameters_changed"] = n_changed
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    steps: int = 3000
    lr: float = 3e-3
    weight_decay_lambda: float = 1e-4
    seed: int = 0


def train_edit(base: QNetwork, targets: dict, method: str, objective: str, rank: int = 0,
               edit_bias: bool = True, cfg: TrainConfig | None = None) -> tuple[EditedNet, list]:
    """Full-batch Adam on the given targets; deterministic given cfg.seed."""
    cfg = cfg or TrainConfig()
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective}")
    if objective == "act" and method in ("head", "qbias"):
        raise ValueError("activation matching requires an edit at or below layer 2")
    torch.manual_seed(cfg.seed)
    net = EditedNet(base, method, rank, edit_bias, cfg.seed)
    opt = torch.optim.Adam(net.deltas.parameters(), lr=cfg.lr)
    obs, h_t, q_t, a_t = targets["obs"], targets["h2"], targets["q"], targets["a"]
    log = []
    for step in range(cfg.steps):
        h2, q = net.forward_all(obs)
        if objective == "act":
            fit = F.mse_loss(h2, h_t)
        elif objective == "q":
            fit = F.mse_loss(q, q_t)
        else:
            fit = F.cross_entropy(q, a_t)
        loss = fit + cfg.weight_decay_lambda * net.reg()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == cfg.steps - 1:
            log.append({"step": step, "fit": float(fit), "loss": float(loss)})
    return net, log


def matching_errors(model: QNetwork, targets: dict) -> dict:
    """Activation and Q matching of a standalone model against the patch targets."""
    with torch.no_grad():
        h1 = F.relu(model.fc1(targets["obs"]))
        h2 = F.relu(model.fc2(h1))
        q = model.fc3(h2)
    d = targets["h2"] - targets["h_rec"]
    act_mse = float(((h2 - targets["h2"]) ** 2).mean())
    return {"act_mse": act_mse,
            "act_relative_error": float(((h2 - targets["h2"]) ** 2).sum() / (d ** 2).sum()) if float((d ** 2).sum()) > 0 else 0.0,
            "q_mse": float(((q - targets["q"]) ** 2).mean()),
            "action_agreement": float((q.argmax(1) == targets["a"]).float().mean())}


def random_lowrank_model(base: QNetwork, rank: int, fro_norm: float, seed: int) -> QNetwork:
    """Base network plus a random rank-`rank` layer-2 weight change of Frobenius norm `fro_norm`."""
    g = torch.Generator().manual_seed(seed)
    out2, in2 = base.fc2.weight.shape
    A = torch.randn(out2, rank, generator=g)
    B = torch.randn(in2, rank, generator=g)
    dW = A @ B.T
    dW = dW * (fro_norm / float(torch.linalg.norm(dW)))
    net = copy.deepcopy(base)
    with torch.no_grad():
        net.fc2.weight.add_(dW)
    net.eval()
    return net


def trajectory_observations(positions_trace: list, max_steps: int = 100, width: int = 8, height: int = 8,
                            goal=(6, 1), reward=(3, 5)) -> np.ndarray:
    """Observations visited along a recorded positions list (index = step)."""
    nx, ny = width - 1, height - 1
    rows = []
    for t, (x, y) in enumerate(positions_trace[:-1]):  # the last position is post-terminal
        rows.append([x / nx, y / ny, goal[0] / nx, goal[1] / ny, reward[0] / nx, reward[1] / ny,
                     max(0.0, (max_steps - t) / max_steps)])
    return np.asarray(rows, dtype=np.float32)
