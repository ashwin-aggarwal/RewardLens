"""Experiment 2, Phase 2: full-layer activation transplantation between same-seed networks.

`h_patched = h_recipient + alpha * (h_donor - h_recipient)` at post-ReLU layer 1
or layer 2. At alpha = 1 on layer 2 this is the recipient's output layer
reading the donor's layer-2 activation.

Donor modes:
  matched         donor activation for the recipient's current observation
  wrong_position  donor sees the right goal and tile but a random agent position and time
  shuffled        donor activation for a random state of a random training layout
  q_bias          no activation change; a constant vector is added to the recipient's Q-values
                  (the mean Q-value change the matched patch causes on fitting states)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

MODES = ("matched", "wrong_position", "shuffled", "q_bias")


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def donor_activation(donor: torch.nn.Module, obs: torch.Tensor, layer: int) -> torch.Tensor:
    h1 = torch.relu(donor.fc1(obs))
    return h1 if layer == 1 else torch.relu(donor.fc2(h1))


def patched_q(recipient: torch.nn.Module, obs: torch.Tensor, donor_act: torch.Tensor, layer: int,
              alpha: float = 1.0) -> torch.Tensor:
    """Recipient Q-values with its layer activation moved toward `donor_act`."""
    h1 = torch.relu(recipient.fc1(obs))
    if layer == 1:
        h1 = h1 + alpha * (donor_act - h1)
        return recipient.fc3(torch.relu(recipient.fc2(h1)))
    if layer == 2:
        h2 = torch.relu(recipient.fc2(h1))
        return recipient.fc3(h2 + alpha * (donor_act - h2))
    raise ValueError("layer must be 1 or 2")


def mean_q_shift(recipient: torch.nn.Module, donor: torch.nn.Module, fit_obs: np.ndarray, layer: int,
                 alpha: float = 1.0) -> np.ndarray:
    """Average change in the recipient's Q-values caused by the matched patch, over fitting states."""
    with torch.no_grad():
        obs = _t(fit_obs)
        return (patched_q(recipient, obs, donor_activation(donor, obs, layer), layer, alpha)
                - recipient(obs)).mean(dim=0).numpy()


def make_policy(recipient: torch.nn.Module, donor: torch.nn.Module | None = None, layer: int = 2,
                mode: str = "matched", alpha: float = 1.0, pool_obs: np.ndarray | None = None,
                q_bias: np.ndarray | None = None, seed: int = 0,
                grid: tuple[int, int] = (8, 8)) -> Callable[[np.ndarray], np.ndarray]:
    """A greedy policy for the batched simulator, with the chosen intervention applied."""
    if donor is None:
        return lambda obs: _greedy(recipient(_t(obs)))
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.RandomState(seed)
    pool = _t(pool_obs) if pool_obs is not None else None
    bias = _t(q_bias) if q_bias is not None else None

    def act(obs_np: np.ndarray) -> np.ndarray:
        obs = _t(obs_np)
        with torch.no_grad():
            if mode == "q_bias":
                return (recipient(obs) + bias).argmax(dim=1).numpy()
            if mode == "matched":
                donor_obs = obs
            elif mode == "wrong_position":
                donor_obs = obs.clone()
                n = len(obs)
                donor_obs[:, 0] = _t(rng.randint(0, grid[0], n) / (grid[0] - 1))
                donor_obs[:, 1] = _t(rng.randint(0, grid[1], n) / (grid[1] - 1))
                donor_obs[:, 6] = _t(rng.random_sample(n))
            else:
                donor_obs = pool[rng.randint(0, len(pool), len(obs))]
            return patched_q(recipient, obs, donor_activation(donor, donor_obs, layer), layer, alpha).argmax(dim=1).numpy()

    return act


def _greedy(q: torch.Tensor) -> np.ndarray:
    return q.argmax(dim=1).detach().numpy()
