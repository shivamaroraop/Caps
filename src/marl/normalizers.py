"""Running normalisation for observations and value targets.

Value normalisation is not a cosmetic detail here.  In the legacy
implementation the undiscounted episode return was around -550 and the critic
was initialised near zero, so the first value loss was O(10^5).  Actor and
critic shared one optimiser and one global gradient-norm clip of 0.5, which
meant the critic's gradient of norm ~253 and the actor's of norm ~1.07 were
rescaled together by a factor of 0.002.  The measured effective actor learning
rate was 5.9e-7 -- roughly 500x too small.  The policy was therefore not
learning at all; the run-to-run spread the project observed was a random walk
around initialisation, not optimisation variance.

Normalising the regression target to unit scale removes the pathology at
source.  Yu et al. (2022) reach the same conclusion empirically and recommend
value normalisation unconditionally for MAPPO.
"""
from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """Welford-style running moments, used for observation whitening."""

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, *self.mean.shape)
        batch_mean, batch_var, n = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m_a = self.var * self.count
        m_b = batch_var * n
        self.var = (m_a + m_b + delta ** 2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class ValueNormalizer:
    """Normalise value regression targets to zero mean and unit variance.

    The critic predicts *normalised* returns.  ``denormalize`` is applied
    wherever a value enters the advantage computation, so bootstrapping and GAE
    operate in the environment's own reward units, while the regression loss
    operates on a target whose scale is stationary by construction.
    """

    def __init__(self, beta: float = 0.99995, epsilon: float = 1e-5, device="cpu"):
        self.beta = beta
        self.epsilon = epsilon
        self.device = device
        self.running_mean = torch.zeros(1, device=device)
        self.running_sq = torch.ones(1, device=device)
        self.debias = torch.zeros(1, device=device)

    @torch.no_grad()
    def update(self, targets: torch.Tensor) -> None:
        m = targets.mean()
        s = (targets ** 2).mean()
        self.running_mean.mul_(self.beta).add_((1 - self.beta) * m)
        self.running_sq.mul_(self.beta).add_((1 - self.beta) * s)
        self.debias.mul_(self.beta).add_(1 - self.beta)

    def _stats(self):
        d = self.debias.clamp_min(self.epsilon)
        mean = self.running_mean / d
        var = (self.running_sq / d) - mean ** 2
        return mean, var.clamp_min(self.epsilon).sqrt()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats()
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats()
        return x * std + mean

    def state_dict(self):
        return {"running_mean": self.running_mean, "running_sq": self.running_sq,
                "debias": self.debias}

    def load_state_dict(self, d):
        self.running_mean = d["running_mean"].to(self.device)
        self.running_sq = d["running_sq"].to(self.device)
        self.debias = d["debias"].to(self.device)
