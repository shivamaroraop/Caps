"""Policy and value networks.

Actor
-----
A shared (parameter-tied) Gaussian policy over the pre-squash variable ``z``,
which is then mapped to the action box by ``tanh`` on the four bipolar
dimensions and ``sigmoid`` on the price quote.  The rollout buffer stores ``z``
itself, not the squashed action.

That choice fixes a real numerical fault.  Recovering ``z`` at update time by
inverting the squash -- ``atanh(a)`` -- is only stable while ``a`` stays away
from the boundary.  Once the policy pushes ``|a|`` toward 1 the inverse
diverges: at the clamp of 0.999999 used in the legacy code, ``atanh`` returns
7.25 against a distribution whose standard deviation is ~0.17, so the recovered
log-probability is off by tens of nats and the PPO importance ratio explodes.
Storing ``z`` makes the ratio exact by construction.

Critic
------
A permutation-equivariant attention critic that returns **one value per agent**
rather than a single community value.  Each agent's query is formed from its own
observation together with the global state; keys and values are the encoded
observations of every household.

Per-agent values are what make per-agent advantages possible, and per-agent
advantages are what make credit assignment possible.  With a single shared
scalar value -- as in the legacy implementation, which additionally averaged
the reward vector before storing it -- every household in a 25-agent community
received an identical advantage at every timestep, so the gradient carried no
information about which household's action was responsible for the outcome.

This is attention over an unordered set, not a graph neural network: the
dataset supplies no feeder topology, and claiming a graph without an adjacency
structure would be unsupportable.  Yu et al. (2022) find that giving the critic
both local and global information is the most effective configuration, which is
exactly the input used here.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

N_TANH = 4          # battery, EV1, EV2, P2P quantity
N_SIGMOID = 1       # P2P price quote


def _init(layer: nn.Linear, gain: float = math.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    """Shared Gaussian policy over pre-squash actions."""

    def __init__(self, obs_dim: int, action_dim: int = 5, hidden: int = 128,
                 log_std_init: float = -1.0, layer_norm: bool = True):
        super().__init__()
        act = nn.Tanh
        layers = [_init(nn.Linear(obs_dim, hidden)), act()]
        if layer_norm:
            layers.insert(1, nn.LayerNorm(hidden))
        layers += [_init(nn.Linear(hidden, hidden)), act()]
        if layer_norm:
            layers.insert(-1, nn.LayerNorm(hidden))
        self.trunk = nn.Sequential(*layers)
        self.mu = _init(nn.Linear(hidden, action_dim), gain=0.01)
        # A near-zero output layer means the initial policy is the *reference
        # controller* exactly, because the residual correction it emits is zero.
        # Training therefore starts from a known-good operating point instead of
        # from random battery and EV dispatch.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.mu(h), self.log_std.clamp(-5.0, 1.0).expand_as(self.mu(h))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mu, log_std = self.forward(obs)
        return Normal(mu, log_std.exp())

    @staticmethod
    def squash(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.tanh(z[..., :N_TANH]),
                          torch.sigmoid(z[..., N_TANH:])], dim=-1)

    @staticmethod
    def log_prob(dist: Normal, z: torch.Tensor) -> torch.Tensor:
        """Exact log-density of the squashed action, evaluated at stored ``z``."""
        base = dist.log_prob(z).sum(-1)
        t = torch.tanh(z[..., :N_TANH])
        log_det_tanh = torch.log((1 - t.pow(2)).clamp_min(1e-6)).sum(-1)
        s = torch.sigmoid(z[..., N_TANH:])
        log_det_sig = torch.log((s * (1 - s)).clamp_min(1e-6)).sum(-1)
        return base - log_det_tanh - log_det_sig

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        z = dist.mean if deterministic else dist.rsample()
        return self.squash(z), z, self.log_prob(dist, z)


class AttentionCritic(nn.Module):
    """Agent-specific centralised value function with set attention."""

    def __init__(self, obs_dim: int, state_dim: int, hidden: int = 128,
                 heads: int = 4, layer_norm: bool = True, n_heads_out: int = 1):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.n_heads_out = n_heads_out
        norm = (lambda: nn.LayerNorm(hidden)) if layer_norm else nn.Identity
        self.obs_enc = nn.Sequential(_init(nn.Linear(obs_dim, hidden)), norm(), nn.Tanh(),
                                     _init(nn.Linear(hidden, hidden)), nn.Tanh())
        self.state_enc = nn.Sequential(_init(nn.Linear(state_dim, hidden)), norm(), nn.Tanh())
        self.query = _init(nn.Linear(2 * hidden, hidden))
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden) if layer_norm else nn.Identity()
        # Output head. With ``n_heads_out=2`` the second channel is a *safety
        # critic*: it predicts the discounted future constraint cost, so the
        # Lagrange multiplier can be driven by where the policy is heading
        # rather than only by the cost measured in the batch just collected
        # (cf. MA-DELC, Safety-Constrained MARL for Active Voltage Control).
        self.head = nn.Sequential(_init(nn.Linear(3 * hidden, hidden)), nn.Tanh(),
                                  _init(nn.Linear(hidden, n_heads_out), gain=1.0))

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``obs`` (B, N, obs_dim), ``state`` (B, state_dim).

        Returns ``(B, N)`` when ``n_heads_out == 1``, else ``(B, N, H)``.
        """
        tokens = self.obs_enc(obs)                            # (B, N, H)
        g = self.state_enc(state).unsqueeze(1).expand_as(tokens)
        q = self.query(torch.cat([tokens, g], dim=-1))        # (B, N, H)
        attended, _ = self.attn(q, tokens, tokens, need_weights=False)
        attended = self.ln(attended + q)                      # residual connection
        out = self.head(torch.cat([attended, tokens, g], dim=-1))
        return out.squeeze(-1) if self.n_heads_out == 1 else out

    @torch.no_grad()
    def attention_weights(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Expose the attention matrix for interpretability figures."""
        tokens = self.obs_enc(obs)
        g = self.state_enc(state).unsqueeze(1).expand_as(tokens)
        q = self.query(torch.cat([tokens, g], dim=-1))
        _, w = self.attn(q, tokens, tokens, need_weights=True, average_attn_weights=True)
        return w


class MLPCritic(nn.Module):
    """Ablation critic: identical inputs, no attention over the household set."""

    def __init__(self, obs_dim: int, state_dim: int, hidden: int = 128,
                 n_heads_out: int = 1, **_):
        super().__init__()
        self.n_heads_out = n_heads_out
        self.net = nn.Sequential(_init(nn.Linear(obs_dim + state_dim, hidden)), nn.Tanh(),
                                 _init(nn.Linear(hidden, hidden)), nn.Tanh(),
                                 _init(nn.Linear(hidden, n_heads_out), gain=1.0))

    def forward(self, obs, state):
        g = state.unsqueeze(1).expand(-1, obs.shape[1], -1)
        out = self.net(torch.cat([obs, g], dim=-1))
        return out.squeeze(-1) if self.n_heads_out == 1 else out
