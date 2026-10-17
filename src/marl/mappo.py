"""Multi-Agent PPO with per-agent credit assignment and constrained updates.

Every departure from the legacy implementation below is a response to a
measured failure, not a stylistic preference.  The diagnostic script
``scripts/00_diagnose_legacy.py`` reproduces the measurements.

===========================  ==================================================
Legacy behaviour             Consequence and correction
===========================  ==================================================
``rewards.mean()`` stored     All 25 agents received an identical advantage.
                              Corrected: per-agent rewards, per-agent values
                              from an agent-specific critic, per-agent GAE.
Unnormalised value targets    Initial value loss ~1.4e5; critic gradient norm
                              253 against actor 1.07.
                              Corrected: running value normalisation.
One shared optimiser and a    Joint clipping rescaled both by 0.002, giving a
joint ``clip_grad_norm(0.5)`` measured effective actor LR of 5.9e-7.
                              Corrected: separate optimisers, separate clips.
One episode per update        96 correlated samples reused for 8 epochs.
                              Corrected: E parallel episodes per update.
``atanh`` to recover ``z``     Log-probabilities diverge at the action boundary.
                              Corrected: the pre-squash variable is stored.
Fixed constraint penalties     Wrong shadow price at every stage of training.
                              Corrected: PID-Lagrangian duals.
No KL control or LR decay      Nothing stopped a destructive update.
                              Corrected: KL early stopping and linear decay.
===========================  ==================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .lagrangian import ConstraintSet, DEFAULT_CONSTRAINTS
from .networks import Actor, AttentionCritic, MLPCritic
from .normalizers import RunningMeanStd, ValueNormalizer

CRITICS = {"attention": AttentionCritic, "mlp": MLPCritic}


@dataclass
class MAPPOConfig:
    hidden_size: int = 128
    attention_heads: int = 4
    critic: str = "attention"
    layer_norm: bool = True
    log_std_init: float = -1.0

    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.001
    max_grad_norm_actor: float = 0.5
    max_grad_norm_critic: float = 1.0
    ppo_epochs: int = 6
    num_minibatches: int = 4
    target_kl: float = 0.02
    anneal_lr: bool = True
    normalize_obs: bool = True
    normalize_value: bool = True
    per_agent_advantage: bool = True
    shared_reward: bool = False          # ablation: collapse to a team scalar
    use_lagrangian: bool = True
    # Safety critic: predict discounted future constraint cost and drive the
    # dual update from that forecast instead of the batch-measured cost alone.
    # Safety critic: predicts discounted future constraint cost so the dual
    # update can be driven by a forecast rather than the batch measurement.
    # DEFAULT OFF.  A/B tested at 40 updates it was neutral-to-slightly-worse
    # (EUR 149.47 vs 148.37 with everything else fixed), so it is retained as a
    # switchable ablation rather than adopted.  See docs/AB_TEST.md.
    safety_critic: bool = False
    safety_critic_coef: float = 0.5
    # Fraction of updates trained at a *randomly sampled* multiplier so that one
    # policy generalises across constraint pressure.  DEFAULT OFF.  On a short
    # budget this spends training data on pressures the deployed policy never
    # sees, and measured clearly worse (EUR 157.80 vs 148.37, and it broke the
    # safety gate).  Worth revisiting at a much larger update budget.
    lam_sample_prob: float = 0.0
    # Ablation switch: reproduce the legacy coupling exactly -- one optimiser
    # over actor+critic parameters and one joint gradient-norm clip.  This is
    # the configuration whose measured effective actor learning rate is 5.9e-7.
    legacy_joint_optimizer: bool = False
    constraints: Dict = field(default_factory=lambda: dict(DEFAULT_CONSTRAINTS))
    device: str = "cpu"
    seed: int = 0


class RolloutBuffer:
    """Stores ``(T, E, N)`` transitions with all reward components intact."""

    def __init__(self, T: int, E: int, N: int, obs_dim: int, state_dim: int,
                 action_dim: int, component_keys: List[str]):
        self.T, self.E, self.N = T, E, N
        self.obs = np.zeros((T, E, N, obs_dim), np.float32)
        self.state = np.zeros((T, E, state_dim), np.float32)
        self.z = np.zeros((T, E, N, action_dim), np.float32)
        self.logp = np.zeros((T, E, N), np.float32)
        self.value = np.zeros((T, E, N), np.float32)
        self.cost_value = np.zeros((T, E, N), np.float32)
        self.components = {k: np.zeros((T, E, N), np.float32) for k in component_keys}
        self.ptr = 0

    def add(self, obs, state, z, logp, value, components, cost_value=None):
        i = self.ptr
        self.obs[i], self.state[i], self.z[i] = obs, state, z
        self.logp[i], self.value[i] = logp, value
        if cost_value is not None:
            self.cost_value[i] = cost_value
        for k, v in components.items():
            if k in self.components:
                self.components[k][i] = v
        self.ptr += 1

    def reset(self):
        self.ptr = 0


class MAPPO:
    """Constrained multi-agent PPO with a centralised agent-specific critic."""

    COMPONENTS = ["econ", "team", "shaping", "ev_shortfall", "terminal", "c_ir", "c_grid"]

    def __init__(self, obs_dim: int, state_dim: int, action_dim: int,
                 cfg: Optional[MAPPOConfig] = None):
        self.cfg = cfg or MAPPOConfig()
        c = self.cfg
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        self.device = torch.device(c.device)
        self.obs_dim, self.state_dim, self.action_dim = obs_dim, state_dim, action_dim

        self.actor = Actor(obs_dim, action_dim, c.hidden_size, c.log_std_init,
                           c.layer_norm).to(self.device)
        critic_cls = CRITICS[c.critic]
        self.n_value_heads = 2 if c.safety_critic else 1
        self.critic = critic_cls(obs_dim, state_dim, c.hidden_size,
                                 heads=c.attention_heads, layer_norm=c.layer_norm,
                                 n_heads_out=self.n_value_heads).to(self.device)
        # Separate optimisers and separate gradient clipping.  This is the single
        # change that restores a usable actor learning rate.
        if c.legacy_joint_optimizer:
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            self.opt_actor = torch.optim.Adam(params, lr=c.learning_rate_actor, eps=1e-5)
            self.opt_critic = self.opt_actor
        else:
            self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=c.learning_rate_actor, eps=1e-5)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=c.learning_rate_critic, eps=1e-5)

        self.obs_rms = RunningMeanStd((obs_dim,)) if c.normalize_obs else None
        self.value_norm = ValueNormalizer(device=self.device) if c.normalize_value else None
        # The safety critic regresses a *different* quantity on a different
        # scale, so it needs its own normaliser.  Sharing the reward normaliser
        # (or using none) reintroduces exactly the unbalanced-gradient failure
        # this project was built to fix: measured here as value loss 107 and
        # critic gradient 98 against an actor gradient of 0.86.
        self.cost_norm = (ValueNormalizer(device=self.device)
                          if (c.normalize_value and c.safety_critic) else None)
        self.constraints = ConstraintSet(c.constraints, enabled=c.use_lagrangian)
        self.progress = 0.0                    # 0 -> 1 over training, drives annealing

    # ------------------------------------------------------------------
    def _prep_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs.astype(np.float32)
        return self.obs_rms.normalize(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray, deterministic: bool = False):
        """``obs`` (E, N, obs_dim), ``state`` (E, state_dim)."""
        o = torch.as_tensor(self._prep_obs(obs), device=self.device)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        flat = o.reshape(-1, self.obs_dim)
        action, z, logp = self.actor.act(flat, deterministic)
        E, N = obs.shape[:2]
        out = self.critic(o, s)
        if self.n_value_heads > 1:
            value, cost_value = out[..., 0], out[..., 1]
        else:
            value, cost_value = out, torch.zeros_like(out)
        if self.value_norm is not None:
            value = self.value_norm.denormalize(value)
        return (action.reshape(E, N, -1).cpu().numpy(),
                z.reshape(E, N, -1).cpu().numpy(),
                logp.reshape(E, N).cpu().numpy(),
                value.reshape(E, N).cpu().numpy(),
                cost_value.reshape(E, N).cpu().numpy())

    @torch.no_grad()
    def cost_value_of(self, obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Safety critic prediction: discounted future constraint cost."""
        if self.n_value_heads == 1:
            return np.zeros(obs.shape[:2], dtype=np.float32)
        o = torch.as_tensor(self._prep_obs(obs), device=self.device)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.critic(o, s)[..., 1].cpu().numpy()

    @torch.no_grad()
    def value_of(self, obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(self._prep_obs(obs), device=self.device)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        out = self.critic(o, s)
        v = out[..., 0] if self.n_value_heads > 1 else out
        if self.value_norm is not None:
            v = self.value_norm.denormalize(v)
        return v.cpu().numpy()

    # ------------------------------------------------------------------
    def compose_reward(self, comp: Dict[str, np.ndarray], env_cfg,
                       lam: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Combine reward components with the Lagrange multipliers in force.

        ``lam`` is passed explicitly so that the reward is priced with exactly
        the multipliers the policy *observed* during the rollout; otherwise a
        state-augmented policy would be told one constraint pressure and graded
        against another.
        """
        lam = self.constraints.multipliers() if lam is None else lam
        r = (env_cfg.w_econ * comp["econ"]
             + env_cfg.w_team * comp["team"]
             + env_cfg.w_shape * comp["shaping"]
             - env_cfg.w_ev_shortfall * comp["ev_shortfall"]
             + comp["terminal"]
             - lam.get("ir", 0.0) * comp["c_ir"]
             - lam.get("grid", 0.0) * comp["c_grid"])
        if self.cfg.shared_reward:
            # Ablation reproducing the legacy signal: every agent receives the
            # team mean, destroying per-agent credit assignment.
            r = np.repeat(r.mean(axis=-1, keepdims=True), r.shape[-1], axis=-1)
        return r.astype(np.float32)

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                    last_value: np.ndarray) -> tuple:
        """Per-agent GAE over ``(T, E, N)`` arrays.

        The episode is a fixed-length day with no intermediate termination, so
        the only boundary is the final step, handled through ``last_value``.
        """
        T = rewards.shape[0]
        adv = np.zeros_like(rewards)
        gae = np.zeros_like(last_value)
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        for t in range(T - 1, -1, -1):
            nxt = last_value if t == T - 1 else values[t + 1]
            nonterminal = 0.0 if t == T - 1 else 1.0
            delta = rewards[t] + g * nxt * nonterminal - values[t]
            gae = delta + g * lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + values

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer, rewards: np.ndarray,
               last_value: np.ndarray,
               cost_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        c = self.cfg
        T, E, N = buf.T, buf.E, buf.N
        adv, ret = self.compute_gae(rewards, buf.value, last_value)

        if not c.per_agent_advantage:
            # Ablation: broadcast the team-mean advantage, as the legacy code did.
            adv = np.repeat(adv.mean(axis=-1, keepdims=True), N, axis=-1)

        if self.obs_rms is not None:
            self.obs_rms.update(buf.obs.reshape(-1, self.obs_dim))
        obs_n = self._prep_obs(buf.obs)

        dev = self.device
        B = T * E
        obs_t = torch.as_tensor(obs_n.reshape(B, N, self.obs_dim), device=dev)
        state_t = torch.as_tensor(buf.state.reshape(B, self.state_dim), device=dev)
        z_t = torch.as_tensor(buf.z.reshape(B, N, self.action_dim), device=dev)
        logp_t = torch.as_tensor(buf.logp.reshape(B, N), device=dev)
        old_v_t = torch.as_tensor(buf.value.reshape(B, N), device=dev)
        adv_t = torch.as_tensor(adv.reshape(B, N), device=dev)
        ret_t = torch.as_tensor(ret.reshape(B, N), device=dev)

        if self.value_norm is not None:
            self.value_norm.update(ret_t)
            ret_target = self.value_norm.normalize(ret_t)
            old_v_norm = self.value_norm.normalize(old_v_t)
        else:
            ret_target, old_v_norm = ret_t, old_v_t

        if cost_returns is None:
            cost_returns = np.zeros_like(rewards)
        cost_ret_t = torch.as_tensor(cost_returns.reshape(B, N), device=dev)
        if self.cost_norm is not None:
            self.cost_norm.update(cost_ret_t)
            cost_ret_t = self.cost_norm.normalize(cost_ret_t)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        if c.anneal_lr:
            frac = max(0.0, 1.0 - self.progress)
            for g_ in self.opt_actor.param_groups:
                g_["lr"] = c.learning_rate_actor * frac
            for g_ in self.opt_critic.param_groups:
                g_["lr"] = c.learning_rate_critic * frac
        ent_coef = c.entropy_coef + (c.entropy_coef_final - c.entropy_coef) * min(1.0, self.progress)

        mb = max(1, B // c.num_minibatches)
        idx = np.arange(B)
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "kl",
                                 "clip_frac", "grad_actor", "grad_critic")}
        stop = False
        for epoch in range(c.ppo_epochs):
            np.random.shuffle(idx)
            for s0 in range(0, B, mb):
                sel = torch.as_tensor(idx[s0:s0 + mb], device=dev)
                o, st = obs_t[sel], state_t[sel]
                if c.legacy_joint_optimizer:
                    stats_mb = self._legacy_step(o, st, sel, z_t, logp_t, adv_t,
                                                 ret_target, ent_coef, stats)
                    continue
                dist = self.actor.distribution(o.reshape(-1, self.obs_dim))
                new_logp = self.actor.log_prob(dist, z_t[sel].reshape(-1, self.action_dim))
                new_logp = new_logp.reshape(len(sel), N)
                ratio = (new_logp - logp_t[sel]).exp()
                a = adv_t[sel]
                pl = -torch.min(ratio * a,
                                ratio.clamp(1 - c.clip_range, 1 + c.clip_range) * a).mean()
                entropy = dist.entropy().sum(-1).mean()
                actor_loss = pl - ent_coef * entropy

                self.opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                ga = nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm_actor)
                self.opt_actor.step()

                out = self.critic(o, st)
                if self.n_value_heads > 1:
                    v, vc = out[..., 0], out[..., 1]
                else:
                    v, vc = out, None
                v_clipped = old_v_norm[sel] + (v - old_v_norm[sel]).clamp(
                    -c.value_clip_range, c.value_clip_range)
                vl = torch.max((v - ret_target[sel]).pow(2),
                               (v_clipped - ret_target[sel]).pow(2)).mean()
                if vc is not None:
                    vl = vl + c.safety_critic_coef * (vc - cost_ret_t[sel]).pow(2).mean()
                self.opt_critic.zero_grad(set_to_none=True)
                vl.backward()
                gc = nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm_critic)
                self.opt_critic.step()

                with torch.no_grad():
                    log_ratio = new_logp - logp_t[sel]
                    approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1).abs() > c.clip_range).float().mean().item()
                stats["policy_loss"].append(pl.item())
                stats["value_loss"].append(vl.item())
                stats["entropy"].append(entropy.item())
                stats["kl"].append(approx_kl)
                stats["clip_frac"].append(clip_frac)
                stats["grad_actor"].append(float(ga))
                stats["grad_critic"].append(float(gc))
            # Early stopping on KL: the guard the legacy trainer lacked entirely.
            if c.target_kl > 0 and np.mean(stats["kl"][-c.num_minibatches:]) > c.target_kl:
                stop = True
                break
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        out.update({"epochs_run": epoch + 1, "early_stop": float(stop),
                    "mean_return": float(ret.mean()), "mean_adv_abs": float(np.abs(adv).mean()),
                    "log_std": float(self.actor.log_std.mean().item()),
                    "entropy_coef": ent_coef})
        return out

    def _legacy_step(self, o, st, sel, z_t, logp_t, adv_t, ret_target, ent_coef, stats):
        """One update step under the legacy coupling, for the ablation."""
        c = self.cfg
        N = o.shape[1]
        dist = self.actor.distribution(o.reshape(-1, self.obs_dim))
        new_logp = self.actor.log_prob(dist, z_t[sel].reshape(-1, self.action_dim)).reshape(len(sel), N)
        ratio = (new_logp - logp_t[sel]).exp()
        a = adv_t[sel]
        pl = -torch.min(ratio * a, ratio.clamp(1 - c.clip_range, 1 + c.clip_range) * a).mean()
        entropy = dist.entropy().sum(-1).mean()
        v = self.critic(o, st)
        vl = (v - ret_target[sel]).pow(2).mean()
        loss = pl + 0.5 * vl - ent_coef * entropy
        self.opt_actor.zero_grad(set_to_none=True)
        loss.backward()
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        g = nn.utils.clip_grad_norm_(params, c.max_grad_norm_actor)
        self.opt_actor.step()
        with torch.no_grad():
            lr_ = new_logp - logp_t[sel]
            stats["kl"].append(((lr_.exp() - 1) - lr_).mean().item())
            stats["clip_frac"].append(((ratio - 1).abs() > c.clip_range).float().mean().item())
        stats["policy_loss"].append(pl.item()); stats["value_loss"].append(vl.item())
        stats["entropy"].append(entropy.item()); stats["grad_actor"].append(float(g))
        stats["grad_critic"].append(float(g))
        return None

    # ------------------------------------------------------------------
    def save(self, path: str | Path, extra: Optional[dict] = None):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
            "value_norm": self.value_norm.state_dict() if self.value_norm else None,
            "cost_norm": self.cost_norm.state_dict() if self.cost_norm else None,
            "constraints": self.constraints.state_dict(),
            "cfg": self.cfg.__dict__,
            "dims": (self.obs_dim, self.state_dim, self.action_dim),
            "extra": extra or {},
        }, p)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu"):
        d = torch.load(path, map_location=device, weights_only=False)
        cfg = MAPPOConfig(**{**d["cfg"], "device": device})
        obs_dim, state_dim, action_dim = d["dims"]
        m = cls(obs_dim, state_dim, action_dim, cfg)
        m.actor.load_state_dict(d["actor"])
        m.critic.load_state_dict(d["critic"])
        if d.get("obs_rms") and m.obs_rms:
            m.obs_rms.load_state_dict(d["obs_rms"])
        if d.get("value_norm") and m.value_norm:
            m.value_norm.load_state_dict(d["value_norm"])
        if d.get("cost_norm") and m.cost_norm:
            m.cost_norm.load_state_dict(d["cost_norm"])
        m.constraints.load_state_dict(d.get("constraints", {}))
        return m, d.get("extra", {})
