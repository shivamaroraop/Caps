"""Training loop, evaluation, and the deployment safety gate.

The trust-region curriculum
---------------------------
The learned policy emits a bounded correction to the rule-based controller.
The bound ``kappa`` is annealed from a small starting value to its final value
over the first part of training.  Early on the policy is therefore constrained
to stay close to a controller that is known to be safe, which is what makes the
initial rollouts informative instead of being dominated by random battery and
EV dispatch.  As the critic becomes accurate the trust region opens and the
policy is free to depart from the heuristic.

This is a schedule on the *action space*, not on the reward, so it cannot bias
the objective -- only the region of policy space explored at each stage.

The safety gate
---------------
A learned policy is deployed only if it passes acceptance tests on held-out
evaluation days: individual rationality above threshold, no feeder violations,
no EV shortfall, and cost no worse than the rule-based controller.  Otherwise
the system falls back to that controller.  The gate is evaluated over multiple
seeds, and a policy that passes on average but fails on some seeds is reported
as failing -- which is the distinction the original study was missing.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..env.community_env import EnvConfig, P2PCommunityEnv
from .mappo import MAPPO, MAPPOConfig, RolloutBuffer


@dataclass
class TrainConfig:
    updates: int = 200
    n_envs: int = 8
    eval_every: int = 20
    eval_episodes: int = 8
    residual_scale_start: float = 0.15
    # The trust region opens to 1.0: at the end of training the policy has
    # full authority over the action box and the reference controller acts only
    # as an initialisation prior.  Capping it below 1.0 would leave actions the
    # policy provably needs -- such as fully suspending EV charging during a
    # feeder constraint -- permanently out of reach.
    residual_scale_final: float = 1.0
    residual_anneal_frac: float = 0.5
    log_every: int = 10
    seed: int = 0


@dataclass
class AcceptanceCriteria:
    """Deployment gate. Values are deliberately explicit and auditable."""
    min_individual_rationality: float = 0.90
    max_unserved_kwh: float = 1e-3     # 1 Wh: below float32 resolution on ~1000 kWh sums; machine-independent
    max_ev_shortfall_kwh: float = 1e-3
    max_cost_ratio_vs_reference: float = 1.0     # must not be worse than the heuristic
    min_fairness: float = 0.90
    require_all_seeds: bool = True


def collect_rollout(env: P2PCommunityEnv, agent: MAPPO, buf: RolloutBuffer,
                    deterministic: bool = False):
    obs, state = env.reset()
    buf.reset()
    comp_sum = {k: 0.0 for k in MAPPO.COMPONENTS}
    for t in range(env.n_steps):
        action, z, logp, value, cost_value = agent.act(obs, state, deterministic)
        nobs, nstate, _, done, info = env.step(action)
        buf.add(obs, state, z, logp, value, info["components"], cost_value)
        for k in comp_sum:
            comp_sum[k] += float(info["components"][k].mean())
        obs, state = nobs, nstate
    last_value = agent.value_of(obs, state)
    return last_value, info["episode"], comp_sum


def discounted_cost_returns(cost: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted future constraint cost, the safety critic's regression target."""
    out = np.zeros_like(cost)
    run = np.zeros_like(cost[0])
    for t in range(cost.shape[0] - 1, -1, -1):
        run = cost[t] + gamma * run
        out[t] = run
    return out


def evaluate(env: P2PCommunityEnv, policy, episodes: int = 8,
             deterministic: bool = True, seed: int = 12345,
             is_agent: bool = True) -> Dict[str, np.ndarray]:
    """Run ``episodes`` evaluation days and return stacked per-episode metrics."""
    results: Dict[str, List[float]] = {}
    n_batches = max(1, int(np.ceil(episodes / env.E)))
    for b in range(n_batches):
        obs, state = env.reset(seed=seed + b, deterministic=True)
        done = False
        while not done:
            if is_agent:
                action = policy.act(obs, state, deterministic)[0]
            else:
                action = policy.act(obs, state, env)
            obs, state, _, done, info = env.step(action)
        for k, v in info["episode"].items():
            results.setdefault(k, []).extend(np.atleast_1d(v).tolist())
    return {k: np.asarray(v[:episodes if episodes <= len(v) else len(v)]) for k, v in results.items()}


def passes_gate(metrics: Dict[str, np.ndarray], reference: Dict[str, np.ndarray],
                crit: AcceptanceCriteria) -> Dict[str, object]:
    """Evaluate the acceptance criteria and report *why* a policy fails."""
    agg = (np.min if crit.require_all_seeds else np.mean)
    checks = {
        "individual_rationality": (float(agg(metrics["individual_rationality"])),
                                   crit.min_individual_rationality, "ge"),
        "fairness_jain": (float(agg(metrics["fairness_jain"])), crit.min_fairness, "ge"),
        "unserved_kwh": (float(np.max(metrics["unserved_kwh"])), crit.max_unserved_kwh, "le"),
        "ev_shortfall_kwh": (float(np.max(metrics["ev_shortfall_kwh"])), crit.max_ev_shortfall_kwh, "le"),
        "cost_ratio": (float(np.mean(metrics["net_cost_eur"]) /
                             max(np.mean(reference["net_cost_eur"]), 1e-9)),
                       crit.max_cost_ratio_vs_reference, "le"),
    }
    failures = [k for k, (val, lim, op) in checks.items()
                if (val < lim if op == "ge" else val > lim)]
    return {"passed": len(failures) == 0, "failures": failures,
            "checks": {k: {"value": v[0], "limit": v[1], "op": v[2]} for k, v in checks.items()}}


class Trainer:
    def __init__(self, tensors, env_cfg: EnvConfig, algo_cfg: MAPPOConfig,
                 train_cfg: TrainConfig, out_dir: str | Path = "results/run"):
        self.tensors = tensors
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        env_cfg.n_envs = train_cfg.n_envs
        self.env = P2PCommunityEnv(tensors, env_cfg, seed=train_cfg.seed)
        # Evaluation runs on a *deterministic* copy of the day: no measurement
        # noise, so policy comparisons are not confounded by sampling luck.
        eval_cfg = EnvConfig(**{**asdict(env_cfg), "n_envs": train_cfg.eval_episodes,
                                "noise_std": 0.0})
        self.eval_env = P2PCommunityEnv(tensors, eval_cfg, seed=train_cfg.seed + 9999)
        self.eval_env_raw = P2PCommunityEnv(
            tensors, EnvConfig(**{**asdict(eval_cfg), "residual": False}),
            seed=train_cfg.seed + 9999)

        self.agent = MAPPO(self.env.observation_dim, self.env.state_dim,
                           self.env.action_dim, algo_cfg)
        self.buf = RolloutBuffer(self.env.n_steps, self.env.E, self.env.N,
                                 self.env.observation_dim, self.env.state_dim,
                                 self.env.action_dim, MAPPO.COMPONENTS)
        self.history: List[dict] = []
        self.reference_metrics: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    def _anneal_trust_region(self, frac: float) -> float:
        tc = self.train_cfg
        a = min(1.0, frac / max(tc.residual_anneal_frac, 1e-6))
        k = tc.residual_scale_start + a * (tc.residual_scale_final - tc.residual_scale_start)
        for e in (self.env, self.eval_env):
            e.cfg.residual_scale = k
            e.cfg.residual_price_scale = k
        # The evaluation environment must use the *same* trust-region scale.
        # Leaving it at the config default evaluates the policy with its
        # corrections shrunk by up to 6x -- i.e. it measures a crippled version
        # of the agent that was never trained, and reports it as the agent.
        return k

    def reference_baseline(self) -> Dict[str, np.ndarray]:
        from ..env.baselines import TariffAwareP2P
        if self.reference_metrics is None:
            self.reference_metrics = evaluate(self.eval_env, TariffAwareP2P(),
                                              self.train_cfg.eval_episodes,
                                              is_agent=False)
        return self.reference_metrics

    def train(self, verbose: bool = True) -> Dict:
        tc = self.train_cfg
        ref = self.reference_baseline()
        best = {"score": -np.inf, "update": -1}
        t0 = time.time()
        for update in range(1, tc.updates + 1):
            frac = (update - 1) / max(tc.updates - 1, 1)
            self.agent.progress = frac
            kappa = self._anneal_trust_region(frac)

            # State augmentation: the policy must see the constraint pressure it
            # is acting under.  On a fraction of updates we deliberately train
            # at a *random* multiplier so that one policy generalises across the
            # whole range rather than tracking whatever value the PID happens to
            # be at right now.
            lam_now = self.agent.constraints.multipliers()
            if (self.agent.cfg.lam_sample_prob > 0
                    and np.random.rand() < self.agent.cfg.lam_sample_prob):
                cap = self.env.cfg.lam_max_obs
                lam_train = {k: float(np.random.uniform(0, cap)) for k in lam_now}
            else:
                lam_train = lam_now
            self.env.set_multipliers(lam_train)
            self.eval_env.set_multipliers(lam_now)

            last_value, ep, comp = collect_rollout(self.env, self.agent, self.buf)
            rewards = self.agent.compose_reward(
                {k: self.buf.components[k] for k in MAPPO.COMPONENTS}, self.env.cfg,
                lam=lam_train)

            # Dual update.  With the safety critic enabled the multiplier is
            # driven by the *predicted discounted* constraint cost -- where the
            # policy is heading -- rather than only by what this batch happened
            # to incur, which is a one-step, high-variance signal.
            g = self.agent.cfg.gamma
            cost_ir = discounted_cost_returns(self.buf.components["c_ir"], g)
            cost_grid = discounted_cost_returns(self.buf.components["c_grid"], g)
            if self.agent.cfg.safety_critic:
                horizon = (1.0 - g ** self.env.n_steps) / (1.0 - g)
                raw = self.buf.cost_value.mean()
                if self.agent.cost_norm is not None:
                    import torch as _t
                    raw = float(self.agent.cost_norm.denormalize(
                        _t.tensor([float(raw)], device=self.agent.device)).item())
                sig_ir = float(raw) / horizon
                measured_ir = float(self.buf.components["c_ir"].mean())
                # Blend forecast with measurement: the critic is unreliable early.
                w = min(1.0, frac * 2.0)
                dual_ir = (1 - w) * measured_ir + w * max(sig_ir, 0.0)
            else:
                dual_ir = float(self.buf.components["c_ir"].mean())
            duals = self.agent.constraints.update({
                "ir": dual_ir,
                "grid": float(self.buf.components["c_grid"].mean()),
            })
            cost_ret = cost_ir + cost_grid
            stats = self.agent.update(self.buf, rewards, last_value, cost_ret)

            row = {"update": update, "kappa": kappa, "time_s": time.time() - t0,
                   "reward_mean": float(rewards.mean()),
                   **{f"comp_{k}": v / self.env.n_steps for k, v in comp.items()},
                   **{f"lam_{k}": v for k, v in duals.items()},
                   **stats,
                   **{f"train_{k}": float(np.mean(v)) for k, v in ep.items()}}

            if update % tc.eval_every == 0 or update == tc.updates:
                m = evaluate(self.eval_env, self.agent, tc.eval_episodes)
                gate = passes_gate(m, ref, AcceptanceCriteria())
                score = self._score(m, ref, gate)
                row.update({f"eval_{k}": float(np.mean(v)) for k, v in m.items()})
                row["eval_gate_passed"] = bool(gate["passed"])
                row["eval_score"] = score
                if score > best["score"]:
                    best = {"score": score, "update": update,
                            "metrics": {k: v.tolist() for k, v in m.items()}, "gate": gate}
                    self.agent.save(self.out_dir / "best.pt",
                                    extra={"update": update, "gate": gate,
                                           "residual_scale": kappa,
                                           "multipliers": dict(lam_now)})
                if verbose:
                    print(f"  [eval @ {update}] cost={np.mean(m['net_cost_eur']):.2f} "
                          f"IR={np.mean(m['individual_rationality']):.3f} "
                          f"Jain={np.mean(m['fairness_jain']):.3f} "
                          f"unserved={np.mean(m['unserved_kwh']):.2f} "
                          f"gate={'PASS' if gate['passed'] else 'FAIL ' + ','.join(gate['failures'])}")
            if verbose and (update % tc.log_every == 0 or update == 1):
                print(f"update {update:4d}/{tc.updates} k={kappa:.2f} "
                      f"R={row['reward_mean']:+.3f} vloss={stats['value_loss']:.4f} "
                      f"kl={stats['kl']:.4f} ent={stats['entropy']:.2f} "
                      f"gA={stats['grad_actor']:.2f} gC={stats['grad_critic']:.2f} "
                      f"lam_ir={duals.get('ir', 0):.2f} lam_grid={duals.get('grid', 0):.2f}")
            self.history.append(row)

        self.agent.save(self.out_dir / "final.pt", extra={"updates": tc.updates})
        with open(self.out_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=1)
        return {"best": best, "history": self.history, "wall_time_s": time.time() - t0}

    @staticmethod
    def _score(m, ref, gate) -> float:
        """Model-selection score: cost improvement, zeroed out if the gate fails.

        Selecting the checkpoint with the lowest cost regardless of constraint
        satisfaction is how an unsafe policy gets promoted.  A failed gate makes
        the score strictly worse than any passing checkpoint.
        """
        rel = (np.mean(ref["net_cost_eur"]) - np.mean(m["net_cost_eur"])) / \
              max(abs(np.mean(ref["net_cost_eur"])), 1e-9)
        if not gate["passed"]:
            return -1.0 + 0.001 * float(rel)
        return float(rel + 0.1 * np.mean(m["fairness_jain"]) +
                     0.1 * np.mean(m["individual_rationality"]))
