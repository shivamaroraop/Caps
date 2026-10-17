"""One-call comparison of every policy on a fixed community and fixed days."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

from ..env.baselines import BASELINES, perfect_foresight_lp
from ..env.community_env import EnvConfig, P2PCommunityEnv
from ..marl.trainer import AcceptanceCriteria, evaluate, passes_gate

HEADLINE = ["net_cost_eur", "cost_saving_pct", "individual_rationality", "fairness_jain",
            "p2p_volume_kwh", "market_utilisation", "self_sufficiency",
            "peak_import_kw", "unserved_kwh", "ev_shortfall_kwh", "congestion_steps"]


def compare_policies(tensors, env_cfg: EnvConfig, agents: Optional[Dict] = None,
                     episodes: int = 8, seed: int = 12345,
                     include_lp: bool = True) -> Dict[str, dict]:
    """Evaluate rule-based baselines, optional learned agents, and the LP bound.

    Every policy sees the *same* deterministic days, so differences are due to
    control and not to which weather draw a method happened to get.
    """
    results: Dict[str, dict] = {}
    base = asdict(env_cfg)

    for name, cls in BASELINES.items():
        pol = cls()
        env = P2PCommunityEnv(tensors, EnvConfig(**{**base, "n_envs": episodes,
                                                    "noise_std": 0.0,
                                                    "residual": pol.residual}), seed=seed)
        results[name] = {k: v.tolist() for k, v in
                         evaluate(env, pol, episodes, seed=seed, is_agent=False).items()}

    for name, agent in (agents or {}).items():
        # A learned residual policy is only meaningful at the trust-region scale
        # it was trained under, so the scale travels with the checkpoint.
        k = getattr(agent, "residual_scale", None)
        over = {} if k is None else {"residual_scale": k, "residual_price_scale": k}
        env = P2PCommunityEnv(tensors, EnvConfig(**{**base, "n_envs": episodes,
                                                    "noise_std": 0.0,
                                                    "residual": True, **over}), seed=seed)
        # A state-augmented policy reads the multipliers out of its observation.
        # Evaluating it in an environment reporting lambda = 0 asks it to act
        # under a constraint pressure it never trained at, which is the same
        # class of error as evaluating a residual policy at the wrong trust
        # region. The multipliers therefore travel with the checkpoint.
        lam = getattr(agent, "multipliers", None)
        if lam:
            env.set_multipliers(lam)
        results[name] = {k: v.tolist() for k, v in
                         evaluate(env, agent, episodes, seed=seed, is_agent=True).items()}

    ref = results.get("tariff_aware_p2p")
    crit = AcceptanceCriteria()
    for name, m in results.items():
        arr = {k: np.asarray(v) for k, v in m.items()}
        gate = passes_gate(arr, {k: np.asarray(v) for k, v in ref.items()}, crit)
        m["_gate"] = gate

    if include_lp:
        lp = perfect_foresight_lp(tensors, env_cfg)
        if lp:
            results["perfect_foresight_lp"] = {"net_cost_eur": [lp["optimal_cost_eur"]],
                                               "unserved_kwh": [lp["unserved_kwh"]],
                                               "peak_import_kw": [lp["peak_import_kw"]],
                                               "_bound": True}
    return results


def headline_rows(results: Dict[str, dict]) -> list:
    rows = []
    base = float(np.mean(results["no_flexibility"]["net_cost_eur"]))
    opt = results.get("perfect_foresight_lp", {}).get("net_cost_eur", [None])[0]
    for name, m in results.items():
        cost = float(np.mean(m["net_cost_eur"]))
        row = {"policy": name, "cost_eur": f"{cost:.2f}",
               "vs_grid_only_%": f"{100 * (base - cost) / abs(base):.1f}"}
        if opt is not None and name != "perfect_foresight_lp":
            row["savings_recovered_%"] = f"{100 * (base - cost) / max(base - opt, 1e-9):.1f}"
        for k in ("individual_rationality", "fairness_jain", "p2p_volume_kwh",
                  "self_sufficiency", "peak_import_kw", "unserved_kwh", "ev_shortfall_kwh"):
            if k in m:
                row[k] = f"{float(np.mean(m[k])):.3f}"
        g = m.get("_gate")
        row["gate"] = "-" if g is None else ("PASS" if g["passed"] else "FAIL:" + ",".join(g["failures"]))
        rows.append(row)
    return rows
