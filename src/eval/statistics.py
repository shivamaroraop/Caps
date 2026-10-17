"""Aggregation and comparison of policies across seeds.

Reporting a single mean over a handful of seeds is not enough to support a
claim about which method is better; the deep-RL literature has converged on
interval estimates and robust location statistics for exactly this reason
(Agarwal et al., 2021, *Deep Reinforcement Learning at the Edge of the
Statistical Precipice*).  This module reports, for every metric:

* the mean and the **interquartile mean** (IQM), which discards the top and
  bottom quartile and is far less sensitive to a single lucky or catastrophic
  seed than the mean, and less wasteful than the median;
* a **bootstrap confidence interval** over seeds;
* the **worst seed**, because a policy that is excellent on average and unsafe
  on one seed in five is not deployable.

The last point is what the original study was missing: it observed that MAPPO
was "sometimes very good, sometimes unstable" without turning that observation
into a decision rule.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05, statistic=np.mean, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = statistic(x[idx], axis=1)
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def iqm(x) -> float:
    """Interquartile mean: mean of the middle 50% of the samples."""
    x = np.sort(np.asarray(x, dtype=float))
    k = x.size // 4
    core = x[k:x.size - k] if x.size >= 4 else x
    return float(core.mean())


LOWER_IS_BETTER = {
    "net_cost_eur", "grid_import_kwh", "peak_import_kw", "unserved_kwh",
    "curtailed_kwh", "congestion_steps", "ev_shortfall_kwh", "worst_burden",
    "bess_throughput_kwh",
}


def summarize(values, lower_is_better: bool = False) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(v)
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "iqm": iqm(v), "median": float(np.median(v)),
            "ci_low": lo, "ci_high": hi,
            "min": float(v.min()), "max": float(v.max()),
            "worst": float(v.max() if lower_is_better else v.min()),
            "n": int(v.size)}


def _numeric(v) -> bool:
    """Metric dictionaries also carry bookkeeping entries such as the gate
    report; aggregating those would raise rather than be silently wrong, but
    filtering them here keeps callers from having to know that."""
    if isinstance(v, dict) or isinstance(v, (str, bool)):
        return False
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.size > 0 and np.isfinite(arr).all()


def aggregate_runs(runs: List[Dict[str, Sequence[float]]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for r in runs for k in r if not k.startswith("_")})
    out = {}
    for k in keys:
        vals = [float(np.mean(r[k])) for r in runs if k in r and _numeric(r[k])]
        if vals:
            out[k] = summarize(vals, lower_is_better=k in LOWER_IS_BETTER)
    return out


def probability_of_improvement(a, b, lower_is_better: bool = True) -> float:
    """P(a beats b) over independent seed draws.

    A distribution-free effect size answering the question a practitioner
    actually asks -- "if I run each method once, how often does A win?" --
    without assuming normality or equal variance.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    comp = (a[:, None] < b[None, :]) if lower_is_better else (a[:, None] > b[None, :])
    ties = (a[:, None] == b[None, :])
    return float(comp.mean() + 0.5 * ties.mean())


def savings_recovered(policy_cost: float, baseline_cost: float, optimal_cost: float) -> float:
    """Fraction of the *achievable* savings a policy captures.

    ``(baseline - policy) / (baseline - optimal)``.  "We beat our own
    heuristic" is close to unfalsifiable; "we recover 68% of what perfect
    foresight could achieve" is not.
    """
    denom = baseline_cost - optimal_cost
    return float("nan") if abs(denom) < 1e-9 else float((baseline_cost - policy_cost) / denom)


def format_table(rows: List[Dict], columns: List[str], title: str = "") -> str:
    if not rows:
        return f"\n{title}\n(no rows)\n"
    widths = {c: max([len(str(c))] + [len(f"{r.get(c, '')}") for r in rows]) for c in columns}
    line = "  ".join(str(c).ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join("  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in columns) for r in rows)
    head = f"\n{title}\n{'=' * len(title)}\n" if title else "\n"
    return f"{head}{line}\n{sep}\n{body}\n"
