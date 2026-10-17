"""PID-controlled Lagrange multipliers for constrained multi-agent RL.

Motivation
----------
The legacy reward priced individual-rationality violations and feeder
congestion with fixed hand-chosen weights (1.5 and 3.0).  A fixed weight is a
guess at the shadow price of a constraint, and the guess is wrong in both
directions at different points in training: too small early, when the policy is
violating badly and needs pressure; too large late, when the constraint is
slack and the penalty is only distorting the economic objective.  The observed
symptom was exactly what that predicts -- some seeds converging to policies
that satisfied the constraints, others to policies that traded them away.

Formulation
-----------
The problem is a constrained Markov game: maximise expected return subject to

.. math::  J_{C_k}(\\pi) \\le d_k \\qquad k = 1 \\dots K

Solving the Lagrangian dual

.. math::  \\min_{\\lambda \\ge 0} \\max_{\\pi}\\ J_R(\\pi) - \\sum_k \\lambda_k (J_{C_k}(\\pi) - d_k)

by gradient ascent on :math:`\\pi` and descent on :math:`\\lambda` makes the
penalty weight a *learned shadow price* rather than a hyperparameter.

Naive dual ascent on :math:`\\lambda` oscillates, because the multiplier only
responds to the accumulated integral of the violation and therefore always
overshoots.  Stooke, Achiam and Abbeel (2020), *Responsive Safety in
Reinforcement Learning by PID Lagrangian Methods*, show that adding
proportional and derivative terms damps the oscillation:

.. math::  \\lambda \\leftarrow \\left[ K_P e + K_I I + K_D (\\Delta e)_+ \\right]_+

with :math:`e = J_C - d` the constraint violation and :math:`I` its integral.
That is what is implemented here.

Two constraints are used:

``ir``   mean per-agent cost in excess of the grid-only outside option.  This
         is the economic participation constraint: a household that would be
         better off leaving the community will leave.
``grid`` unserved plus curtailed energy, normalised by community scale.  This
         is the feeder constraint.

Both are *soft* by design.  Constraints that must never be violated -- EV
departure feasibility, battery SoC limits, connection limits -- are enforced by
projection in :mod:`src.env.safety` and never reach this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PIDLagrangian:
    """One dual variable with PID-damped updates."""

    limit: float
    kp: float = 0.05
    ki: float = 0.0005
    kd: float = 0.01
    lam_init: float = 0.0
    lam_max: float = 50.0
    integral: float = 0.0
    prev_cost: float | None = None
    lam: float = field(init=False)

    def __post_init__(self):
        self.lam = float(self.lam_init)

    def update(self, cost: float) -> float:
        """Feed the measured constraint cost, return the updated multiplier."""
        error = float(cost) - self.limit
        self.integral = max(0.0, self.integral + error)
        delta = 0.0 if self.prev_cost is None else max(0.0, float(cost) - self.prev_cost)
        self.prev_cost = float(cost)
        self.lam = float(min(self.lam_max,
                             max(0.0, self.kp * error + self.ki * self.integral + self.kd * delta)))
        return self.lam

    def state_dict(self):
        return {"lam": self.lam, "integral": self.integral, "prev_cost": self.prev_cost}

    def load_state_dict(self, d):
        self.lam, self.integral, self.prev_cost = d["lam"], d["integral"], d["prev_cost"]


class ConstraintSet:
    """A named collection of PID-Lagrangian duals."""

    def __init__(self, spec: Dict[str, dict] | None = None, enabled: bool = True):
        spec = spec or {}
        self.enabled = enabled
        self.duals: Dict[str, PIDLagrangian] = {
            name: PIDLagrangian(**kwargs) for name, kwargs in spec.items()
        }
        self.history: List[Dict[str, float]] = []

    def multipliers(self) -> Dict[str, float]:
        if not self.enabled:
            return {k: 0.0 for k in self.duals}
        return {k: d.lam for k, d in self.duals.items()}

    def update(self, costs: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for name, dual in self.duals.items():
            if name in costs:
                out[name] = dual.update(costs[name]) if self.enabled else 0.0
        self.history.append({**out, **{f"cost_{k}": v for k, v in costs.items()}})
        return out

    def state_dict(self):
        return {k: d.state_dict() for k, d in self.duals.items()}

    def load_state_dict(self, d):
        for k, v in d.items():
            if k in self.duals:
                self.duals[k].load_state_dict(v)


DEFAULT_CONSTRAINTS = {
    # Mean per-agent, per-step cost above the outside option, in units of the
    # household's own typical step cost.  A small positive budget is allowed:
    # requiring exactly zero would forbid any transient loss even when it is
    # repaid within the same day.
    "ir": {"limit": 0.02, "kp": 0.5, "ki": 0.01, "kd": 0.1, "lam_max": 20.0},
    # Unserved + curtailed energy, kWh per household per step.  The limit is
    # zero because the perfect-foresight LP shows a dispatch with no unserved
    # energy exists: any violation is a controller failure, not a physical
    # necessity.
    "grid": {"limit": 0.0, "kp": 3.0, "ki": 0.15, "kd": 0.5, "lam_max": 60.0},
}
