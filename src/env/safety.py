"""Hard safety projection applied between the policy and the physics.

Three layers guard the system, in increasing order of strictness:

1. **Residual trust region** (in the environment): the learned action is a
   bounded correction around a transparent rule-based controller.
2. **Feasibility projection** (this module): device limits and, critically, EV
   departure feasibility are enforced *before* dispatch.
3. **Lagrangian constraints** (in the learner): soft economic constraints —
   individual rationality and feeder congestion — are priced by dual variables.

The distinction matters academically.  Layer 2 encodes constraints that are
physically or contractually non-negotiable and that a policy should never be
asked to discover by trial and error.  If an EV must leave at 07:30 with 30 kWh
on board, no amount of reward shaping should be trusted to guarantee it; the
projection simply removes infeasible actions from the action set.  Layer 3
encodes constraints that are genuinely economic trade-offs, where the correct
shadow price is unknown a priori and is therefore *learned*.

Reference: this is the "shielding"/"safety layer" pattern (Dalal et al., 2018,
*Safe Exploration in Continuous Action Spaces*; Alshiekh et al., 2018), adapted
to a multi-agent dispatch setting.
"""
from __future__ import annotations

import numpy as np


class EVFeasibilityProjection:
    """Force-charge EVs that can no longer meet their departure target.

    The test is *least laxity*, expressed in steps rather than in energy.  At
    each step we ask what the vehicle could still gain if it declined to charge
    right now and then charged flat out for every remaining connected step::

        gain_if_deferred = (usable_steps - 1) * P_max * eta * dt

    If that is less than the remaining deficit, this step is no longer
    optional and the setpoint is raised to the smallest value that keeps the
    target reachable.

    Testing the energy slack over *all* remaining steps instead, as a first
    implementation of this class did, silently fails on the final step: with
    one step left and a 0.4 kWh deficit the slack is still positive, so the
    shield declines to fire and the target is missed by exactly the amount it
    was protecting.  That defect produced ~10 kWh of unmet mobility demand per
    day across the fleet while reporting the shield as active.

    Raising the setpoint only to the level required -- rather than to full
    power -- matters too: forcing every shielded vehicle to maximum draw
    manufactures precisely the synchronised peak the feeder constraint is
    meant to avoid.
    """

    def __init__(self, margin_kwh: float = 0.5, enabled: bool = True):
        self.margin_kwh = float(margin_kwh)
        self.enabled = bool(enabled)

    def __call__(self, ev_action: np.ndarray, ev_soc: np.ndarray, ctx) -> tuple:
        """Project EV setpoints.

        Parameters
        ----------
        ev_action : (2, E, N) desired setpoints in [-1, 1]
        ev_soc    : (2, E, N) current state of charge in kWh
        ctx       : object exposing ``ev_target``, ``ev_charge_max``, ``ev_eta``,
                    ``ev_at_home`` (2, T, N), ``ev_departure`` (2, N), ``dt``, ``t``

        Returns
        -------
        (projected_action, must_charge_mask)
        """
        if not self.enabled:
            return ev_action, np.zeros_like(ev_action, dtype=bool)

        t = ctx.t
        T = ctx.n_steps
        # Steps remaining at home before departure (or before end of horizon).
        dep = ctx.ev_departure                                   # (2, N)
        horizon = np.where(dep > t, dep - t, T - t)              # (2, N)
        horizon = np.maximum(horizon, 0)
        # Only steps in which the vehicle is actually connected can be used.
        avail = ctx.ev_home_remaining(t)                         # (2, N) counts
        usable = np.minimum(horizon, avail).astype(np.float32)

        eta = np.sqrt(ctx.ev_eta)                                # (2, N)
        per_step = ctx.ev_charge_max * eta * ctx.dt              # (2, N) kWh per step
        gain_if_deferred = np.maximum(usable - 1.0, 0.0) * per_step
        deficit = np.maximum(ctx.ev_target[:, None, :] - ev_soc, 0.0)  # (2,E,N)
        at_home = ctx.ev_at_home[:, t, :][:, None, :] > 0.5
        must = ((gain_if_deferred[:, None, :] < deficit + self.margin_kwh)
                & (deficit > 1e-6) & at_home)

        required = np.clip(deficit / np.maximum(per_step[:, None, :], 1e-9), 0.0, 1.0)
        projected = np.where(must, np.maximum(ev_action, required), ev_action)
        return projected.astype(np.float32), must


def project_p2p_quotes(quantity: np.ndarray, price: np.ndarray,
                       buy_price: np.ndarray, sell_price: np.ndarray):
    """Map raw actions to admissible market quotes.

    A rational agent never bids above the grid import tariff (it could buy from
    the grid instead) and never asks below the grid export tariff (it could sell
    to the grid instead).  Restricting quotes to ``[sell_price, buy_price]``
    therefore removes only dominated actions.  This keeps the P2P price inside
    the tariff spread by construction, which is the standard individual
    rationality condition for a P2P market and removes an entire failure mode
    in which an untrained policy quotes nonsense and destroys welfare.
    """
    q = np.abs(np.clip(quantity, -1.0, 1.0))
    frac = np.clip(price, 0.0, 1.0)
    spread = np.maximum(buy_price - sell_price, 1e-6)
    quote = sell_price + frac * spread
    return q, quote
