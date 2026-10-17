"""Non-learned reference policies and a centralised optimality bound.

Reporting "the RL agent beat a do-nothing baseline" is close to meaningless.
This module supplies a ladder of references so that a learned policy can be
placed on an interpretable scale:

============================  ================================================
Policy                        What it establishes
============================  ================================================
``NoFlexibility``             the grid-only outside option (lower bound)
``GreedySelfConsumption``     value of storage alone, with no market
``TariffAwareP2P``            value of storage + a transparent P2P heuristic
``PerfectForesightLP``        the *upper* bound: the best any policy could do
============================  ================================================

The LP bound matters most.  Without it a capstone can only say "our method is
better than our heuristics"; with it, the claim becomes "our method recovers
X% of the achievable savings", which is a falsifiable statement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .community_env import ACTION_DIM, A_BESS, A_EV1, A_EV2, A_PRICE, A_QTY


class RulePolicy:
    """Base class. ``residual`` declares which action convention it expects."""
    residual = False
    name = "rule"

    def act(self, obs: np.ndarray, state: np.ndarray, env=None) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        pass


class NoFlexibility(RulePolicy):
    """The grid-only outside option, reproduced inside the simulator.

    No storage dispatch, no P2P, and *uncontrolled* EV charging: every vehicle
    draws full power on arrival until its departure target is met.  This is
    exactly the schedule used to build the cost counterfactual, so a correct
    implementation reproduces ``baseline_cost_eur`` to numerical precision.
    The test suite asserts this, which turns the counterfactual from an
    assumption into a checked invariant.
    """
    residual = False
    name = "no_flexibility"

    def act(self, obs, state, env=None):
        E, N = obs.shape[:2]
        a = np.zeros((E, N, ACTION_DIM), dtype=np.float32)
        for j, slot in enumerate((A_EV1, A_EV2)):
            home = obs[:, :, 8 + 5 * j]
            deficit = obs[:, :, 9 + 5 * j]
            a[:, :, slot] = np.where((home > 0.5) & (deficit > 1e-6), 1.0, 0.0)
        a[:, :, A_PRICE] = 0.5
        return a


class DeadlineCharging(RulePolicy):
    """No storage, no P2P, but EVs charge as *late* as feasibility allows.

    Isolates the value of the safety projection on its own.  Any saving it
    shows is pure load shifting into cheap hours, with no market and no
    battery, and it is therefore the correct reference against which to judge
    how much of a learned policy's gain comes from trading rather than from the
    shielding layer it inherits.
    """
    residual = False
    name = "deadline_charging"

    def act(self, obs, state, env=None):
        E, N = obs.shape[:2]
        a = np.zeros((E, N, ACTION_DIM), dtype=np.float32)
        a[:, :, A_PRICE] = 0.5
        return a


class GreedySelfConsumption(RulePolicy):
    """Storage absorbs surplus and covers deficit; no trading with peers."""
    residual = False
    name = "greedy_self_consumption"

    def act(self, obs, state, env=None):
        E, N = obs.shape[:2]
        a = np.zeros((E, N, ACTION_DIM), dtype=np.float32)
        net = obs[:, :, 2]                     # net demand, household-normalised
        room = obs[:, :, 4]
        reserve = obs[:, :, 5]
        a[:, :, A_BESS] = np.where((net < -1e-3) & (room > 0.01), 1.0,
                                   np.where((net > 1e-3) & (reserve > 0.02), -1.0, 0.0))
        # EVs charge whenever connected and short of target.
        for j, slot in enumerate((A_EV1, A_EV2)):
            home = obs[:, :, 8 + 5 * j]
            deficit = obs[:, :, 9 + 5 * j]
            a[:, :, slot] = np.where((home > 0.5) & (deficit > 1e-4), 1.0, 0.0)
        a[:, :, A_PRICE] = 0.5
        return a


class TariffAwareP2P(RulePolicy):
    """The transparent controller that also centres the residual trust region.

    Implemented by returning a zero residual, so that it is *by construction*
    the same controller the learner corrects.  In the legacy code the baseline
    policy and the residual reference were written twice with slightly
    different thresholds, which meant "MAPPO vs the baseline" was not a
    controlled comparison.
    """
    residual = True
    name = "tariff_aware_p2p"

    def act(self, obs, state, env=None):
        E, N = obs.shape[:2]
        a = np.zeros((E, N, ACTION_DIM), dtype=np.float32)
        a[:, :, A_PRICE] = 0.5                 # zero correction to the reference quote
        return a


# ----------------------------------------------------------------------
# Perfect-foresight centralised bound
# ----------------------------------------------------------------------
def perfect_foresight_lp(tensors, cfg, verbose: bool = False) -> Optional[dict]:
    """Least-cost community dispatch with perfect knowledge of the whole day.

    Formulation
    -----------
    Peer-to-peer trades are *internal transfers*: whatever one household pays a
    neighbour, the neighbour receives.  The welfare-optimal P2P market therefore
    minimises the community's residual exchange with the grid, so the community
    optimum is the linear program

    .. math::
        \\min \\sum_t \\left( \\pi^{buy}_t G^{imp}_t - \\pi^{sell}_t G^{exp}_t
                             + \\mu\\, \\pi^{buy}_t U_t \\right)

    over battery and EV charge/discharge power, community import/export, and
    unserved energy :math:`U_t`, subject to community power balance, storage
    dynamics with charge/discharge efficiency, EV connectivity, EV departure and
    end-of-day energy requirements, and feeder import/export limits.  Unserved
    energy carries the same scarcity multiple :math:`\\mu` the simulator applies,
    so the bound is attainable in principle rather than an artefact of a softer
    model.

    State of charge is carried as an explicit variable with a banded linking
    equality rather than as a cumulative sum over the control history.  The two
    are mathematically identical; the banded form has O(T) non-zeros per device
    instead of O(T^2), which is the difference between a solve of seconds and
    one of minutes at 250 households.

    The program is a *relaxation* in one respect: it nets community imports and
    exports at the same instant, which the agent-level simulation also permits
    through the P2P market.  It is therefore a valid upper bound on achievable
    savings, and the gap between a policy and this bound -- rather than the gap
    to a hand-written heuristic -- is the quantity worth reporting.

    Returns ``None`` if SciPy is unavailable or the solver fails.
    """
    try:
        from scipy.optimize import linprog
        from scipy import sparse
    except ImportError:                                    # pragma: no cover
        return None

    ct = tensors
    T, N, dt = ct.n_steps, ct.n_agents, ct.dt
    eta_b = np.sqrt(ct.bess_eta)
    eta_e = np.sqrt(ct.ev_eta)
    dep_target = getattr(ct, "ev_departure_target", ct.ev_target)
    fin_target = getattr(ct, "ev_final_target", ct.ev_target)
    mu = getattr(cfg, "unserved_price_multiple", 3.0)
    buy = ct.buy_price.mean(axis=1)
    sell = ct.sell_price.mean(axis=1)

    nb, ne = T * N, 2 * T * N
    off = {}
    cur = 0
    for key, size in (("bc", nb), ("bd", nb), ("eb", nb),
                      ("ec", ne), ("ed", ne), ("ee", ne),
                      ("gi", T), ("ge", T), ("us", T), ("cu", T)):
        off[key] = cur
        cur += size
    nvar = cur

    def ib(t, i):     return off["bc"] + t * N + i
    def ibd(t, i):    return off["bd"] + t * N + i
    def ieb(t, i):    return off["eb"] + t * N + i
    def iec(j, t, i): return off["ec"] + j * T * N + t * N + i
    def ied(j, t, i): return off["ed"] + j * T * N + t * N + i
    def iee(j, t, i): return off["ee"] + j * T * N + t * N + i

    c = np.zeros(nvar)
    c[off["gi"]:off["gi"] + T] = buy * dt
    c[off["ge"]:off["ge"] + T] = -sell * dt
    c[off["us"]:off["us"] + T] = buy * dt * mu
    c[off["cu"]:off["cu"] + T] = 1e-3 * dt
    # A vanishing throughput cost stops the solver from cycling storage for free
    # when the objective is flat, which would otherwise report a bound no causal
    # controller could match for reasons unrelated to foresight.
    c[off["bc"]:off["bc"] + nb] += 1e-4 * dt
    c[off["bd"]:off["bd"] + nb] += 1e-4 * dt

    lo = np.zeros(nvar)
    hi = np.full(nvar, np.inf)
    for t in range(T):
        hi[ib(t, 0):ib(t, 0) + N] = ct.bess_charge_max
        hi[ibd(t, 0):ibd(t, 0) + N] = ct.bess_discharge_max
        lo[ieb(t, 0):ieb(t, 0) + N] = np.minimum(cfg.bess_min_soc * ct.bess_capacity, ct.bess_soc0)
        hi[ieb(t, 0):ieb(t, 0) + N] = np.maximum(cfg.bess_max_soc * ct.bess_capacity, ct.bess_soc0)
        for j in range(2):
            home = ct.ev_at_home[j, t] > 0.5
            hi[iec(j, t, 0):iec(j, t, 0) + N] = np.where(home, ct.ev_charge_max[j], 0.0)
            hi[ied(j, t, 0):ied(j, t, 0) + N] = np.where(
                home, np.minimum(ct.ev_discharge_max[j], ct.ev_v2g_max[j, t]), 0.0)
            hi[iee(j, t, 0):iee(j, t, 0) + N] = ct.ev_capacity[j]
    hi[off["gi"]:off["gi"] + T] = cfg.feeder_import_kw_per_hh * N
    hi[off["ge"]:off["ge"] + T] = cfg.feeder_export_kw_per_hh * N

    R, C, V, beq = [], [], [], []
    r = 0
    # (1) community power balance
    for t in range(T):
        for i in range(N):
            R += [r, r]; C += [ib(t, i), ibd(t, i)]; V += [1.0, -1.0]
            for j in range(2):
                R += [r, r]; C += [iec(j, t, i), ied(j, t, i)]; V += [1.0, -1.0]
        R += [r, r, r, r]
        C += [off["gi"] + t, off["ge"] + t, off["us"] + t, off["cu"] + t]
        V += [-1.0, 1.0, -1.0, 1.0]
        beq.append(float((ct.pv_kw[t] - ct.load_kw[t]).sum()))
        r += 1
    # (2) battery state of charge dynamics
    for t in range(T):
        for i in range(N):
            R += [r, r, r]; C += [ieb(t, i), ib(t, i), ibd(t, i)]
            V += [1.0, -dt * eta_b[i], dt / eta_b[i]]
            if t > 0:
                R += [r]; C += [ieb(t - 1, i)]; V += [-1.0]
                beq.append(0.0)
            else:
                beq.append(float(ct.bess_soc0[i]))
            r += 1
    # (3) EV state of charge dynamics, with the trip drawdown at departure
    for j in range(2):
        for t in range(T):
            for i in range(N):
                R += [r, r, r]; C += [iee(j, t, i), iec(j, t, i), ied(j, t, i)]
                V += [1.0, -dt * eta_e[j, i], dt / eta_e[j, i]]
                trip = ct.ev_trip_kwh[j, i] if int(ct.ev_departure[j, i]) == t else 0.0
                if t > 0:
                    R += [r]; C += [iee(j, t - 1, i)]; V += [-1.0]
                    beq.append(float(-trip))
                else:
                    beq.append(float(ct.ev_soc0[j, i] - trip))
                r += 1
    A_eq = sparse.coo_matrix((V, (R, C)), shape=(r, nvar)).tocsr()

    # (4) EV energy requirements as simple variable bounds
    for j in range(2):
        for i in range(N):
            d = int(np.clip(ct.ev_departure[j, i], 0, T - 1))
            need = float(dep_target[j, i])
            if need > 1e-6 and d > 0:
                # required *before* the trip is deducted at step d
                lo[iee(j, d - 1, i)] = min(max(lo[iee(j, d - 1, i)], need),
                                           float(ct.ev_capacity[j, i]))
            fin = float(min(fin_target[j, i], ct.ev_capacity[j, i]))
            if fin > 1e-6:
                lo[iee(j, T - 1, i)] = min(max(lo[iee(j, T - 1, i)], fin),
                                           float(ct.ev_capacity[j, i]))

    res = linprog(c, A_eq=A_eq, b_eq=np.asarray(beq), bounds=list(zip(lo, hi)),
                  method="highs")
    if not res.success:
        if verbose:
            print("LP failed:", res.message)
        return None
    gi = res.x[off["gi"]:off["gi"] + T]
    ge = res.x[off["ge"]:off["ge"] + T]
    us = res.x[off["us"]:off["us"] + T]
    cost = float(((buy * gi - sell * ge + buy * us * mu) * dt).sum())
    return {"optimal_cost_eur": cost,
            "grid_import_kwh": float(gi.sum() * dt),
            "grid_export_kwh": float(ge.sum() * dt),
            "unserved_kwh": float(us.sum() * dt),
            "peak_import_kw": float(gi.max()),
            "status": str(res.message)}


BASELINES = {
    "no_flexibility": NoFlexibility,
    "deadline_charging": DeadlineCharging,
    "greedy_self_consumption": GreedySelfConsumption,
    "tariff_aware_p2p": TariffAwareP2P,
}
