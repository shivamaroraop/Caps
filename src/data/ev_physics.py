"""Corrections to the EV representation read out of the workbook.

Three defects in the original parsing materially changed the physics of the
problem.  They are documented here rather than silently patched, because the
corrected model is what makes EV charging a genuine control problem instead of
an unconstrained power sink.

1. Home charge point rating
---------------------------
The ``EVs`` sheet row *Charge (kW)* reports 55-121 kW.  Those are **vehicle DC
charge-acceptance** ratings, not the rating of the charge point the vehicle is
plugged into at home.  The sheets ``Max EV1/EV2 Dis-Charge`` carry a constant
7.4 or 11 kW per household -- the standard single-phase (32 A) and three-phase
(16 A) AC home charge point ratings.  Home charging power is therefore

    P_home = min(vehicle acceptance, charge point rating)

Using the uncorrected 121 kW figure lets a *single* household saturate a feeder
sized for the whole community, which made the congestion metrics meaningless
and gave the learner an action with absurd leverage.

2. Departure feasibility
------------------------
*Initial battery level* is 20% of capacity, and *Final battery level* is the
same figure.  The legacy loader treated the final level as a **departure**
target, which produced a target already satisfied at t=0 and hence zero EV
flexibility requirement.  But the recorded trips consume 3-14 kWh, which is
*more* than the 20% starting charge for most vehicles: as parsed, the fleet
could not physically complete its own trips.

The coherent reading is two separate requirements:

* **departure**: enough energy for the round trip plus a reserve, required at
  the departure period,
* **end of day**: back to the *Final battery level* by the last period, so the
  vehicle is ready for the next day.

This creates a morning charging window and an evening recharging window, which
is precisely the structure that makes coordinated charging valuable.

3. Trip timing
--------------
Vehicles leave between 08:15 and 11:30 and return between 19:45 and 21:30; the
``at Home`` sheets are authoritative for connectivity and are used directly.
"""
from __future__ import annotations

import numpy as np

DEFAULT_HOME_CHARGER_KW = 7.4
TRIP_RESERVE_FRACTION = 0.10     # keep 10% of capacity as arrival reserve


def apply_ev_corrections(ct, home_charger_kw: float | None = None,
                         reserve_fraction: float = TRIP_RESERVE_FRACTION) -> dict:
    """Rewrite EV fields of a :class:`CommunityTensors` in place.

    Returns a report describing what changed, so experiments can log the exact
    physical assumptions they ran under.
    """
    before_charge = ct.ev_charge_max.copy()

    # --- 1. charge point rating ---------------------------------------
    # The ``Max EV Dis-Charge`` profile is *time varying*: it carries the home
    # charge point rating (7.4 / 11 kW) while the vehicle is at home and the
    # public fast-charger rating (22-60 kW) while it is out.  Taking the max,
    # as a naive read does, therefore imports a DC fast charger into the living
    # room.  The home rating is the smallest positive value observed on the
    # steps where the vehicle is actually connected at home.
    fallback = home_charger_kw or DEFAULT_HOME_CHARGER_KW
    at_home = ct.ev_at_home > 0.5                            # (2, T, N)
    prof = np.where(at_home & (ct.ev_v2g_max > 1e-3), ct.ev_v2g_max, np.inf)
    rating = prof.min(axis=1)                                # (2, N)
    rating = np.where(np.isfinite(rating), rating, fallback)
    has_ev = (ct.ev_capacity > 1e-2) & (before_charge > 1e-6)
    ct.ev_charge_max = np.where(has_ev, np.minimum(before_charge, rating), 0.0).astype(np.float32)
    ct.ev_discharge_max = np.where(has_ev, np.minimum(ct.ev_discharge_max, rating), 0.0).astype(np.float32)
    # Cap the V2G profile at the home rating too: energy can only reach the
    # community through the household connection.
    ct.ev_v2g_max = np.minimum(ct.ev_v2g_max, rating[:, None, :]).astype(np.float32)
    ct.ev_v2g_max = np.where(at_home, ct.ev_v2g_max, 0.0).astype(np.float32)
    ct.home_charger_kw = rating.astype(np.float32)

    # --- 2. departure and end-of-day targets --------------------------
    reserve = reserve_fraction * ct.ev_capacity
    ct.ev_departure_target = np.where(
        has_ev, np.minimum(ct.ev_trip_kwh + reserve, ct.ev_capacity), 0.0).astype(np.float32)
    ct.ev_final_target = np.where(has_ev, np.minimum(ct.ev_target, ct.ev_capacity), 0.0).astype(np.float32)
    # ``ev_target`` remains the *active* deadline target used by the feasibility
    # projection; the environment swaps it for the end-of-day target once the
    # vehicle has departed.
    ct.ev_target = ct.ev_departure_target.copy()

    # --- 3. feasibility audit -----------------------------------------
    infeasible = 0
    for j in range(2):
        for i in range(ct.n_agents):
            if not has_ev[j, i]:
                continue
            dep = int(ct.ev_departure[j, i])
            steps = ct.ev_at_home[j, :max(dep, 1), i].sum()
            gain = steps * ct.dt * ct.ev_charge_max[j, i] * np.sqrt(ct.ev_eta[j, i])
            if ct.ev_soc0[j, i] + gain + 1e-6 < ct.ev_departure_target[j, i]:
                infeasible += 1

    return {
        "home_charge_kw_before": (float(before_charge[has_ev].min()) if has_ev.any() else 0.0,
                                  float(before_charge[has_ev].max()) if has_ev.any() else 0.0),
        "home_charge_kw_after": (float(ct.ev_charge_max[has_ev].min()) if has_ev.any() else 0.0,
                                 float(ct.ev_charge_max[has_ev].max()) if has_ev.any() else 0.0),
        "n_ev": int(has_ev.sum()),
        "mean_trip_kwh": float(ct.ev_trip_kwh[has_ev].mean()) if has_ev.any() else 0.0,
        "total_ev_energy_kwh": float((ct.ev_trip_kwh * has_ev).sum()),
        "infeasible_departures": infeasible,
    }
