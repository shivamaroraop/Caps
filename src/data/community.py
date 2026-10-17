"""Vectorised community tensors.

The legacy environment stored households as a list of Python objects and looped
over them at every one of the 96 timesteps.  With 25-250 agents that loop
dominated wall-clock time and made multi-seed studies impractical.

This module flattens a :class:`CommunityData` into a dataclass of dense NumPy
arrays laid out *timestep-first* ``(T, N)`` so that a single simulation step is
one vectorised expression over all households.

It also caches the parsed workbook to ``.npz`` so repeated experiments do not
re-parse 19 Excel sheets (~3 s) on every launch.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .ev_physics import apply_ev_corrections
from .loader import DatasetLoader
from .schemas import CommunityData

T_DAY = 96
DT_HOURS = 0.25


@dataclass
class CommunityTensors:
    """Dense, timestep-first arrays describing one energy community."""

    # --- identity -----------------------------------------------------
    player_ids: np.ndarray                 # (N,)
    n_agents: int
    n_steps: int = T_DAY
    dt: float = DT_HOURS

    # --- exogenous timeseries, shape (T, N) ---------------------------
    load_kw: np.ndarray = None             # inflexible household demand
    pv_kw: np.ndarray = None               # PV generation
    buy_price: np.ndarray = None           # EUR/kWh grid import tariff
    sell_price: np.ndarray = None          # EUR/kWh grid export tariff

    # --- static per-household, shape (N,) -----------------------------
    has_pv: np.ndarray = None
    has_bess: np.ndarray = None
    buy_limit_kw: np.ndarray = None
    sell_limit_kw: np.ndarray = None

    # --- BESS, shape (N,) ---------------------------------------------
    bess_capacity: np.ndarray = None
    bess_charge_max: np.ndarray = None
    bess_discharge_max: np.ndarray = None
    bess_eta: np.ndarray = None            # round-trip efficiency
    bess_soc0: np.ndarray = None

    # --- EVs, shape (2, N) or (2, T, N) --------------------------------
    ev_capacity: np.ndarray = None         # (2, N)
    ev_charge_max: np.ndarray = None       # (2, N)
    ev_discharge_max: np.ndarray = None    # (2, N)
    ev_eta: np.ndarray = None              # (2, N)
    ev_soc0: np.ndarray = None             # (2, N)
    ev_target: np.ndarray = None           # (2, N) required SoC at departure
    ev_trip_kwh: np.ndarray = None         # (2, N)
    ev_departure: np.ndarray = None        # (2, N) int timestep index
    ev_at_home: np.ndarray = None          # (2, T, N) binary
    ev_v2g_max: np.ndarray = None          # (2, T, N) time-varying V2G cap

    # --- derived normalisation constants -------------------------------
    cost_scale: np.ndarray = None          # (N,) EUR/step, per-household scale
    energy_scale: np.ndarray = None        # (N,) kWh/step, per-household scale
    ev_departure_target: np.ndarray = None  # (2, N) kWh needed at departure
    ev_final_target: np.ndarray = None      # (2, N) kWh needed at end of day
    ev_report: Dict = field(default_factory=dict)
    home_charger_kw: np.ndarray = None      # (2, N) AC home charge point rating
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def grid_only_cost(self) -> np.ndarray:
        """(N,) EUR: cost of the *no-flexibility, no-P2P* outside option.

        Every fairness and individual-rationality statement in this project is
        measured against this counterfactual, so it is computed once, from the
        exogenous data alone, and never depends on any policy.
        """
        net = self.load_kw - self.pv_kw                      # (T, N)
        imp = np.maximum(net, 0.0) * self.dt
        exp = np.maximum(-net, 0.0) * self.dt
        return (imp * self.buy_price - exp * self.sell_price).sum(axis=0)

    @property
    def ev_baseline_energy(self) -> np.ndarray:
        """(N,) kWh that EVs must absorb over the day.

        Driven by trip consumption, not by the start-of-day gap: in this dataset
        every vehicle begins at or above its departure target, leaves in the
        morning, and returns depleted.  The recharge that follows is the real
        EV load and it is what creates the evening peak.

        Charging an EV is a service, not a cost overrun, so the outside option
        must include it: otherwise every EV-owning household is trivially
        'worse off than grid-only' and individual rationality is meaningless.
        """
        start_gap = np.maximum(self.ev_target - self.ev_soc0, 0.0)
        return (start_gap + self.ev_trip_kwh).sum(axis=0)


def _stack_attr(households: Sequence, path: str, default=0.0, dtype=np.float32):
    out = []
    for h in households:
        obj = h
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        out.append(default if obj is None else obj)
    return np.asarray(out, dtype=dtype)


def _stack_series(households: Sequence, path: str, n_steps: int) -> np.ndarray:
    """Stack a (96,) timeseries attribute into a (T, N) array."""
    cols = []
    for h in households:
        obj = h
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        cols.append(np.zeros(n_steps) if obj is None else np.asarray(obj, dtype=np.float64))
    return np.stack(cols, axis=1).astype(np.float32)


def build_tensors(community: CommunityData) -> CommunityTensors:
    """Flatten a :class:`CommunityData` into dense arrays."""
    hs: List = community.households
    n = len(hs)
    T = community.timesteps_per_day
    dt = community.timestep_minutes / 60.0

    ct = CommunityTensors(
        player_ids=np.asarray([h.player_id for h in hs], dtype=np.int32),
        n_agents=n, n_steps=T, dt=dt,
    )
    ct.load_kw = _stack_series(hs, "load_profile_kw", T)
    ct.pv_kw = _stack_series(hs, "pv_profile_kw", T)
    ct.buy_price = _stack_series(hs, "buy_price_eur", T)
    ct.sell_price = _stack_series(hs, "sell_price_eur", T)

    ct.has_pv = np.asarray([bool(h.has_pv) for h in hs], dtype=np.float32)
    ct.has_bess = np.asarray([bool(h.has_bess) for h in hs], dtype=np.float32)
    ct.buy_limit_kw = _stack_attr(hs, "power_buy_limit_kw", 17.25)
    ct.sell_limit_kw = _stack_attr(hs, "power_sell_limit_kw", 17.25)

    ct.bess_capacity = _stack_attr(hs, "bess.capacity_kwh", 0.0)
    ct.bess_charge_max = _stack_attr(hs, "bess.max_charge_kw", 0.0)
    ct.bess_discharge_max = _stack_attr(hs, "bess.max_discharge_kw", 0.0)
    ct.bess_eta = np.clip(_stack_attr(hs, "bess.efficiency", 0.95), 0.5, 1.0)
    ct.bess_soc0 = _stack_attr(hs, "bess.initial_soc_kwh", 0.0)
    # A household without storage is represented by a zero-power battery rather
    # than by a branch in the simulation loop.
    no_bess = ct.has_bess < 0.5
    ct.bess_capacity = np.where(no_bess, 1.0, np.maximum(ct.bess_capacity, 1e-3))
    ct.bess_charge_max = np.where(no_bess, 0.0, ct.bess_charge_max)
    ct.bess_discharge_max = np.where(no_bess, 0.0, ct.bess_discharge_max)
    ct.bess_soc0 = np.where(no_bess, 0.0, np.clip(ct.bess_soc0, 0.0, ct.bess_capacity))

    def ev(attr, default=0.0):
        return np.stack([_stack_attr(hs, f"ev1.{attr}", default),
                         _stack_attr(hs, f"ev2.{attr}", default)])

    ct.ev_capacity = np.maximum(ev("capacity_kwh"), 1e-3)
    ct.ev_charge_max = ev("max_charge_kw")
    ct.ev_discharge_max = ev("max_discharge_kw")
    ct.ev_eta = np.clip(ev("efficiency", 0.90), 0.5, 1.0)
    ct.ev_soc0 = np.clip(ev("initial_soc_kwh"), 0.0, ct.ev_capacity)
    ct.ev_target = np.clip(ev("target_final_soc_kwh"), 0.0, ct.ev_capacity)
    ct.ev_trip_kwh = ev("trip_consumption_kwh")
    ct.ev_departure = np.stack([_stack_attr(hs, "ev1.departure_period", 0, dtype=np.int64),
                                _stack_attr(hs, "ev2.departure_period", 0, dtype=np.int64)])
    ct.ev_at_home = np.stack([_stack_series(hs, "ev1.at_home", T),
                              _stack_series(hs, "ev2.at_home", T)])
    ct.ev_v2g_max = np.stack([_stack_series(hs, "ev1.max_discharge_profile", T),
                              _stack_series(hs, "ev2.max_discharge_profile", T)])
    # A vehicle with no capacity or no charger is inert.
    inert = (ct.ev_capacity <= 1e-2) | (ct.ev_charge_max <= 1e-6)
    ct.ev_charge_max = np.where(inert, 0.0, ct.ev_charge_max)
    ct.ev_discharge_max = np.where(inert, 0.0, ct.ev_discharge_max)
    ct.ev_target = np.where(inert, 0.0, ct.ev_target)

    # --- per-household scales -----------------------------------------
    # Rewards are divided by these so that a 12 kWh/day flat and a 60 kWh/day
    # house with two EVs produce comparable gradient magnitudes.  Without this
    # the policy is dominated by the few largest consumers.
    daily_energy = ((np.abs(ct.load_kw).sum(axis=0) + np.abs(ct.pv_kw).sum(axis=0)) * dt
                    + ct.ev_baseline_energy)
    ct.energy_scale = np.maximum(daily_energy / T, 0.05).astype(np.float32)
    mean_price = ct.buy_price.mean(axis=0)
    ct.cost_scale = np.maximum(ct.energy_scale * mean_price, 1e-3).astype(np.float32)

    ct.ev_report = apply_ev_corrections(ct)
    ct.meta = {
        "n_pv": int(ct.has_pv.sum()), "n_bess": int(ct.has_bess.sum()),
        "n_ev": int((ct.ev_charge_max > 0).sum()),
        "daily_load_kwh": float(ct.load_kw.sum() * dt),
        "daily_pv_kwh": float(ct.pv_kw.sum() * dt),
        "ev_physics": ct.ev_report,
    }
    return ct


# ----------------------------------------------------------------------
# Loading with cache
# ----------------------------------------------------------------------
def load_community(path: str | Path, cache_dir: str | Path = ".cache") -> CommunityData:
    """Parse the workbook, caching the result keyed on file content hash."""
    path = Path(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(path.read_bytes()).hexdigest()[:16]
    cache = cache_dir / f"community_{digest}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    community = DatasetLoader(str(path)).load_all()
    with open(cache, "wb") as fh:
        pickle.dump(community, fh)
    return community


def stratified_subset(community: CommunityData, n: int, seed: int = 0) -> CommunityData:
    """Select ``n`` households preserving the (PV, BESS, EV) population mix.

    Uniform random sampling of 25 out of 250 households routinely produced
    communities with almost no consumers, which makes the P2P market trivially
    one-sided and inflates the apparent benefit of trading.  Stratifying on the
    flexibility archetype keeps the buyer/seller balance representative.
    """
    import copy
    if n >= len(community.households):
        return community
    rng = np.random.default_rng(seed)
    strata: Dict[tuple, List] = {}
    for h in community.households:
        has_ev = bool(getattr(h.ev1, "max_charge_kw", 0) or getattr(h.ev2, "max_charge_kw", 0))
        strata.setdefault((bool(h.has_pv), bool(h.has_bess), has_ev), []).append(h)

    total = len(community.households)
    picked: List = []
    # Largest-remainder apportionment keeps the mix exact rather than drifting.
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    floors = {k: int(np.floor(q)) for k, q in quotas.items()}
    for k, f in floors.items():
        f = min(f, len(strata[k]))
        if f:
            picked.extend(rng.choice(strata[k], f, replace=False).tolist())
    remainder = sorted(strata, key=lambda k: quotas[k] - floors[k], reverse=True)
    ri = 0
    while len(picked) < n and ri < len(remainder) * 4:
        k = remainder[ri % len(remainder)]
        pool = [h for h in strata[k] if h not in picked]
        if pool:
            picked.append(pool[int(rng.integers(len(pool)))])
        ri += 1
    if len(picked) < n:  # pathological fallback
        rest = [h for h in community.households if h not in picked]
        picked.extend(rng.choice(rest, n - len(picked), replace=False).tolist())

    sub = copy.copy(community)
    sub.households = picked[:n]
    sub.compute_aggregates()
    return sub
