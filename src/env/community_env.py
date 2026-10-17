"""Multi-agent peer-to-peer energy community environment.

Design notes that differ materially from the legacy implementation
--------------------------------------------------------------------
1. **Per-agent rewards are preserved.**  The environment returns an ``(E, N)``
   reward matrix and, separately, every additive component of it.  The learner
   is responsible for combining components; the environment never collapses the
   matrix to a scalar.  In the legacy code the trainer called ``rewards.mean()``
   before storing the transition, which erased all cross-agent credit
   assignment: every household received an identical advantage regardless of
   what it did.

2. **The reward is Markovian.**  Cumulative cost burden, community saving rate
   and fairness are exposed in the observation and global state, so the terms
   that depend on them are functions of the observed state rather than of
   hidden history.  Previously the reward depended on cumulative quantities the
   critic could not see, which put a floor on value error.

3. **The outside option includes EV charging.**  The grid-only counterfactual
   simulates uncontrolled ("dumb") EV charging at full power on arrival.  Under
   the legacy counterfactual, which ignored EVs entirely, every EV-owning
   household was worse off than baseline by construction and the individual
   rationality metric was not interpretable.

4. **Rewards are scaled per household.**  A 60 kWh/day house with two EVs and a
   12 kWh/day flat produce comparable gradient magnitudes, so the shared policy
   is not dominated by the largest consumers.

5. **Everything is vectorised over ``E`` parallel episodes.**  The legacy env
   ran one episode per PPO update: 96 timesteps of data for eight epochs of
   optimisation, which is the second-largest source of the instability the
   project reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..data.community import CommunityTensors
from .market import clear_double_auction
from .safety import EVFeasibilityProjection, project_p2p_quotes

# Observation layout is documented in ``OBS_NAMES`` and asserted in tests so
# that any change to the feature set fails loudly instead of silently shifting
# the indices the rule-based baselines read.
OBS_NAMES = [
    # local physical state (0-6)
    "load_norm", "pv_norm", "net_norm",
    "bess_soc_frac", "bess_charge_room", "bess_discharge_room", "bess_power_norm",
    # EV1 (7-11)
    "ev1_soc_frac", "ev1_at_home", "ev1_deficit_frac", "ev1_time_to_dep", "ev1_urgency",
    # EV2 (12-16)
    "ev2_soc_frac", "ev2_at_home", "ev2_deficit_frac", "ev2_time_to_dep", "ev2_urgency",
    # prices (17-20)
    "buy_price_z", "sell_price_z", "price_percentile", "spread_norm",
    # market / community (21-24)
    "last_clearing_price_z", "last_volume_norm", "community_net_norm", "congestion_ratio",
    # time (25-27)
    "sin_phase", "cos_phase", "day_remaining",
    # economic state -- makes the reward Markovian (28-31)
    "own_burden", "own_cost_norm", "community_burden", "ir_margin",
    # forecasts (32-37)
    "fc_net_15m", "fc_net_60m", "fc_net_180m",
    "fc_price_15m", "fc_price_60m", "fc_price_180m",
    # reference action context for the residual policy (38-42)
    "ref_bess", "ref_ev1", "ref_ev2", "ref_qty", "ref_price",
    # constraint pressure -- state augmentation (43-44)
    "lam_ir", "lam_grid",
]
OBS_DIM = len(OBS_NAMES)

STATE_NAMES = [
    "tot_load", "tot_pv", "tot_net", "mean_bess_soc", "mean_ev_soc",
    "buy_price_z", "sell_price_z", "last_price_z", "last_volume", "congestion",
    "sin_phase", "cos_phase", "day_remaining",
    "community_saving_rate", "fairness_jain", "ir_fraction",
    "market_utilisation", "mean_burden", "std_burden", "grid_import_norm",
]
STATE_DIM = len(STATE_NAMES)

ACTION_DIM = 5
A_BESS, A_EV1, A_EV2, A_QTY, A_PRICE = range(5)


@dataclass
class EnvConfig:
    n_envs: int = 8
    noise_std: float = 0.02
    include_evs: bool = True
    residual: bool = True
    residual_scale: float = 0.35          # trust region around the safe policy
    residual_price_scale: float = 0.40
    market_mechanism: str = "uniform"     # or "midpoint"
    p2p_fee: float = 0.01
    bess_min_soc: float = 0.10
    bess_max_soc: float = 0.95
    bess_degradation_eur_per_kwh: float = 0.02
    # After-diversity feeder sizing.  Per-household *connection* capacity in
    # this dataset is ~17.3 kW, but an LV feeder is never sized at N x 17.3.
    # 8 kW/household import is a realistic EV-era after-diversity figure: it is
    # comfortably above the 2.5 kW/hh non-EV peak and above the 1.4 kW/hh daily
    # mean, but well below the 11.3 kW/hh peak that uncontrolled simultaneous
    # EV charging produces.  The constraint therefore binds exactly when
    # coordination fails, which is the regime the project is about.
    feeder_import_kw_per_hh: float = 8.0
    feeder_export_kw_per_hh: float = 5.0
    unserved_price_multiple: float = 3.0
    ev_safety_projection: bool = True
    ev_safety_margin_kwh: float = 0.5
    # State augmentation (Calvo-Fullana et al. 2024; Amaya-Corredor et al. 2026).
    # The current Lagrange multipliers are exposed to the policy so that it can
    # condition its behaviour on how much constraint pressure the system is
    # under.  Without this the policy only feels lambda through the reward and
    # is, at any instant, a fixed-multiplier policy -- the configuration whose
    # brittleness the 2026 TMLR ablation quantifies (7 feasible out of 414).
    state_augmented: bool = True
    lam_max_obs: float = 20.0
    # Time-varying grid carbon intensity, used for the emissions KPI.
    carbon_aware: bool = True
    # reward weights (constraint terms are priced by the learner's duals)
    w_econ: float = 1.0
    w_team: float = 0.30
    w_shape: float = 0.25
    w_ev_shortfall: float = 2.0
    w_terminal_fairness: float = 8.0
    w_terminal_ir: float = 8.0
    gamma: float = 0.99                   # must match the learner for shaping


class P2PCommunityEnv:
    """``E`` parallel one-day episodes of an ``N``-household energy community."""

    observation_dim = OBS_DIM
    state_dim = STATE_DIM
    action_dim = ACTION_DIM

    def __init__(self, tensors: CommunityTensors, cfg: Optional[EnvConfig] = None,
                 seed: int = 0):
        self.ct = tensors
        self.cfg = cfg or EnvConfig()
        self.N = tensors.n_agents
        self.E = int(self.cfg.n_envs)
        self.n_steps = tensors.n_steps
        self.dt = tensors.dt
        self.rng = np.random.default_rng(seed)
        self.t = 0

        ct = self.ct
        # ---- price normalisation statistics (computed once, from data) ----
        self._buy_mu = ct.buy_price.mean()
        self._buy_sd = max(ct.buy_price.std(), 1e-4)
        self._sell_mu = ct.sell_price.mean()
        self._sell_sd = max(ct.sell_price.std(), 1e-4)
        order = np.argsort(ct.buy_price.mean(axis=1))
        pct = np.empty(self.n_steps)
        pct[order] = np.linspace(0.0, 1.0, self.n_steps)
        self._price_percentile = pct.astype(np.float32)

        # ---- EV connectivity look-ahead --------------------------------
        self._precompute_ev_schedule()

        self.feeder_import_kw = self.cfg.feeder_import_kw_per_hh * self.N
        self.feeder_export_kw = self.cfg.feeder_export_kw_per_hh * self.N

        # Live dual variables, written by the trainer each update.  They are
        # part of the observation, not just the reward.
        self.lam = {"ir": 0.0, "grid": 0.0}
        self._carbon = self._carbon_profile()
        self.ev_projection = EVFeasibilityProjection(self.cfg.ev_safety_margin_kwh,
                                                     self.cfg.ev_safety_projection)

        # ---- grid-only counterfactual (the outside option) -------------
        self.baseline_net_kw, self.baseline_cost_step = self._grid_only_reference()
        self.baseline_cost_total = self.baseline_cost_step.sum(axis=0)      # (N,)
        # A household whose outside option is ~0 EUR (tiny load, large PV
        # export credit) would otherwise produce an unbounded burden ratio.
        self._burden_denom = np.maximum(np.abs(self.baseline_cost_total), 0.25)

        self.reset(seed)

    # ------------------------------------------------------------------
    # pre-computation
    # ------------------------------------------------------------------
    def set_multipliers(self, lam: Dict[str, float]) -> None:
        """Publish the current dual variables to the observation."""
        self.lam = {k: float(v) for k, v in lam.items()}

    def _carbon_profile(self) -> np.ndarray:
        """(T,) gCO2/kWh of grid electricity, inferred from the tariff shape.

        The dataset carries no emissions series.  Rather than invent one, we use
        the (observable) fact that in a tariff-following system the expensive
        hours are the thermally-dispatched ones: carbon intensity is taken as an
        affine function of the within-day price percentile, spanning a range
        typical of a European grid (150-500 gCO2/kWh).  This is a *proxy*, and
        it is labelled as one wherever it is reported; it is included because
        emissions are a standard KPI in the 2025-26 literature and omitting them
        entirely is worse than reporting a clearly-stated proxy.
        """
        return (150.0 + 350.0 * self._price_percentile).astype(np.float32)

    def _precompute_ev_schedule(self):
        """Two-phase EV requirements, resolved per timestep.

        Before departure the binding requirement is *enough energy for the
        trip*; after the vehicle returns it is *back to the overnight level by
        the end of the day*.  Both are expressed as ``(2, T, N)`` arrays of an
        active target and the number of connected steps that remain before the
        corresponding deadline, so the feasibility projection and the
        observation never have to branch on the phase.
        """
        ct = self.ct
        T, N = self.n_steps, self.N
        dep_target = getattr(ct, "ev_departure_target", ct.ev_target)
        fin_target = getattr(ct, "ev_final_target", ct.ev_target)
        home = ct.ev_at_home                                     # (2, T, N)
        suffix = np.cumsum(home[:, ::-1, :], axis=1)[:, ::-1, :]  # at-home steps from t to end
        target = np.zeros((2, T, N), dtype=np.float32)
        remaining = np.zeros((2, T, N), dtype=np.float32)
        idx = np.arange(N)
        for j in range(2):
            dep = np.clip(ct.ev_departure[j], 0, T - 1)
            for t in range(T):
                pre = t <= dep
                target[j, t] = np.where(pre, dep_target[j], fin_target[j])
                after_dep = suffix[j, dep, idx]
                remaining[j, t] = np.where(pre,
                                           np.maximum(suffix[j, t] - after_dep, 0.0),
                                           suffix[j, t])
        self._ev_target_t = target
        self._home_remaining = remaining.astype(np.float32)

    def _dumb_ev_schedule(self) -> np.ndarray:
        """(T, N) kW drawn by uncontrolled EV charging.

        The reference behaviour of an unmanaged household: on arrival, charge at
        maximum power until the departure target is met.  This is the honest
        outside option for an EV owner and it is what the P2P community must
        beat in order to claim individual rationality.
        """
        ct = self.ct
        power = np.zeros((self.n_steps, self.N), dtype=np.float32)
        for j in range(2):
            soc = ct.ev_soc0[j].copy()
            eta = np.sqrt(ct.ev_eta[j])
            cap = ct.ev_capacity[j]
            for t in range(self.n_steps):
                home = ct.ev_at_home[j, t] > 0.5
                need = np.maximum(self._ev_target_t[j, t] - soc, 0.0)
                room = np.maximum(cap - soc, 0.0)
                # Full power until the target is reached, then stop.  Real AC
                # charge points do not taper to land exactly on a setpoint, and
                # matching the bang-bang behaviour exactly is what lets the
                # NoFlexibility policy reproduce this counterfactual inside the
                # simulator (asserted in tests/test_environment.py).
                p = np.where(home & (need > 1e-9),
                             np.minimum(ct.ev_charge_max[j], room / (self.dt * eta)), 0.0)
                soc = soc + p * self.dt * eta
                power[t] += p
                dep = ct.ev_departure[j] == t
                soc = np.where(dep, np.maximum(soc - ct.ev_trip_kwh[j], 0.0), soc)
        return power

    def _grid_only_reference(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cost of the unmanaged household, obtained by *simulating* it.

        The counterfactual is produced by running an uncontrolled-charging
        policy through this very ``step`` function, so it is settled under
        exactly the physics -- feeder curtailment, scarcity pricing, conversion
        losses -- that the learned policy faces.  Any hand-written parallel
        formula, however carefully derived, can drift away from the simulator
        as the model changes; this cannot.  ``tests/test_all.py`` asserts that
        the NoFlexibility policy reproduces it to 1e-4.
        """
        return self._simulate_uncontrolled()

    def _simulate_uncontrolled(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cost of the unmanaged household, settled under the *same* physics.

        Crucially the counterfactual is passed through the same feeder
        constraint and the same scarcity pricing as the controlled system.
        Computing it against an implicit infinite grid, as the legacy code did,
        makes the baseline cheaper than it can physically be and understates
        every saving attributed to coordination.  The test suite asserts that
        running :class:`NoFlexibility` inside the simulator reproduces this
        array exactly, so the counterfactual is a checked invariant rather than
        a parallel implementation that can silently drift.
        """
        from dataclasses import replace as _replace
        ct = self.ct
        shadow = object.__new__(P2PCommunityEnv)
        shadow.__dict__.update(self.__dict__)
        shadow.cfg = _replace(self.cfg, n_envs=1, noise_std=0.0, residual=False)
        shadow.E = 1
        shadow.baseline_cost_step = np.zeros((self.n_steps, self.N), np.float32)
        shadow.baseline_cost_total = np.zeros(self.N, np.float32)
        shadow._burden_denom = np.ones(self.N, np.float32)
        shadow.reset(seed=0, deterministic=True)
        costs = np.zeros((self.n_steps, self.N), np.float32)
        net = np.zeros((self.n_steps, self.N), np.float32)
        for t in range(self.n_steps):
            a = np.zeros((1, self.N, ACTION_DIM), np.float32)
            # Uncontrolled charging: full power whenever connected and short of
            # the active target.  No storage dispatch, no market participation.
            for j, slot in enumerate((A_EV1, A_EV2)):
                home = ct.ev_at_home[j, t][None, :] > 0.5
                deficit = shadow._ev_target_t[j, t][None, :] - shadow.ev_soc[j]
                a[:, :, slot] = np.where(home & (deficit > 1e-9), 1.0, 0.0)
            a[:, :, A_PRICE] = 0.5
            prev = shadow.cum_cost.copy()
            net[t] = (shadow.load[:, t, :] - shadow.pv[:, t, :])[0]
            shadow.step(a)
            costs[t] = (shadow.cum_cost - prev)[0]
        return net, costs

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, deterministic: bool = False):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ct = self.ct
        E, N, T = self.E, self.N, self.n_steps
        self.t = 0

        # Forecast/measurement noise: multiplicative, per episode, so that the
        # E parallel episodes are genuinely distinct draws rather than E copies
        # of the same deterministic day.
        if deterministic or self.cfg.noise_std <= 0:
            lf = np.ones((E, 1, 1), dtype=np.float32)
            pf = np.ones((E, 1, 1), dtype=np.float32)
        else:
            s = self.cfg.noise_std
            lf = np.clip(self.rng.normal(1.0, s, (E, 1, N)), 0.7, 1.3).astype(np.float32)
            pf = np.clip(self.rng.normal(1.0, s * 2, (E, 1, N)), 0.5, 1.5).astype(np.float32)
        self.load = (ct.load_kw[None] * lf).astype(np.float32)        # (E, T, N)
        self.pv = (ct.pv_kw[None] * pf).astype(np.float32)

        self.bess_soc = np.tile(ct.bess_soc0, (E, 1)).astype(np.float32)      # (E, N)
        self.ev_soc = np.tile(ct.ev_soc0[:, None, :], (1, E, 1)).astype(np.float32)  # (2,E,N)
        self.ev_trip_taken = np.zeros((2, E, N), dtype=np.float32)

        self.cum_cost = np.zeros((E, N), dtype=np.float64)
        self.cum_baseline = np.zeros((E, N), dtype=np.float64)
        self.last_price = np.full(E, float(ct.sell_price.mean()), dtype=np.float32)
        self.last_volume = np.zeros(E, dtype=np.float32)
        self.last_congestion = np.zeros(E, dtype=np.float32)
        self.last_net_community = np.zeros(E, dtype=np.float32)
        self.cum_matched = np.zeros(E, dtype=np.float64)
        self.cum_feasible = np.zeros(E, dtype=np.float64)
        self._ref_cache = None

        self.metrics = {k: np.zeros(E) for k in (
            "grid_import_kwh", "grid_export_kwh", "p2p_volume_kwh", "p2p_feasible_kwh",
            "grid_cost_eur", "p2p_cost_eur", "p2p_revenue_eur", "self_consumed_kwh",
            "pv_to_peers_kwh", "pv_total_kwh", "bess_throughput_kwh",
            "unserved_kwh", "curtailed_kwh", "congestion_steps", "ev_shortfall_kwh",
            "peak_import_kw", "degradation_eur", "emissions_kg",
        )}
        return self._observe(), self._global_state()

    # ------------------------------------------------------------------
    # rule-based reference controller
    # ------------------------------------------------------------------
    def reference_actions(self) -> np.ndarray:
        """(E, N, 5) tariff-aware solar-first controller.

        This is simultaneously (a) the deployable non-learned baseline, (b) the
        centre of the residual policy's trust region, and (c) the fallback if a
        learned policy fails acceptance testing.  Keeping one implementation for
        all three roles removes the inconsistency in the legacy code, where the
        baseline policy and the residual reference used slightly different
        thresholds and were therefore not the same controller.
        """
        if self._ref_cache is not None:
            return self._ref_cache
        t = min(self.t, self.n_steps - 1)
        ct = self.ct
        load, pv = self.load[:, t, :], self.pv[:, t, :]
        net = load - pv                                            # (E, N)
        buy = ct.buy_price[t][None, :]
        sell = ct.sell_price[t][None, :]
        pct = self._price_percentile[t]

        soc_frac = self.bess_soc / ct.bess_capacity[None, :]
        headroom = np.maximum(self.cfg.bess_max_soc - soc_frac, 0.0)
        reserve = np.maximum(soc_frac - self.cfg.bess_min_soc, 0.0)

        a = np.zeros((self.E, self.N, ACTION_DIM), dtype=np.float32)
        # Battery: absorb surplus first; discharge only into expensive hours.
        surplus = np.maximum(-net, 0.0)
        deficit = np.maximum(net, 0.0)
        charge_cmd = np.minimum(surplus / np.maximum(ct.bess_charge_max[None, :], 1e-6), 1.0)
        charge_cmd = np.where(headroom > 0.01, charge_cmd, 0.0)
        discharge_cmd = np.minimum(deficit / np.maximum(ct.bess_discharge_max[None, :], 1e-6), 1.0)
        expensive = pct > 0.6
        discharge_cmd = np.where((reserve > 0.02) & expensive, -discharge_cmd, 0.0)
        a[:, :, A_BESS] = np.where(surplus > 1e-6, charge_cmd, discharge_cmd)

        # EVs: least-laxity charging, modulated by the tariff.
        #
        # "Charge at full power whenever connected and cheap" synchronises the
        # whole fleet onto the same cheap hours and is the single largest cause
        # of feeder congestion in this dataset.  Instead the setpoint tracks
        # *urgency* -- the fraction of the remaining charging capability that
        # the vehicle actually needs -- so vehicles with slack charge gently and
        # vehicles near their deadline charge hard.  The tariff term shifts
        # energy earlier when power is cheap without ever synchronising the
        # fleet, because urgency differs across households by construction.
        #
        # This is the classic Least-Laxity-First rule from deadline scheduling,
        # applied to charging.
        for j, slot in enumerate((A_EV1, A_EV2)):
            home = ct.ev_at_home[j, t][None, :] > 0.5
            deficit_ev = np.maximum(self._ev_target_t[j, t][None, :] - self.ev_soc[j], 0.0)
            capability = np.maximum(
                self._home_remaining[j, t][None, :] * ct.ev_charge_max[j][None, :]
                * np.sqrt(ct.ev_eta[j])[None, :] * self.dt, 1e-6)
            urgency = deficit_ev / capability
            setpoint = np.clip(urgency * (2.0 - 2.0 * pct), 0.0, 1.0)
            setpoint = np.maximum(setpoint, np.where(surplus > 1e-6, 0.5, 0.0))
            a[:, :, slot] = np.where(home & (deficit_ev > 1e-6), setpoint, 0.0)

        # P2P: always offer the full residual imbalance to the market and quote
        # just inside the tariff spread on the profitable side.
        a[:, :, A_QTY] = np.where(np.abs(net) > 1e-6, 1.0, 0.0)
        a[:, :, A_PRICE] = np.where(net < 0, 0.25, 0.75)
        self._ref_cache = a
        return a

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """Advance one 15-minute interval.

        ``action`` is ``(E, N, 5)`` with the first four entries in [-1, 1] and
        the last in [0, 1].  When ``cfg.residual`` is set the action is
        interpreted as a bounded correction to :meth:`reference_actions`.
        """
        cfg, ct = self.cfg, self.ct
        E, N = self.E, self.N
        t = self.t
        action = np.asarray(action, dtype=np.float32).reshape(E, N, ACTION_DIM)

        if cfg.residual:
            ref = self.reference_actions()
            a = ref.copy()
            a[:, :, :4] = np.clip(ref[:, :, :4] + cfg.residual_scale * np.clip(action[:, :, :4], -1, 1), -1, 1)
            a[:, :, 4] = np.clip(ref[:, :, 4] + cfg.residual_price_scale * (np.clip(action[:, :, 4], 0, 1) - 0.5), 0, 1)
        else:
            a = action.copy()
            a[:, :, :4] = np.clip(a[:, :, :4], -1, 1)
            a[:, :, 4] = np.clip(a[:, :, 4], 0, 1)

        # ---------------- desired device dispatch ---------------------
        # Powers are computed first *without* committing state, because the
        # feeder may curtail them before they are actually delivered.
        cap = ct.bess_capacity[None, :]
        eta = np.sqrt(ct.bess_eta)[None, :]
        cmd = a[:, :, A_BESS]
        want_kw = np.where(cmd >= 0, cmd * ct.bess_charge_max[None, :],
                           cmd * ct.bess_discharge_max[None, :])
        room_kwh = np.maximum(cfg.bess_max_soc * cap - self.bess_soc, 0.0)
        avail_kwh = np.maximum(self.bess_soc - cfg.bess_min_soc * cap, 0.0)
        b_charge = np.minimum(np.maximum(want_kw, 0.0), room_kwh / (self.dt * eta))
        b_discharge = np.minimum(np.maximum(-want_kw, 0.0), avail_kwh * eta / self.dt)

        ev_cmd = np.stack([a[:, :, A_EV1], a[:, :, A_EV2]])                  # (2,E,N)
        ev_cmd, must_charge = self.ev_projection(ev_cmd, self.ev_soc, self)
        e_charge = np.zeros((2, E, N), np.float32)
        e_discharge = np.zeros((2, E, N), np.float32)
        e_required = np.zeros((2, E, N), np.float32)
        if cfg.include_evs:
            for j in range(2):
                home = (ct.ev_at_home[j, t] > 0.5)[None, :]
                ee = np.sqrt(ct.ev_eta[j])[None, :]
                capj = ct.ev_capacity[j][None, :]
                v2g_cap = np.minimum(ct.ev_discharge_max[j], ct.ev_v2g_max[j, t])[None, :]
                want = np.where(ev_cmd[j] >= 0, ev_cmd[j] * ct.ev_charge_max[j][None, :],
                                ev_cmd[j] * v2g_cap) * home
                room = np.maximum(capj - self.ev_soc[j], 0.0)
                floor = np.minimum(self._ev_target_t[j, t][None, :], capj)
                availj = np.maximum(self.ev_soc[j] - floor, 0.0)
                e_charge[j] = np.minimum(np.maximum(want, 0.0), room / (self.dt * ee))
                e_discharge[j] = np.minimum(np.maximum(-want, 0.0), availj * ee / self.dt)
                # Charging demanded by the feasibility shield is *firm*: it is
                # the minimum that still meets a mobility deadline and must not
                # be curtailed by the feeder controller.
                e_required[j] = np.where(must_charge[j], e_charge[j], 0.0)

        # ---------------- feeder-level flexibility curtailment --------
        # A distribution operator facing an overloaded feeder curtails flexible
        # load before it sheds firm demand.  Modelling the constraint purely as
        # an accounting penalty -- letting the battery charge anyway and then
        # recording the excess as "unserved" -- does not conserve energy: the
        # stored kWh appear from nowhere.  Here the constraint acts on the
        # dispatch itself, pro rata across curtailable charging, and only the
        # residue that no flexibility can absorb is counted as unserved.
        load, pv_avail = self.load[:, t, :], self.pv[:, t, :]
        pv = pv_avail.copy()
        net_kw = (load + (b_charge - b_discharge)
                  + (e_charge - e_discharge).sum(axis=0) - pv).sum(axis=1)      # (E,)

        over_import = np.maximum(net_kw - self.feeder_import_kw, 0.0)
        flex_pool = (b_charge.sum(axis=1)
                     + np.maximum(e_charge - e_required, 0.0).sum(axis=(0, 2)))
        s_imp = np.clip(1.0 - np.minimum(1.0, over_import / np.maximum(flex_pool, 1e-9)),
                        0.0, 1.0)[:, None]
        b_charge = b_charge * s_imp
        e_charge = e_required + (e_charge - e_required) * s_imp[None, :, :]

        over_export = np.maximum(-net_kw - self.feeder_export_kw, 0.0)
        disc_pool = b_discharge.sum(axis=1) + e_discharge.sum(axis=(0, 2))
        s_exp = np.clip(1.0 - np.minimum(1.0, over_export / np.maximum(disc_pool, 1e-9)),
                        0.0, 1.0)[:, None]
        b_discharge = b_discharge * s_exp
        e_discharge = e_discharge * s_exp[None, :, :]

        # Export headroom still exceeded is met by curtailing PV, which is what
        # an inverter under export limitation actually does.
        net_kw = (load + (b_charge - b_discharge)
                  + (e_charge - e_discharge).sum(axis=0) - pv).sum(axis=1)
        residual_export = np.maximum(-net_kw - self.feeder_export_kw, 0.0)
        pv_cut = np.minimum(residual_export / np.maximum(pv.sum(axis=1), 1e-9), 1.0)[:, None]
        curtailed_kwh = pv * pv_cut * self.dt
        pv = pv * (1.0 - pv_cut)

        # ---------------- commit device state -------------------------
        bess_kw = b_charge - b_discharge
        self.bess_soc = np.clip(
            self.bess_soc + b_charge * self.dt * eta - b_discharge * self.dt / eta, 0.0, cap)
        throughput = (b_charge + b_discharge) * self.dt
        degradation_eur = cfg.bess_degradation_eur_per_kwh * throughput

        ev_kw = np.zeros((E, N), dtype=np.float32)
        ev_shortfall = np.zeros((E, N), dtype=np.float32)
        if cfg.include_evs:
            for j in range(2):
                ee = np.sqrt(ct.ev_eta[j])[None, :]
                capj = ct.ev_capacity[j][None, :]
                self.ev_soc[j] = np.clip(
                    self.ev_soc[j] + e_charge[j] * self.dt * ee - e_discharge[j] * self.dt / ee,
                    0.0, capj)
                ev_kw += e_charge[j] - e_discharge[j]
                deadline = ((ct.ev_departure[j] == t) | (t == self.n_steps - 1))[None, :]
                ev_shortfall += np.where(
                    deadline, np.maximum(self._ev_target_t[j, t][None, :] - self.ev_soc[j], 0.0), 0.0)
                departing = (ct.ev_departure[j] == t)[None, :]
                trip = np.where(departing, ct.ev_trip_kwh[j][None, :], 0.0)
                self.ev_trip_taken[j] += trip
                self.ev_soc[j] = np.maximum(self.ev_soc[j] - trip, 0.0)

        # ---------------- residual imbalance and P2P market -----------
        residual_kw = load + bess_kw + ev_kw - pv                            # (E,N)
        need_kwh = np.maximum(residual_kw, 0.0) * self.dt
        surp_kwh = np.maximum(-residual_kw, 0.0) * self.dt

        qty_frac, quote = project_p2p_quotes(
            a[:, :, A_QTY], a[:, :, A_PRICE],
            ct.buy_price[t][None, :], ct.sell_price[t][None, :])
        bid_q = np.minimum(need_kwh, qty_frac * ct.buy_limit_kw[None, :] * self.dt)
        ask_q = np.minimum(surp_kwh, qty_frac * ct.sell_limit_kw[None, :] * self.dt)
        mk = clear_double_auction(bid_q, quote, ask_q, quote,
                                  np.full(E, float(ct.sell_price[t].mean())),
                                  cfg.market_mechanism)

        # ---------------- grid settlement ------------------------------
        # Net community exchange equals the sum of household residuals, because
        # peer trades cancel inside the community.  Anything still above the
        # feeder limit after flexibility curtailment is firm demand the network
        # cannot deliver: recorded as unserved and priced at a scarcity multiple.
        imp_kwh = np.maximum(need_kwh - mk.buy_kwh, 0.0)
        exp_kwh = np.maximum(surp_kwh - mk.sell_kwh, 0.0)
        net_exchange_kw = (imp_kwh.sum(axis=1) - exp_kwh.sum(axis=1)) / self.dt
        over = np.maximum(net_exchange_kw - self.feeder_import_kw, 0.0)
        share = imp_kwh / np.maximum(imp_kwh.sum(axis=1, keepdims=True), 1e-9)
        unserved = np.minimum(share * (over * self.dt)[:, None], imp_kwh)
        imp_kwh = imp_kwh - unserved
        congestion = np.maximum(
            net_exchange_kw / max(self.feeder_import_kw, 1e-8),
            -net_exchange_kw / max(self.feeder_export_kw, 1e-8)).astype(np.float32)

        buy_p = ct.buy_price[t][None, :]
        sell_p = ct.sell_price[t][None, :]
        grid_cost = imp_kwh * buy_p - exp_kwh * sell_p
        p2p_cost = mk.buy_kwh * mk.price[:, None] * (1 + cfg.p2p_fee)
        p2p_rev = mk.sell_kwh * mk.price[:, None] * (1 - cfg.p2p_fee)
        # Unserved energy is not free: the household still has to meet the load,
        # so it is charged at the grid tariff plus a scarcity multiple.
        actual_cost = (grid_cost + p2p_cost - p2p_rev + degradation_eur
                       + unserved * buy_p * cfg.unserved_price_multiple)

        base_step = self.baseline_cost_step[t][None, :]
        self.cum_cost += actual_cost
        self.cum_baseline += base_step

        # ---------------- reward components ---------------------------
        scale = ct.cost_scale[None, :]
        econ = ((base_step - actual_cost) / scale).astype(np.float32)
        team = float(0.0)
        team_vec = ((self.cum_baseline.sum(axis=1) - self.cum_cost.sum(axis=1))
                    / np.maximum(np.abs(self.baseline_cost_total.sum()), 1.0)).astype(np.float32)
        team = np.repeat(team_vec[:, None], N, axis=1)

        # Potential-based storage valuation.  Phi is the stored energy expressed
        # in units of the household's own typical per-step throughput; the
        # shaping term gamma*Phi' - Phi therefore rewards *keeping* energy that
        # will be worth more later without changing the ranking of policies
        # within a step (Ng et al., 1999).  At the terminal step Phi is retained
        # rather than forced to zero: end-of-day stored energy has genuine value
        # and zeroing it makes the policy dump the battery at 23:45.
        stored = (self.bess_soc + (self.ev_soc + self.ev_trip_taken).sum(axis=0))
        phi_new = (stored / (ct.energy_scale[None, :] * 20.0)).astype(np.float32)
        phi_old = getattr(self, "_phi", phi_new)
        shaping = (cfg.gamma * phi_new - phi_old).astype(np.float32)
        self._phi = phi_new

        ev_pen = (ev_shortfall / np.maximum(ct.ev_capacity.sum(axis=0)[None, :], 1e-3)).astype(np.float32)

        # Constraint costs, returned separately so the learner can price them.
        c_ir = (np.maximum(actual_cost - base_step, 0.0) / scale).astype(np.float32)
        # Reported in kWh per household per step.  An earlier version divided by
        # the total community energy scale, which made a 27 kWh/day violation
        # register as a constraint cost of 0.018 -- so small that the dual
        # variable never left zero and the constraint was in practice absent.
        # A constraint cost must live on a scale where a violation that matters
        # operationally is a number that matters numerically.
        c_grid = ((unserved + curtailed_kwh).sum(axis=1) / self.N).astype(np.float32)
        c_grid = np.repeat(c_grid[:, None], N, axis=1)

        # ---------------- bookkeeping ---------------------------------
        m = self.metrics
        m["grid_import_kwh"] += imp_kwh.sum(axis=1)
        m["grid_export_kwh"] += exp_kwh.sum(axis=1)
        m["p2p_volume_kwh"] += mk.volume_kwh
        m["p2p_feasible_kwh"] += mk.feasible_kwh
        m["grid_cost_eur"] += grid_cost.sum(axis=1)
        m["p2p_cost_eur"] += p2p_cost.sum(axis=1)
        m["p2p_revenue_eur"] += p2p_rev.sum(axis=1)
        self_cons = np.minimum(pv, load + np.maximum(bess_kw, 0) + np.maximum(ev_kw, 0)) * self.dt
        m["self_consumed_kwh"] += self_cons.sum(axis=1)
        m["pv_to_peers_kwh"] += np.minimum(mk.sell_kwh, np.maximum(pv - load, 0) * self.dt).sum(axis=1)
        m["pv_total_kwh"] += (pv_avail * self.dt).sum(axis=1)
        m["bess_throughput_kwh"] += throughput.sum(axis=1)
        m["unserved_kwh"] += unserved.sum(axis=1)
        m["curtailed_kwh"] += curtailed_kwh.sum(axis=1)
        m["congestion_steps"] += (congestion > 1.0).astype(np.float64)
        m["ev_shortfall_kwh"] += ev_shortfall.sum(axis=1)
        m["degradation_eur"] += degradation_eur.sum(axis=1)
        m["emissions_kg"] += imp_kwh.sum(axis=1) * self._carbon[t] / 1000.0
        # The feeder sees the *net* community exchange, since peer trades
        # cancel internally; that is the quantity the limit applies to.
        m["peak_import_kw"] = np.maximum(
            m["peak_import_kw"], (imp_kwh.sum(axis=1) - exp_kwh.sum(axis=1)) / self.dt)
        self.cum_matched += mk.volume_kwh
        self.cum_feasible += mk.feasible_kwh

        self.last_price = mk.price.astype(np.float32)
        self.last_volume = mk.volume_kwh.astype(np.float32)
        self.last_congestion = congestion
        self.last_net_community = residual_kw.sum(axis=1).astype(np.float32)

        self.t += 1
        done = self.t >= self.n_steps
        self._ref_cache = None

        terminal = np.zeros((E, N), dtype=np.float32)
        if done:
            fair = self.fairness_jain()
            ir = self.ir_fraction()
            bonus = (cfg.w_terminal_fairness * (fair - 0.75) +
                     cfg.w_terminal_ir * (ir - 0.75)).astype(np.float32)
            terminal += bonus[:, None]

        components = {
            "econ": econ, "team": team, "shaping": shaping,
            "ev_shortfall": ev_pen, "terminal": terminal,
            "c_ir": c_ir, "c_grid": c_grid,
        }
        reward = (cfg.w_econ * econ + cfg.w_team * team + cfg.w_shape * shaping
                  - cfg.w_ev_shortfall * ev_pen + terminal)

        info = {"components": components,
                "clearing_price": mk.price, "volume_kwh": mk.volume_kwh,
                "congestion": congestion}
        if done:
            info["episode"] = self.episode_summary()
        return self._observe(), self._global_state(), reward.astype(np.float32), done, info

    # ------------------------------------------------------------------
    # derived statistics
    # ------------------------------------------------------------------
    def burden(self) -> np.ndarray:
        """(E, N) cumulative cost as a multiple of the grid-only outside option."""
        denom = self._burden_denom[None, :]
        base = self.cum_baseline
        # Shift both sides by the (constant) worst-case negative baseline so the
        # ratio is well behaved for net-exporting households whose outside
        # option is a *credit* rather than a cost.
        offset = np.maximum(-base, 0.0) + denom
        return ((self.cum_cost + offset) / (base + offset)).astype(np.float32)

    def fairness_jain(self) -> np.ndarray:
        b = np.clip(self.burden(), 0.0, 4.0)
        return (b.sum(axis=1) ** 2 / (self.N * np.square(b).sum(axis=1) + 1e-9)).astype(np.float32)

    def ir_fraction(self) -> np.ndarray:
        """Fraction of households no worse off than the grid-only counterfactual."""
        tol = 1e-6 + 0.01 * np.abs(self.cum_baseline)
        return (self.cum_cost <= self.cum_baseline + tol).mean(axis=1).astype(np.float32)

    def episode_summary(self) -> Dict[str, np.ndarray]:
        m = self.metrics
        net_cost = self.cum_cost.sum(axis=1)
        base_cost = self.cum_baseline.sum(axis=1)
        return {
            "net_cost_eur": net_cost,
            "baseline_cost_eur": base_cost,
            "cost_saving_eur": base_cost - net_cost,
            "cost_saving_pct": 100.0 * (base_cost - net_cost) / np.maximum(np.abs(base_cost), 1e-6),
            "fairness_jain": self.fairness_jain(),
            "individual_rationality": self.ir_fraction(),
            "p2p_volume_kwh": m["p2p_volume_kwh"],
            "market_utilisation": m["p2p_volume_kwh"] / np.maximum(m["p2p_feasible_kwh"], 1e-6),
            "self_sufficiency": (m["self_consumed_kwh"] + m["pv_to_peers_kwh"]) /
                                np.maximum(m["pv_total_kwh"], 1e-6),
            "grid_import_kwh": m["grid_import_kwh"],
            "peak_import_kw": m["peak_import_kw"],
            "unserved_kwh": m["unserved_kwh"],
            "curtailed_kwh": m["curtailed_kwh"],
            "congestion_steps": m["congestion_steps"],
            "ev_shortfall_kwh": m["ev_shortfall_kwh"],
            "bess_throughput_kwh": m["bess_throughput_kwh"],
            "emissions_kg": m["emissions_kg"],
            "worst_burden": np.clip(self.burden(), 0, 4).max(axis=1),
        }

    # ------------------------------------------------------------------
    # helpers used by the safety projection
    # ------------------------------------------------------------------
    @property
    def ev_departure(self):
        """Active deadline: the departure period before it, the horizon after."""
        t = min(self.t, self.n_steps - 1)
        dep = self.ct.ev_departure
        return np.where(dep >= t, dep, self.n_steps - 1)

    @property
    def ev_target(self):
        """Active (phase-dependent) EV target at the current timestep."""
        return self._ev_target_t[:, min(self.t, self.n_steps - 1), :]

    @property
    def ev_charge_max(self):
        return self.ct.ev_charge_max

    @property
    def ev_eta(self):
        return self.ct.ev_eta

    @property
    def ev_at_home(self):
        return self.ct.ev_at_home

    def ev_home_remaining(self, t: int) -> np.ndarray:
        return self._home_remaining[:, min(t, self.n_steps - 1), :]

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        ct = self.ct
        E, N = self.E, self.N
        t = min(self.t, self.n_steps - 1)
        es = ct.energy_scale[None, :] / self.dt          # kW scale per household

        load, pv = self.load[:, t, :], self.pv[:, t, :]
        net = load - pv
        cap = ct.bess_capacity[None, :]
        soc_frac = self.bess_soc / cap

        cols = [
            np.clip(load / es, 0, 8), np.clip(pv / es, 0, 8), np.clip(net / es, -8, 8),
            soc_frac,
            np.clip(self.cfg.bess_max_soc - soc_frac, 0, 1),
            np.clip(soc_frac - self.cfg.bess_min_soc, 0, 1),
            np.tile(ct.bess_charge_max / np.maximum(ct.energy_scale / self.dt, 1e-3), (E, 1)).clip(0, 10),
        ]
        for j in range(2):
            capj = ct.ev_capacity[j][None, :]
            soc = self.ev_soc[j]
            home = ct.ev_at_home[j, t][None, :]
            deficit = np.maximum(self._ev_target_t[j, t][None, :] - soc, 0.0)
            rem = self._home_remaining[j, t][None, :]
            gain = np.maximum(rem * ct.ev_charge_max[j][None, :] * np.sqrt(ct.ev_eta[j])[None, :] * self.dt, 1e-6)
            cols += [
                soc / capj,
                np.tile(home, (E, 1)),
                deficit / capj,
                np.tile(rem / self.n_steps, (E, 1)),
                np.clip(deficit / gain, 0, 2),          # urgency: 1.0 == must charge now
            ]
        buy_z = (ct.buy_price[t].mean() - self._buy_mu) / self._buy_sd
        sell_z = (ct.sell_price[t].mean() - self._sell_mu) / self._sell_sd
        spread = (ct.buy_price[t] - ct.sell_price[t])[None, :] / max(self._buy_mu, 1e-6)
        cols += [
            np.full((E, N), buy_z, np.float32), np.full((E, N), sell_z, np.float32),
            np.full((E, N), self._price_percentile[t], np.float32),
            np.tile(spread, (E, 1)),
        ]
        cols += [
            np.repeat(((self.last_price - self._buy_mu) / self._buy_sd)[:, None], N, 1),
            np.repeat((self.last_volume / max(ct.energy_scale.sum(), 1e-3))[:, None], N, 1).clip(0, 5),
            np.repeat((self.last_net_community / max(ct.energy_scale.sum() / self.dt, 1e-3))[:, None], N, 1).clip(-5, 5),
            np.repeat(np.minimum(self.last_congestion, 3.0)[:, None], N, 1),
        ]
        phase = 2 * np.pi * t / self.n_steps
        cols += [
            np.full((E, N), np.sin(phase), np.float32), np.full((E, N), np.cos(phase), np.float32),
            np.full((E, N), 1.0 - t / self.n_steps, np.float32),
        ]
        burden = np.clip(self.burden(), 0, 3)
        cols += [
            burden,
            np.clip(self.cum_cost / np.maximum(self._burden_denom[None, :], 1e-3), -3, 3).astype(np.float32),
            np.repeat(burden.mean(axis=1)[:, None], N, 1),
            np.clip((self.cum_baseline - self.cum_cost) / np.maximum(self._burden_denom[None, :], 1e-3), -3, 3).astype(np.float32),
        ]
        for h in (1, 4, 12):
            k = min(t + h, self.n_steps - 1)
            cols.append(np.clip((self.load[:, k, :] - self.pv[:, k, :]) / es, -8, 8))
        for h in (1, 4, 12):
            k = min(t + h, self.n_steps - 1)
            cols.append(np.full((E, N), (ct.buy_price[k].mean() - self._buy_mu) / self._buy_sd, np.float32))

        ref = self.reference_actions()
        for i in range(ACTION_DIM):
            cols.append(ref[:, :, i])

        lam_scale = max(self.cfg.lam_max_obs, 1e-6)
        if self.cfg.state_augmented:
            li = min(self.lam.get("ir", 0.0), lam_scale) / lam_scale
            lg = min(self.lam.get("grid", 0.0), lam_scale) / lam_scale
        else:
            li = lg = 0.0
        cols.append(np.full((E, N), li, np.float32))
        cols.append(np.full((E, N), lg, np.float32))

        obs = np.stack(cols, axis=-1).astype(np.float32)
        assert obs.shape == (E, N, OBS_DIM), f"{obs.shape} != {(E, N, OBS_DIM)}"
        return np.nan_to_num(obs, nan=0.0, posinf=8.0, neginf=-8.0)

    def _global_state(self) -> np.ndarray:
        ct = self.ct
        t = min(self.t, self.n_steps - 1)
        E = self.E
        load = self.load[:, t, :].sum(axis=1)
        pv = self.pv[:, t, :].sum(axis=1)
        tot_scale = max(ct.energy_scale.sum() / self.dt, 1e-3)
        burden = np.clip(self.burden(), 0, 3)
        s = np.stack([
            load / tot_scale, pv / tot_scale, (load - pv) / tot_scale,
            (self.bess_soc / ct.bess_capacity[None, :]).mean(axis=1),
            (self.ev_soc / ct.ev_capacity[:, None, :]).mean(axis=(0, 2)),
            np.full(E, (ct.buy_price[t].mean() - self._buy_mu) / self._buy_sd),
            np.full(E, (ct.sell_price[t].mean() - self._sell_mu) / self._sell_sd),
            (self.last_price - self._buy_mu) / self._buy_sd,
            self.last_volume / max(ct.energy_scale.sum(), 1e-3),
            np.minimum(self.last_congestion, 3.0),
            np.full(E, np.sin(2 * np.pi * t / self.n_steps)),
            np.full(E, np.cos(2 * np.pi * t / self.n_steps)),
            np.full(E, 1.0 - t / self.n_steps),
            (self.cum_baseline.sum(1) - self.cum_cost.sum(1)) / max(abs(self.baseline_cost_total.sum()), 1.0),
            self.fairness_jain(), self.ir_fraction(),
            self.cum_matched / np.maximum(self.cum_feasible, 1e-6),
            burden.mean(axis=1), burden.std(axis=1),
            self.metrics["grid_import_kwh"] / max(ct.energy_scale.sum() * self.n_steps, 1e-3),
        ], axis=-1).astype(np.float32)
        assert s.shape == (E, STATE_DIM)
        return np.nan_to_num(s, nan=0.0, posinf=5.0, neginf=-5.0)
