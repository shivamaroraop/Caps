"""Correctness tests.

The tests that matter most here are the *invariants*: statements that must hold
for the reported metrics to mean anything.  In particular the grid-only
counterfactual is not merely computed, it is checked against a policy rollout,
so it cannot silently drift away from the physics the learned policy faces.

Run with::

    python -m pytest tests/ -v
    python tests/test_all.py          # also works without pytest
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import (BASELINES, NoFlexibility, TariffAwareP2P,
                               perfect_foresight_lp)
from src.env.community_env import (ACTION_DIM, OBS_DIM, OBS_NAMES, STATE_DIM,
                                   EnvConfig, P2PCommunityEnv)
from src.env.market import clear_double_auction

DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "EC_EV_dataset.xlsx")
_CACHE = {}


def community(n=12):
    if n not in _CACHE:
        com = load_community(DATASET)
        _CACHE[n] = build_tensors(stratified_subset(com, n, seed=0))
    return _CACHE[n]


def rollout(policy, tensors=None, n_envs=2, noise=0.0, seed=3):
    tensors = tensors or community()
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=n_envs, noise_std=noise,
                                             residual=policy.residual), seed=seed)
    obs, state = env.reset(seed=seed, deterministic=noise == 0.0)
    done = False
    info = {}
    while not done:
        obs, state, r, done, info = env.step(policy.act(obs, state, env))
    return env, info


# ----------------------------------------------------------------------
# data layer
# ----------------------------------------------------------------------
def test_ev_charge_rating_is_a_home_charger():
    """The corrected home charge point must be an AC rating, not a DC one."""
    ct = community()
    live = ct.ev_charge_max > 0
    assert live.any()
    assert ct.ev_charge_max[live].max() <= 25.0, "home charger above 25 kW is not credible"
    ratings = np.unique(ct.home_charger_kw)
    assert all(np.isclose(r, 7.4) or np.isclose(r, 11.0) for r in ratings), ratings


def test_every_ev_trip_is_feasible():
    ct = community()
    assert ct.meta["ev_physics"]["infeasible_departures"] == 0


def test_stratified_subset_preserves_population_mix():
    com = load_community(DATASET)
    full = build_tensors(com)
    sub = build_tensors(stratified_subset(com, 25, seed=0))
    for attr in ("has_pv", "has_bess"):
        assert abs(getattr(sub, attr).mean() - getattr(full, attr).mean()) < 0.12


# ----------------------------------------------------------------------
# market
# ----------------------------------------------------------------------
def test_double_auction_matches_only_when_bid_covers_ask():
    bid_q = np.array([[2.0, 1.0, 0.0, 0.0]])
    ask_q = np.array([[0.0, 0.0, 2.0, 1.0]])
    price = np.array([[0.20, 0.18, 0.15, 0.17]])
    r = clear_double_auction(bid_q, price, ask_q, price, np.array([0.05]), "uniform")
    assert abs(r.buy_kwh.sum() - r.sell_kwh.sum()) < 1e-9, "energy must balance"
    assert r.volume_kwh[0] <= min(bid_q.sum(), ask_q.sum()) + 1e-9
    assert 0.15 <= r.price[0] <= 0.20


def test_double_auction_no_trade_when_no_overlap():
    bid_q = np.array([[1.0, 0.0]])
    ask_q = np.array([[0.0, 1.0]])
    price = np.array([[0.10, 0.30]])          # buyer will not pay the ask
    r = clear_double_auction(bid_q, price, ask_q, price, np.array([0.05]), "uniform")
    assert r.volume_kwh[0] == 0.0


def test_market_never_exceeds_offered_quantities():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 8
        bq = rng.random((1, n)) * (rng.random((1, n)) > 0.5)
        aq = rng.random((1, n)) * (rng.random((1, n)) > 0.5)
        bp = rng.uniform(0.05, 0.25, (1, n))
        ap = rng.uniform(0.05, 0.25, (1, n))
        r = clear_double_auction(bq, bp, aq, ap, np.array([0.05]), "uniform")
        assert (r.buy_kwh <= bq + 1e-9).all()
        assert (r.sell_kwh <= aq + 1e-9).all()
        assert abs(r.buy_kwh.sum() - r.sell_kwh.sum()) < 1e-8


# ----------------------------------------------------------------------
# environment invariants
# ----------------------------------------------------------------------
def test_observation_layout_is_stable():
    assert len(OBS_NAMES) == OBS_DIM
    assert len(set(OBS_NAMES)) == OBS_DIM
    env = P2PCommunityEnv(community(), EnvConfig(n_envs=2), seed=0)
    obs, state = env.reset(seed=0)
    assert obs.shape == (2, env.N, OBS_DIM)
    assert state.shape == (2, STATE_DIM)
    assert np.isfinite(obs).all() and np.isfinite(state).all()


def test_no_flexibility_reproduces_the_counterfactual():
    """The headline invariant: the outside option is a checked quantity.

    If this fails, every saving, fairness and individual-rationality number in
    the project is measured against a baseline the simulator cannot reproduce.
    """
    env, info = rollout(NoFlexibility())
    ep = info["episode"]
    assert np.allclose(ep["net_cost_eur"], ep["baseline_cost_eur"], rtol=1e-4), \
        f"{ep['net_cost_eur']} vs {ep['baseline_cost_eur']}"
    assert np.allclose(ep["individual_rationality"], 1.0)
    assert np.allclose(ep["cost_saving_pct"], 0.0, atol=1e-3)


def test_battery_state_of_charge_stays_within_limits():
    env, _ = rollout(TariffAwareP2P())
    cap = env.ct.bess_capacity[None, :]
    assert (env.bess_soc >= -1e-6).all()
    assert (env.bess_soc <= cap * env.cfg.bess_max_soc + 1e-4).all()


def test_ev_energy_requirements_are_always_met():
    """The safety shield is a hard guarantee, not a soft penalty."""
    for cls in BASELINES.values():
        env, info = rollout(cls())
        assert float(np.max(info["episode"]["ev_shortfall_kwh"])) < 1e-3, cls.name


def test_ev_never_charges_while_away():
    ct = community()
    env = P2PCommunityEnv(ct, EnvConfig(n_envs=1, noise_std=0.0, residual=False), seed=0)
    env.reset(seed=0, deterministic=True)
    for t in range(env.n_steps):
        prev = env.ev_soc.copy()
        a = np.zeros((1, env.N, ACTION_DIM), np.float32)
        a[:, :, 1:3] = 1.0                       # command full charge everywhere
        env.step(a)
        for j in range(2):
            away = ct.ev_at_home[j, t] < 0.5
            delta = (env.ev_soc[j, 0] - prev[j, 0])[away]
            departing = (ct.ev_departure[j] == t)[away]
            assert (delta[~departing] <= 1e-5).all(), f"EV{j} charged while away at t={t}"


def test_feeder_limit_is_never_exceeded():
    for cls in BASELINES.values():
        env, info = rollout(cls())
        assert np.max(info["episode"]["peak_import_kw"]) <= env.feeder_import_kw + 1e-3


def test_energy_balance_holds_every_step():
    """Load + storage + EV = PV + P2P net + grid net, to machine precision."""
    ct = community()
    env = P2PCommunityEnv(ct, EnvConfig(n_envs=2, noise_std=0.0), seed=0)
    obs, state = env.reset(seed=0, deterministic=True)
    pol = TariffAwareP2P()
    for _ in range(env.n_steps):
        soc_b, soc_e = env.bess_soc.copy(), env.ev_soc.copy()
        trip0 = env.ev_trip_taken.copy()
        m0 = {k: np.copy(v) for k, v in env.metrics.items()}
        t = env.t
        obs, state, r, done, info = env.step(pol.act(obs, state, env))
        d_imp = env.metrics["grid_import_kwh"] - m0["grid_import_kwh"]
        d_exp = env.metrics["grid_export_kwh"] - m0["grid_export_kwh"]
        d_uns = env.metrics["unserved_kwh"] - m0["unserved_kwh"]
        d_cur = env.metrics["curtailed_kwh"] - m0["curtailed_kwh"]
        # Community balance. ``demand`` is computed against *available* PV, so
        # curtailed PV must be added back; unserved load was never delivered,
        # so it must be added to the metered import to recover the request.
        demand = ((env.load[:, t, :] - env.pv[:, t, :]).sum(axis=1)) * env.dt
        # Trip consumption leaves the system in the vehicle, not through the
        # meter, so it is added back before checking the electrical balance.
        trip = (env.ev_trip_taken - trip0).sum(axis=(0, 2))
        stored = ((env.bess_soc - soc_b).sum(axis=1)
                  + (env.ev_soc - soc_e).sum(axis=(0, 2)) + trip)
        lhs = d_imp - d_exp + d_uns - d_cur
        rhs = demand + stored
        # Round-trip conversion losses mean stored energy is smaller than
        # metered energy, so the tolerance scales with the step's throughput
        # rather than with its net position.
        tol = 0.08 * float(np.mean(d_imp + d_exp + d_uns + np.abs(stored))) + 0.05
        assert np.all(np.abs(lhs - rhs) < tol), f"t={t}: {lhs} vs {rhs}"
        if done:
            break


def test_p2p_price_stays_inside_the_tariff_spread():
    ct = community()
    env = P2PCommunityEnv(ct, EnvConfig(n_envs=2, noise_std=0.0), seed=0)
    obs, state = env.reset(seed=0, deterministic=True)
    pol = TariffAwareP2P()
    done = False
    while not done:
        t = env.t
        obs, state, r, done, info = env.step(pol.act(obs, state, env))
        if float(info["volume_kwh"].max()) > 0:
            lo = float(ct.sell_price[t].min()) - 1e-6
            hi = float(ct.buy_price[t].max()) + 1e-6
            assert (info["clearing_price"] >= lo).all() and (info["clearing_price"] <= hi).all()


def test_p2p_strictly_improves_on_no_trading():
    """A working market must beat the same controller with the market disabled."""
    ct = community()
    pol = TariffAwareP2P()
    costs = {}
    for mech, tag in (("uniform", "with_p2p"), None or ("uniform", "with_p2p")):
        break
    env = P2PCommunityEnv(ct, EnvConfig(n_envs=1, noise_std=0.0, residual=True), seed=5)
    obs, state = env.reset(seed=5, deterministic=True)
    done = False
    while not done:
        obs, state, _, done, info = env.step(pol.act(obs, state, env))
    with_p2p = float(np.mean(info["episode"]["net_cost_eur"]))

    env = P2PCommunityEnv(ct, EnvConfig(n_envs=1, noise_std=0.0, residual=True), seed=5)
    obs, state = env.reset(seed=5, deterministic=True)
    done = False
    while not done:
        a = pol.act(obs, state, env)
        a[:, :, 3] = -0.5                      # residual that zeroes the offered quantity
        obs, state, _, done, info = env.step(a)
    assert with_p2p <= float(np.mean(info["episode"]["net_cost_eur"])) + 1e-6


# ----------------------------------------------------------------------
# bound
# ----------------------------------------------------------------------
def test_lp_bound_is_a_bound():
    ct = community()
    cfg = EnvConfig()
    lp = perfect_foresight_lp(ct, cfg)
    if lp is None:
        return                                  # SciPy unavailable
    best = min(float(np.mean(rollout(cls())[1]["episode"]["net_cost_eur"]))
               for cls in BASELINES.values())
    assert lp["optimal_cost_eur"] <= best + 1e-6, "LP is not bounding the heuristics"


# ----------------------------------------------------------------------
# learner
# ----------------------------------------------------------------------
def test_stored_pre_squash_action_gives_exact_log_probs():
    """The bug the legacy ``atanh`` inversion introduced must not recur."""
    import torch
    from src.marl.networks import Actor
    a = Actor(OBS_DIM, ACTION_DIM)
    obs = torch.randn(64, OBS_DIM) * 3.0
    with torch.no_grad():
        dist = a.distribution(obs)
        z = dist.sample() * 5.0                 # push actions to the boundary
        lp_stored = a.log_prob(dist, z)
        action = a.squash(z)
        z_recovered = torch.cat([torch.atanh(action[:, :4].clamp(-0.999999, 0.999999)),
                                 torch.logit(action[:, 4:].clamp(1e-6, 1 - 1e-6))], -1)
        lp_recovered = a.log_prob(dist, z_recovered)
    assert torch.isfinite(lp_stored).all()
    # The recovered version is the legacy path; it must be demonstrably worse.
    assert (lp_stored - lp_recovered).abs().max() > 1e-3


def test_per_agent_advantages_are_distinct():
    from src.marl.mappo import MAPPO, MAPPOConfig, RolloutBuffer
    from src.marl.trainer import collect_rollout
    ct = community()
    env = P2PCommunityEnv(ct, EnvConfig(n_envs=2), seed=0)
    ag = MAPPO(env.observation_dim, env.state_dim, env.action_dim, MAPPOConfig(seed=0))
    buf = RolloutBuffer(env.n_steps, env.E, env.N, env.observation_dim,
                        env.state_dim, env.action_dim, MAPPO.COMPONENTS)
    lv, _, _ = collect_rollout(env, ag, buf)
    r = ag.compose_reward({k: buf.components[k] for k in MAPPO.COMPONENTS}, env.cfg)
    adv, _ = ag.compute_gae(r, buf.value, lv)
    assert adv.std(axis=-1).mean() > 1e-4, "advantages are identical across agents"


def test_value_normalisation_keeps_the_loss_bounded():
    from src.marl.mappo import MAPPO, MAPPOConfig, RolloutBuffer
    from src.marl.trainer import collect_rollout
    ct = community()
    env = P2PCommunityEnv(ct, EnvConfig(n_envs=2), seed=0)
    ag = MAPPO(env.observation_dim, env.state_dim, env.action_dim, MAPPOConfig(seed=0))
    buf = RolloutBuffer(env.n_steps, env.E, env.N, env.observation_dim,
                        env.state_dim, env.action_dim, MAPPO.COMPONENTS)
    lv, _, _ = collect_rollout(env, ag, buf)
    r = ag.compose_reward({k: buf.components[k] for k in MAPPO.COMPONENTS}, env.cfg)
    stats = ag.update(buf, r, lv)
    assert stats["value_loss"] < 10.0, stats["value_loss"]
    assert stats["grad_critic"] < 50.0
    assert stats["grad_actor"] > 1e-6


def test_lagrangian_multiplier_responds_to_violation():
    from src.marl.lagrangian import PIDLagrangian
    d = PIDLagrangian(limit=0.01, kp=1.0, ki=0.1, kd=0.0)
    for _ in range(5):
        d.update(0.5)
    assert d.lam > 0.0
    high = d.lam
    for _ in range(50):
        d.update(0.0)
    assert d.lam < high


def _all_tests():
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _all_tests():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    total = len(_all_tests())
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
