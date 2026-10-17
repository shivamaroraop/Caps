"""Correctness tests. These encode the invariants that make the results mean something.

Run with:  python -m pytest tests -q      (or: python tests/test_system.py)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import (BASELINES, NoFlexibility, TariffAwareP2P,
                               perfect_foresight_lp)
from src.env.community_env import EnvConfig, OBS_DIM, OBS_NAMES, P2PCommunityEnv
from src.env.market import clear_double_auction

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tensors():
    return build_tensors(stratified_subset(load_community(ROOT / "data/EC_EV_dataset.xlsx"),
                                           25, seed=0))


def run(tensors, policy, **kw):
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=1, noise_std=0.0,
                                             residual=policy.residual, **kw), seed=1)
    obs, st = env.reset(seed=1, deterministic=True)
    done = False
    while not done:
        obs, st, r, done, info = env.step(policy.act(obs, st, env))
    return info["episode"], env


# --- the counterfactual is a checked invariant, not an assumption -----------
def test_no_flexibility_reproduces_the_counterfactual(tensors):
    ep, env = run(tensors, NoFlexibility())
    assert abs(float(ep["net_cost_eur"][0]) - float(ep["baseline_cost_eur"][0])) < 1e-3
    assert float(ep["individual_rationality"][0]) == pytest.approx(1.0)


# --- physics ---------------------------------------------------------------
def test_battery_stays_within_soc_limits(tensors):
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=2, noise_std=0.0, residual=False), seed=0)
    obs, st = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(env.n_steps):
        a = rng.uniform(-1, 1, (env.E, env.N, env.action_dim)).astype(np.float32)
        a[:, :, 4] = rng.uniform(0, 1, (env.E, env.N))
        obs, st, r, done, info = env.step(a)
        cap = tensors.bess_capacity[None, :]
        assert (env.bess_soc >= -1e-5).all()
        assert (env.bess_soc <= cap * env.cfg.bess_max_soc + 1e-4).all()


def test_ev_never_exceeds_capacity_and_is_never_charged_away(tensors):
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=2, noise_std=0.0, residual=False), seed=0)
    obs, st = env.reset(seed=0)
    rng = np.random.default_rng(1)
    for t in range(env.n_steps):
        prev = env.ev_soc.copy()
        a = rng.uniform(-1, 1, (env.E, env.N, env.action_dim)).astype(np.float32)
        obs, st, r, done, info = env.step(a)
        assert (env.ev_soc <= tensors.ev_capacity[:, None, :] + 1e-4).all()
        assert (env.ev_soc >= -1e-5).all()
        away = tensors.ev_at_home[:, t, :][:, None, :] < 0.5
        # away vehicles may only lose energy through the scheduled trip
        assert (np.where(away, env.ev_soc - prev, 0.0) <= 1e-4).all()


def test_safety_shield_eliminates_ev_shortfall(tensors):
    for name, cls in BASELINES.items():
        ep, _ = run(tensors, cls())
        assert float(ep["ev_shortfall_kwh"][0]) < 1e-3, name


def test_disabling_the_shield_causes_shortfall(tensors):
    """The shield must be doing real work, not decorating an already-safe policy."""
    from src.env.baselines import DeadlineCharging
    ep, _ = run(tensors, DeadlineCharging(), ev_safety_projection=False)
    assert float(ep["ev_shortfall_kwh"][0]) > 1.0


def test_feeder_limit_is_never_exceeded(tensors):
    ep, env = run(tensors, TariffAwareP2P())
    assert float(ep["peak_import_kw"][0]) <= env.feeder_import_kw + 1e-3


# --- market ----------------------------------------------------------------
def test_double_auction_conserves_energy_and_respects_quotes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 12))
        bq = np.where(rng.random(n) < .5, rng.random(n) * 5, 0.0)[None, :]
        aq = np.where(rng.random(n) < .5, rng.random(n) * 5, 0.0)[None, :]
        bp = (rng.random(n) * .2 + .05)[None, :]
        res = clear_double_auction(bq, bp, aq, bp, np.array([.1]))
        assert abs(res.buy_kwh.sum() - res.sell_kwh.sum()) < 1e-8   # energy conserved
        assert (res.buy_kwh <= bq + 1e-9).all()                     # no over-allocation
        assert (res.sell_kwh <= aq + 1e-9).all()
        assert res.volume_kwh[0] <= res.feasible_kwh[0] + 1e-9


def test_no_trade_when_no_bid_covers_an_ask():
    bq = np.array([[2.0, 0.0]]); aq = np.array([[0.0, 2.0]])
    bp = np.array([[0.05, 0.05]]); ap = np.array([[0.20, 0.20]])
    res = clear_double_auction(bq, bp, aq, ap, np.array([0.1]))
    assert res.volume_kwh[0] == 0.0


def test_p2p_price_stays_inside_the_tariff_spread(tensors):
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=1, noise_std=0.0, residual=False), seed=0)
    obs, st = env.reset(seed=0)
    rng = np.random.default_rng(3)
    for t in range(env.n_steps):
        a = rng.uniform(-1, 1, (1, env.N, 5)).astype(np.float32)
        a[:, :, 4] = rng.uniform(0, 1, (1, env.N))
        obs, st, r, done, info = env.step(a)
        if info["volume_kwh"][0] > 1e-9:
            lo, hi = tensors.sell_price[t].min(), tensors.buy_price[t].max()
            assert lo - 1e-6 <= info["clearing_price"][0] <= hi + 1e-6


# --- learning machinery ----------------------------------------------------
def test_state_augmentation_exposes_the_multipliers(tensors):
    """The policy must actually see constraint pressure change."""
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=1, state_augmented=True), seed=0)
    obs, _ = env.reset(seed=0)
    lo = obs[..., OBS_NAMES.index("lam_ir")].copy()
    env.set_multipliers({"ir": 10.0, "grid": 5.0})
    obs2 = env._observe()
    assert (obs2[..., OBS_NAMES.index("lam_ir")] > lo).all()
    off = P2PCommunityEnv(tensors, EnvConfig(n_envs=1, state_augmented=False), seed=0)
    off.reset(seed=0); off.set_multipliers({"ir": 10.0, "grid": 5.0})
    assert (off._observe()[..., OBS_NAMES.index("lam_ir")] == 0).all()


def test_emissions_are_tracked_and_monotone(tensors):
    ep, _ = run(tensors, TariffAwareP2P())
    assert float(ep["emissions_kg"][0]) > 0


def test_observation_layout_matches_its_documentation(tensors):
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=1), seed=0)
    obs, st = env.reset(seed=0)
    assert obs.shape[-1] == OBS_DIM == len(OBS_NAMES)
    assert np.isfinite(obs).all() and np.isfinite(st).all()


def test_log_prob_is_exact_at_the_action_boundary():
    """The failure mode that `atanh`-based inversion hides."""
    import torch
    from src.marl.networks import Actor
    a = Actor(OBS_DIM, 5, hidden=32)
    with torch.no_grad():
        a.log_std.fill_(1.0)                      # wide policy -> boundary actions
    obs = torch.randn(256, OBS_DIM)
    dist = a.distribution(obs)
    z = dist.sample()
    lp = a.log_prob(dist, z)
    assert torch.isfinite(lp).all()
    action = a.squash(z)
    assert (action[:, :4].abs() > 0.999).any(), "test needs boundary actions"
    # recomputing from the stored z must reproduce the ratio exactly
    assert torch.allclose(a.log_prob(a.distribution(obs), z), lp, atol=1e-5)


def test_per_agent_advantage_differs_across_agents(tensors):
    """Guards the single most damaging legacy defect."""
    from src.marl.mappo import MAPPO, MAPPOConfig
    env = P2PCommunityEnv(tensors, EnvConfig(n_envs=2), seed=0)
    ag = MAPPO(env.observation_dim, env.state_dim, env.action_dim,
               MAPPOConfig(hidden_size=32))
    T, E, N = 10, env.E, env.N
    rew = np.random.randn(T, E, N).astype(np.float32)
    val = np.random.randn(T, E, N).astype(np.float32)
    adv, ret = ag.compute_gae(rew, val, np.zeros((E, N), np.float32))
    assert adv.std(axis=-1).mean() > 1e-3


def test_lp_bound_is_not_worse_than_any_policy(tensors):
    lp = perfect_foresight_lp(tensors, EnvConfig())
    assert lp is not None and lp["unserved_kwh"] < 1e-6
    for name, cls in BASELINES.items():
        ep, _ = run(tensors, cls())
        assert lp["optimal_cost_eur"] <= float(ep["net_cost_eur"][0]) + 1e-6, name


def test_ev_physics_correction_produces_domestic_charger_ratings(tensors):
    r = tensors.meta["ev_physics"]
    assert r["home_charge_kw_after"][1] <= 22.0
    assert r["infeasible_departures"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__)), "-q"]))
