"""Reproduce the measurements that explain the original result.

Run this against the *legacy* package to see, in numbers, why the rule-based
controller beat MAPPO there.  Point ``--legacy`` at the old project root (the
folder containing ``src/rl/mappo.py``).

    python scripts/diagnose_legacy.py --legacy ../CAPstone_Final_Advanced

It measures four quantities and prints them beside the corrected values:

1. how much of the reward signal is destroyed by ``rewards.mean()``;
2. the scale of the initial value-regression target;
3. the actor and critic gradient norms;
4. the factor by which a *joint* ``clip_grad_norm_`` rescales them, and hence
   the effective actor learning rate.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", required=True)
    args = ap.parse_args()
    legacy = Path(args.legacy).resolve()
    sys.path.insert(0, str(legacy))
    import torch
    from torch import nn
    from src.data.loader import DatasetLoader
    from src.env.marl_energy_env import P2PEnergyMARLEnv
    from src.rl.mappo import MAPPO, Rollout
    from src.rl.approaches import with_approach
    from src.utils.config import load_config
    from scripts.train_marl import representative_community

    cfg = with_approach(load_config(legacy / "configs/mappo.yaml"), "balanced")
    ds = next(legacy.glob("*.xlsx"))
    community = representative_community(DatasetLoader(str(ds)).load_all(), 25, 42)
    env = P2PEnergyMARLEnv(community, cfg)
    model = MAPPO(env.observation_dim, env.action_dim, env.global_state_dim, cfg["training"])

    obs, state, _ = env.reset()
    r = Rollout([], [], [], [], [], [], [])
    per_agent, done = [], False
    while not done:
        a, lp = model.act(obs); v = model.value(obs, state)
        nobs, nstate, rew, done, info = env.step(a)
        r.observations.append(obs.copy()); r.states.append(state.copy())
        r.actions.append(a.copy()); r.log_probs.append(lp.copy())
        r.rewards.append(float(rew.mean())); r.values.append(v); r.dones.append(done)
        per_agent.append(rew.copy()); obs, state = nobs, nstate
    R = np.array(per_agent)

    print("\n1. CREDIT ASSIGNMENT")
    print(f"   per-agent reward matrix                : {R.shape}")
    print(f"   std ACROSS AGENTS at a timestep        : {R.std(axis=1).mean():.3f}")
    print(f"   std ACROSS TIME for an agent           : {R.std(axis=0).mean():.3f}")
    print(f"   -> the first number is discarded by `rewards.mean()`;")
    print(f"      every agent then receives an identical advantage.")

    T = len(r.rewards)
    values = np.asarray(r.values + [0.0], np.float32)
    rewards = np.asarray(r.rewards, np.float32)
    adv = np.zeros(T, np.float32); gae = 0.0
    for t in range(T - 1, -1, -1):
        nt = 1.0 - float(r.dones[t])
        d = rewards[t] + .99 * values[t + 1] * nt - values[t]
        gae = d + .99 * .95 * nt * gae; adv[t] = gae
    ret = adv + values[:-1]
    print("\n2. VALUE TARGET SCALE")
    print(f"   discounted return at t=0               : {ret[0]:.1f}")
    print(f"   critic is initialised near 0, so the first value MSE is ~{ret[0]**2:.0f}")

    n = R.shape[1]
    obs_seq = np.stack(r.observations)
    obs_t = torch.tensor(obs_seq.reshape(T * n, -1))
    act_t = torch.tensor(np.concatenate(r.actions))
    olp = torch.tensor(np.concatenate(r.log_probs))
    advn = (adv - adv.mean()) / (adv.std() + 1e-8)
    advt = torch.tensor(np.repeat(advn, n))
    ids = torch.arange(0, min(1024, T * n))
    dist = model._distribution(obs_t[ids]); z = model._unsquash(act_t[ids])
    nlp = dist.log_prob(z).sum(-1) - model._log_det_jacobian(z)
    ratio = (nlp - olp[ids]).exp()
    pl = -torch.minimum(ratio * advt[ids], ratio.clamp(.8, 1.2) * advt[ids]).mean()
    ent = dist.entropy().sum(-1).mean()
    tid = torch.unique(torch.div(ids, n, rounding_mode="floor"))
    vl = nn.functional.mse_loss(model.critic(torch.tensor(obs_seq)[tid],
                                             torch.tensor(np.stack(r.states))[tid]),
                                torch.tensor(ret)[tid])
    ap_, cp = list(model.actor.parameters()), list(model.critic.parameters())
    model.opt.zero_grad(); (pl - .01 * ent).backward(retain_graph=True)
    gA = torch.sqrt(sum((p.grad ** 2).sum() for p in ap_ if p.grad is not None)).item()
    model.opt.zero_grad(); (.5 * vl).backward(retain_graph=True)
    gC = torch.sqrt(sum((p.grad ** 2).sum() for p in cp if p.grad is not None)).item()
    joint = float(np.hypot(gA, gC)); scale = 0.5 / joint
    print("\n3. GRADIENT NORMS (one shared optimiser, one shared clip)")
    print(f"   actor  : {gA:.3f}")
    print(f"   critic : {gC:.1f}")
    print(f"   joint  : {joint:.1f}")
    print("\n4. EFFECT OF clip_grad_norm_(actor+critic, 0.5)")
    print(f"   rescale factor applied to BOTH         : {scale:.2e}")
    print(f"   effective actor learning rate          : {3e-4 * scale:.3e}")
    print(f"   a working actor learning rate would be : ~3e-04")
    print("\n   The actor was frozen near initialisation. The run-to-run spread")
    print("   reported as 'MAPPO instability' is a random walk, not optimisation.\n")


if __name__ == "__main__":
    main()
