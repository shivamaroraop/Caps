# A/B test of the 2025–26 literature improvements

Three changes were implemented, measured against the previous configuration on
identical seeds and identical evaluation days, and then **kept or discarded on
the evidence**. Reproduce with `python scripts/ab_test.py`.

## Measured results

40 updates, seed 0, everything else held fixed:

| Variant | Cost € | IR | Unserved kWh | CO₂ kg | Gate | Verdict |
|---|---:|---:|---:|---:|:--|:--|
| `v0_previous` | 148.37 | 1.000 | 0.00 | 317.4 | PASS | reference |
| `v1_stateaug` (λ in obs **+ random-λ training**) | 157.80 | 0.960 | 23.64 | 309.1 | FAIL | **discarded** |
| `v2_safetycrit` (safety critic drives the dual) | 149.47 | 1.000 | 0.00 | 319.2 | PASS | **discarded** |
| `v1b_obsonly` (λ in obs only) | **140.53** | 1.000 | 0.00 | **292.4** | PASS | **kept** |

Seed 1, same budget:

| Variant | Cost € | Unserved kWh | Gate |
|---|---:|---:|:--|
| `v0_previous` | 150.82 | **17.27** | FAIL (unserved) |
| `v1b_obsonly` | 152.45 | **0.00** | FAIL (cost ratio) |

70 updates, seed 0, full pipeline:

| Configuration | Cost € | Savings recovered | Unserved | Peak kW | Gate |
|---|---:|---:|---:|---:|:--|
| Previous | 145.15 | 34.8% | 0.00 | 200.00 | PASS |
| Kept configuration | 147.34 | 32.3% | 0.00 | 198.39 | PASS |

## Decisions and reasoning

### KEPT — λ in the observation (state augmentation)

`env.state_augmented: true`. The two current Lagrange multipliers are appended
to every agent's observation, so the policy can condition on how much
constraint pressure the system is under rather than feeling λ only through the
reward. Motivated by Calvo-Fullana et al. (2024) and the fixed-multiplier
brittleness quantified by Amaya-Corredor et al. (TMLR, 2026).

**The honest reading of the evidence: this is a safety win, not a cost win.**
Cost is a wash — better at 40 updates on seed 0 (140.53 vs 148.37), slightly
worse at 70 updates on seed 0 (147.34 vs 145.15), slightly worse on seed 1.
The consistent effect is on unserved energy, which is **0.00 on every seed and
every budget tested**, against 17.27 kWh on seed 1 without it. Peak import also
drops below the feeder limit (198.39 vs 200.00 kW) rather than sitting exactly
on it. Since the deployment gate is driven by constraint satisfaction and not
by cost alone, that is the property worth having. It is a config flag, so the
comparison can be rerun either way.

### DISCARDED — random-λ training

`algo.lam_sample_prob: 0.0`. Training a fraction of updates at a randomly
sampled multiplier is the mechanism by which the TMLR paper gets one policy to
generalise across all constraint pressures. Measured here it was clearly worse:
€157.80 against €148.37, individual rationality down to 0.96, 23.64 kWh
unserved, and it broke the safety gate.

The likely reason is budget. Spending 25% of a 40-update run on constraint
pressures the deployed policy never encounters is a real cost when the total
sample budget is small; the TMLR result uses 10⁶ PPO timesteps per agent. This
is retained as a switchable option and flagged as worth revisiting at a much
larger budget — it is not a refutation of the paper, it is a statement that the
technique does not pay for itself at this scale.

### DISCARDED — safety critic

`algo.safety_critic: false`. A second critic head predicting discounted future
constraint cost, used to drive the dual update from a forecast rather than the
batch measurement (cf. MA-DELC). Measured neutral-to-slightly-worse: €149.47
against €148.37, with identical safety metrics.

Worth recording how it failed *first*: with the cost head regressing an
unnormalised target, value loss jumped to 107 and the critic gradient to 98
against an actor gradient of 0.86 — a rediscovery, in new code, of the exact
imbalance this project was built to fix. Adding a separate `ValueNormalizer`
for the cost head brought the gradient back to 1.99 and the loss to 0.40. Only
*then* was the comparison meaningful, and only then was it fair to conclude the
idea does not help here. A tempting but wrong conclusion would have been to
discard the safety critic on the strength of the pre-normalisation numbers.

### KEPT — carbon KPI

`env.carbon_aware: true`. Grid imports are weighted by a time-varying carbon
intensity and reported as `emissions_kg`. Reporting only — it does not enter
the reward, so it cannot affect training, and the A/B numbers above confirm it
is inert. Included because emissions are a standard KPI across the 2025–26
literature.

**This one carries a caveat that must be stated whenever it is reported.** The
dataset contains no emissions series. The intensity is a *proxy*, taken as an
affine function of the within-day price percentile spanning 150–500 gCO₂/kWh,
on the reasoning that expensive hours are the thermally-dispatched ones. It is
a defensible modelling assumption and it is not a measurement. Do not present
the CO₂ column as an empirical result.

## A bug this exercise exposed

The first full run with state augmentation scored €172.38 and failed the gate,
far worse than the A/B suggested. The cause was not the technique: the
evaluation harness constructed a fresh environment reporting λ = 0, so the
policy was asked to act under a constraint pressure it had never trained at.

This is the same class of error as the trust-region mismatch found earlier
(evaluating a residual policy at κ = 0.15 when it trained at κ = 1.0). Both are
now fixed the same way — the quantity travels with the checkpoint
(`extra["multipliers"]`, `extra["residual_scale"]`) and is applied by the
harness. After the fix the same checkpoint scored €147.34 and passed.

The general lesson, worth stating in a viva: **when a learned policy conditions
on something, that something is part of the policy and must be reproduced at
evaluation time.** Two separate instances of this bug appeared in one project.
