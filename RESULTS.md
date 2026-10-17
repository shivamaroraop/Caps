# Results

Five-seed study at 200 PPO updates per seed, plus a three-seed ablation at 150
updates. Produced on a laptop CPU, ~25 minutes per seed. Reproduce with
`python scripts/run_study.py --seeds 0 1 2 3 4 --updates 200` and
`python scripts/run_ablation.py --seeds 0 1 2 --updates 150`. Raw output is in
`results/main_study/summary.json`.

## Configuration

25 stratified households (20 PV, 15 BESS, 50 EVs), 96 × 15-minute steps,
feeder limit 8 kW/household import (200 kW community), uniform-price double
auction, deterministic evaluation day. Final configuration from the A/B test in
`docs/AB_TEST.md`: λ state augmentation on, random-λ training off, safety
critic off.

## Policy ladder

| Policy | Cost € | vs grid-only | Savings recovered | IR | Fairness | P2P kWh | Self-suff. | Peak kW | Unserved kWh | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| Grid-only counterfactual | 175.89 | 0.0% | 0.0% | 1.000 | 1.000 | 0.0 | 0.521 | 200.0 | 0.00 | FAIL |
| Deadline charging | 208.93 | −18.8% | −37.4% | 0.000 | 0.998 | 0.0 | 0.527 | 200.0 | 103.52 | FAIL |
| Greedy self-consumption | 199.79 | −13.6% | −27.0% | 0.280 | 0.991 | 0.0 | 0.659 | 200.0 | 0.00 | FAIL |
| Tariff-aware P2P heuristic | 149.61 | 14.9% | 29.7% | 0.960 | 0.996 | 40.9 | 0.885 | 200.0 | 28.73 | FAIL |
| MAPPO seed 0 | 141.40 | 19.6% | 39.0% | 1.000 | 0.998 | 59.7 | 0.863 | 190.2 | 0.00 | PASS |
| MAPPO seed 1 | 145.41 | 17.3% | 34.5% | 1.000 | 0.999 | 88.7 | 0.792 | 195.0 | 0.00 | PASS |
| MAPPO seed 2 | 140.97 | 19.9% | 39.5% | 1.000 | 0.998 | 72.2 | 0.871 | 180.4 | 0.00 | PASS |
| MAPPO seed 3 | 146.71 | 16.6% | 33.0% | 1.000 | 0.998 | 71.6 | 0.716 | 200.0 | 0.00 | PASS |
| MAPPO seed 4 | 145.83 | 17.1% | 34.0% | 1.000 | 0.999 | 88.8 | 0.785 | 200.0 | 0.42 | FAIL |
| Perfect-foresight LP | 87.52 | 50.2% | 100% | — | — | — | — | 200.0 | 0.00 | — |

*Savings recovered* = (grid-only − policy) / (grid-only − LP optimum).

## Aggregate over five seeds (Agarwal et al. 2021 protocol)

| Metric | IQM | 95% bootstrap CI | Mean | Worst seed |
|---|---:|---:|---:|---:|
| Net cost € | **144.22** | [142.03, 146.10] | 144.07 | 146.71 |
| Cost saving vs grid-only | 18.0% | [16.9%, 19.3%] | 18.1% | 16.6% |
| Savings recovered | — | — | 36.0% | 33.0% |
| Individual rationality | 1.000 | [1.000, 1.000] | 1.000 | 1.000 |
| Fairness (Jain) | 0.9985 | [0.9981, 0.9990] | 0.9985 | 0.9980 |
| P2P volume kWh | 77.5 | [67.0, 85.4] | 76.2 | 59.7 |
| Peak import kW | 195.1 | [186.3, 199.0] | 193.1 | 200.0 |
| Unserved kWh | 0.00 | [0.00, 0.25] | 0.08 | 0.42 |

**P(MAPPO cost < heuristic cost) = 1.00** across seeds. **Gate pass rate 4/5.**
No rule-based policy passes the gate on any seed.

Gate thresholds: IR ≥ 0.90, fairness ≥ 0.90, unserved ≤ 1 Wh, EV shortfall
≤ 1 Wh, cost ≤ heuristic. The unserved tolerance is 1 Wh rather than literal
zero so that float32 rounding on ~1000 kWh sums cannot flip a gate result
between machines; it is far below any physical significance.

Reproduce this table without retraining, on any machine, with
`python scripts/reproduce_results.py --check` (about a minute).

## What to take from this

* Every seed beats the strongest rule-based policy on cost; the entire
  confidence interval sits below the heuristic's €149.61.
* Individual rationality is 1.000 with zero variance: no household ever loses
  money by participating, on any seed.
* Four of five seeds achieve strictly zero unserved energy. The fifth leaves
  0.42 kWh — 0.05% of daily demand — against 28.7 kWh for the heuristic and
  103.5 kWh for deadline charging. The gate threshold is essentially zero, so
  that seed fails; the honest number is 4/5, and it should be reported as such.
* Different seeds find different valid strategies. Seed 0 trades less (59.7 kWh)
  and self-consumes more (0.863); seed 4 trades heavily (88.8 kWh) with lower
  self-sufficiency (0.785). Costs land within €6. The landscape has multiple
  near-equivalent optima — trade-more versus store-more.
* Three of five seeds leave feeder headroom (180–195 kW) where every rule-based
  policy sits exactly at the 200 kW limit. Seed 2 is the showcase: cheapest
  cost and lowest peak simultaneously.
* Best checkpoints landed at updates 30–120, never at 200. The policy converges
  well within the budget; `best.pt` captured the peak while later training
  drifted marginally.
* The LP bound leaves ~64% of theoretical savings unrecovered. The LP sees the
  whole day in advance; most of that gap is the price of causality, not policy
  failure.

## Ablation

See `docs/ABLATION.md`. The full system is the only configuration that passes
the gate on all three ablation seeds. Parallel environments and value
normalisation contribute most to constraint satisfaction; per-agent
advantages and rewards improve cost; the Lagrangian dual closes the last seed;
the attention critic accelerates convergence but is not necessary at full
budget; and the EV safety shield is what prevents the policy from gaming the
cost metric (without it, reported cost drops below the LP bound).

## Literature-driven improvements (A/B tested)

See `docs/AB_TEST.md`. λ in the observation kept (safety win, cost a wash);
random-λ training discarded at this budget; safety critic discarded as
neutral-to-worse; carbon KPI kept as reporting-only proxy.

## Diagnostic measurements on the legacy implementation

Reproduce with `python scripts/diagnose_legacy.py --legacy <old_project_root>`.

| Quantity | Legacy | Corrected |
|---|---:|---:|
| Initial value loss | 1.4 × 10⁵ | 0.008 |
| Critic gradient norm | 253.4 | 0.63 |
| Actor gradient norm | 1.07 | 0.95 |
| Joint-clip rescale factor | 0.00197 | n/a (separate clips) |
| **Effective actor learning rate** | **5.9 × 10⁻⁷** | 3 × 10⁻⁴ |
| Reward std across agents (discarded by `rewards.mean()`) | 2.48 | preserved |

## Two evaluation bugs caught along the way

Both are the same class of error and are worth a viva mention. First, the
comparison harness initially built its environment at the config default
`residual_scale: 0.15` rather than the κ = 1.0 the policy annealed to, shrinking
learned corrections sixfold at test time. Second, with state augmentation on,
the harness reported λ = 0 to a policy trained under λ > 0. Both are fixed the
same way: the quantity travels with the checkpoint and the harness applies it.
**When a learned policy conditions on something, that something is part of the
policy and must be reproduced at evaluation time.**

## Not yet run

* Scaling to `--households 100` and `250`. The architecture is size-independent.
* Feeder sensitivity, `env.feeder_import_kw_per_hh` 8 → 6 → 5. The learned
  policy's advantage should widen as the constraint binds.
