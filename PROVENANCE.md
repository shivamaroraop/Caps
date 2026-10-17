# Provenance

## Origin

Rebuilt from an earlier capstone submission (`CAPstone_Final_Advanced`) that
reported a rule-based tariff heuristic outperforming MAPPO. The dataset
(`EC_EV_dataset.xlsx`, 250 households × 96 quarter-hours, 19 sheets) is carried
over unchanged. Nothing else is.

## Carried over unchanged

* `data/EC_EV_dataset.xlsx`
* `src/data/loader.py`, `src/data/schemas.py` — Excel parsing, verified against
  the raw workbook. Their *interpretation* is corrected downstream in
  `src/data/ev_physics.py` rather than by editing the parser, so the original
  read remains inspectable.

## Written for this project

Everything under `src/env/`, `src/marl/`, `src/eval/`, `src/utils/`,
`scripts/`, `tests/`, `configs/`, `docs/`.

## Corrections applied to the inherited data interpretation

| Field | Original reading | Corrected reading |
|---|---|---|
| EV charge power | `EVs` sheet *Charge (kW)*, 55–121 kW | `min(vehicle acceptance, home charge point)` = 7.4/11 kW, taken from the at-home portion of the time-varying `Max EV Dis-Charge` profile |
| EV target SoC | *Final battery level* treated as a departure target (already satisfied at t=0) | two-phase: trip energy + 10% reserve at departure, *Final battery level* at end of day |
| Cost counterfactual | load − PV settled against an infinite grid, EVs ignored | uncontrolled charge-on-arrival, settled through the same feeder limit and scarcity pricing; asserted equal to the `NoFlexibility` policy in `tests/test_system.py` |

## External ideas used, with attribution

* Value normalisation and agent-specific centralised critic input — Yu et al.
  (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*.
* PID-damped dual ascent — Stooke, Achiam & Abbeel (2020), *Responsive Safety in
  Reinforcement Learning by PID Lagrangian Methods*.
* Action-space projection / shielding — Dalal et al. (2018), *Safe Exploration in
  Continuous Action Spaces*; Alshiekh et al. (2018).
* Potential-based reward shaping — Ng, Harada & Russell (1999).
* IQM and interval estimates over seeds — Agarwal et al. (2021), *Deep RL at the
  Edge of the Statistical Precipice*.
* Least-laxity-first is standard deadline scheduling, applied here to charging.

## What this project does *not* claim

* It is **not** a graph neural network. The dataset provides no feeder adjacency,
  so the critic uses attention over an unordered set. Claiming a GNN without an
  adjacency matrix would be unsupportable.
* The feeder is a single aggregate import/export limit, **not** a power-flow
  model. There is no voltage, no line impedance, no unbalanced phases.
* The LP is a relaxation (perfect foresight, community-level netting within an
  interval). It bounds achievable savings; it is not a schedule a causal
  controller could execute.
* Results are a five-seed study (200 updates) plus a three-seed ablation (150 updates). See `RESULTS.md` and `docs/ABLATION.md`.
