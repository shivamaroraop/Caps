# Presentation guide

**Title:** *Safe Constrained Multi-Agent Reinforcement Learning for Fair,
Grid-Aware Peer-to-Peer Energy Trading*

**One-sentence contribution:** a P2P energy community in which 25–250 household
agents learn battery, EV and bidding policies under a three-layer safety
architecture — shielded physical constraints, a residual trust region, and
PID-Lagrangian shadow prices for economic constraints — benchmarked against a
perfect-foresight LP bound rather than against its own heuristic.

---

## Headline numbers (final study — put these on the slides)

* **Five seeds, 200 updates each.** MAPPO cost **€144.2 IQM, 95% CI [142.0, 146.1]**
  vs **€149.6** for the tariff-aware heuristic. P(improvement) = 1.00.
* **18.1% saving vs grid-only; 36% of perfect-foresight savings recovered**
  (heuristic: 29.7%). LP bound €87.5.
* **IR = 1.000 on every seed, zero variance.** Fairness ≥ 0.998.
* **Gate pass 4/5.** Worst seed: 0.42 kWh unserved (0.05% of demand) vs 28.7 kWh
  for the heuristic. No rule-based policy passes on any seed.
* **Ablation (3 seeds, 150 updates): full system is the only configuration
  passing the gate on all seeds.** Parallel envs and value normalisation matter
  most for safety; per-agent advantages/rewards for cost; the shield keeps the
  cost metric honest (without it: €74, below the LP bound — impossible).
* Three of five seeds leave feeder headroom (180–195 kW); every rule-based
  policy sits at the 200 kW limit.

Full tables: `RESULTS.md`, `docs/ABLATION.md`, `docs/AB_TEST.md`.

---

## Suggested slide order

**1 — The problem.** 25 homes, 15 batteries, 50 EVs, one shared feeder. Every
home can trade surplus solar with neighbours instead of exporting cheaply to the
grid. Sell price is flat at €0.045/kWh; buy price ranges €0.107–0.234. That
spread is the entire economic opportunity.

**2 — The first result, stated plainly.** "A rule-based tariff heuristic beat
MAPPO." Do not hide this. It is the setup.

**3 — The diagnosis.** Live-run `scripts/diagnose_legacy.py`. Land on one number:

> **effective actor learning rate = 5.9 × 10⁻⁷, about 500× too small**

Then: the reward vector was averaged before storage, so all 25 agents shared one
advantage. MAPPO was not unstable — it was frozen at initialisation, which
(because of the residual parameterisation) means it was running the heuristic
plus noise.

**4 — The data bugs.** EV chargers read at 55–121 kW instead of the true 7.4/11
kW home rating; the fleet as parsed could not physically complete its own trips;
the cost counterfactual ignored EVs and the feeder. Physics errors, not tuning.

**5 — The rebuild.** The architecture diagram from the README. Emphasise the
three constraint layers and *why they are different kinds of constraint*.

**6 — The safety gate.** IR ≥ 0.90 on every seed, zero unserved energy, zero EV
shortfall, fairness ≥ 0.90, cost ≤ heuristic. Note that the **heuristic fails
this gate** — it overloads the feeder.

**7 — Results.** The ladder table, with `savings_recovered_%` as the headline.
The LP bound is what makes the claim falsifiable.

**8 — Ablation.** Which fix mattered: turn each one back off and watch the
result degrade. This is what turns "I rewrote it" into evidence.

**9 — Limitations.** One day of data; attention over a set, not a GNN, because
there is no adjacency matrix; the LP is a relaxation. Owning these is worth more
than overclaiming.

---

## Questions you will be asked

**"Isn't this just PPO with extra steps?"**
No — the corrections are specific to the *multi-agent* setting. Per-agent
advantages from an agent-specific centralised critic are the MARL credit
assignment problem; a single-agent PPO implementation would never have had this
bug because there is only one agent to assign credit to.

**"Why attention and not a GNN?"**
The dataset provides no feeder topology or adjacency. Attention over an unordered
set is permutation-equivariant and makes no unsupported structural claim. Calling
it a GNN without a graph would be a fabrication — and an examiner who knows the
area will ask exactly this.

**"How do you know the LP bound is right?"**
`test_lp_bound_is_not_worse_than_any_policy` asserts it lower-bounds the cost of
every simulated policy. It is a relaxation (perfect foresight, community-level
netting), so it is an upper bound on achievable savings, not an attainable
schedule.

**"Why does the heuristic fail the safety gate?"**
Uncontrolled and naively price-driven EV charging synchronises the fleet. 50 EVs
at 7.4–11 kW is 400+ kW against a 200 kW feeder. Least-laxity scheduling helps
but does not eliminate it; only the constrained learned policy drives unserved
energy to zero.

**"What if MAPPO fails the gate on a future seed?"**
Then it is not deployed, and the system falls back to the rule-based controller.
That is the architecture, not a caveat. The gate is evaluated on the **worst**
seed, not the mean.

**"Why 25 households and not 250?"**
Compute. The architecture is size-independent — parameter-shared policy,
permutation-equivariant critic — and `--households 250` runs unchanged. Report
whichever scaling runs you completed.

---

## Numbers worth memorising

| | |
|---|---:|
| Households / batteries / EVs | 25 / 15 / 50 |
| Timesteps per episode | 96 (15 min × 24 h) |
| Observation / action dimension | 43 / 5 |
| Daily load / PV / EV energy | 717 / 285 / 430 kWh |
| Grid tariff spread | €0.107–0.234 buy, €0.045 sell |
| Feeder limit | 8 kW/household import (200 kW) |
| Grid-only counterfactual cost | €191.42 |
| Improved heuristic | €151.42 (38.5% of achievable savings) |
| Perfect-foresight LP optimum | €87.52 |
| Legacy effective actor LR | 5.9 × 10⁻⁷ |
| Corrected actor / critic gradient norms | 0.95 / 0.63 |
| Value loss, legacy → corrected | 12,080 → 0.008 |
