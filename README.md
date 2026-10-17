# Safe Constrained Multi-Agent RL for Peer-to-Peer Energy Trading

A 25–250 household energy community in which every home is an independent
learning agent controlling its battery, two EVs, and its bids in a peer-to-peer
double auction, under a shared feeder constraint.

This is a **rebuild** of an earlier capstone whose headline finding was that a
hand-written tariff heuristic beat MAPPO. That finding was an artefact of five
implementation defects plus three data-interpretation bugs, each of which is
measured, documented and fixed here.

**New here? Read `QUICKSTART.md` — install, verify, and reproduce every number in three commands. Or just open `RESULTS.html`.**

**Read `docs/WHY_THE_BASELINE_WON.md` first.** It is the strongest material in
the project and the core of the presentation.

---

## 1. Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests -q                            # 34 correctness tests (~50 s)
python scripts/run_study.py --seeds 0 --updates 60   # ~5 min on one CPU core
```

The last command prints a comparison of every policy against the
perfect-foresight optimum.

> **Note on compute.** Run training *alone*. Everything here fits in ~1 GB, but
> running the test suite concurrently with training on a 1-core / 3 GB box will
> get the trainer OOM-killed.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────┐
  EC_EV_dataset ──► │ loader → EV physics correction → tensors │
                    └───────────────────┬──────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────┐
        │  P2PCommunityEnv  (8 parallel days, fully vectorised) │
        │  ├── battery + EV physics (hard limits)               │
        │  ├── EV least-laxity safety shield                    │
        │  ├── uniform-price double auction                     │
        │  ├── feeder limits → unserved / curtailed energy      │
        │  └── PER-AGENT reward, decomposed into components     │
        └───────────────┬───────────────────────┬───────────────┘
                        ▼                       ▼
     ┌───────────────────────────┐   ┌────────────────────────────┐
     │ Actor (shared, residual)  │   │ Attention critic           │
     │ correction to the rule    │   │ ONE VALUE PER HOUSEHOLD    │
     │ controller, κ: 0.15 → 1.0 │   │ → per-agent GAE            │
     └───────────────────────────┘   └────────────────────────────┘
                        └──────────┬────────────┘
                                   ▼
             ┌──────────────────────────────────────────┐
             │ PID-Lagrangian duals: learned shadow     │
             │ prices for IR and feeder constraints     │
             └────────────────────┬─────────────────────┘
                                  ▼
             ┌──────────────────────────────────────────┐
             │ Safety gate → deploy, or fall back       │
             └──────────────────────────────────────────┘
```

**Agents.** One per household. 43 local features (own load, PV, battery SoC,
both EVs' SoC / connectivity / deficit / *urgency*, tariffs, market state, own
cumulative cost burden, three forecast horizons, and the reference controller's
current action). 5 actions: battery setpoint, EV1, EV2, P2P quantity, P2P price.

**Market.** Uniform-price double auction cleared by merit order, vectorised with
`searchsorted` instead of a `while` loop. Quotes are constrained to the tariff
spread `[sell_price, buy_price]` — this removes only dominated actions and makes
P2P individually rational by construction.

**Constraints, in three deliberate layers.**

| Layer | Enforces | Mechanism |
|---|---|---|
| Trust region | stay near a known-good controller early | κ annealed 0.15 → 1.0 |
| Projection | EV departure feasibility, SoC/power limits | hard shield, never learned |
| Lagrangian | individual rationality, feeder congestion | PID-controlled duals |

Mobility requirements are non-negotiable and are *shielded*. Economic trade-offs
have unknown shadow prices and are *learned*. Making that distinction explicit is
the safety contribution.

---

## 3. Running every experiment

### Main multi-seed study
```bash
python scripts/run_study.py --seeds 0 1 2 3 4 --updates 200 --households 25
```
Trains one agent per seed, evaluates every policy on the **same deterministic
days**, writes `results/main_study/summary.json` and a formatted table. Headline
metric:

> savings recovered = (grid-only − policy) / (grid-only − LP optimum)

≈ 4.4 s/update on one core; 200 updates ≈ 15 min per seed.

### Ablation — which fix actually mattered
```bash
python scripts/run_ablation.py --seeds 0 1 --updates 80
```
Each variant re-enables exactly one legacy defect: `shared_reward`,
`team_advantage`, `no_value_norm`, `single_env`, `no_lagrangian`, `mlp_critic`,
`no_shield`.

### Reproduce the diagnosis of the original code
```bash
python scripts/diagnose_legacy.py --legacy ../CAPstone_Final_Advanced
```
Prints the four measurements that explain the original result, including the
effective actor learning rate. **Run this live in your presentation** — a
measurement is far stronger than a slide claiming the bug existed.

### Scale to the full population
```bash
python scripts/run_study.py --households 100 --seeds 0 --updates 150
python scripts/run_study.py --households 250 --seeds 0 --updates 150
```
The policy is parameter-shared and the critic is permutation-equivariant, so the
networks are unchanged by `N`; only the environment tensors grow.

### Feeder sensitivity
Copy `configs/default.yaml`, change `env.feeder_import_kw_per_hh` (8.0 → 6.0 →
5.0), and rerun with `--config`. Coordination is worth more the tighter the
feeder; this sweep shows the learned policy's advantage growing as the
constraint binds.

---

## 4. Reading the results table

```
policy                  cost_eur  vs_grid_only_%  savings_recovered_%  IR     unserved  gate
no_flexibility          191.42     0.0             0.0                 1.000   78.07    FAIL
deadline_charging       208.93    -9.2           -16.8                 0.040  103.52    FAIL
greedy_self_consumption 208.28    -8.8           -16.2                 0.440   38.94    FAIL
tariff_aware_p2p        151.42    20.9            38.5                 1.000   34.02    FAIL
mappo_seed*                 …        …                …                    …       …    see results/
perfect_foresight_lp     87.52    54.3           100.0                    –    0.00       –
```

* **`no_flexibility` is the grid-only counterfactual.** The test suite asserts it
  reproduces the analytic baseline to 1e-3. IR and fairness are measured against it.
* **`savings_recovered_%` is the number to quote.** "We beat our heuristic" is
  unfalsifiable; "we recover X% of what perfect foresight achieves" is not.
* **`gate`** is the deployment decision: IR ≥ 0.90 on *every* seed, zero unserved
  energy, zero EV shortfall, fairness ≥ 0.90, cost no worse than the heuristic.
  A policy that passes on average but fails on one seed in five is reported as
  failing — that is the distinction the original study was missing.

**The rule-based heuristic fails the gate too.** It overloads the feeder. On this
data no non-learned controller in the ladder satisfies the constraints.

---

## 5. Layout

```
src/data/    loader · schemas · ev_physics.py (corrections) · community.py (tensors)
src/env/     community_env.py · market.py · safety.py · baselines.py (+ LP bound)
src/marl/    networks.py · normalizers.py · lagrangian.py · mappo.py · trainer.py
src/eval/    harness.py · statistics.py (IQM, bootstrap CI, P(improvement))
scripts/     reproduce_results.py · run_study.py · run_ablation.py · ab_test.py · diagnose_legacy.py
tests/       34 tests: physics, market, counterfactual invariant, learner
docs/        WHY_THE_BASELINE_WON.md · AB_TEST.md · ABLATION.md · DESIGN.md · PRESENTATION.md · LITERATURE_POSITIONING.md
configs/     default.yaml (every hyperparameter, commented)
```

---

## 6. Limitations — state these, they are not weaknesses if you own them

* **One day.** The dataset is a single 96-step day. Generalisation is tested over
  stochastic load/PV perturbations and held-out evaluation seeds, not seasons.
* **No network topology.** The data supplies no feeder adjacency, so the critic
  uses attention over an unordered set — **not** a graph neural network. Claiming
  a GNN without an adjacency matrix would be unsupportable. The feeder is a single
  aggregate import/export limit, not a power-flow model.
* **The LP bound is a relaxation.** Perfect foresight, and it nets community
  import/export within an interval. It is an upper bound on achievable savings,
  not a schedule a causal controller could execute.
* **Modest seed count**, because results were produced on one CPU core. This is
  why reporting uses IQM and bootstrap intervals rather than bare means.

## 7. References

- Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games* — value normalisation; agent-specific critic input.
- Stooke, Achiam & Abbeel (2020), *Responsive Safety in RL by PID Lagrangian
  Methods* — damped dual updates.
- Dalal et al. (2018), *Safe Exploration in Continuous Action Spaces* — shielding.
- Ng, Harada & Russell (1999), *Policy Invariance under Reward Transformations*
  — potential-based shaping.
- Agarwal et al. (2021), *Deep RL at the Edge of the Statistical Precipice* —
  IQM and interval estimates over seeds.
