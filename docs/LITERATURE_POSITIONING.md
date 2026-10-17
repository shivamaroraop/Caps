# Positioning against the 2024–2026 P2P MARL literature

A short map of where this project sits relative to recent published work, for
use in the report's related-work section and in a viva. Each row states what
the strand does, and what this project takes from it or deliberately does not.

## Strands in the recent literature and this project's stance

### 1. Constrained / safe MARL for energy communities

The dominant recent shift is from unconstrained cost minimisation to explicit
constraint handling — feeder limits, unserved energy, deployment gates.

* **PID-Lagrangian dual ascent** (Stooke, Achiam & Abbeel 2020) — adopted as
  the core constraint mechanism.
* **State augmentation: λ in the observation** (Calvo-Fullana et al. 2024;
  Amaya-Corredor et al., TMLR 2026) — implemented, A/B tested, **kept** as a
  safety win (zero unserved energy on every seed tested). See `AB_TEST.md`.
* **Random-λ training for pressure-general policies** (same TMLR 2026 line) —
  implemented, A/B tested, **discarded at this sample budget** with the honest
  caveat that the original result uses ~10⁶ timesteps per agent.
* **Safety critics driving the dual** (MA-DELC line) — implemented, A/B
  tested, **discarded** as neutral-to-worse here.

The differentiator to state explicitly: this project does not just cite these
techniques, it measures them against each other on one dataset and keeps only
what pays. Very little of the published work does head-to-head A/B testing.

### 2. Market mechanism design

* **Double-auction clearing with MARL bidders** (Qiu et al., IJCAI 2021, and
  the auction-based clearing in Shah et al., Applied Energy 2026 on dairy-farm
  P2P) — this project uses a uniform-price double auction, consistent with
  that strand.
* **Mean-field approaches for scale** (Qiu et al., IEEE TPWRS 2023) — not
  used; at 25–250 households an attention critic over the agent set is
  sufficient and easier to defend. Worth citing as the scaling route beyond
  ~10³ agents.

### 3. Fairness and equity

FairMarket-RL (2025) and the distributionally-fair trading line (Ruiz Irusta &
Morales 2025) push fairness into the training objective, sometimes with
LLM-guided reward shaping. This project reports fairness (Jain-style index and
individual rationality per household) as first-class evaluation metrics and
gates on IR, but does not shape rewards toward fairness. That is a defensible
scope decision: fairness here is measured, not optimised, and the measured
values (IR = 1.000, fairness ≈ 0.999) suggest the auction plus constraints
already deliver it on this data.

### 4. LLM-enhanced MARL

A visible 2025–26 strand (e.g. LLM expert workflows guiding CTDE training,
arXiv 2507.14995) uses LLMs to generate personalised bidding strategies or
shape rewards. Deliberately out of scope: it adds an unauditable component to
a project whose thesis is that careful, inspectable engineering beats added
complexity. Cite it as future work, not as an omission.

### 5. Federated / privacy-preserving RL

Federated RL for P2P trading (Applied Energy 2025–26) addresses data privacy
across households. Out of scope here — the dataset is centrally available and
the contribution is elsewhere — but one sentence in future work covers it.

### 6. Network-aware trading

Guerrero et al. (2019) through Feng et al. (2023–2025) incorporate voltage and
line constraints. This project's feeder is a single aggregate import/export
limit by design, and `PROVENANCE.md` already states that no power-flow claim
is made. Do not let a slide imply otherwise.

## The one-paragraph version for the report

Recent work on P2P energy trading with MARL splits into constraint-aware
training (Lagrangian methods, state augmentation, safety critics), market
mechanism design (double auctions, mean-field clearing), fairness-aware
objectives, and LLM- or federation-enhanced variants. This project sits in the
first two strands: a constrained MAPPO trader in a uniform-price double
auction, with the 2024–2026 state-augmentation technique adopted after
measurement and two other recent techniques implemented and rejected on
evidence at this sample budget. Fairness and emissions are reported as
evaluation metrics rather than training objectives, and network physics beyond
an aggregate feeder limit is explicitly out of scope.
