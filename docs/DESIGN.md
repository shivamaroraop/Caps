# Design decisions

This document records *why* the system is built the way it is. Every entry
names the alternative that was rejected and the reason. It is written to be
defended in a viva.

---

## 1. Problem formulation

### 1.1 Why a constrained Markov game, not a plain MDP

The community optimum is not "minimise total cost". A policy that saves the
community €40 by making six households worse off than buying from the grid is
not deployable: those households leave, and the market they leave is smaller
and less efficient. The objective is therefore constrained:

> maximise expected community return, subject to
> (a) each household at least as well off as its grid-only outside option,
> (b) feeder import/export within capacity,
> (c) every EV meeting its mobility requirement.

Constraints (a) and (b) are economic trade-offs whose correct shadow price is
unknown before training. Constraint (c) is not negotiable at any price.
They are therefore handled by different mechanisms (§4).

### 1.2 Why the outside option includes EV charging

Charging an EV is a *service delivered*, not a cost overrun. If the
counterfactual is "load minus PV settled with the grid" and excludes EVs, then
every EV-owning household spends money the counterfactual does not, and is
worse off by construction. Individual rationality then measures EV ownership,
not market design.

The counterfactual is uncontrolled ("dumb") charging: full power on arrival
until the target is met, no storage dispatch, no market.

### 1.3 Why the counterfactual is simulated rather than derived

It is produced by running an uncontrolled-charging policy through the *same*
`step()` function the learned policy uses, so it is settled under identical
physics: feeder curtailment, scarcity pricing, conversion losses. A parallel
closed-form expression, however carefully derived, drifts away from the
simulator as the model changes — which is exactly what happened in the earlier
version, where the counterfactual assumed an infinite grid.

`test_no_flexibility_reproduces_the_counterfactual` asserts agreement to 1e-4.
This turns the baseline from an assumption into a checked invariant.

---

## 2. Data layer

### 2.1 Home charge point rating

`EVs!Charge (kW)` reports 55–121 kW. These are **vehicle DC charge-acceptance**
ratings. The sheets `Max EV1/EV2 Dis-Charge` are time-varying: 7.4 or 11 kW
while the vehicle is at home (single-phase 32 A and three-phase 16 A AC home
charge points) and 22–60 kW while it is at a public station.

Taking the maximum of that profile — the naive read — installs a DC fast
charger in the living room. One household could then saturate a feeder sized
for twenty-five. The home rating is recovered as the smallest positive value
observed on steps where `at Home` is true.

*Rejected alternative:* hard-coding 7.4 kW everywhere. The dataset genuinely
distinguishes single- and three-phase households, and that heterogeneity is
useful — it makes some households more flexible than others, which is what
gives the P2P market something to arbitrage.

### 2.2 Two-phase EV targets

`Initial battery level` = `Final battery level` = 20% of capacity, while
recorded trips consume 3–14 kWh. Read as a departure target, the requirement is
satisfied at t = 0 and EV flexibility demand is zero; read as a *physical
starting state*, the fleet cannot complete its own trips.

The coherent reading is two requirements:

* **departure**: `trip_consumption + 10% reserve`, required at the departure
  period;
* **end of day**: back to `Final battery level`, required at t = 95.

This creates a morning charging window and an evening recharging window. The
evening window is where the interesting control problem lives: vehicles return
between 19:45 and 21:30, and if they all charge on arrival the feeder overloads.

### 2.3 Stratified subsetting

Uniform sampling of 25 from 250 households routinely produced communities with
almost no pure consumers, making the market one-sided and inflating the
apparent value of trading. Households are stratified on the (PV, BESS, EV)
archetype with largest-remainder apportionment, so the 25-household community
has the same 80% PV / 60% BESS / 100% EV mix as the full population.

### 2.4 Per-household reward scaling

Rewards are divided by a per-household cost scale derived from its own daily
energy throughput. Without this the shared policy is dominated by the largest
consumers: a 60 kWh/day house with two EVs generates gradients an order of
magnitude larger than a 12 kWh/day flat, and the policy simply ignores the
flat.

---

## 3. Environment

### 3.1 Uniform-price double auction

Clearing sorts bids descending and asks ascending, then finds the largest
quantity at which the marginal bid still covers the marginal ask, via two
`searchsorted` calls over the union of cumulative-quantity breakpoints. All
matched energy settles at one price.

*Why uniform rather than pay-as-bid:* under uniform pricing a quote determines
*whether* an agent trades, not *what* it pays. The incentive to shade is far
weaker, which keeps the learning problem well posed — under pay-as-bid the
policy spends its capacity learning to game its own price signal rather than
learning to schedule energy. Pay-as-bid remains available
(`market_mechanism: midpoint`) for ablation.

*Why quotes are clipped to `[sell_price, buy_price]`:* a rational agent never
bids above the grid import tariff (it could buy from the grid) and never asks
below the export tariff (it could sell to the grid). Restricting quotes to the
spread removes only dominated actions, and removes a whole failure mode in
which an untrained policy quotes nonsense and destroys welfare.

### 3.2 Feeder-level flexibility curtailment

The earlier model applied the feeder limit as pure accounting: devices
dispatched whatever they wanted, and the excess import was labelled "unserved"
and priced. That does not conserve energy — the battery charges from kWh that
never crossed the meter.

Here the constraint acts on the dispatch:

1. curtail flexible charging (battery, non-firm EV) pro rata;
2. curtail discharging, then PV export, on the export side;
3. only the residue that no flexibility can absorb is unserved, priced at
   3× the tariff.

Charging demanded by the safety shield is **firm** and is not curtailed, since
it is the minimum that still meets a mobility deadline.

`test_energy_balance_holds_every_step` checks conservation to within
round-trip conversion losses.

### 3.3 Feeder sizing

Per-household *connection* capacity in the dataset is ~17.3 kW, but a feeder is
never sized at N × 17.3. The default is 8 kW/household import and 5 kW/household
export — above the 2.5 kW/hh non-EV peak and the 1.4 kW/hh daily mean, and well
below the 11.3 kW/hh peak that uncontrolled simultaneous EV charging produces.

The constraint therefore binds exactly when coordination fails, which is the
regime the project is about. It is exposed in the config so the sensitivity can
be swept.

### 3.4 Markovian reward

Cumulative cost burden, community saving rate, fairness and the individual
rationality margin are all included in the observation and the global state.
The earlier reward depended on cumulative quantities the critic could not
observe, which places an irreducible floor on value error no amount of training
removes.

### 3.5 Potential-based storage shaping

Φ(s) = stored energy expressed in units of the household's own typical per-step
throughput; the shaping term is γΦ(s′) − Φ(s) (Ng, Harada & Russell, 1999).

**Honest caveat:** strict policy invariance in the episodic setting requires
Φ(terminal) = 0. Here Φ is *retained* at the terminal step, because end-of-day
stored energy has genuine value and forcing Φ to zero makes the policy dump the
battery at 23:45. The term is therefore a deliberate **terminal value function
for storage**, not a neutral shaping term, and it is declared as such.

---

## 4. Three-layer safety architecture

| layer | what it is | what it enforces | why at this layer |
|---|---|---|---|
| 1 | residual trust region, κ annealed 0.15 → 1.0 | exploration stays near a known-good controller early | acts on the *action space*, so it cannot bias the objective — only the region explored |
| 2 | least-laxity EV shield, device limits | mobility deadlines, SoC bounds, connection limits | never negotiable; a policy should not discover them by trial and error |
| 3 | PID-Lagrangian duals | individual rationality, feeder congestion | genuine economic trade-offs whose shadow price is unknown a priori, so it is *learned* |

### 4.1 Why κ anneals to 1.0, not to 0.6

With κ capped below 1, a reference action of 1.0 (charge at full power) can only
be corrected down to 1 − κ. The policy is then *structurally incapable* of
suspending EV charging during a feeder constraint — the action it provably
needs is outside its reach. Annealing to 1.0 makes the reference controller an
initialisation prior rather than a permanent cage.

### 4.2 Why the shield tests laxity in steps, not energy slack

An earlier version fired when
`remaining_steps × P × η × dt − deficit ≤ margin`. On the final step, with one
step left and a 0.4 kWh deficit, the slack is still positive, so the shield
declines to fire and the target is missed by exactly the amount it was
protecting. That defect produced ~10 kWh/day of unmet mobility demand while
reporting the shield as active.

The correct test asks what the vehicle could gain if it *declined* to charge
now and then charged flat out for every remaining connected step. If that is
less than the deficit, this step is no longer optional.

The shield also raises the setpoint only to the level *required*, not to full
power: forcing every shielded vehicle to maximum draw manufactures precisely
the synchronised peak the feeder constraint exists to prevent.

### 4.3 Why PID-Lagrangian rather than fixed penalties

A fixed weight is a guess at a shadow price, and the guess is wrong in opposite
directions at different stages: too small early, when violation is large and
pressure is needed; too large late, when the constraint is slack and the penalty
only distorts the economic objective. Plain dual ascent oscillates because the
multiplier responds only to the accumulated integral of violation and therefore
always overshoots. Proportional and derivative terms damp it (Stooke, Achiam &
Abbeel, 2020).

### 4.4 The reference controller uses least-laxity charging

"Charge at full power whenever connected and cheap" synchronises the whole
fleet onto the same cheap hours and is the single largest source of congestion
in this dataset. The reference instead tracks urgency — the fraction of
remaining charging capability the vehicle actually needs — modulated by the
tariff percentile. Vehicles with slack charge gently; vehicles near a deadline
charge hard. Because urgency differs across households by construction, the
fleet does not synchronise.

Replacing the naive rule with this one reduced heuristic cost by ~12% and
unserved energy by ~77% *before any learning*. It is important to say so: part
of the improvement over the original project is a better baseline, not the RL.

---

## 5. Learner

### 5.1 Agent-specific attention critic

One value per household. Query is formed from the household's own observation
concatenated with the global state; keys and values are the encoded
observations of every household. Yu et al. (2022) find that giving the critic
both local and global information is the most effective configuration.

Per-agent values make per-agent advantages possible; per-agent advantages make
credit assignment possible.

**It is not a GNN.** The dataset supplies no feeder topology, so there is no
adjacency matrix. Attention over an unordered set is the honest description,
and `MLPCritic` is included as the no-attention ablation.

### 5.2 Storing the pre-squash action

The buffer stores `z`, not `tanh(z)`. Recovering `z` by inverting the squash is
stable only while actions stay away from the boundary; at the usual clamp,
`atanh(0.999999) = 7.25` against a policy with σ ≈ 0.17, so the recovered
log-probability is wrong by tens of nats and the PPO ratio explodes.
`test_stored_pre_squash_action_gives_exact_log_probs` demonstrates the
divergence.

### 5.3 Separate optimisers and separate clips

The single change that restores a usable actor learning rate. See README §3.

### 5.4 Model selection

The checkpoint with the lowest cost is *not* selected. Selection uses a score
that is strictly worse for any policy failing the deployment gate, so an unsafe
checkpoint can never be promoted over a safe one.

---

## 6. Evaluation

### 6.1 The LP bound is the important reference

Without it, the strongest available claim is "our method beats our heuristics",
which is unfalsifiable — a weaker heuristic makes the method look better. The
LP gives the least-cost dispatch under perfect foresight, so results can be
stated as *share of achievable saving recovered*.

Formulated with explicit SoC variables and banded linking equalities rather
than cumulative sums over the control history: mathematically identical, O(T)
non-zeros per device instead of O(T²), and the difference between a one-second
solve and a several-minute one at scale.

It is a relaxation in one respect — it nets community import and export at the
same instant — which the agent-level simulation also permits through the P2P
market. It is a valid upper bound, not an attainable policy.

### 6.2 IQM and bootstrap intervals

With four to five seeds the sample mean is dominated by outliers and point
estimates without intervals are not interpretable (Agarwal et al., 2021). The
interquartile mean and a percentile bootstrap interval are reported for every
metric, along with the probability that MAPPO beats the heuristic on a *new*
seed — which is the question a reader actually has.

### 6.3 The deployment gate

Explicit, auditable criteria: individual rationality, fairness, zero unserved
energy, zero EV shortfall, and cost no worse than the reference controller.
`require_all_seeds` defaults to true, so a policy that passes on average but
fails on one seed is reported as failing. That distinction is precisely what
the original study elided when it described MAPPO as "sometimes very good".

---

## 7. What would change with more resources

* **Multiple days.** One 96-step day cannot demonstrate seasonal
  generalisation. A year of data would allow proper train/validation/test
  splits across days.
* **A power flow model.** The feeder is a single aggregate capacity. Replacing
  it with a linearised DistFlow model would allow voltage constraints and
  genuine line-level congestion, and would justify a graph critic.
* **Strategic agents.** Households are cooperative here. Modelling them as
  self-interested and measuring the price of anarchy against this cooperative
  optimum would be the natural next study.
* **Longer training and more seeds.** The reported runs are budget-limited;
  every number improves monotonically with updates over the range tested, so
  the reported figures are lower bounds on what the method reaches.
