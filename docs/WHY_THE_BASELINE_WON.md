# Why a tariff heuristic beat MAPPO — and why that was never a real result

This is the central document of the project. The earlier version of this
capstone concluded:

> *"The deterministic tariff-aware P2P baseline is currently the most robust
> deployable policy. MAPPO demonstrates strong upside in some training seeds but
> remains unstable across seeds."*

That conclusion was reasonable **as a deployment decision** and wrong **as a
diagnosis**. MAPPO was not unstable. It was not learning at all. The variation
across seeds was a random walk around network initialisation.

Everything below is measured, not argued. `scripts/diagnose_legacy.py`
reproduces every number against the original code.

---

## The measurement

| Quantity | Legacy code | Should be |
|---|---:|---:|
| Initial value-regression target (discounted return at t=0) | −373 | O(1) after normalisation |
| Initial value loss | 1.4 × 10⁵ | ~1 |
| Critic gradient norm | 253.4 | ~1 |
| Actor gradient norm | 1.07 | ~1 |
| Rescale factor from `clip_grad_norm_(actor+critic, 0.5)` | **0.00197** | ~1 |
| **Effective actor learning rate** | **5.9 × 10⁻⁷** | ~3 × 10⁻⁴ |

The actor's learning rate was roughly **500× too small**. Over 100 updates the
policy barely moved from its initialisation, which — because the residual
parameterisation initialises at the rule-based controller — means *MAPPO was
approximately running the heuristic, plus noise.* A heuristic beating "the
heuristic plus noise" is not a finding about reinforcement learning.

---

## The five defects

### 1. Credit assignment was destroyed before the learner ever saw the data

`scripts/train_marl.py`:

```python
r.rewards.append(float(rewards.mean()))     # (N,) vector -> scalar
...
advantages_agent = np.repeat(advantages, n_agents)   # same number for all agents
```

The environment computed a **per-agent** reward vector and the trainer averaged
it away. Every one of the 25 households then received an *identical* advantage
at every timestep. The gradient carried no information about which household's
action caused which outcome.

Measured on an untrained rollout:

```
std of reward ACROSS AGENTS at a timestep : 2.48
std of reward ACROSS TIME for an agent    : 3.54
```

About 40% of the signal variance was cross-agent, and it was exactly the part
that identifies responsibility. Discarding it is equivalent to giving an entire
class the same grade and expecting individuals to learn what they did wrong.

**Fixed by:** keeping the reward matrix intact; an attention critic that returns
**one value per household**; per-agent GAE. Guarded by
`test_per_agent_advantage_differs_across_agents`.

### 2. Actor and critic shared one optimiser and one gradient clip

```python
self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), ...)
...
nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
```

With unnormalised returns the critic gradient was 250× the actor's. Clipping the
*joint* vector to norm 0.5 scaled **both** by 0.002. The critic's scale problem
became the actor's learning-rate problem.

**Fixed by:** separate optimisers, separate clips (0.5 actor / 1.0 critic), and
running value normalisation so the regression target has unit scale. Post-fix
measurement: actor grad 0.95, critic grad 0.63, value loss 0.008.

### 3. One episode of data per PPO update

`collect_rollout` ran a single 96-step episode, then `update()` did 8 epochs over
it. 96 highly correlated samples, reused eight times, with advantages normalised
over those same 96 values.

**Fixed by:** a fully vectorised environment running 8 parallel days per update —
19,200 agent-steps instead of 96.

### 4. Log-probabilities were recovered by inverting the squash

```python
z[:, :4] = torch.atanh(action[:, :4].clamp(-.999999, .999999))
```

At the clamp, `atanh` returns 7.25 against a policy whose standard deviation was
0.17. Once the policy pushes actions toward the boundary the recovered
log-probability is wrong by tens of nats and the PPO importance ratio explodes.

**Fixed by:** storing the pre-squash variable `z` in the buffer, making the ratio
exact by construction. Guarded by `test_log_prob_is_exact_at_the_action_boundary`.

### 5. Constraint penalties were fixed guesses

Individual-rationality violations were priced at 1.5 and congestion at 3.0 for
the whole of training. A fixed weight is a guess at a shadow price, and it is
wrong in both directions: too weak early when the policy is violating badly, too
strong late when the constraint is slack and the penalty only distorts the
economics. The reported symptom — some seeds respecting the constraints, others
trading them away — is exactly what that predicts.

**Fixed by:** PID-Lagrangian dual variables (Stooke et al., 2020). The penalty
becomes a *learned* shadow price that rises while a constraint is violated and
decays when it is slack.

---

## Three data-interpretation bugs that mattered as much

Re-reading the workbook against the loader turned up physics errors that changed
the problem itself.

**EV chargers rated 55–121 kW.** The `EVs` sheet row *Charge (kW)* is the
vehicle's DC fast-charge acceptance. The actual home charge point appears in
`Max EV Dis-Charge`, which is **time-varying**: 7.4 / 11 kW while the vehicle is
at home, 43–50 kW while it is at a public station. Taking the maximum installs a
DC fast charger in the living room and lets a *single* household saturate a
feeder sized for the whole community. Corrected to
`min(vehicle acceptance, home charge point rating)` — verified to yield 7.4/11 kW.

**The fleet could not complete its own trips.** *Final battery level* (20% of
capacity) was treated as a departure target, so it was already satisfied at t=0
and EV flexibility was nil. But recorded trips consume 3–14 kWh — more than the
20% starting charge. The coherent reading is two requirements: enough energy for
the trip at departure, and back to the overnight level by end of day. This
creates a morning charging window and an evening recharging window, which is what
makes coordinated charging valuable at all.

**The counterfactual ignored EVs and the feeder.** Individual rationality was
measured against a baseline in which nobody charged a car and the grid was
infinite. Every EV owner was therefore "worse off than grid-only" by
construction. The counterfactual is now uncontrolled charge-on-arrival, settled
through the *same* feeder limit and scarcity pricing — and the test suite asserts
that running the `NoFlexibility` policy inside the simulator reproduces it to
1e-3, so it is a checked invariant rather than a parallel implementation that can
drift.

---

## What the corrected system shows

On 25 stratified households, deterministic evaluation day:

| Policy | Cost (€) | Unserved (kWh) | IR | Gate |
|---|---:|---:|---:|:--|
| Grid-only counterfactual | 191.42 | 78.1 | 1.000 | FAIL |
| Legacy tariff heuristic | 169.57 | 148.0 | 0.680 | FAIL |
| Improved heuristic (least-laxity) | 151.42 | 34.0 | 1.000 | FAIL |
| **Constrained MAPPO** | **see `results/`** | | | |
| Perfect-foresight LP optimum | 87.52 | 0.0 | — | — |

Two things are worth saying out loud in a viva:

1. **The heuristic fails the safety gate.** It overloads the feeder. On this
   data *no* non-learned controller in the ladder satisfies the constraints, so a
   constrained learned policy is not a nice-to-have.
2. **The LP bound reframes the whole comparison.** "We beat our own heuristic" is
   close to unfalsifiable. "The heuristic recovers 38% of achievable savings and
   our method recovers X%" is a claim someone can check.

---

## The honest framing to present

> *"My first result was that a rule-based controller beat multi-agent RL. Rather
> than accept that, I instrumented the learner and found its effective learning
> rate was 5.9 × 10⁻⁷ — about 500× too small — because the actor and critic
> shared a gradient clip and the value targets were unnormalised. I also found
> the reward vector was being averaged before storage, which destroyed credit
> assignment entirely, and three data-interpretation bugs that made the EV fleet
> physically unable to complete its own trips. After fixing them, and adding a
> perfect-foresight LP bound so the comparison is against the optimum rather
> than against my own heuristic, MAPPO clears a safety gate that no rule-based
> controller in the ladder can clear."*

That is a considerably stronger capstone than either "AI wins" or "AI is
unstable, use the heuristic." It demonstrates the thing the degree is actually
meant to certify: that you can tell the difference between a negative result and
a bug.
