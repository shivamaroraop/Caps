# Ablation study

Which components of the constrained MAPPO system are responsible for the
result. Each row removes exactly one component from the full configuration;
everything else — seeds, budget, evaluation day, environment — is held fixed.
Reproduce with `python scripts/run_ablation.py --seeds 0 1 2 --updates 150`.

An earlier pass at 80 updates and 2 seeds (retained in the appendix below) was
too short to converge — the `full` reference itself failed one seed — so it was
rerun at 150 updates and 3 seeds. Only the longer run is used for conclusions.

## Results (3 seeds, 150 updates)

| Variant | Removes | Mean cost € | Range € | Mean unserved kWh | Max unserved | Gate pass |
|---|---|---:|---:|---:|---:|:--:|
| **full** | — | **143.79** | 141.30 – 147.04 | **0.00** | 0.00 | **3/3** |
| mlp_critic | attention critic → plain MLP | 145.96 | 143.74 – 147.07 | 0.00 | 0.00 | 3/3 |
| no_lagrangian | PID-Lagrangian dual | 144.90 | 141.94 – 146.98 | 0.90 | 2.69 | 2/3 |
| shared_reward | per-agent reward → team mean | 148.39 | 143.24 – 153.30 | 0.00 | 0.00 | 2/3 |
| team_advantage | per-agent advantage → team mean | 147.28 | 146.17 – 148.21 | 6.61 | 19.84 | 2/3 |
| no_value_norm | value normalisation | 146.26 | 142.28 – 149.15 | 2.17 | 6.04 | 1/3 |
| single_env | 8 parallel envs → 1 | 145.61 | 143.50 – 148.19 | 11.23 | 18.10 | 1/3 |
| no_shield | EV safety projection | 74.37 | 74.30 – 74.44 | 0.00 | 0.00 | 0/3 |

Individual rationality was 1.000 for every variant and seed and is omitted.

## The headline

**The full system is the only configuration that passes the deployment gate on
every seed.** Every ablation breaks at least one seed. No component is dead
weight — though, as the rows below make clear, they matter for different
reasons and by different amounts.

## Reading each row

### no_shield — an integrity finding, not a cost result

€74 is *below* the perfect-foresight LP bound of €87.52. That is impossible for
a legitimate policy. What happened is that without the EV feasibility
projection the network learned to simply not charge the EVs: load is served,
individual rationality holds, cost collapses, and the gate fails on EV
shortfall. **The shield is what makes the cost metric non-gameable.** This row
must never be reported as a cost number without this explanation attached; it
belongs on a slide about why the gate has an EV-shortfall check.

### single_env — the most damaging legitimate ablation

Two of three seeds leave 15–18 kWh unserved. Parallel environments are what
give each PPO update enough diverse experience to learn the feeder constraint;
one environment per update produces a policy that is cost-competitive but
unsafe. This maps directly onto a legacy-code correction (the inherited
implementation used a single environment).

### no_value_norm — most important under a small budget

Two of three seeds fail; cost is worse on two of three. At 80 updates (appendix)
this was the worst legitimate row by a wide margin (€157–163, 18–35 kWh
unserved). At 150 updates the critic partly recovers, so the effect narrows.
Honest phrasing: value normalisation matters most when the sample budget is
small — which is precisely the regime the legacy code was in when its
1.4 × 10⁵ value loss starved the actor.

### team_advantage — coordination signal lost

Cost worse on all three seeds; one seed leaves 19.8 kWh unserved. Averaging
advantages across agents (the legacy `rewards.mean()` behaviour) discards the
individual signal each household needs to learn its share of feeder
coordination.

### shared_reward — safe but a worse credit signal

Zero unserved on every seed, but seed 2 costs €153.30 — worse than the
heuristic — and fails the gate on cost ratio. A team-mean reward is safe here,
just less informative than per-agent reward for cost.

### no_lagrangian — a robustness component, not a cost driver

Resolved, and subtle: cost is essentially tied with `full`, and two seeds are
fine; one seed leaves 2.7 kWh unserved. The honest reading is that the shield
and potential-based shaping already do most of the constraint work, and the
Lagrangian dual is what closes the last seed. It is worth stating plainly that
the Lagrangian earns its place on robustness across seeds, not on cost.
The earlier A/B test (`AB_TEST.md`) points the same way: putting λ *in the
observation* was a safety win with cost a wash.

### mlp_critic — attention accelerates, but is not necessary at full budget

At 80 updates the MLP critic left 15–17 kWh unserved on both seeds. At 150
updates it passes 3/3 with cost roughly tied (mean €145.96 vs €143.79). The
conclusion is that attention over the agent set speeds convergence and gives a
small cost edge, but is not required once the budget is large enough. Attention
is retained — it is faster, slightly cheaper, and its weights are inspectable —
but it should not be claimed as essential.

## The one-paragraph version for the report

Ablating one component at a time over three seeds shows that the full system
is the only configuration that satisfies the deployment gate on every seed.
Parallel environments and value normalisation are the largest contributors to
constraint satisfaction; per-agent advantages and per-agent rewards improve
cost; the PID-Lagrangian dual closes the last seed rather than driving cost;
the attention critic accelerates convergence but is not necessary at full
budget; and the EV safety projection is what prevents the policy from gaming
the cost metric by abandoning EV charging — without it the reported cost drops
below the perfect-foresight bound, which is impossible for a legitimate policy.

## Appendix — the 80-update, 2-seed pass

Kept for transparency. Not used for conclusions because the `full` reference
itself failed one seed at this budget.

| Variant | Cost € (s0, s1) | Unserved kWh (s0, s1) | Gate |
|---|---|---|---|
| full | 147.01, 148.13 | 0.00, 18.00 | 1/2 |
| shared_reward | 143.23, 150.34 | 0.00, 0.00 | 1/2 |
| team_advantage | 142.41, 145.75 | 1.32, 15.66 | 0/2 |
| no_value_norm | 157.32, 163.31 | 18.15, 34.50 | 0/2 |
| no_lagrangian | 145.27, 148.38 | 0.00, 20.78 | 1/2 |
| mlp_critic | 147.08, 147.48 | 16.54, 15.39 | 0/2 |
| single_env | 146.40, 142.89 | 0.00, 10.03 | 1/2 |
| no_shield | 81.32, 81.44 | 0.00, 0.00 | 0/2 |
