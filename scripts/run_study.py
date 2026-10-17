"""Main experiment: multi-seed MAPPO training plus full policy comparison.

    python scripts/run_study.py --seeds 0 1 2 --updates 150 --households 25
"""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)

from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import perfect_foresight_lp
from src.env.community_env import EnvConfig
from src.eval.harness import compare_policies, headline_rows
from src.eval.statistics import aggregate_runs, format_table, probability_of_improvement, savings_recovered
from src.marl.mappo import MAPPO, MAPPOConfig
from src.marl.trainer import Trainer, TrainConfig
from src.utils.config import load_config, build_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--households", type=int, default=None)
    ap.add_argument("--out", default="results/main_study")
    ap.add_argument("--tag", default="mappo")
    ap.add_argument("--overrides", default=None,
                    help="JSON dict of algo-config overrides, e.g. '{\"shared_reward\": true}'")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    if args.updates: cfg["train"]["updates"] = args.updates
    if args.households: cfg["data"]["households"] = args.households
    if args.overrides: cfg["algo"].update(json.loads(args.overrides))

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    community = load_community(root / cfg["data"]["path"])
    tensors = build_tensors(stratified_subset(community, cfg["data"]["households"],
                                              seed=cfg["data"]["subset_seed"]))
    print(f"Community: {tensors.n_agents} households | {tensors.meta}")

    env_cfg, algo_cfg, train_cfg = build_configs(cfg)
    lp = perfect_foresight_lp(tensors, env_cfg)
    if lp: print(f"Perfect-foresight LP optimum: EUR {lp['optimal_cost_eur']:.2f}")

    per_seed, agents = [], {}
    for s in args.seeds:
        print(f"\n===== seed {s} =====")
        a_cfg = MAPPOConfig(**{**asdict(algo_cfg), "seed": s})
        t_cfg = TrainConfig(**{**asdict(train_cfg), "seed": s})
        tr = Trainer(tensors, EnvConfig(**asdict(env_cfg)), a_cfg, t_cfg,
                     out_dir=out / f"seed{s}")
        res = tr.train(verbose=True)
        agent, extra = MAPPO.load(out / f"seed{s}" / "best.pt")
        agent.residual_scale = extra.get("residual_scale", train_cfg.residual_scale_final)
        agent.multipliers = extra.get("multipliers", {})
        agents[f"{args.tag}_seed{s}"] = agent
        per_seed.append({"seed": s, "best": res["best"]["update"],
                         "wall_min": res["wall_time_s"] / 60})

    results = compare_policies(tensors, env_cfg, agents,
                               episodes=cfg["train"]["eval_episodes"])
    rows = headline_rows(results)
    cols = ["policy", "cost_eur", "vs_grid_only_%", "savings_recovered_%",
            "individual_rationality", "fairness_jain", "p2p_volume_kwh",
            "self_sufficiency", "peak_import_kw", "unserved_kwh", "gate"]
    table = format_table(rows, cols, f"Policy comparison ({tensors.n_agents} households)")
    print(table)

    learned = [results[k] for k in agents]
    agg = aggregate_runs(learned)
    base = float(np.mean(results["no_flexibility"]["net_cost_eur"]))
    heur = [float(np.mean(results["tariff_aware_p2p"]["net_cost_eur"]))] * len(learned)
    mine = [float(np.mean(r["net_cost_eur"])) for r in learned]
    lp_cost = lp["optimal_cost_eur"] if lp else None
    summary = {
        "lp": lp, "per_seed": per_seed,
        "aggregate_learned": agg,
        "grid_only_cost": base,
        "heuristic_cost": heur[0],
        "savings_recovered_mean": savings_recovered(agg["net_cost_eur"]["mean"], base, lp_cost) if lp else None,
        "savings_recovered_heuristic": savings_recovered(heur[0], base, lp_cost) if lp else None,
        "p_improvement_vs_heuristic": probability_of_improvement(mine, heur),
        "gate_pass_rate": float(np.mean([results[k]["_gate"]["passed"] for k in agents])),
        "table": table,
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(out / "comparison.txt", "w") as f:
        f.write(table)
    print(f"\nSaved -> {out}/summary.json")
    if lp:
        print(f"Savings recovered: heuristic {100*summary['savings_recovered_heuristic']:.1f}% "
              f"| MAPPO {100*summary['savings_recovered_mean']:.1f}%")
    print(f"Safety-gate pass rate across seeds: {100*summary['gate_pass_rate']:.0f}%")


if __name__ == "__main__":
    main()
