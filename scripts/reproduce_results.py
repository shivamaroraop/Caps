"""Reproduce the reported policy comparison from the SAVED checkpoints.

No training. Loads results/main_study/seed*/best.pt, runs every policy on the
deterministic evaluation day, and prints the same table as RESULTS.md.
Takes about a minute on a laptop CPU.

    python scripts/reproduce_results.py
    python scripts/reproduce_results.py --study results/main_study --check

--check compares the freshly computed costs against the stored summary.json
and reports PASS/FAIL per policy (tolerance 0.05 EUR).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)

from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import perfect_foresight_lp
from src.eval.harness import compare_policies, headline_rows
from src.eval.statistics import format_table
from src.marl.mappo import MAPPO
from src.utils.config import load_config, build_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--study", default="results/main_study")
    ap.add_argument("--check", action="store_true",
                    help="verify against the stored summary.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    study = root / args.study

    community = load_community(root / cfg["data"]["path"])
    tensors = build_tensors(stratified_subset(community, cfg["data"]["households"],
                                              seed=cfg["data"]["subset_seed"]))
    env_cfg, algo_cfg, train_cfg = build_configs(cfg)
    print(f"Community: {tensors.n_agents} households")

    lp = perfect_foresight_lp(tensors, env_cfg)
    if lp:
        print(f"Perfect-foresight LP optimum: EUR {lp['optimal_cost_eur']:.2f}")

    agents = {}
    for ck in sorted(study.glob("seed*/best.pt")):
        s = ck.parent.name.replace("seed", "")
        agent, extra = MAPPO.load(ck)
        agent.residual_scale = extra.get("residual_scale", train_cfg.residual_scale_final)
        agent.multipliers = extra.get("multipliers", {})
        agents[f"mappo_seed{s}"] = agent
    if not agents:
        sys.exit(f"No checkpoints found under {study}/seed*/best.pt")
    print(f"Loaded {len(agents)} saved MAPPO checkpoints")

    results = compare_policies(tensors, env_cfg, agents,
                               episodes=cfg["train"]["eval_episodes"])
    rows = headline_rows(results)
    cols = ["policy", "cost_eur", "vs_grid_only_%", "savings_recovered_%",
            "individual_rationality", "fairness_jain", "p2p_volume_kwh",
            "self_sufficiency", "peak_import_kw", "unserved_kwh", "gate"]
    print(format_table(rows, cols, f"Policy comparison ({tensors.n_agents} households)"))

    if args.check:
        stored = json.load(open(study / "summary.json"))["results"]
        print("\nVerification against stored summary.json (tolerance 0.05 EUR):")
        ok = True
        for k, v in results.items():
            if k not in stored or "net_cost_eur" not in v:
                continue
            new = float(np.mean(v["net_cost_eur"]))
            old = float(np.mean(stored[k]["net_cost_eur"]))
            good = abs(new - old) < 0.05
            if "_gate" in v and "_gate" in stored[k]:
                good &= (v["_gate"]["passed"] == stored[k]["_gate"]["passed"])
            ok &= good
            print(f"  {k:24s} stored {old:8.2f}  now {new:8.2f}  {'PASS' if good else 'FAIL'}")
        print("\nALL RESULTS REPRODUCED" if ok else "\nMISMATCH — see rows above")


if __name__ == "__main__":
    main()
