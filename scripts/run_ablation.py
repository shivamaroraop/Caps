"""Ablation study: turn each correction off and measure what breaks.

Every variant below re-enables one specific defect of the legacy
implementation, holding everything else fixed.  This is what converts "we
rewrote it and it works better" into evidence about *which* change mattered.

    python scripts/run_ablation.py --seeds 0 1 --updates 80
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)

from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import EnvConfig
from src.eval.harness import compare_policies
from src.eval.statistics import format_table
from src.marl.mappo import MAPPO, MAPPOConfig
from src.marl.trainer import Trainer, TrainConfig
from src.utils.config import build_configs, load_config

VARIANTS = {
    "full": ({}, {}, "All corrections enabled (proposed method)"),
    "shared_reward": ({"shared_reward": True}, {},
                      "Legacy defect: reward averaged across agents before storage"),
    "team_advantage": ({"per_agent_advantage": False}, {},
                       "Legacy defect: team-mean advantage broadcast to every agent"),
    "no_value_norm": ({"normalize_value": False}, {},
                      "Legacy defect: unnormalised value targets"),
    "no_lagrangian": ({"use_lagrangian": False}, {},
                      "Fixed penalties instead of learned shadow prices"),
    "mlp_critic": ({"critic": "mlp"}, {}, "No attention over the household set"),
    "single_env": ({}, {"n_envs": 1}, "Legacy defect: one episode per PPO update"),
    "no_shield": ({}, {}, "EV feasibility projection disabled"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--updates", type=int, default=80)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    cfg["train"]["updates"] = args.updates
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    community = load_community(root / cfg["data"]["path"])
    tensors = build_tensors(stratified_subset(community, cfg["data"]["households"],
                                              seed=cfg["data"]["subset_seed"]))
    env_cfg, algo_cfg, train_cfg = build_configs(cfg)

    rows, raw = [], {}
    for name in args.variants:
        algo_over, train_over, desc = VARIANTS[name]
        costs, irs, unserved, passes = [], [], [], []
        for s in args.seeds:
            e = EnvConfig(**asdict(env_cfg))
            if name == "no_shield":
                e.ev_safety_projection = False
            a = MAPPOConfig(**{**asdict(algo_cfg), **algo_over, "seed": s})
            t = TrainConfig(**{**asdict(train_cfg), **train_over, "seed": s})
            if "n_envs" in train_over:
                e.n_envs = train_over["n_envs"]
            tr = Trainer(tensors, e, a, t, out_dir=out / name / f"seed{s}")
            tr.train(verbose=False)
            agent, extra = MAPPO.load(out / name / f"seed{s}" / "best.pt")
            agent.residual_scale = extra.get("residual_scale", t.residual_scale_final)
            agent.multipliers = extra.get("multipliers", {})
            res = compare_policies(tensors, e, {"x": agent},
                                   episodes=cfg["train"]["eval_episodes"],
                                   include_lp=False)["x"]
            costs.append(float(np.mean(res["net_cost_eur"])))
            irs.append(float(np.mean(res["individual_rationality"])))
            unserved.append(float(np.mean(res["unserved_kwh"])))
            passes.append(bool(res["_gate"]["passed"]))
            print(f"  {name} seed{s}: cost={costs[-1]:.2f} IR={irs[-1]:.3f} "
                  f"unserved={unserved[-1]:.2f} gate={'PASS' if passes[-1] else 'FAIL'}")
        raw[name] = {"cost": costs, "ir": irs, "unserved": unserved, "gate": passes}
        rows.append({"variant": name, "description": desc,
                     "cost_eur": f"{np.mean(costs):.2f}",
                     "cost_spread": f"{np.max(costs) - np.min(costs):.2f}",
                     "IR": f"{np.mean(irs):.3f}",
                     "unserved_kwh": f"{np.mean(unserved):.2f}",
                     "gate_pass": f"{int(np.sum(passes))}/{len(passes)}"})

    table = format_table(rows, ["variant", "cost_eur", "cost_spread", "IR",
                                "unserved_kwh", "gate_pass", "description"],
                         "Ablation: each row disables one correction")
    print(table)
    (out / "ablation.txt").write_text(table)
    (out / "ablation.json").write_text(json.dumps(raw, indent=1))


if __name__ == "__main__":
    main()
