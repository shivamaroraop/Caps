"""A/B test the proposed improvements against the previous configuration.

Each variant is trained on identical seeds and evaluated on identical days, so
the only difference is the change under test.  A change is kept only if it
improves the outcome; this script is the evidence for that decision.
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
    "v0_previous":   {"state_augmented": False, "safety_critic": False, "lam_sample_prob": 0.0},
    "v1_stateaug":   {"state_augmented": True,  "safety_critic": False, "lam_sample_prob": 0.25},
    "v2_safetycrit": {"state_augmented": False, "safety_critic": True,  "lam_sample_prob": 0.0},
    "v3_both":       {"state_augmented": True,  "safety_critic": True,  "lam_sample_prob": 0.25},
    # Does the *observation* of lambda help on its own, without spending part
    # of the training budget on randomly-sampled constraint pressure?
    "v1b_obsonly":   {"state_augmented": True,  "safety_critic": False, "lam_sample_prob": 0.0},
}
ENV_KEYS = {"state_augmented"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default="results/ab_test")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/default.yaml")
    cfg["train"]["updates"] = args.updates
    out = root / args.out; out.mkdir(parents=True, exist_ok=True)

    tensors = build_tensors(stratified_subset(
        load_community(root / cfg["data"]["path"]), cfg["data"]["households"],
        seed=cfg["data"]["subset_seed"]))
    env_cfg, algo_cfg, train_cfg = build_configs(cfg)

    rows, raw = [], {}
    for name in args.variants:
        over = VARIANTS[name]
        costs, irs, uns, gates, carbon = [], [], [], [], []
        for s in args.seeds:
            e = EnvConfig(**{**asdict(env_cfg),
                             **{k: v for k, v in over.items() if k in ENV_KEYS}})
            a = MAPPOConfig(**{**asdict(algo_cfg),
                               **{k: v for k, v in over.items() if k not in ENV_KEYS},
                               "seed": s})
            t = TrainConfig(**{**asdict(train_cfg), "seed": s})
            tr = Trainer(tensors, e, a, t, out_dir=out / name / f"seed{s}")
            tr.train(verbose=False)
            agent, extra = MAPPO.load(out / name / f"seed{s}" / "best.pt")
            agent.residual_scale = extra.get("residual_scale", t.residual_scale_final)
            agent.multipliers = extra.get("multipliers", {})
            r = compare_policies(tensors, e, {"x": agent},
                                 episodes=cfg["train"]["eval_episodes"],
                                 include_lp=False)["x"]
            costs.append(float(np.mean(r["net_cost_eur"])))
            irs.append(float(np.mean(r["individual_rationality"])))
            uns.append(float(np.mean(r["unserved_kwh"])))
            carbon.append(float(np.mean(r.get("emissions_kg", [0]))))
            gates.append(bool(r["_gate"]["passed"]))
            print(f"  {name} seed{s}: cost={costs[-1]:7.2f} IR={irs[-1]:.3f} "
                  f"unserved={uns[-1]:6.2f} CO2={carbon[-1]:6.1f}kg "
                  f"gate={'PASS' if gates[-1] else 'FAIL'}", flush=True)
        raw[name] = {"cost": costs, "ir": irs, "unserved": uns,
                     "carbon": carbon, "gate": gates}
        rows.append({"variant": name,
                     "cost_eur": f"{np.mean(costs):.2f}",
                     "spread": f"{np.max(costs)-np.min(costs):.2f}",
                     "IR": f"{np.mean(irs):.3f}",
                     "unserved": f"{np.mean(uns):.2f}",
                     "CO2_kg": f"{np.mean(carbon):.1f}",
                     "gate": f"{int(np.sum(gates))}/{len(gates)}"})

    table = format_table(rows, ["variant", "cost_eur", "spread", "IR", "unserved",
                                "CO2_kg", "gate"], "A/B test of proposed improvements")
    print(table)
    (out / "ab_test.txt").write_text(table)
    (out / "ab_test.json").write_text(json.dumps(raw, indent=1))


if __name__ == "__main__":
    main()
