# Quickstart — run it, see it, verify it

Works on Windows, macOS and Linux. Python 3.10+ and ~2 GB free for PyTorch.
Everything runs on CPU. No GPU, no accounts, no internet after install.

## 0. See the results with zero setup

Open **`RESULTS.html`** in any browser. Every table in the project is on that
page — the policy comparison, the multi-seed statistics, the ablation, and the
A/B test — rendered from the stored result files. Nothing to install.

## 1. Install (once, ~5 minutes)

Open a terminal inside the `P2P_MARL_Capstone` folder.

    python -m venv venv
    venv\Scripts\activate          <- Windows (PowerShell: venv\Scripts\Activate.ps1)
    source venv/bin/activate       <- macOS / Linux
    pip install -r requirements.txt

You should see `(venv)` at the start of the prompt.

## 2. Verify the code works (20 seconds)

    python -m pytest tests/ -q

Expected: `36 passed`.

## 3. Reproduce the reported numbers from the saved models (~1 minute)

    python scripts/reproduce_results.py --check

This loads the five trained checkpoints in `results/main_study/seed*/best.pt`,
runs every policy on the deterministic evaluation day, prints the comparison
table, and checks it against the stored `summary.json`. Expected last line:

    ALL RESULTS REPRODUCED

Evaluation is fully deterministic (deterministic policy, deterministic day),
so this gives the same numbers on any machine.

## 4. Retrain from scratch (optional, ~25 min per seed)

    python scripts/run_study.py --seeds 0 1 2 3 4 --updates 200
    python scripts/run_ablation.py --seeds 0 1 2 --updates 150

**Note on retraining:** training is seeded, so retraining on the *same* machine
and library versions gives identical results. Across different hardware or
PyTorch versions, floating-point differences can shift costs by a few tenths of
a euro; the ordering, the gate outcomes and every conclusion are stable. If you
want the exact reported numbers, use step 3 — that is what the saved
checkpoints are for.

Retraining overwrites `results/main_study/` and `results/ablation/`. Copy them
first if you want to keep the originals.

## Where things are

| What | Where |
|---|---|
| Dataset (250 households, 19 sheets) | `data/EC_EV_dataset.xlsx` |
| Headline results and tables | `RESULTS.md`, `RESULTS.html` |
| Ablation | `docs/ABLATION.md`, `results/ablation/` |
| A/B test of literature techniques | `docs/AB_TEST.md` |
| Trained models | `results/main_study/seed*/best.pt` |
| Per-update training curves | `results/main_study/seed*/history.json` |
| Environment, market, safety layers | `src/env/` |
| MAPPO, networks, Lagrangian | `src/marl/` |
| Design rationale | `docs/DESIGN.md`, `docs/WHY_THE_BASELINE_WON.md` |
| Slide guide + viva questions | `docs/PRESENTATION.md` |
| What was and was not built | `PROVENANCE.md` |

## Common problems

* `'source' is not recognized` — you are on Windows; use `venv\Scripts\activate`.
* PowerShell refuses to activate — run once:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
* Torch install fails — you need ~2 GB free and Python 3.10–3.12.
* Long silence during training is normal; check Task Manager for CPU use.
