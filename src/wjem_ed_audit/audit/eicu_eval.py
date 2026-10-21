"""E5 (real component) — external validation on the REAL eICU-CRD demo.

Baseline sklearn objects were never persisted (runs store metrics/preds only),
so this module re-trains the clin_only (C+D) flat baselines on the SYNTHETIC
cache (data.cache_dir of configs/full_synth.yaml) via the exact feature path of
``models.trainer.train_baseline`` (``api.load_flat``, split_random=train) and
then predicts on the re-anchored eICU flat matrix (``data.eicu.load_eicu_flat``).

Outputs ``results_synth_full/audit/E5_eicu_real.csv`` (E5_transfer.csv is left
untouched) with one row per task x model x seed plus seed="ALL" summaries:
  task, model, seed, auc_random_mimic, auc_eicu_real, decay, lo, hi, n_eicu,
  cohort, note
where auc_random_mimic is the in-distribution random-test AUC read back from
``runs/death/{model}/clin_only/seed*/metrics.json`` and decay is
auc_random_mimic - auc_eicu_real.  Per-seed CIs are stay-level bootstrap
(audit.stats.bootstrap_auc_ci); the ALL row aggregates across seeds
(audit.stats.aggregate_seeds).

CLI:
    python -m wjem_ed_audit.audit.eicu_eval \
        --config configs/full_synth.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wjem_ed_audit.data import api, eicu
from wjem_ed_audit.models.baselines import fit_predict_proba
from wjem_ed_audit.models.trainer import train_baseline
from wjem_ed_audit.audit import stats as astats

TASK = "death"
MODELS = ("lr", "lgbm")
GROUPS = ("C", "D")  # clin_only

NOTE = ("REAL eICU-CRD demo v2.0.1 (ODbL); models trained on real MIMIC-IV-ED "
        "(cache_real) — re-anchored external validation; demo-scale cohort")


# ------------------------------------------------------------------ core
def evaluate_eicu(cfg: dict, raw_dir=None, cache_path=None) -> pd.DataFrame:
    """Retrain clin_only baselines on the configured cache (synthetic or real MIMIC) and score the eICU cohort.

    Returns the detail+summary DataFrame that is written to E5_eicu_real.csv.
    """
    cache_dir = cfg["data"]["cache_dir"]
    results_root = Path(cfg.get("results_root", "results_synth_full"))
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    n_boot = int(cfg.get("audit", {}).get("bootstrap", 2000))

    X_e, y_e, _ids = eicu.load_eicu_flat(raw_dir=raw_dir, cache_path=cache_path)
    ok = ~np.isnan(y_e)
    y_eval = y_e[ok].astype(int)
    n_eicu = int(ok.sum())
    # sensitivity endpoint: ICU (unit) mortality on the same scores
    cache = pd.read_parquet(Path(cache_path) if cache_path else eicu.CACHE_PATH)
    y_u = cache["y_death_unit"].to_numpy(dtype=np.float64)
    ok_u = ~np.isnan(y_u)
    y_unit = y_u[ok_u].astype(int)

    rows: list[dict] = []
    sens: dict[str, list[float]] = {m: [] for m in MODELS}
    for model in MODELS:
        for seed in seeds:
            # same feature path as trainer.train_baseline; also (re)writes the
            # random-split run so runs/death/{model}/clin_only/seed*/ exists
            train_baseline(cfg, TASK, model, GROUPS, seed, cache_dir,
                           results_root, temporal_eval=False)
            run_dir = api.run_dir_for(results_root, TASK, model, "clin_only", seed)
            with open(run_dir / "metrics.json") as f:
                auc_random = float(json.load(f)["auroc"])

            Xtr, ytr, _ = api.load_flat(TASK, "split_random", "train", cache_dir,
                                        groups=GROUPS)
            Xe = X_e.reindex(columns=Xtr.columns, fill_value=0.0)
            scores = fit_predict_proba(model, Xtr, ytr, Xe, task=TASK, seed=seed)
            boot = astats.bootstrap_auc_ci(y_eval, scores[ok], n=n_boot, seed=seed)
            rows.append({"task": TASK, "model": model, "seed": seed,
                         "auc_random_mimic": auc_random,
                         "auc_eicu_real": boot["auc"],
                         "decay": auc_random - boot["auc"],
                         "lo": boot["lo"], "hi": boot["hi"],
                         "n_eicu": n_eicu})
            sens[model].append(astats._safe_auc(y_unit, scores[ok_u]))
        sub = pd.DataFrame([r for r in rows if r["model"] == model])
        agg_e = astats.aggregate_seeds(sub.assign(v=sub["auc_eicu_real"]), "v")
        agg_r = astats.aggregate_seeds(sub.assign(v=sub["auc_random_mimic"]), "v")
        rows.append({"task": TASK, "model": model, "seed": "ALL",
                     "auc_random_mimic": agg_r["mean"],
                     "auc_eicu_real": agg_e["mean"],
                     "decay": agg_r["mean"] - agg_e["mean"],
                     "lo": agg_e["lo"], "hi": agg_e["hi"],
                     "n_eicu": n_eicu})
    df = pd.DataFrame(rows)
    df["cohort"] = eicu.COHORT
    df["note"] = NOTE
    df = df[["task", "model", "seed", "auc_random_mimic", "auc_eicu_real",
             "decay", "lo", "hi", "n_eicu", "cohort", "note"]]
    df.attrs["sens_unit_death_auc"] = {m: float(np.nanmean(v))
                                       for m, v in sens.items()}
    return df


# ------------------------------------------------------------------ driver
def run_eicu_eval(cfg: dict, raw_dir=None, cache_path=None,
                  out_path=None) -> pd.DataFrame:
    results_root = Path(cfg.get("results_root", "results_synth_full"))
    out_path = Path(out_path) if out_path else results_root / "audit" / "E5_eicu_real.csv"
    df = evaluate_eicu(cfg, raw_dir=raw_dir, cache_path=cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[eicu_eval] wrote {out_path}  ({len(df)} rows)")
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print("\n================ E5 real eICU summary (clin_only, task=death) ================")
    view = df.copy()
    for c in ("auc_random_mimic", "auc_eicu_real", "decay", "lo", "hi"):
        view[c] = view[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
    print(view[["model", "seed", "auc_random_mimic", "auc_eicu_real", "decay",
                "lo", "hi", "n_eicu"]].to_string(index=False))
    sens = df.attrs.get("sens_unit_death_auc", {})
    if sens:
        print("sensitivity (unit/ICU mortality, mean AUC over seeds): "
              + ", ".join(f"{m}={v:.4f}" for m, v in sens.items()))
    print("=" * 80)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = api.load_config(args.config)
    df = run_eicu_eval(cfg)
    _print_summary(df)
    return df


if __name__ == "__main__":
    main()
