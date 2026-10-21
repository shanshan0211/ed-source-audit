"""E2 — field-level ablation within the J and P groups (SPEC §5).

Flat-feature scenario on lgbm: api.load_flat provides group-prefixed columns
(C__/J__/P__/D__), so removing a *field* = dropping its column prefix family.
Includes the redundancy contrast: whole-group removal ΔAUC vs the sum of
per-field removal ΔAUCs within that group.

The lgbm fit/predict step is injectable (`fit_predict`); by default it calls
models.baselines.fit_predict_proba("lgbm", ...) (lazy import)
and falls back to a local lightgbm trainer so the module is testable
standalone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import api
from . import stats as astats
from .masks import cls_tasks

# field -> flat-column prefix family (a field may span several columns)
FIELD_PREFIXES: dict[str, tuple[str, ...]] = {
    "acuity": ("J__acuity",),            # J__acuity + J__acuity_missing
    "arrival_transport": ("P__at_",),    # one-hot family
    "hour_of_day": ("P__hour_of_day",),
    "day_of_week": ("P__day_of_week",),
    "n_ed_visits_365d": ("P__n_ed_visits_365d",),
}
FIELD_GROUP: dict[str, str] = {
    "acuity": "J",
    "arrival_transport": "P",
    "hour_of_day": "P",
    "day_of_week": "P",
    "n_ed_visits_365d": "P",
}


def columns_for_field(all_cols, field: str) -> list[str]:
    prefixes = FIELD_PREFIXES[field]
    return [c for c in all_cols if any(c.startswith(p) for p in prefixes)]


def columns_for_group(all_cols, group: str) -> list[str]:
    return [c for c in all_cols if c.startswith(f"{group}__")]


# ------------------------------------------------------------- lgbm engine
def _local_fit_predict_lgbm(Xtr, ytr, Xte, seed: int = 0) -> np.ndarray:
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                         subsample=0.9, subsample_freq=1,
                         colsample_bytree=0.9,
                         random_state=seed, verbose=-1, n_jobs=4)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def default_fit_predict(Xtr, ytr, Xte, seed: int = 0,
                        task: str = "critical") -> np.ndarray:
    """Prefer the unified baseline trainer when available."""
    try:
        from wjem_ed_audit.models.baselines import fit_predict_proba  # lazy
    except ImportError:
        return _local_fit_predict_lgbm(Xtr, ytr, Xte, seed=seed)
    return np.asarray(fit_predict_proba("lgbm", Xtr, ytr, Xte, task=task,
                                        seed=seed), dtype=float)


# -------------------------------------------------------------------- core
def _drop_cols(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return X.drop(columns=cols) if cols else X


def ablate_once(task: str, seed: int, fields: list[str], cache_dir,
                fit_predict=None) -> list[dict]:
    """One task x seed: full vs each field removed vs whole J/P groups removed.

    Returns per-field dicts with redundancy contrast columns.
    """
    fp = fit_predict or default_fit_predict
    Xtr, ytr, _ = api.load_flat(task, "split_random", "train", cache_dir)
    Xte, yte, _ = api.load_flat(task, "split_random", "test", cache_dir)
    all_cols = list(Xtr.columns)
    auc_full = astats._safe_auc(yte, fp(Xtr, ytr, Xte, seed=seed, task=task))

    group_delta: dict[str, float] = {}
    for group in ("J", "P"):
        cols = columns_for_group(all_cols, group)
        auc = astats._safe_auc(yte, fp(_drop_cols(Xtr, cols), ytr,
                                       _drop_cols(Xte, cols), seed=seed,
                                       task=task))
        group_delta[group] = auc_full - auc

    field_delta: dict[str, float] = {}
    for field in fields:
        cols = columns_for_field(all_cols, field)
        auc = astats._safe_auc(yte, fp(_drop_cols(Xtr, cols), ytr,
                                       _drop_cols(Xte, cols), seed=seed,
                                       task=task))
        field_delta[field] = auc_full - auc

    rows = []
    for field in fields:
        group = FIELD_GROUP.get(field, "?")
        g_sum = sum(field_delta[f] for f in fields if FIELD_GROUP.get(f) == group)
        g_del = group_delta.get(group, float("nan"))
        rows.append({
            "task": task, "seed": int(seed), "field": field, "group": group,
            "auc_full": auc_full, "delta_field": field_delta[field],
            "delta_group_remove": g_del,
            "delta_group_field_sum": g_sum,
            "redundancy_gap": g_sum - g_del,
        })
    return rows


def run_e2(cfg: dict, cache_dir=None, results_root=None, out_dir=None,
           fit_predict=None) -> pd.DataFrame:
    """E2 driver over tasks x seeds; writes E2_field_ablation.csv.

    Detail rows per task x seed x field plus summary rows (seed="ALL") with
    cross-seed mean ΔAUC, t-based 95% CI, one-sample t-test p (SPEC §5) and
    BH-FDR adjusted p (family: all ablated fields within each task).
    """
    cache_dir = cache_dir or cfg.get("data", {}).get("cache_dir")
    fields = list(cfg["audit"].get("e2_fields", FIELD_PREFIXES.keys()))
    rows: list[dict] = []
    for task in cls_tasks(cfg):
        for seed in cfg.get("seeds", [0]):
            rows.extend(ablate_once(task, int(seed), fields, cache_dir,
                                    fit_predict=fit_predict))
    df = pd.DataFrame(rows)
    summary_rows = []
    for (task, field), g in df.groupby(["task", "field"], sort=True):
        agg = astats.aggregate_seeds(g.assign(v=g["delta_field"]), "v")
        _, p = astats.ttest_one_sample(g["delta_field"].to_numpy())
        summary_rows.append({
            "task": task, "seed": "ALL", "field": field,
            "group": g["group"].iloc[0], "auc_full": np.nan,
            "delta_field": agg["mean"], "delta_lo": agg["lo"],
            "delta_hi": agg["hi"], "p": p, "p_adj": np.nan,
            "delta_group_remove": np.nan, "delta_group_field_sum": np.nan,
            "redundancy_gap": np.nan,
        })
    summ = pd.DataFrame(summary_rows)
    # BH-FDR within each task family (the per-field removal tests).
    for _, gidx in summ.groupby("task").groups.items():
        summ.loc[gidx, "p_adj"] = astats.bh_fdr(summ.loc[gidx, "p"].to_numpy())
    out = pd.concat([df, summ], ignore_index=True)
    if out_dir is None and results_root is not None:
        out_dir = Path(results_root) / "audit"
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "E2_field_ablation.csv", index=False)
    return out
