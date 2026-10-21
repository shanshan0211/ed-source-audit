"""SDS / CSR metric family (SPEC §5).

Definitions (AUC_full = full combo, AUC_clin = clin_only, AUC_meta = meta_only):
    SDS_mask = (AUC_full - AUC_clin) / (AUC_full - 0.5)
    SDS_solo = (AUC_meta - 0.5)      / (AUC_full - 0.5)
    CSR      = (AUC_full - AUC_meta) / (AUC_full - 0.5)
    R        = (AUC_clin + AUC_meta - AUC_full - 0.5) / (AUC_full - 0.5)
Bootstrap CIs resample stays within each seed's predictions (SPEC §5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .runs_index import load_preds
from .stats import _percentile_ci, _safe_auc

COMBOS = ("full", "clin_only", "meta_only")
METRICS = ("SDS_mask", "SDS_solo", "CSR", "R")


def compute_sds_family(auc_full: float, auc_clin: float, auc_meta: float) -> dict:
    """Point estimates of the SDS/CSR family from three combo AUCs."""
    denom = auc_full - 0.5
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        nan = float("nan")
        return {"SDS_mask": nan, "SDS_solo": nan, "CSR": nan, "R": nan,
                "auc_full": float(auc_full), "auc_clin": float(auc_clin),
                "auc_meta": float(auc_meta)}
    return {
        "SDS_mask": float((auc_full - auc_clin) / denom),
        "SDS_solo": float((auc_meta - 0.5) / denom),
        "CSR": float((auc_full - auc_meta) / denom),
        "R": float((auc_clin + auc_meta - auc_full - 0.5) / denom),
        "auc_full": float(auc_full),
        "auc_clin": float(auc_clin),
        "auc_meta": float(auc_meta),
    }


# --------------------------------------------------------------- bootstrap
def _preds_per_seed(runs_index: pd.DataFrame, task: str, model: str,
                    combo: str, bootstrap_cache: dict | None) -> list[tuple]:
    """[(y_true, y_score, stay_ids_or_None), ...] in sorted seed order.

    Sources, in order: bootstrap_cache[(task, model, combo)] then preds.parquet
    files referenced by runs_index.
    """
    key = (task, model, combo)
    if bootstrap_cache is not None and key in bootstrap_cache:
        out = []
        for item in bootstrap_cache[key]:
            if isinstance(item, pd.DataFrame):
                out.append((item["y_true"].to_numpy(), item["y_score"].to_numpy(),
                            item["stay_id"].to_numpy() if "stay_id" in item else None))
            else:
                yt, ys, *ids = item
                out.append((np.asarray(yt), np.asarray(ys),
                            np.asarray(ids[0]) if ids else None))
        return out
    if runs_index is None or runs_index.empty:
        return []
    sub = runs_index[(runs_index["task"] == task)
                     & (runs_index["model"] == model)
                     & (runs_index["config_tag"] == combo)
                     & (~runs_index["temporal"])]
    out = []
    for _, row in sub.sort_values("seed").iterrows():
        preds = load_preds(row["path"])
        if not preds.empty and {"y_true", "y_score"} <= set(preds.columns):
            if "stay_id" not in preds:
                raise ValueError("Stored predictions require stay_id for paired bootstrap")
            out.append((preds["y_true"].to_numpy(), preds["y_score"].to_numpy(),
                        preds["stay_id"].to_numpy()))
    return out


def _mean_auc(preds_per_seed: list[tuple], indices=None) -> float:
    """Average seed AUCs using one shared row sample, never independent samples."""
    aucs = []
    for yt, ys, _ids in preds_per_seed:
        if indices is not None:
            yt, ys = yt[indices], ys[indices]
        aucs.append(_safe_auc(yt, ys))
    aucs = np.asarray(aucs, dtype=float)
    if np.all(np.isnan(aucs)):
        return float("nan")
    return float(np.nanmean(aucs))


def _align_predictions(per_combo: dict) -> tuple[dict, np.ndarray | None]:
    """Require equal seed counts and identical labeled test stays across runs.

    In-memory two-array fixtures use an explicit positional contract. Stored
    predictions always require IDs; mixing identified and positional data fails.
    """
    counts = {len(v) for v in per_combo.values()}
    if len(counts) != 1:
        raise ValueError("All SDS combinations must use the same seed count")
    identified = {ids is not None for runs in per_combo.values() for _, _, ids in runs}
    if len(identified) != 1:
        raise ValueError("Cannot mix identified and positional predictions")
    ref_y, ref_ids = None, None
    aligned = {}
    for combo, runs in per_combo.items():
        aligned[combo] = []
        for y, score, ids in runs:
            if len(y) != len(score) or not np.all(np.isfinite(score)):
                raise ValueError("Prediction lengths or scores are invalid")
            if ids is not None:
                if len(ids) != len(y) or len(np.unique(ids)) != len(ids):
                    raise ValueError("stay_id must be unique and match prediction length")
                order = np.argsort(ids)
                y, score, ids = y[order], score[order], ids[order]
            if ref_y is None:
                ref_y, ref_ids = y, ids
            elif not np.array_equal(y, ref_y) or (
                    ids is not None and not np.array_equal(ids, ref_ids)):
                raise ValueError("SDS comparisons require identical labeled test stays")
            aligned[combo].append((y, score, ids))
    return aligned, ref_ids


def sds_table(runs_index: pd.DataFrame, bootstrap_cache: dict | None = None,
              n_boot: int = 500, seed: int = 0, alpha: float = 0.05,
              patient_ids: pd.Series | None = None) -> pd.DataFrame:
    """Summarise the SDS/CSR family per (task, model): point value + bootstrap CI.

    AUCs are averaged across seeds first (mean AUC per combo), then the SDS
    formulas are applied. One bootstrap sample is shared across ALL combinations
    and seeds. Optional patient_ids is a Series indexed by stay_id; whole patients
    are sampled with replacement, retaining their stays and cross-model pairing.
    These intervals condition on fitted models, not retraining uncertainty.

    Returns long-format DataFrame: task, model, metric, value, lo, hi, n_seeds.
    Metrics include SDS_mask/SDS_solo/CSR/R plus component AUCs
    (auc_full/auc_clin/auc_meta).
    """
    if runs_index is None or runs_index.empty:
        return pd.DataFrame(columns=["task", "model", "metric", "value", "lo", "hi",
                                     "n_seeds"])
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    pairs = (runs_index.loc[~runs_index["temporal"], ["task", "model"]]
             .drop_duplicates().sort_values(["task", "model"]))
    for _, pr in pairs.iterrows():
        task, model = pr["task"], pr["model"]
        per_combo = {c: _preds_per_seed(runs_index, task, model, c, bootstrap_cache)
                     for c in COMBOS}
        if any(len(per_combo[c]) == 0 for c in COMBOS):
            continue
        if "seed" in runs_index.columns:
            sub = runs_index[(runs_index["task"] == task)
                             & (runs_index["model"] == model)
                             & (~runs_index["temporal"])]
            seed_sets = [tuple(sub.loc[sub["config_tag"] == c, "seed"].sort_values())
                         for c in COMBOS]
            if any(s != seed_sets[0] for s in seed_sets[1:]):
                raise ValueError("SDS combinations must use identical seed identifiers")
        per_combo, stay_ids = _align_predictions(per_combo)
        n_seeds = len(per_combo["full"])
        n_stays = len(per_combo["full"][0][0])
        cluster_rows = None
        if patient_ids is not None:
            if stay_ids is None or not patient_ids.index.is_unique:
                raise ValueError("Patient bootstrap requires unique stay_id mapping")
            patients = patient_ids.reindex(stay_ids)
            if patients.isna().any():
                raise ValueError("Patient identifier missing for a test stay")
            codes, _uniques = pd.factorize(patients)
            order = np.argsort(codes, kind="stable")
            cluster_rows = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
        point = {c: _mean_auc(per_combo[c]) for c in COMBOS}
        fam = compute_sds_family(point["full"], point["clin_only"], point["meta_only"])
        boots: dict[str, list[float]] = {m: [] for m in fam}
        for _ in range(n_boot):
            if cluster_rows is None:
                idx = rng.integers(0, n_stays, n_stays)
            else:
                sampled = rng.integers(0, len(cluster_rows), len(cluster_rows))
                idx = np.concatenate([cluster_rows[k] for k in sampled])
            b = {c: _mean_auc(per_combo[c], idx) for c in COMBOS}
            bf = compute_sds_family(b["full"], b["clin_only"], b["meta_only"])
            for m, v in bf.items():
                boots[m].append(v)
        for m in ("SDS_mask", "SDS_solo", "CSR", "R",
                  "auc_full", "auc_clin", "auc_meta"):
            lo, hi = _percentile_ci(np.asarray(boots[m]), alpha)
            rows.append({"task": task, "model": model, "metric": m,
                         "value": fam[m], "lo": lo, "hi": hi, "n_seeds": n_seeds})
    return pd.DataFrame(rows, columns=["task", "model", "metric", "value",
                                       "lo", "hi", "n_seeds"])
