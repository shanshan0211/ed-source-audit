"""Subgroup performance: per-level AUC and counts per config eval.subgroups.

Binning rules (per task contract):
- anchor_age: <65 vs >=65
- acuity: 1-2 / 3 / 4-5 (NaN -> "missing")
- everything else: categorical levels as-is (NaN -> "UNKNOWN")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _bin_anchor_age(s: pd.Series) -> pd.Series:
    return pd.cut(s.astype(float), [-np.inf, 65.0, np.inf], labels=["<65", ">=65"])


def _bin_acuity(s: pd.Series) -> pd.Series:
    b = pd.cut(s.astype(float), [0.0, 2.0, 3.0, 5.0], labels=["1-2", "3", "4-5"])
    b = b.cat.add_categories(["missing"])
    return b.fillna("missing")


_BINNERS = {"anchor_age": _bin_anchor_age, "acuity": _bin_acuity}


def subgroup_performance(stays: pd.DataFrame, y_true, y_score,
                         subgroups, min_n: int = 30) -> pd.DataFrame:
    """AUC + n per subgroup level. `stays` must be row-aligned with y_true/y_score.

    Returns DataFrame: variable, level, n, n_pos, prevalence, auc.
    AUC is reported as NaN (never a fabricated number) when n < min_n
    (default 30) or only one class is present.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(stays) != len(y_true):
        raise ValueError("stays must be row-aligned with y_true/y_score")
    rows = []
    for var in subgroups:
        if var not in stays.columns:
            continue
        if var in _BINNERS:
            levels = _BINNERS[var](stays[var])
        else:
            levels = stays[var].astype("object").where(stays[var].notna(), "UNKNOWN").astype(str)
        for lv in pd.unique(levels):
            m = (levels == lv).to_numpy()
            yt, ys = y_true[m], y_score[m]
            n = int(m.sum())
            n_pos = int(yt.sum())
            auc = np.nan
            if n >= min_n and 0 < n_pos < n:
                auc = float(roc_auc_score(yt, ys))
            rows.append({
                "variable": var,
                "level": str(lv),
                "n": n,
                "n_pos": n_pos,
                "prevalence": float(n_pos / n) if n else np.nan,
                "auc": auc,
            })
    return pd.DataFrame(rows)
