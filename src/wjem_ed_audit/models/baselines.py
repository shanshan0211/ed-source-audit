"""Flat baselines (LR / XGBoost / LightGBM) and clinical score models.

All estimators expose one unified entry point:

    fit_predict_proba(model_name, Xtr, ytr, Xte, task, seed) -> scores (n_test,)

Classification tasks (critical/hosp/revisit72) use the *Classifier variant and
return P(y=1).  The `edlos` task uses the *Regressor variant; the caller always
passes the raw (untransformed) target, and this module internally applies
log1p before fitting and expm1 on predictions (clipped at >= 0).

Clinical scores (NEWS2 / qSOFA) need no training: they map the 7 triage vitals
to a risk score that is used directly as y_score.  Missing vitals are scored
as normal (0 points), per common practice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.api import TASKS as _API_TASKS

# classification = all contract tasks except the edlos regression
CLASSIFICATION_TASKS = tuple(t for t in _API_TASKS if t != "edlos")


def is_classification(task: str) -> bool:
    return task in CLASSIFICATION_TASKS


# ------------------------------------------------------------------ sklearn
def _make_estimator(model_name: str, task: str, seed: int):
    """Return a sklearn-compatible estimator (pipeline for LR)."""
    cls = is_classification(task)
    if model_name == "lr":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if cls:
            est = LogisticRegression(max_iter=2000, random_state=seed)
        else:
            est = LinearRegression()
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    if model_name == "xgb":
        if cls:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9,
                                 random_state=seed, n_jobs=4, eval_metric="logloss")
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=seed, n_jobs=4)
    if model_name == "lgbm":
        # subsample/colsample_bytree give per-seed stochasticity: LightGBM is
        # deterministic without them, which collapses cross-seed variance to 0
        # (zero-width CIs / +-inf t stats downstream).  subsample_freq=1 is
        # required -- bagging is disabled when freq=0 (the default) even when
        # subsample<1.
        if cls:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=300, learning_rate=0.1,
                                  subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8,
                                  random_state=seed, n_jobs=4, verbose=-1)
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.1,
                             subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8,
                             random_state=seed, n_jobs=4, verbose=-1)
    raise ValueError(f"unknown baseline model {model_name!r}; choose lr/xgb/lgbm")


def fit_predict_proba(model_name: str, Xtr: pd.DataFrame, ytr: np.ndarray,
                      Xte: pd.DataFrame, task: str = "critical",
                      seed: int = 0) -> np.ndarray:
    """Fit `model_name` on (Xtr, ytr) and return scores for Xte.

    Classification -> predicted probability of the positive class.
    edlos (regression) -> predicted LOS in hours on the raw scale; the target
    is log1p-transformed internally before fitting and expm1 back afterwards.
    """
    est = _make_estimator(model_name, task, seed)
    if is_classification(task):
        est.fit(Xtr, ytr.astype(int))
        return est.predict_proba(Xte)[:, 1].astype(np.float64)
    est.fit(Xtr, np.log1p(np.asarray(ytr, dtype=np.float64)))
    pred = np.expm1(est.predict(Xte))
    return np.clip(pred, 0.0, None).astype(np.float64)


# ------------------------------------------------------------- clinical scores
def _band(value: np.ndarray, bands: list[tuple[float, float, int]]) -> np.ndarray:
    """Score `value` against (lo, hi, points) bands; NaN -> 0 points (normal)."""
    out = np.zeros(len(value), dtype=np.float64)
    for lo, hi, pts in bands:
        m = (~np.isnan(value)) & (value >= lo) & (value < hi)
        out = np.where(m, pts, out)
    return out


def news2_score(df_vitals: pd.DataFrame) -> np.ndarray:
    """NEWS2 aggregate score from triage vitals (SpO2 scale 1, room air assumed).

    Consciousness level and supplemental-oxygen status are not recorded in the
    stays table; both are assumed normal (alert, room air) -> 0 points.
    Missing vitals contribute 0 points (treated as normal).
    Higher score = higher risk; usable directly as y_score.
    """
    v = {c: pd.to_numeric(df_vitals.get(c), errors="coerce").to_numpy(dtype=np.float64)
         for c in ["resprate", "o2sat", "sbp", "heartrate", "temperature"]}
    inf = np.inf
    score = np.zeros(len(df_vitals), dtype=np.float64)
    score += _band(v["resprate"], [(-inf, 9, 3), (9, 12, 1), (12, 21, 0),
                                   (21, 25, 2), (25, inf, 3)])
    score += _band(v["o2sat"], [(-inf, 92, 3), (92, 94, 2), (94, 96, 1), (96, inf, 0)])
    score += _band(v["sbp"], [(-inf, 91, 3), (91, 101, 2), (101, 111, 1),
                              (111, 220, 0), (220, inf, 3)])
    score += _band(v["heartrate"], [(-inf, 41, 3), (41, 51, 1), (51, 91, 0),
                                    (91, 111, 1), (111, 131, 2), (131, inf, 3)])
    score += _band(v["temperature"], [(-inf, 35.1, 3), (35.1, 36.1, 1), (36.1, 38.1, 0),
                                      (38.1, 39.1, 1), (39.1, inf, 2)])
    return score


def qsofa_score(df_vitals: pd.DataFrame) -> np.ndarray:
    """qSOFA score (0-2 here): RR >= 22 and/or SBP <= 100 give 1 point each.

    Altered mentation (GCS < 15) is not recorded in the stays table and is
    assumed absent (0 points).  Missing vitals contribute 0 points.
    Higher score = higher risk; usable directly as y_score.
    """
    rr = pd.to_numeric(df_vitals.get("resprate"), errors="coerce").to_numpy(dtype=np.float64)
    sbp = pd.to_numeric(df_vitals.get("sbp"), errors="coerce").to_numpy(dtype=np.float64)
    score = np.zeros(len(df_vitals), dtype=np.float64)
    score += np.where(~np.isnan(rr) & (rr >= 22), 1.0, 0.0)
    score += np.where(~np.isnan(sbp) & (sbp <= 100), 1.0, 0.0)
    return score


CLINICAL_SCORES = {"news2": news2_score, "qsofa": qsofa_score}
