"""Statistical utilities for the audit engine (SPEC §5).

All bootstrap procedures resample *stays* (rows) with replacement, matching the
"对 stay 重抽样" requirement of SPEC §5. Functions are pure and unit-testable.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------ helpers
def _as_1d(a) -> np.ndarray:
    return np.asarray(a).ravel()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """roc_auc_score that returns NaN on degenerate (single-class) resamples."""
    y_true = _as_1d(y_true)
    y_score = _as_1d(y_score)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _sign_boot_p(deltas: np.ndarray) -> float:
    """Two-sided p from a bootstrap distribution of deltas centred under H0: d=0."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return float("nan")
    p_le = float(np.mean(deltas <= 0.0))
    p_ge = float(np.mean(deltas >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


# ------------------------------------------------------------- public API
def bootstrap_auc_ci(y_true, y_score, n: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for AUC, resampling stays with replacement.

    Returns {"auc", "lo", "hi", "n_boot"} where `auc` is the point estimate
    (identical to sklearn roc_auc_score on the full sample).
    """
    y_true = _as_1d(y_true)
    y_score = _as_1d(y_score)
    auc = _safe_auc(y_true, y_score)
    rng = np.random.default_rng(seed)
    m = y_true.size
    boots = np.empty(n, dtype=float)
    for b in range(n):
        idx = rng.integers(0, m, m)
        boots[b] = _safe_auc(y_true[idx], y_score[idx])
    lo, hi = _percentile_ci(boots, alpha)
    return {"auc": auc, "lo": lo, "hi": hi, "n_boot": int((~np.isnan(boots)).sum())}


def bootstrap_metric_ci(fn, y_true, y_score, n: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Generic bootstrap CI for a metric fn(y_true, y_score) -> float."""
    y_true = _as_1d(y_true)
    y_score = _as_1d(y_score)
    val = float(fn(y_true, y_score))
    rng = np.random.default_rng(seed)
    m = y_true.size
    boots = np.empty(n, dtype=float)
    for b in range(n):
        idx = rng.integers(0, m, m)
        try:
            boots[b] = float(fn(y_true[idx], y_score[idx]))
        except Exception:
            boots[b] = float("nan")
    lo, hi = _percentile_ci(boots, alpha)
    return {"value": val, "lo": lo, "hi": hi, "n_boot": int((~np.isnan(boots)).sum())}


def paired_delta_auc(y_true, s1, s2, n: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Bootstrap distribution of paired ΔAUC = AUC(s1) - AUC(s2) over stays.

    Resamples the same stay indices for both score vectors (paired design).
    Returns {"delta", "lo", "hi", "p"} with a two-sided sign-based p value.
    """
    y_true = _as_1d(y_true)
    s1 = _as_1d(s1)
    s2 = _as_1d(s2)
    delta = _safe_auc(y_true, s1) - _safe_auc(y_true, s2)
    rng = np.random.default_rng(seed)
    m = y_true.size
    boots = np.empty(n, dtype=float)
    for b in range(n):
        idx = rng.integers(0, m, m)
        boots[b] = _safe_auc(y_true[idx], s1[idx]) - _safe_auc(y_true[idx], s2[idx])
    lo, hi = _percentile_ci(boots, alpha)
    return {"delta": float(delta), "lo": lo, "hi": hi, "p": _sign_boot_p(boots)}


def bh_fdr(pvals, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values within ONE family of tests.

    The caller defines the family (the set of tests whose error rate is
    controlled jointly); the families used by the audit engine are:
      * E1: per (task, model), the delta-AUC tests of the non-full combos;
      * E2: per task, the per-field removal delta-AUC tests (J/P fields);
      * E3: per task, the 4 query-group suppressions (J/P/D/random_C);
      * E4: per (task, model), the J and P permutation tests.
    `alpha` is the nominal FDR level (kept in the signature for call-site
    clarity; the adjusted values themselves do not depend on it).

    Returns an array aligned with `pvals` (NaN in -> NaN out). Adjusted values
    are monotone non-decreasing in the rank of the raw p-values.
    """
    p = np.asarray(list(pvals), dtype=float).ravel()
    adj = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return adj
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = q
    adj[ok] = out
    return adj


def ttest_one_sample(deltas) -> tuple[float, float]:
    """One-sample t test of mean(deltas) == 0 (SPEC §5). Returns (t, p).

    Guards degenerate inputs: n<2 -> (nan, nan); zero variance -> exact
    answer (p=1 if mean==0 else p=0 with t=+-inf).
    """
    d = np.asarray(list(deltas), dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan"), float("nan")
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    if sd == 0.0:
        if mean == 0.0:
            return 0.0, 1.0
        return float(np.sign(mean) * np.inf), 0.0
    res = sps.ttest_1samp(d, 0.0)
    return float(res.statistic), float(res.pvalue)


def aggregate_seeds(rows, value_col: str, alpha: float = 0.05) -> dict:
    """Aggregate a metric across seeds: mean + t-based CI.

    `rows` may be a DataFrame or an iterable of dicts. Returns
    {"mean", "lo", "hi", "sd", "n"}; CI is mean +/- t(1-alpha/2, n-1)*sd/sqrt(n).
    With n<2 the CI collapses to the point estimate bounds (nan sd -> nan CI).
    """
    if hasattr(rows, "__getitem__") and not isinstance(rows, (list, tuple)):
        # DataFrame-like
        vals = np.asarray(rows[value_col], dtype=float)
    else:
        vals = np.asarray([r[value_col] for r in rows], dtype=float)
    vals = vals[~np.isnan(vals)]
    n = int(vals.size)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "sd": float("nan"), "n": 0}
    mean = float(vals.mean())
    if n < 2:
        return {"mean": mean, "lo": float("nan"), "hi": float("nan"),
                "sd": float("nan"), "n": n}
    sd = float(vals.std(ddof=1))
    half = float(sps.t.ppf(1 - alpha / 2, n - 1)) * sd / np.sqrt(n)
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "sd": sd, "n": n}
