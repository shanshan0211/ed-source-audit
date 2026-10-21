"""Calibration assessment: binned calibration data, ECE, and reliability plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calibration_curve_data(y_true, y_score, n_bins: int = 15) -> pd.DataFrame:
    """Uniform-width binning over [0, 1].

    Returns a DataFrame with one row per bin:
    bin, bin_left, bin_right, bin_center, count, pred_mean, frac_pos.
    Empty bins have count=0 and NaN pred_mean/frac_pos.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        rows.append({
            "bin": b,
            "bin_left": edges[b],
            "bin_right": edges[b + 1],
            "bin_center": 0.5 * (edges[b] + edges[b + 1]),
            "count": cnt,
            "pred_mean": float(y_score[m].mean()) if cnt else np.nan,
            "frac_pos": float(y_true[m].mean()) if cnt else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_score, n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / N) * |frac_pos_b - pred_mean_b|; in [0, 1]."""
    df = calibration_curve_data(y_true, y_score, n_bins)
    n = int(df["count"].sum())
    if n == 0:
        return float("nan")
    gap = (df["frac_pos"] - df["pred_mean"]).abs()
    return float((df["count"] * gap).sum(skipna=True) / n)


def plot_calibration(preds_by_label: dict, out_png, n_bins: int = 15) -> Path:
    """Reliability diagram + score histogram for several labelled (y_true, y_score) pairs.

    preds_by_label: {label: (y_true, y_score)}.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="perfect calibration")
    for label, (yt, ys) in preds_by_label.items():
        df = calibration_curve_data(yt, ys, n_bins)
        ece = expected_calibration_error(yt, ys, n_bins)
        ax.plot(df["pred_mean"], df["frac_pos"], "o-", ms=4, lw=1.2,
                label=f"{label} (ECE={ece:.3f})")
        axh.hist(np.clip(np.asarray(ys, dtype=float), 0, 1), bins=n_bins,
                 range=(0, 1), alpha=0.35, label=label)
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    axh.set_xlabel("Predicted probability")
    axh.set_ylabel("Count")
    for a in (ax, axh):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png
