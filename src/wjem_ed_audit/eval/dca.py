"""Decision curve analysis: net benefit across threshold probabilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def decision_curve(y_true, y_score, thresholds) -> pd.DataFrame:
    """Net benefit at each threshold probability pt.

    NB_model    = TP/n - FP/n * pt/(1-pt)
    NB_treat_all = prevalence - (1-prevalence) * pt/(1-pt)
    NB_treat_none = 0

    Returns DataFrame: threshold, net_benefit, treat_all, treat_none.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    th = np.asarray(list(thresholds), dtype=float)
    n = len(y_true)
    pi = float(y_true.mean())
    rows = []
    for pt in th:
        pt = float(np.clip(pt, 0.0, 1.0 - 1e-9))
        odds = pt / (1.0 - pt)
        pred_pos = y_score >= pt
        tp = float((pred_pos & (y_true == 1)).sum())
        fp = float((pred_pos & (y_true == 0)).sum())
        rows.append({
            "threshold": pt,
            "net_benefit": tp / n - fp / n * odds,
            "treat_all": pi - (1.0 - pi) * odds,
            "treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def plot_dca(preds_by_label: dict, thresholds, out_png) -> Path:
    """Plot net-benefit curves for {label: (y_true, y_score)} with treat-all/none references."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    first = True
    for label, (yt, ys) in preds_by_label.items():
        df = decision_curve(yt, ys, thresholds)
        ax.plot(df["threshold"], df["net_benefit"], lw=1.5, label=label)
        if first:  # treat-all depends only on prevalence; draw once
            ax.plot(df["threshold"], df["treat_all"], color="grey", ls="--",
                    lw=1.0, label="treat all")
            ax.axhline(0.0, color="black", lw=1.0, label="treat none")
            first = False
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png
