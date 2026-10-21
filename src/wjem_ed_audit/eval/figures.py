"""Paper-level figures (matplotlib, 300 dpi, clean style).

Expected audit CSV schemas (SPEC §5). Both the audit engine's native schema
and the legacy long schema are accepted (normalised by the _norm_* helpers):
- E1_branch_mask.csv:        task, [model,] combo, seed, delta_auc
                             (audit engine: delta_vs_full; seed="ALL" summary
                              rows are excluded from the per-seed plotting)
- E3_repr_intervention.csv:  task, condition|intervention, seed, delta_auc, probe_auc
                             (audit engine: intervention/auc/delta_vs_none --
                              esi_probe rows carry the probe AUC in `auc`,
                              suppressions carry delta in `delta_vs_none`)
- E5_transfer.csv:           task, [model,] combo, split, seed, auc
                             (audit engine: kind=detail rows with
                              auc_random/auc_temporal, melted to long form;
                              eICU placeholder rows are excluded)
- sds_csr.csv:               task, [model,] metric, value, ci_lo, ci_hi
                             (audit engine: lo/hi)

Each fig_* returns the output Path, or None (with a warning) when the input
CSV is missing, so the report can run on partial results.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DPI = 300
_PALETTE = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3", "#937860"]


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, out_png: Path) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _read(csv_path) -> pd.DataFrame | None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        warnings.warn(f"[figures] input missing, skipped: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def _drop_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" in df.columns:
        df = df[df["seed"].astype(str) != "ALL"]
    return df


def _norm_e1(df: pd.DataFrame) -> pd.DataFrame:
    """Audit engine writes delta_vs_full (+model column); legacy writes delta_auc."""
    df = df.copy()
    if "delta_auc" not in df.columns and "delta_vs_full" in df.columns:
        df = df.rename(columns={"delta_vs_full": "delta_auc"})
    df = _drop_summary_rows(df)
    if "model" in df.columns and df["model"].nunique() > 1:
        df["task"] = df["task"].astype(str) + "/" + df["model"].astype(str)
    return df


def _norm_e3(df: pd.DataFrame) -> pd.DataFrame:
    """Audit engine writes intervention/auc/delta_vs_none; legacy writes
    condition/delta_auc/probe_auc."""
    df = df.copy()
    if "intervention" in df.columns:
        df = df.rename(columns={"intervention": "condition"})
        is_probe = df["condition"] == "esi_probe"
        df["probe_auc"] = np.where(is_probe, df["auc"].to_numpy(dtype=float),
                                   np.nan)
        df["delta_auc"] = np.where(is_probe, np.nan,
                                   df["delta_vs_none"].to_numpy(dtype=float))
    df = _drop_summary_rows(df)
    return df


def _norm_e5(df: pd.DataFrame) -> pd.DataFrame:
    """Audit engine writes kind=detail rows with auc_random/auc_temporal
    (+ eICU placeholder); legacy writes long split/auc rows."""
    df = df.copy()
    if "auc_random" in df.columns:
        d = df[df["kind"] == "detail"]
        frames = []
        for col, sp in (("auc_random", "random_test"),
                        ("auc_temporal", "temporal_test")):
            f = d[["task", "combo", "seed"]].copy()
            if "model" in d.columns:
                f["model"] = d["model"]
            f["split"] = sp
            f["auc"] = d[col].to_numpy(dtype=float)
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        df["status"] = np.nan
        if "model" in df.columns and df["model"].nunique() > 1:
            df["combo"] = df["combo"].astype(str) + "/" + df["model"].astype(str)
    if "status" in df.columns:  # drop eICU placeholder rows
        df = df[df["status"].isna() | (df["status"] == "ok")]
    df = _drop_summary_rows(df)
    return df


def _norm_sds(df: pd.DataFrame) -> pd.DataFrame:
    """Audit engine writes lo/hi (+model column); legacy writes ci_lo/ci_hi."""
    df = df.copy()
    if "ci_lo" not in df.columns and "lo" in df.columns:
        df = df.rename(columns={"lo": "ci_lo", "hi": "ci_hi"})
    if "model" in df.columns and df["model"].nunique() > 1:
        df["task"] = df["task"].astype(str) + "/" + df["model"].astype(str)
    return df


def _mean_ci(g: pd.DataFrame, col: str, by: str, order):
    agg = g.groupby(by)[col]
    m = agg.mean().reindex(order)
    s = agg.std().reindex(order).fillna(0.0)
    k = agg.count().reindex(order).fillna(1).clip(lower=1)
    ci = (1.96 * s / np.sqrt(k)).fillna(0.0)
    return m, ci


def fig2_branch_mask(e1_csv, out_png) -> Path | None:
    """Fig 2: dependency structure — mean delta-AUC vs full per combo, grouped by task."""
    df = _read(e1_csv)
    if df is None:
        return None
    df = _norm_e1(df)
    combos = list(dict.fromkeys(df["combo"]))
    tasks = list(dict.fromkeys(df["task"]))
    x = np.arange(len(combos))
    w = 0.8 / max(len(tasks), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, t in enumerate(tasks):
        m, ci = _mean_ci(df[df["task"] == t], "delta_auc", "combo", combos)
        ax.bar(x + (i - (len(tasks) - 1) / 2) * w, m.values, w, yerr=ci.values,
               capsize=3, label=t, color=_PALETTE[i % len(_PALETTE)])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(combos)
    ax.set_ylabel("ΔAUC vs full")
    ax.set_xlabel("Input combination")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)
    return _save(fig, out_png)


def fig3_repr_intervention(e3_csv, out_png) -> Path | None:
    """Fig 3: representation intervention — suppress J/P/random delta-AUC + ESI probe AUC."""
    df = _read(e3_csv)
    if df is None:
        return None
    df = _norm_e3(df)
    supp = df[df["delta_auc"].notna()]
    probe = df[df["probe_auc"].notna()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4),
                                   gridspec_kw={"width_ratios": [2, 1]})
    if not supp.empty:
        conds = list(dict.fromkeys(supp["condition"]))
        tasks = list(dict.fromkeys(supp["task"]))
        x = np.arange(len(conds))
        w = 0.8 / max(len(tasks), 1)
        for i, t in enumerate(tasks):
            m, ci = _mean_ci(supp[supp["task"] == t], "delta_auc", "condition", conds)
            ax1.bar(x + (i - (len(tasks) - 1) / 2) * w, m.values, w, yerr=ci.values,
                    capsize=3, label=t, color=_PALETTE[i % len(_PALETTE)])
        ax1.set_xticks(x)
        ax1.set_xticklabels(conds, rotation=15)
    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.set_ylabel("ΔAUC vs unsuppressed")
    ax1.set_title("Query-group suppression")
    ax1.legend(frameon=False, fontsize=8)
    if not probe.empty:
        tasks = list(dict.fromkeys(probe["task"]))
        m, ci = _mean_ci(probe, "probe_auc", "task", tasks)
        ax2.bar(np.arange(len(tasks)), m.values, 0.6, yerr=ci.values, capsize=3,
                color=_PALETTE[: len(tasks)])
        ax2.set_xticks(np.arange(len(tasks)))
        ax2.set_xticklabels(tasks, rotation=15)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("ESI probe reconstruction AUC")
    ax2.set_title("Acuity probe")
    _clean(ax1)
    _clean(ax2)
    return _save(fig, out_png)


def fig4_transfer(e5_csv, out_png) -> Path | None:
    """Fig 4: transfer decay — random vs temporal test AUC, grouped bars per task."""
    df = _read(e5_csv)
    if df is None:
        return None
    df = _norm_e5(df)
    if df.empty:
        warnings.warn("[figures] E5 CSV has no usable rows, skipped")
        return None
    tasks = list(dict.fromkeys(df["task"]))
    combos = list(dict.fromkeys(df["combo"]))
    splits = list(dict.fromkeys(df["split"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 3.8),
                             squeeze=False, sharey=True)
    x = np.arange(len(combos))
    w = 0.8 / max(len(splits), 1)
    for ax, t in zip(axes[0], tasks):
        d = df[df["task"] == t]
        for i, sp in enumerate(splits):
            m, ci = _mean_ci(d[d["split"] == sp], "auc", "combo", combos)
            ax.bar(x + (i - (len(splits) - 1) / 2) * w, m.values, w, yerr=ci.values,
                   capsize=2, label=sp, color=_PALETTE[i % len(_PALETTE)])
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=20)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(t)
        _clean(ax)
    axes[0][0].set_ylabel("AUROC")
    axes[0][-1].legend(frameon=False, fontsize=8)
    return _save(fig, out_png)


def fig5_sds_csr(sds_csv, out_png) -> Path | None:
    """Fig 5: SDS / CSR summary — grouped bars per metric with bootstrap 95% CI."""
    df = _read(sds_csv)
    if df is None:
        return None
    df = _norm_sds(df)
    order = [m for m in ["SDS_mask", "SDS_solo", "CSR", "R"] if m in set(df["metric"])]
    order += [m for m in dict.fromkeys(df["metric"]) if m not in order]
    tasks = list(dict.fromkeys(df["task"]))
    x = np.arange(len(order))
    w = 0.8 / max(len(tasks), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, t in enumerate(tasks):
        d = df[df["task"] == t].set_index("metric").reindex(order)
        vals = d["value"].to_numpy(dtype=float)
        lo = np.clip(vals - d["ci_lo"].to_numpy(dtype=float), 0, None)
        hi = np.clip(d["ci_hi"].to_numpy(dtype=float) - vals, 0, None)
        ax.bar(x + (i - (len(tasks) - 1) / 2) * w, vals, w,
               yerr=np.vstack([lo, hi]), capsize=3, label=t,
               color=_PALETTE[i % len(_PALETTE)])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Value")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)
    return _save(fig, out_png)


def make_all_figures(results_root, out_dir) -> list[Path]:
    """Run fig2–fig5 against results_root/audit/; missing inputs are skipped."""
    results_root = Path(results_root)
    out_dir = Path(out_dir)
    audit = results_root / "audit"
    made = []
    for fn, csv_name, png_name in [
        (fig2_branch_mask, "E1_branch_mask.csv", "fig2_branch_mask.png"),
        (fig3_repr_intervention, "E3_repr_intervention.csv", "fig3_repr_intervention.png"),
        (fig4_transfer, "E5_transfer.csv", "fig4_transfer.png"),
        (fig5_sds_csr, "sds_csr.csv", "fig5_sds_csr.png"),
    ]:
        p = fn(audit / csv_name, out_dir / png_name)
        if p is not None:
            made.append(p)
    return made
