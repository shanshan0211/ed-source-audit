"""E5 — temporal transfer decay (SPEC §5).

Compares each combo's AUC on split_random=test vs split_temporal=test
(runs_index rows with temporal=False/True), quantifies decay, and runs an
approximate combo x split interaction test: the decay of each combo is
contrasted against the full combo via a bootstrap over stays (resampling the
four preds sets of every seed jointly, then averaging across seeds).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as astats
from .masks import cls_tasks
from .runs_index import load_preds, scan_runs

EICU_ROW = {
    "kind": "external", "task": "ALL", "model": "ALL", "combo": "eicu_reanchor",
    "seed": "ALL", "auc_random": np.nan, "auc_temporal": np.nan,
    "decay": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan,
    "status": "not_available_pending_dua",
    "note": "eICU-CRD 无 ED 阶段数据/无 ESI，待 DUA 后按 §3.3 再锚定协议执行",
}


# ------------------------------------------------------------------ pieces
def _preds_pair(idx: pd.DataFrame, task: str, model: str, combo: str,
                seed: int):
    """(preds_random, preds_temporal) DataFrames for one cell, or (None, None)."""
    sub = idx[(idx["task"] == task) & (idx["model"] == model)
              & (idx["config_tag"] == combo) & (idx["seed"] == seed)]
    out = []
    for temporal in (False, True):
        row = sub[sub["temporal"] == temporal]
        if row.empty:
            out.append(None)
        else:
            preds = load_preds(row.iloc[0]["path"])
            out.append(preds if not preds.empty
                       and {"y_true", "y_score"} <= set(preds.columns) else None)
    return out[0], out[1]


def _auc_of(preds: pd.DataFrame | None, rng: np.random.Generator | None = None) -> float:
    if preds is None:
        return float("nan")
    yt = preds["y_true"].to_numpy()
    ys = preds["y_score"].to_numpy()
    if rng is not None:
        idx = rng.integers(0, len(yt), len(yt))
        yt, ys = yt[idx], ys[idx]
    return astats._safe_auc(yt, ys)


def decay_table(idx: pd.DataFrame, tasks: list[str] | None = None) -> pd.DataFrame:
    """Per task x model x combo x seed: auc_random, auc_temporal, decay."""
    rows = []
    if idx.empty:
        return pd.DataFrame(columns=["task", "model", "combo", "seed",
                                     "auc_random", "auc_temporal", "decay"])
    cells = (idx[["task", "model", "config_tag", "seed"]].drop_duplicates())
    for _, c in cells.iterrows():
        if tasks is not None and c["task"] not in tasks:
            continue
        sub = idx[(idx["task"] == c["task"]) & (idx["model"] == c["model"])
                  & (idx["config_tag"] == c["config_tag"])
                  & (idx["seed"] == c["seed"])]
        ar = sub.loc[~sub["temporal"], "auroc"]
        at = sub.loc[sub["temporal"], "auroc"]
        if ar.empty and at.empty:
            continue
        auc_r = float(ar.iloc[0]) if not ar.empty else float("nan")
        auc_t = float(at.iloc[0]) if not at.empty else float("nan")
        rows.append({"task": c["task"], "model": c["model"],
                     "combo": c["config_tag"], "seed": int(c["seed"]),
                     "auc_random": auc_r, "auc_temporal": auc_t,
                     "decay": auc_r - auc_t})
    return pd.DataFrame(rows)


def interaction_contrast(idx: pd.DataFrame, task: str, model: str, combo: str,
                         seeds: list[int], ref: str = "full", n_boot: int = 500,
                         seed: int = 0) -> dict:
    """Bootstrap interaction: decay(combo) - decay(ref), averaged across seeds.

    Per bootstrap replicate and seed, one stay-level index set is drawn per
    split (random / temporal) and shared by the combo and ref score vectors
    (paired within split); decays are computed per seed and averaged, then
    contrasted. CI is the percentile interval; p is the two-sided sign-based
    bootstrap p (see stats._sign_boot_p)."""
    rng = np.random.default_rng(seed)
    cells = []
    for sd in seeds:
        pr, pt = _preds_pair(idx, task, model, combo, sd)
        rr, rt = _preds_pair(idx, task, model, ref, sd)
        if any(x is None for x in (pr, pt, rr, rt)):
            continue
        cells.append((pr, pt, rr, rt))
    if not cells:
        return {"interaction": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan"), "n_seeds": 0}

    def _point():
        dec_c = np.nanmean([_auc_of(pr) - _auc_of(pt) for pr, pt, _, _ in cells])
        dec_r = np.nanmean([_auc_of(rr) - _auc_of(rt) for _, _, rr, rt in cells])
        return float(dec_c - dec_r)

    def _resampled_auc(preds: pd.DataFrame, idx: np.ndarray) -> float:
        yt = preds["y_true"].to_numpy()[idx]
        ys = preds["y_score"].to_numpy()[idx]
        return astats._safe_auc(yt, ys)

    boots = []
    for _ in range(n_boot):
        dec_c, dec_r = [], []
        for pr, pt, rr, rt in cells:
            # Paired resampling: combo and ref scores come from the SAME stays
            # within a split, so they share one index set per split; the
            # random and temporal test sets are different cohorts and are
            # resampled independently.
            idx_r = rng.integers(0, len(pr), len(pr))
            idx_t = rng.integers(0, len(pt), len(pt))
            dec_c.append(_resampled_auc(pr, idx_r) - _resampled_auc(pt, idx_t))
            dec_r.append(_resampled_auc(rr, idx_r) - _resampled_auc(rt, idx_t))
        boots.append(np.nanmean(dec_c) - np.nanmean(dec_r))
    boots = np.asarray(boots)
    lo, hi = astats._percentile_ci(boots)
    return {"interaction": _point(), "lo": lo, "hi": hi,
            "p": astats._sign_boot_p(boots), "n_seeds": len(cells)}


# ------------------------------------------------------------------ driver
def run_e5(cfg: dict, results_root, out_dir=None, n_boot: int | None = None,
           seed: int = 0) -> pd.DataFrame:
    """E5 driver; writes E5_transfer.csv (detail + decay summary + interaction
    rows + the eICU placeholder row)."""
    results_root = Path(results_root)
    n_boot = n_boot or int(cfg.get("audit", {}).get("bootstrap", 500))
    idx = scan_runs(results_root)
    tasks = cls_tasks(cfg)
    combos = list(cfg.get("audit", {}).get("e1_combos", {}).keys()) or None
    det = decay_table(idx, tasks=tasks)
    cols = ["kind", "task", "model", "combo", "seed", "auc_random",
            "auc_temporal", "decay", "lo", "hi", "p", "status", "note"]
    rows: list[dict] = []
    for _, r in det.iterrows():
        if combos is not None and r["combo"] not in combos:
            continue
        rows.append({"kind": "detail", "task": r["task"], "model": r["model"],
                     "combo": r["combo"], "seed": r["seed"],
                     "auc_random": r["auc_random"], "auc_temporal": r["auc_temporal"],
                     "decay": r["decay"], "lo": np.nan, "hi": np.nan, "p": np.nan,
                     "status": "ok", "note": ""})
    for (task, model, combo), g in det.groupby(["task", "model", "combo"], sort=True):
        if combos is not None and combo not in combos:
            continue
        agg = astats.aggregate_seeds(g.assign(v=g["decay"]), "v")
        _, p = astats.ttest_one_sample(g["decay"].to_numpy())
        rows.append({"kind": "decay_summary", "task": task, "model": model,
                     "combo": combo, "seed": "ALL", "auc_random": np.nan,
                     "auc_temporal": np.nan, "decay": agg["mean"], "lo": agg["lo"],
                     "hi": agg["hi"], "p": p, "status": "ok",
                     "note": f"n_seeds={agg['n']}"})
        if combo != "full":
            seeds = sorted(g["seed"].unique().tolist())
            inter = interaction_contrast(idx, task, model, combo, seeds,
                                         ref="full", n_boot=n_boot, seed=seed)
            rows.append({"kind": "interaction", "task": task, "model": model,
                         "combo": f"{combo}__vs_full", "seed": "ALL",
                         "auc_random": np.nan, "auc_temporal": np.nan,
                         "decay": inter["interaction"], "lo": inter["lo"],
                         "hi": inter["hi"], "p": inter["p"], "status": "ok",
                         "note": "decay(combo)-decay(full), bootstrap over stays"})
    rows.append(dict(EICU_ROW))
    out = pd.DataFrame(rows, columns=cols)
    out_dir = Path(out_dir) if out_dir else results_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "E5_transfer.csv", index=False)
    return out
