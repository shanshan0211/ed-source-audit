"""E3 — representation-level intervention on TLQ-T (SPEC §5, §6).

Suppress latent-query groups J / P / D (and a random equal-sized C-query
control) in a trained TLQ-T and measure test ΔAUC; plus an ESI-reconstruction
probe (models.probes.train_esi_probe) on the learned representation.

Torch/model imports are lazy; per-run failures become status="skipped" rows
and must be inspected rather than interpreted as completed experiments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as astats
from .masks import cls_tasks
from .runs_index import load_preds

SUPPRESSIONS = (("none", None), ("suppress_J", ["J"]), ("suppress_P", ["P"]),
                ("suppress_D", ["D"]), ("suppress_random_C", ["random"]))


# ------------------------------------------------------ lazy integration IO
def find_tlqt_run(results_root, task: str, seed: int, combo: str = "full") -> Path | None:
    """Run dir of a trained TLQ-T (requires model.pt per SPEC §4)."""
    d = Path(results_root) / "runs" / task / "tlqt" / combo / f"seed{seed}"
    return d if (d / "model.pt").is_file() else None


def pick_device(cfg: dict) -> str:
    want = str(cfg.get("device", "cpu"))
    if want.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
    return want


def load_tlqt_model(run_dir, cfg: dict, device: str = "cpu"):
    """build_model(cfg) + load state_dict from model.pt (SPEC §6)."""
    import torch  # lazy
    from wjem_ed_audit.models.tlqt import build_model  # lazy import
    model = build_model(cfg)
    state = torch.load(Path(run_dir) / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def make_test_dataset(cfg: dict, task: str, cache_dir,
                      split_col: str = "split_random", split_val: str = "test"):
    """SPEC §6: make_tlqt_dataset(cache_dir, task, split_col, split_val, groups, horizon_h, cfg)."""
    from wjem_ed_audit.data.dataset import make_tlqt_dataset  # lazy import
    horizon_h = cfg.get("tlqt", {}).get("horizon_h", 6)
    return make_tlqt_dataset(cache_dir, task, split_col, split_val,
                             ("C", "J", "P", "D"), horizon_h, cfg)


# batch keys consumed by TLQT.forward (mirrors models.trainer._MODEL_KEYS)
MODEL_KEYS = ("hourly", "hourly_mask", "text", "tab_C", "tab_J", "mask_J",
              "tab_P", "tab_D")


def predict_scores(model, dataset, batch_size: int = 256, device: str = "cpu",
                   suppress_groups: list[str] | None = None,
                   batch_hook=None) -> tuple[np.ndarray, np.ndarray]:
    """Forward the dataset; returns (y_true, logits).

    `batch_hook(batch, rng)` may mutate the batch dict in place before forward
    (used by E4 permutation). """
    import torch  # lazy
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ys, ss = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in loader:
            if batch_hook is not None:
                batch = batch_hook(batch, rng)
            b = {k: batch[k].to(device) for k in MODEL_KEYS if k in batch}
            logits = model(b, suppress_groups=suppress_groups)
            ys.append(batch["label"].detach().cpu().numpy().ravel())
            ss.append(logits.detach().cpu().numpy().ravel())
    return np.concatenate(ys), np.concatenate(ss)


def run_esi_probe(model, dataset, device: str = "cpu"):
    """ESI-reconstruction probe AUC via models.probes.train_esi_probe (lazy).

    Interface: train_esi_probe(model, dataset, device) ->
    {"probe_auc", "probe_r2"}. Accepts a float or other dict keys too; NaN
    when unavailable. """
    try:
        from wjem_ed_audit.models.probes import train_esi_probe  # lazy
    except ImportError:
        return float("nan"), "probe_unavailable:models.probes missing"
    try:
        try:
            res = train_esi_probe(model=model, dataset=dataset, device=device)
        except TypeError:
            res = train_esi_probe(model, dataset, device)
    except Exception as exc:
        return float("nan"), f"probe_failed:{type(exc).__name__}: {exc}"
    if isinstance(res, dict):
        for k in ("probe_auc", "auroc", "auc", "probe_auroc"):
            if k in res:
                return float(res[k]), "ok"
        return float("nan"), f"probe_unparsed:{sorted(res.keys())}"
    try:
        return float(res), "ok"
    except (TypeError, ValueError):
        return float("nan"), "probe_unparsed"


# -------------------------------------------------------------------- core
def run_e3(cfg: dict, results_root, cache_dir=None, out_dir=None) -> pd.DataFrame:
    """E3 driver: per task x seed, ΔAUC of each suppression + ESI probe AUC.

    Writes E3_repr_intervention.csv. Rows: task, seed, intervention, auc,
    delta_vs_none, status, note plus lo/hi/p/p_adj. Summary rows (seed="ALL",
    status="summary") carry the cross-seed mean delta with t-based 95% CI,
    one-sample t-test p and BH-FDR adjusted p (family: the 4 suppression
    tests per task). Baseline AUC comes from the run's preds.parquet (random
    test split) when available, else recomputed.
    """
    results_root = Path(results_root)
    cache_dir = cache_dir or cfg.get("data", {}).get("cache_dir")
    device = pick_device(cfg)
    batch_size = int(cfg.get("tlqt", {}).get("batch_size", 256))
    cols = ["task", "seed", "intervention", "auc", "delta_vs_none",
            "status", "note", "lo", "hi", "p", "p_adj"]
    rows: list[dict] = []
    for task in cls_tasks(cfg):
        for seed in cfg.get("seeds", [0]):
            run_dir = find_tlqt_run(results_root, task, int(seed))
            if run_dir is None:
                rows.append({"task": task, "seed": int(seed),
                             "intervention": "ALL", "auc": np.nan,
                             "delta_vs_none": np.nan, "status": "skipped",
                             "note": f"no trained tlqt run for {task}/seed{seed}"})
                continue
            try:
                model = load_tlqt_model(run_dir, cfg, device)
                dataset = make_test_dataset(cfg, task, cache_dir)
                preds = load_preds(run_dir)
                base_auc = np.nan
                if not preds.empty and {"y_true", "y_score"} <= set(preds.columns):
                    base_auc = astats._safe_auc(preds["y_true"].to_numpy(),
                                                preds["y_score"].to_numpy())
                for name, groups in SUPPRESSIONS:
                    y, s = predict_scores(model, dataset, batch_size, device,
                                          suppress_groups=groups)
                    auc = astats._safe_auc(y, s)
                    if name == "none" and not np.isnan(base_auc):
                        auc = base_auc  # trust the recorded test predictions
                    if name == "none":
                        base_auc = auc
                    rows.append({"task": task, "seed": int(seed),
                                 "intervention": name, "auc": auc,
                                 "delta_vs_none": base_auc - auc,
                                 "status": "ok", "note": ""})
                probe_auc, probe_note = run_esi_probe(model, dataset, device)
                rows.append({"task": task, "seed": int(seed),
                             "intervention": "esi_probe", "auc": probe_auc,
                             "delta_vs_none": np.nan,
                             "status": "ok" if probe_note == "ok" else "skipped",
                             "note": probe_note})
            except Exception as exc:
                rows.append({"task": task, "seed": int(seed),
                             "intervention": "ALL", "auc": np.nan,
                             "delta_vs_none": np.nan, "status": "skipped",
                             "note": f"{type(exc).__name__}: {exc}"})
    df = pd.DataFrame(rows)
    for c in ("lo", "hi", "p", "p_adj"):
        if c not in df.columns:
            df[c] = np.nan
    # ---- summary rows: per task x suppression, across seeds -----------------
    # Family for BH-FDR: the 4 query-group suppression tests of one task
    # (the "none" reference and the ESI probe are not delta tests).
    supp = df[df["intervention"].isin(
        [n for n, _g in SUPPRESSIONS if n != "none"]) & (df["status"] == "ok")]
    summary_rows = []
    for (task, interv), g in supp.groupby(["task", "intervention"], sort=True):
        deltas = g["delta_vs_none"].to_numpy()
        agg = astats.aggregate_seeds(g.assign(v=g["delta_vs_none"]), "v")
        _, p = astats.ttest_one_sample(deltas)
        summary_rows.append({"task": task, "seed": "ALL", "intervention": interv,
                             "auc": np.nan, "delta_vs_none": agg["mean"],
                             "status": "summary", "note": f"n_seeds={agg['n']}",
                             "lo": agg["lo"], "hi": agg["hi"], "p": p,
                             "p_adj": np.nan})
    summ = pd.DataFrame(summary_rows, columns=cols)
    if not summ.empty:
        for _, gidx in summ.groupby("task").groups.items():
            summ.loc[gidx, "p_adj"] = astats.bh_fdr(summ.loc[gidx, "p"].to_numpy())
    df = pd.concat([df, summ], ignore_index=True)[cols]
    out_dir = Path(out_dir) if out_dir else results_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "E3_repr_intervention.csv", index=False)
    return df
