"""E4 — test-set group permutation (SPEC §5).

For trained lgbm (flat features) and TLQ-T, permute the J and P fields across
test stays *as whole rows of the group* (joint distribution within the group
is preserved; only the linkage to the stay/outcome is broken), then measure
ΔAUC. Torch/model imports are lazy (module stays import-safe).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import api
from . import stats as astats
from .ablate import columns_for_group, default_fit_predict
from .masks import cls_tasks

GROUPS = ("J", "P")


def permute_group_columns(X: pd.DataFrame, group: str,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Copy of X with the group's columns permuted across rows jointly."""
    cols = columns_for_group(list(X.columns), group)
    if not cols:
        return X.copy()
    Xp = X.copy()
    idx = rng.permutation(len(Xp))
    Xp.loc[:, cols] = Xp[cols].to_numpy()[idx]
    return Xp


def make_group_permute_hook(group: str):
    """Batch hook permuting a TLQ-T group's tensors along the batch dim jointly."""
    keys = {"J": ("tab_J", "mask_J"), "P": ("tab_P",), "D": ("tab_D",),
            "C": ("tab_C",)}[group]

    def hook(batch, rng):
        import torch  # lazy
        first = next((k for k in keys if k in batch), None)
        if first is None:
            return batch
        b = batch[first].shape[0]
        perm = torch.from_numpy(rng.permutation(b))  # seeded -> reproducible
        for k in keys:
            if k in batch:
                batch[k] = batch[k][perm]
        return batch
    return hook


def _e4_lgbm(task: str, seed: int, cache_dir, rng, fit_predict=None) -> list[dict]:
    fp = fit_predict or default_fit_predict
    Xtr, ytr, _ = api.load_flat(task, "split_random", "train", cache_dir)
    Xte, yte, _ = api.load_flat(task, "split_random", "test", cache_dir)
    scores = fp(Xtr, ytr, Xte, seed=seed, task=task)
    auc_base = astats._safe_auc(yte, scores)
    rows = []
    for group in GROUPS:
        Xp = permute_group_columns(Xte, group, rng)
        auc_p = astats._safe_auc(yte, fp(Xtr, ytr, Xp, seed=seed, task=task))
        rows.append({"task": task, "model": "lgbm", "seed": int(seed),
                     "group": group, "auc_base": auc_base, "auc_permuted": auc_p,
                     "delta": auc_base - auc_p, "status": "ok", "note": ""})
    return rows


def _e4_tlqt(task: str, seed: int, cfg, cache_dir, results_root, rng) -> list[dict]:
    from . import repr_interv as ri  # lazy-heavy helpers
    run_dir = ri.find_tlqt_run(results_root, task, seed)
    if run_dir is None:
        return [{"task": task, "model": "tlqt", "seed": int(seed), "group": g,
                 "auc_base": np.nan, "auc_permuted": np.nan, "delta": np.nan,
                 "status": "skipped",
                 "note": f"no trained tlqt run for {task}/seed{seed}"}
                for g in GROUPS]
    device = ri.pick_device(cfg)
    model = ri.load_tlqt_model(run_dir, cfg, device)
    dataset = ri.make_test_dataset(cfg, task, cache_dir)
    bs = int(cfg.get("tlqt", {}).get("batch_size", 256))
    y, s = ri.predict_scores(model, dataset, bs, device)
    auc_base = astats._safe_auc(y, s)
    rows = []
    for group in GROUPS:
        y2, s2 = ri.predict_scores(model, dataset, bs, device,
                                   batch_hook=make_group_permute_hook(group))
        auc_p = astats._safe_auc(y2, s2)
        rows.append({"task": task, "model": "tlqt", "seed": int(seed),
                     "group": group, "auc_base": auc_base, "auc_permuted": auc_p,
                     "delta": auc_base - auc_p, "status": "ok", "note": ""})
    return rows


def run_e4(cfg: dict, results_root, cache_dir=None, out_dir=None,
           fit_predict=None, seed: int = 0) -> pd.DataFrame:
    """E4 driver; writes E4_permutation.csv (detail rows per task x model x
    seed x group plus summary rows seed="ALL" with mean/CI/p across seeds and
    BH-FDR adjusted p; family: the J/P permutation tests per task x model)."""
    results_root = Path(results_root)
    cache_dir = cache_dir or cfg.get("data", {}).get("cache_dir")
    rng = np.random.default_rng(seed)
    cols = ["task", "model", "seed", "group", "auc_base", "auc_permuted",
            "delta", "status", "note", "p", "p_adj"]
    rows: list[dict] = []
    for task in cls_tasks(cfg):
        for sd in cfg.get("seeds", [0]):
            sd = int(sd)
            try:
                rows.extend(_e4_lgbm(task, sd, cache_dir, rng, fit_predict))
            except Exception as exc:
                for g in GROUPS:
                    rows.append({"task": task, "model": "lgbm", "seed": sd,
                                 "group": g, "auc_base": np.nan,
                                 "auc_permuted": np.nan, "delta": np.nan,
                                 "status": "failed",
                                 "note": f"{type(exc).__name__}: {exc}"})
            try:
                rows.extend(_e4_tlqt(task, sd, cfg, cache_dir, results_root, rng))
            except Exception as exc:
                for g in GROUPS:
                    rows.append({"task": task, "model": "tlqt", "seed": sd,
                                 "group": g, "auc_base": np.nan,
                                 "auc_permuted": np.nan, "delta": np.nan,
                                 "status": "skipped",
                                 "note": f"{type(exc).__name__}: {exc}"})
    df = pd.DataFrame(rows)
    for c in ("p", "p_adj"):
        if c not in df.columns:
            df[c] = np.nan
    summary = []
    for (task, model, group), g in df[df["status"] == "ok"].groupby(
            ["task", "model", "group"], sort=True):
        agg = astats.aggregate_seeds(g.assign(v=g["delta"]), "v")
        _, p = astats.ttest_one_sample(g["delta"].to_numpy())
        summary.append({"task": task, "model": model, "seed": "ALL",
                        "group": group, "auc_base": np.nan,
                        "auc_permuted": np.nan, "delta": agg["mean"],
                        "status": f"summary lo={agg['lo']:.4f} hi={agg['hi']:.4f}",
                        "note": f"t-test p={p:.4g}", "p": p, "p_adj": np.nan})
    summ = pd.DataFrame(summary, columns=cols)
    # BH-FDR within each (task, model) family (the J and P permutation tests).
    if not summ.empty:
        for _, gidx in summ.groupby(["task", "model"]).groups.items():
            summ.loc[gidx, "p_adj"] = astats.bh_fdr(summ.loc[gidx, "p"].to_numpy())
    out = pd.concat([df, summ], ignore_index=True)[cols]
    out_dir = Path(out_dir) if out_dir else results_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "E4_permutation.csv", index=False)
    return out
