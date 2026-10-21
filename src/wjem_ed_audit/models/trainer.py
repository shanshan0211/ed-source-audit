"""Training / evaluation drivers for baselines and TLQ-T (SPEC §4, §6).

Random-split protocol: split_random=train for fitting, split_random=val for
early stopping (TLQ-T), split_random=test for the reported metrics.

edlos target convention: data.dataset stores labels as log1p(edlos_hours), so
the TLQ-T optimises MSE on the log scale directly; predictions are expm1'd and
labels inverted back to raw hours for metrics/preds (matching baselines).

Temporal protocol: a *fresh* model is fitted on split_temporal=dev (TLQ-T
carves an internal 80/20 train/val split out of dev for early stopping, per
SPEC §1.1) and evaluated on split_temporal=test; results go to the run's
``temporal/`` subdirectory.

Run directories: ``results/runs/{task}/{model}/{config_tag}/seed{k}[/temporal]``
written via ``data.api.write_run``.  ``config_tag`` is the matching
``audit.e1_combos`` name when the group list matches a canonical combo
(full / clin_only / meta_only / noJ / noP), otherwise ``'_'.join(groups)``.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from wjem_ed_audit.data import api
from wjem_ed_audit.models.baselines import fit_predict_proba, is_classification


# ------------------------------------------------------------------ metrics
def _safe(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return float("nan")


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    m = {"auroc": _safe(roc_auc_score, y_true, y_score),
         "auprc": _safe(average_precision_score, y_true, y_score),
         "brier": _safe(brier_score_loss, y_true, np.clip(y_score, 0, 1)),
         "ece": _ece(y_true, y_score)}
    return m


def _ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    p = np.clip(np.asarray(y_score, dtype=np.float64), 0, 1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {"mae": _safe(mean_absolute_error, y_true, y_pred),
            "rmse": _safe(lambda a, b: math.sqrt(mean_squared_error(a, b)),
                          y_true, y_pred),
            "r2": _safe(r2_score, y_true, y_pred)}


def _mse(a, b):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(a, b)


def _val_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    return _safe(roc_auc_score, y_true, scores)


def _preds_frame(ids, split, y_true, scores, task):
    col = "y_score" if is_classification(task) else "y_pred"
    return pd.DataFrame({"stay_id": ids, "split": split,
                         "y_true": y_true, col: scores})


def _eval_scores(task, y_true, scores):
    if is_classification(task):
        return classification_metrics(y_true, scores)
    return regression_metrics(y_true, scores)


# --------------------------------------------------------------- config tag
def config_tag_for(groups, cfg: dict) -> str:
    g = list(groups)
    for name, combo in (cfg.get("audit", {}).get("e1_combos", {}) or {}).items():
        if list(combo) == g:
            return name
    return "_".join(g)


def _device(cfg: dict) -> torch.device:
    want = str(cfg.get("device", "cuda"))
    return torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")


# ---------------------------------------------------------------- baselines
def train_baseline(cfg: dict, task: str, model_name: str, groups, seed: int,
                   cache_dir, results_root, temporal_eval: bool = True) -> dict:
    """Fit one flat baseline on split_random=train, evaluate on random test
    (and on split_temporal=test with a dev-refit model when temporal_eval).
    Returns the random-test metrics dict."""
    groups = tuple(groups)
    tag = config_tag_for(groups, cfg)
    base_cfg = {"model": model_name, "task": task, "groups": list(groups),
                "seed": seed, "config_tag": tag,
                "hyper": _baseline_hyper(cfg, model_name)}

    Xtr, ytr, _ = api.load_flat(task, "split_random", "train", cache_dir, groups=groups)
    Xte, yte, ids_te = api.load_flat(task, "split_random", "test", cache_dir, groups=groups)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)  # align one-hot levels
    scores = fit_predict_proba(model_name, Xtr, ytr, Xte, task=task, seed=seed)
    metrics = _eval_scores(task, yte, scores)
    metrics.update({"n_test": int(len(yte)), "split": "split_random=test"})
    run_dir = api.run_dir_for(results_root, task, model_name, tag, seed)
    api.write_run(run_dir, metrics, _preds_frame(ids_te, "test", yte, scores, task),
                  {**base_cfg, "split": "split_random"})

    if temporal_eval:
        Xd, yd, _ = api.load_flat(task, "split_temporal", "dev", cache_dir, groups=groups)
        Xt, yt, ids_t = api.load_flat(task, "split_temporal", "test", cache_dir,
                                      groups=groups)
        # align dev/test columns (one-hot levels may differ between periods)
        Xt = Xt.reindex(columns=Xd.columns, fill_value=0.0)
        sc_t = fit_predict_proba(model_name, Xd, yd, Xt, task=task, seed=seed)
        m_t = _eval_scores(task, yt, sc_t)
        m_t.update({"n_test": int(len(yt)), "split": "split_temporal=test"})
        api.write_run(api.run_dir_for(results_root, task, model_name, tag, seed,
                                      temporal=True),
                      m_t, _preds_frame(ids_t, "test", yt, sc_t, task),
                      {**base_cfg, "split": "split_temporal (fit on dev)"})
    return metrics


def _baseline_hyper(cfg: dict, model_name: str) -> dict:
    if model_name == "lr":
        return {"max_iter": 2000, "impute": "median", "standardize": True}
    return {"n_estimators": 300, "max_depth": 6 if model_name == "xgb" else None,
            "learning_rate": 0.1}


# ------------------------------------------------------------------- TLQ-T
def _make_dataset(cache_dir, task, split_col, split_val, groups, horizon_h, cfg):
    # Imported lazily from the dataset module.
    from wjem_ed_audit.data.dataset import make_tlqt_dataset
    return make_tlqt_dataset(cache_dir, task, split_col, split_val,
                             list(groups), horizon_h, cfg)


def _batch_ids(batch: dict, dataset, n: int) -> np.ndarray:
    if "stay_id" in batch:
        return batch["stay_id"].numpy()
    for attr in ("stay_ids", "ids", "stay_id"):
        if hasattr(dataset, attr):
            ids = getattr(dataset, attr)
            return np.asarray(ids[:n] if not callable(ids) else ids())
    return np.arange(n)


@torch.no_grad()
def _predict(model, loader, device, bf16: bool, task: str):
    model.eval()
    all_scores, all_y, all_ids = [], [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items() if k in _MODEL_KEYS}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(b)
        scores = torch.sigmoid(logits.float()) if is_classification(task) \
            else torch.expm1(logits.float()).clamp_min(0)
        all_scores.append(scores.cpu().numpy())
        # dataset stores edlos labels as log1p(hours) (data contract); report
        # the raw hour scale so metrics match the flat baselines.
        y = batch["label"].numpy()
        if not is_classification(task):
            y = np.expm1(y)
        all_y.append(y)
        all_ids.append(_batch_ids(batch, loader.dataset, len(scores)))
    return (np.concatenate(all_scores), np.concatenate(all_y),
            np.concatenate(all_ids))


_MODEL_KEYS = ("hourly", "hourly_mask", "text", "tab_C", "tab_J", "mask_J",
               "tab_P", "tab_D")


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items() if k in _MODEL_KEYS}


def _train_one(cfg, task, groups, seed, cache_dir, device,
               split_col, ds_tr, ds_va, ds_te):
    """Full fit+eval cycle for one (split_col) protocol. Returns
    (model, scores, y, ids) on ds_te."""
    from wjem_ed_audit.models.tlqt import build_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    tc = cfg.get("tlqt", {})
    bs = int(tc.get("batch_size", 256))
    epochs = int(tc.get("epochs", 25))
    patience = int(tc.get("early_stop_patience", 4))
    bf16 = bool(tc.get("bf16", True))
    cls = is_classification(task)
    # Auxiliary per-hour BCE/MSE on the causal risk curve (return_risk_curve);
    # without it the rc_* branch never enters the training graph and stays at
    # random init.  Every cursor position yields a valid prediction (static
    # triage tokens are always visible), so all T positions are supervised.
    rc_w = float(tc.get("risk_curve_loss_weight", 0.3))

    # Worker/pin settings only affect I/O throughput, not numerics: sampler
    # order (shuffle RNG) and the default collate are unchanged.
    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False,
                       num_workers=4, pin_memory=pin, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=bs, num_workers=4, pin_memory=pin,
                       persistent_workers=True)
    dl_te = DataLoader(ds_te, batch_size=bs, num_workers=4, pin_memory=pin,
                       persistent_workers=True)

    model = build_model(cfg).to(device)
    first = next(iter(dl_tr))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        model(_move(first, device))  # materialise LazyLinear before optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc.get("lr", 1e-3)),
                            weight_decay=float(tc.get("weight_decay", 0.01)))
    loss_fn = torch.nn.BCEWithLogitsLoss() if cls else torch.nn.MSELoss()
    rc_loss_fn = (torch.nn.BCEWithLogitsLoss(reduction="none") if cls
                  else torch.nn.MSELoss(reduction="none"))

    def target(batch):
        # dataset labels are log1p(edlos_hours) for the edlos task; applying
        # log1p here again would double-transform the target.
        y = batch["label"].to(device).float()
        return y if cls else y.clamp_min(0)

    def risk_curve_loss(risk: torch.Tensor, batch: dict) -> torch.Tensor:
        y = target(batch).unsqueeze(1).expand_as(risk)
        return rc_loss_fn(risk.float(), y).mean()

    best_state, best_score, bad = None, -np.inf, 0
    for _ep in range(epochs):
        model.train()
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            b = _move(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                if rc_w > 0:
                    logits, risk = model(b, return_risk_curve=True)
                    loss = loss_fn(logits.float(), target(batch)) \
                        + rc_w * risk_curve_loss(risk, batch)
                else:
                    logits = model(b)
                    loss = loss_fn(logits.float(), target(batch))
            loss.backward()
            opt.step()
        va_scores, va_y, _ = _predict(model, dl_va, device, bf16, task)
        crit = (_val_auc(va_y, va_scores) if cls
                else -_safe(_mse, va_y, va_scores))
        print(f"[tlqt:{split_col}] epoch {_ep + 1}/{epochs} val_crit={crit:.5f}",
              flush=True)
        if np.isnan(crit):
            crit = -np.inf
        if crit > best_score:
            best_score, bad = crit, 0
            best_state = copy.deepcopy({k: v.detach().cpu()
                                        for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    scores, y, ids = _predict(model, dl_te, device, bf16, task)
    return model, scores, y, ids


def train_tlqt(cfg: dict, task: str, groups, seed: int, cache_dir, results_root,
               temporal_eval: bool = True) -> dict:
    """Train TLQ-T on split_random=train (val for early stopping), evaluate on
    random test; when temporal_eval, refit on split_temporal=dev (internal
    80/20 val) and evaluate on split_temporal=test. Returns random-test
    metrics."""
    groups = list(groups)
    tag = config_tag_for(groups, cfg)
    horizon = int(cfg.get("tlqt", {}).get("horizon_h", 24))
    device = _device(cfg)
    run_cfg = {"model": "tlqt", "task": task, "groups": groups, "seed": seed,
               "config_tag": tag, "tlqt": cfg.get("tlqt", {}), "device": str(device)}

    ds_tr = _make_dataset(cache_dir, task, "split_random", "train", groups, horizon, cfg)
    ds_va = _make_dataset(cache_dir, task, "split_random", "val", groups, horizon, cfg)
    ds_te = _make_dataset(cache_dir, task, "split_random", "test", groups, horizon, cfg)
    model, scores, y, ids = _train_one(cfg, task, groups, seed, cache_dir, device,
                                       "split_random", ds_tr, ds_va, ds_te)
    metrics = _eval_scores(task, y, scores)
    metrics.update({"n_test": int(len(y)), "split": "split_random=test"})
    api.write_run(api.run_dir_for(results_root, task, "tlqt", tag, seed),
                  metrics, _preds_frame(ids, "test", y, scores, task),
                  {**run_cfg, "split": "split_random"},
                  model_state={k: v.cpu() for k, v in model.state_dict().items()})

    if temporal_eval:
        ds_dev = _make_dataset(cache_dir, task, "split_temporal", "dev", groups,
                               horizon, cfg)
        ds_tt = _make_dataset(cache_dir, task, "split_temporal", "test", groups,
                              horizon, cfg)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ds_dev))
        n_val = max(1, int(0.2 * len(ds_dev)))
        ds_tr2 = Subset(ds_dev, perm[n_val:].tolist())
        ds_va2 = Subset(ds_dev, perm[:n_val].tolist())
        model2, sc_t, y_t, ids_t = _train_one(cfg, task, groups, seed, cache_dir,
                                              device, "split_temporal",
                                              ds_tr2, ds_va2, ds_tt)
        m_t = _eval_scores(task, y_t, sc_t)
        m_t.update({"n_test": int(len(y_t)), "split": "split_temporal=test"})
        api.write_run(api.run_dir_for(results_root, task, "tlqt", tag, seed,
                                      temporal=True),
                      m_t, _preds_frame(ids_t, "test", y_t, sc_t, task),
                      {**run_cfg, "split": "split_temporal (fit on dev)"},
                      model_state={k: v.cpu() for k, v in model2.state_dict().items()})
    return metrics


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
