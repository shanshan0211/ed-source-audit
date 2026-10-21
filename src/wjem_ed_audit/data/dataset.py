"""TLQ-T torch Dataset (SPEC §6): make_tlqt_dataset -> per-sample batch-key dict.

Group-masking design
====================
``groups`` selects which taxonomy groups (C/J/P/D) are *populated*. Tensor
SHAPES ARE FIXED regardless of ``groups``: an excluded group is returned as an
all-zeros tensor of the same dimension (for J, ``mask_J`` is also forced to 0;
for C, ``hourly``/``text``/``tab_C`` are zeroed and ``hourly_mask`` is forced
to 0). The TLQ-T model therefore never sees a dimension change across E1
input combinations (full / clin_only / meta_only / noJ / noP) — one model
class serves all combos, and zeroed source tokens contribute nothing to a
softmax pooler beyond a constant, mirroring the `suppress_groups` contract.

Scalers / encoders
==================
* ``cache_dir/scaler.json``: per-feature mean/std for the 10 hourly features
  + pain, fit ONCE on the training rows of both split schemes
  (split_random=="train" OR split_temporal=="dev") and reused thereafter.
* ``cache_dir/tab_encoder.json``: one-hot category lists for
  arrival_transport/gender/race/insurance/language fit ONCE on the full stays
  table. Encoding style matches api.build_flat_matrix: one-hot for
  categoricals (fillna "UNKNOWN"), numerics raw (hour_of_day, day_of_week,
  n_ed_visits_365d, anchor_age).

Labels: classification tasks -> float 0/1; edlos -> float log1p(edlos_hours).
Hourly: T=horizon_h rows from h=0; stays with fewer hours are zero-padded with
hourly_mask=0; NaN values inside observed hours are 0 after standardization.

Full-cohort tensor cache
========================
The per-stay tensors (hourly/hourly_mask/text/tab_C/tab_J/mask_J/tab_P/tab_D)
are split- and task-independent: they are built ONCE for the full cohort (all
stays sorted by stay_id) with exactly the per-row formulas used historically,
and persisted to ``cache_dir/tlqt_tensor_cache_T{horizon}.npz`` (atomic
tmp-file + rename). The npz also stores the three source-parquet mtimes; any
mtime change invalidates the cache and triggers a rebuild. ``__init__`` then
only loads the cache and selects the rows for ``self.stay_ids`` (via
``np.searchsorted`` on the sorted cache ids) — bit-exact with the legacy
per-split assembly. Labels stay per-task and are never cached.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .api import (HOURLY_FEATS, TRIAGE_VITALS, load_hourly, load_stays,
                  load_text_emb, task_label_col)

EMB_DIM = 768
_ONE_HOT_COLS = ("arrival_transport", "gender", "race", "insurance", "language")
_P_NUMERIC = ("hour_of_day", "day_of_week", "n_ed_visits_365d")
_D_NUMERIC = ("anchor_age",)


# ------------------------------------------------------------------ artifacts
def _fit_or_load_scaler(cache_dir: Path, stays: pd.DataFrame,
                        hourly: pd.DataFrame) -> dict:
    path = cache_dir / "scaler.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    train_ids = stays.loc[
        (stays["split_random"] == "train") | (stays["split_temporal"] == "dev"),
        "stay_id"]
    fit = hourly[hourly["stay_id"].isin(train_ids)]
    stats: dict[str, dict[str, list[float]]] = {"hourly": {}}
    for col in HOURLY_FEATS:
        v = fit[col].to_numpy(dtype=np.float64)
        m = float(np.nanmean(v))
        s = float(np.nanstd(v))
        stats["hourly"][col] = [m, s if s > 1e-6 else 1.0]
    pain = stays.loc[stays["stay_id"].isin(train_ids), "pain"].to_numpy(np.float64)
    ps = float(np.nanstd(pain))
    stats["pain"] = [float(np.nanmean(pain)), ps if ps > 1e-6 else 1.0]
    stats["fit_on"] = "split_random==train OR split_temporal==dev"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _fit_or_load_tab_encoder(cache_dir: Path, stays: pd.DataFrame) -> dict:
    path = cache_dir / "tab_encoder.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    enc = {col: sorted(stays[col].fillna("UNKNOWN").astype(str).unique().tolist())
           for col in _ONE_HOT_COLS}
    enc["numeric_P"] = list(_P_NUMERIC)
    enc["numeric_D"] = list(_D_NUMERIC)
    with open(path, "w") as f:
        json.dump(enc, f, indent=2)
    return enc


def _one_hot(values: pd.Series, cats: list[str]) -> np.ndarray:
    s = values.fillna("UNKNOWN").astype(str).to_numpy()
    out = np.zeros((len(s), len(cats)), dtype=np.float32)
    pos = {c: j for j, c in enumerate(cats)}
    for i, v in enumerate(s):
        j = pos.get(v)
        if j is not None:
            out[i, j] = 1.0
    return out


# -------------------------------------------------------------- tensor cache
def _tensor_cache_path(cache_dir: Path, horizon_h: int) -> Path:
    return cache_dir / f"tlqt_tensor_cache_T{horizon_h}.npz"


def _src_mtimes(cache_dir: Path) -> tuple[float, float, float]:
    def mt(name: str) -> float:
        p = cache_dir / name
        return os.path.getmtime(p) if p.exists() else -1.0
    return mt("stays.parquet"), mt("hourly.parquet"), mt("text_emb.parquet")


_CACHE_KEYS = ("stay_ids", "stays_row_pos", "hourly", "hourly_mask", "text",
               "tab_C", "tab_J", "mask_J", "tab_P", "tab_D")


def _load_tensor_cache(path: Path, mtimes) -> dict | None:
    """Return the cached full-cohort arrays, or None if missing/stale/corrupt."""
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            if not all(float(z[k]) == m for k, m in
                       zip(("mtime_stays", "mtime_hourly", "mtime_text"), mtimes)):
                return None
            return {k: z[k] for k in _CACHE_KEYS}
    except Exception:
        return None


def _save_tensor_cache(path: Path, arrays: dict, mtimes) -> None:
    """Atomic write (tmp file + rename) so concurrent readers never see a
    half-written cache."""
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays,
             mtime_stays=np.float64(mtimes[0]),
             mtime_hourly=np.float64(mtimes[1]),
             mtime_text=np.float64(mtimes[2]))
    os.replace(tmp, path)


def _build_tensor_cache(cache_dir: Path, stays_all: pd.DataFrame,
                        hourly_all: pd.DataFrame, scaler: dict, tab_enc: dict,
                        T: int) -> dict:
    """Assemble the split/task-independent per-stay tensors for the FULL
    cohort (all stays sorted by stay_id). Every per-row formula below is
    identical to the legacy per-split assembly in TLQTDataset.__init__, so any
    split's tensors are bit-exact row subsets of these arrays."""
    order = stays_all.sort_values("stay_id")
    stays_row_pos = order.index.to_numpy(np.int64)  # cache row -> stays_all row
    stays = order.reset_index(drop=True)
    stay_ids = stays["stay_id"].to_numpy(np.int64)
    n = len(stay_ids)

    # ---- hourly (T,10) standardized + mask --------------------------------
    # Per-stay frame content/row order matches the legacy split-filtered
    # assembly (boolean filtering preserves relative row order).
    mean = np.array([scaler["hourly"][c][0] for c in HOURLY_FEATS], np.float32)
    std = np.array([scaler["hourly"][c][1] for c in HOURLY_FEATS], np.float32)
    by_stay = {sid: df for sid, df in
               hourly_all[hourly_all["h"] < T].groupby("stay_id", sort=False)}
    hourly = np.zeros((n, T, len(HOURLY_FEATS)), np.float32)
    hmask = np.zeros((n, T), np.float32)
    sid_row = {sid: i for i, sid in enumerate(stay_ids)}
    for sid, df in by_stay.items():
        i = sid_row.get(sid)
        if i is None:
            continue  # hourly rows for stays outside the stays table: unused
        hs = df["h"].to_numpy(np.int64)
        vals = df[HOURLY_FEATS].to_numpy(np.float32)
        vals = (vals - mean) / std
        vals = np.where(np.isfinite(vals), vals, 0.0)
        hourly[i, hs] = vals
        hmask[i, hs] = 1.0

    # ---- text (768) ---------------------------------------------------------
    try:
        emb = load_text_emb(cache_dir)
        emb_cols = [c for c in emb.columns if c.startswith("emb_")]
        mat = (emb.set_index("stay_id")[emb_cols]
               .reindex(stay_ids).to_numpy(np.float32))
        mat = np.where(np.isfinite(mat), mat, 0.0)
    except Exception as e:
        warnings.warn(f"text_emb unavailable ({type(e).__name__}); using zeros. "
                      "Run wjem_ed_audit.data.text_embed first.")
        mat = np.zeros((n, EMB_DIM), np.float32)

    # ---- tab_C: 7 triage vitals, standardized ------------------------------
    cstats = {c: (scaler["hourly"][c] if c in scaler["hourly"]
                  else scaler["pain"]) for c in TRIAGE_VITALS}
    cm = np.array([cstats[c][0] for c in TRIAGE_VITALS], np.float32)
    cs = np.array([cstats[c][1] for c in TRIAGE_VITALS], np.float32)
    tab_c = stays[TRIAGE_VITALS].to_numpy(np.float32)
    tab_c = (tab_c - cm) / cs
    tab_c = np.where(np.isfinite(tab_c), tab_c, 0.0)

    # ---- tab_J + mask_J ------------------------------------------------------
    acq = stays["acuity"].to_numpy(np.float32)
    mask_j = np.isfinite(acq).astype(np.float32)[:, None]
    tab_j = np.where(np.isfinite(acq), acq, 0.0)[:, None]

    # ---- tab_P / tab_D (style aligned with api.build_flat_matrix) ------------
    intime = pd.to_datetime(stays["intime"])
    p_num = np.stack([
        intime.dt.hour.to_numpy(np.float32),
        intime.dt.dayofweek.to_numpy(np.float32),
        stays["n_ed_visits_365d"].to_numpy(np.float32),
    ], axis=1)
    p_cat = _one_hot(stays["arrival_transport"], tab_enc["arrival_transport"])
    tab_p = np.concatenate([p_num, p_cat], axis=1)

    d_num = stays[list(_D_NUMERIC)].to_numpy(np.float32)
    d_num = np.where(np.isfinite(d_num), d_num, 0.0)
    d_cat = np.concatenate(
        [_one_hot(stays[c], tab_enc[c])
         for c in ("gender", "race", "insurance", "language")], axis=1)
    tab_d = np.concatenate([d_num, d_cat], axis=1)

    return {"stay_ids": stay_ids, "stays_row_pos": stays_row_pos,
            "hourly": hourly, "hourly_mask": hmask, "text": mat,
            "tab_C": tab_c, "tab_J": tab_j, "mask_J": mask_j,
            "tab_P": tab_p, "tab_D": tab_d}


# -------------------------------------------------------------------- dataset
class TLQTDataset(Dataset):
    """Yields SPEC §6 sample dicts; see module docstring for the masking design."""

    def __init__(self, cache_dir: str | Path, task: str, split_col: str,
                 split_val: str, groups=("C", "J", "P", "D"),
                 horizon_h: int = 6, cfg: dict | None = None):
        self.cache_dir = Path(cache_dir)
        self.task = task
        self.groups = frozenset(g.upper() for g in groups)
        self.horizon_h = int(horizon_h or (cfg or {}).get("tlqt", {}).get("horizon_h", 6))

        stays_all = load_stays(self.cache_dir)
        hourly_all = load_hourly(self.cache_dir)
        self.scaler = _fit_or_load_scaler(self.cache_dir, stays_all, hourly_all)
        self.tab_enc = _fit_or_load_tab_encoder(self.cache_dir, stays_all)

        self.stays = (stays_all[stays_all[split_col] == split_val]
                      .sort_values("stay_id").reset_index(drop=True))
        self.stay_ids = self.stays["stay_id"].to_numpy(np.int64)

        # ---- full-cohort tensors (split/task independent; disk-cached) ---------
        mtimes = _src_mtimes(self.cache_dir)
        cache_path = _tensor_cache_path(self.cache_dir, self.horizon_h)
        arrays = _load_tensor_cache(cache_path, mtimes)
        if arrays is None:
            arrays = _build_tensor_cache(self.cache_dir, stays_all, hourly_all,
                                         self.scaler, self.tab_enc, self.horizon_h)
            _save_tensor_cache(cache_path, arrays, mtimes)

        # self.stay_ids is a sorted subset of the sorted full-cohort ids
        idx = np.searchsorted(arrays["stay_ids"], self.stay_ids)
        if not np.array_equal(arrays["stay_ids"][idx], self.stay_ids):
            raise RuntimeError(f"tensor cache {cache_path} out of sync with stays")
        self.hourly = torch.from_numpy(arrays["hourly"][idx])
        self.hourly_mask = torch.from_numpy(arrays["hourly_mask"][idx])
        self.text = torch.from_numpy(arrays["text"][idx])
        self.tab_C = torch.from_numpy(arrays["tab_C"][idx])
        self.tab_J = torch.from_numpy(arrays["tab_J"][idx])
        self.mask_J = torch.from_numpy(arrays["mask_J"][idx])
        self.tab_P = torch.from_numpy(arrays["tab_P"][idx])
        self.tab_D = torch.from_numpy(arrays["tab_D"][idx])

        # ---- labels (per task; never cached) ------------------------------------
        ycol = task_label_col(task)
        y = self.stays[ycol].to_numpy(np.float32)
        self.labels = torch.from_numpy(np.log1p(y) if task == "edlos" else y)

        # group inclusion flags
        self._g = {g: float(g in self.groups) for g in ("C", "J", "P", "D")}

    def __len__(self) -> int:
        return len(self.stays)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        gC, gJ, gP, gD = (self._g["C"], self._g["J"], self._g["P"], self._g["D"])
        return {
            "hourly": self.hourly[i] * gC,
            "hourly_mask": self.hourly_mask[i] * gC,
            "text": self.text[i] * gC,
            "tab_C": self.tab_C[i] * gC,
            "tab_J": self.tab_J[i] * gJ,
            "mask_J": self.mask_J[i] * gJ,
            "tab_P": self.tab_P[i] * gP,
            "tab_D": self.tab_D[i] * gD,
            "label": self.labels[i],
            "stay_id": torch.tensor(self.stay_ids[i], dtype=torch.long),
        }


def make_tlqt_dataset(cache_dir, task, split_col, split_val,
                      groups=("C", "J", "P", "D"), horizon_h=6, cfg=None):
    """SPEC §6 factory: torch Dataset for one (task, split) of the TLQ-T model."""
    return TLQTDataset(cache_dir, task, split_col, split_val,
                       groups=groups, horizon_h=horizon_h, cfg=cfg)
