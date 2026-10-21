"""Shared historical data API for WJEM ED-Audit (see SPEC.md).

Private extraction outputs must match the schemas consumed here. The taxonomy
describes implementation groups, not guaranteed triage-time availability.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

TASKS = ["critical", "hosp", "revisit72", "edlos", "death", "sepsis24h"]
_LABEL_COL = {
    "critical": "y_critical",
    "hosp": "y_hosp",
    "revisit72": "y_revisit72",
    "edlos": "edlos_hours",
    "death": "y_death",
    "sepsis24h": "y_sepsis24h",
}

TRIAGE_VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
HOURLY_FEATS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature",
                "lactate", "wbc", "creatinine", "glucose"]

TAXONOMY: dict[str, list[str]] = {
    "C": TRIAGE_VITALS + ["hourly", "chiefcomplaint"],
    "J": ["acuity"],
    "P": ["arrival_transport", "hour_of_day", "day_of_week", "n_ed_visits_365d"],
    "D": ["anchor_age", "gender", "race", "insurance", "language"],
}


def task_label_col(task: str) -> str:
    if task not in _LABEL_COL:
        raise ValueError(f"unknown task {task}; choose from {TASKS}")
    return _LABEL_COL[task]


def load_config(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- cache I/O
def load_stays(cache_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(cache_dir or CACHE_DIR) / "stays.parquet")


def load_hourly(cache_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(cache_dir or CACHE_DIR) / "hourly.parquet")


def load_text_emb(cache_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(cache_dir or CACHE_DIR) / "text_emb.parquet")


# ------------------------------------------------------------- flat features
def _one_hot(s: pd.Series, prefix: str) -> pd.DataFrame:
    d = pd.get_dummies(s.fillna("UNKNOWN"), prefix=prefix).astype(np.float32)
    return d


def build_flat_matrix(stays: pd.DataFrame,
                      groups: tuple[str, ...] = ("C", "J", "P", "D"),
                      include_text: bool = False,
                      text_emb: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flat per-stay feature matrix with group-prefixed column names."""
    parts: list[pd.DataFrame] = []
    idx = stays.index
    if "C" in groups:
        parts.append(stays[TRIAGE_VITALS].astype(np.float32).add_prefix("C__"))
    if "J" in groups:
        j = pd.DataFrame(index=idx)
        j["J__acuity"] = stays["acuity"].astype(np.float32)
        j["J__acuity_missing"] = stays["acuity"].isna().astype(np.float32)
        parts.append(j)
    if "P" in groups:
        p = pd.DataFrame(index=idx)
        intime = pd.to_datetime(stays["intime"])
        p["P__hour_of_day"] = intime.dt.hour.astype(np.float32)
        p["P__day_of_week"] = intime.dt.dayofweek.astype(np.float32)
        p["P__n_ed_visits_365d"] = stays["n_ed_visits_365d"].astype(np.float32)
        parts += [p, _one_hot(stays["arrival_transport"], "P__at")]
    if "D" in groups:
        d = pd.DataFrame(index=idx)
        d["D__anchor_age"] = stays["anchor_age"].astype(np.float32)
        parts += [d, _one_hot(stays["gender"], "D__gender"),
                  _one_hot(stays["race"], "D__race"),
                  _one_hot(stays["insurance"], "D__ins"),
                  _one_hot(stays["language"], "D__lang")]
    if include_text and "C" in groups and text_emb is not None:
        emb_cols = [c for c in text_emb.columns if c.startswith("emb_")]
        te = text_emb.set_index("stay_id")[emb_cols].reindex(stays["stay_id"].values)
        te.index = idx
        parts.append(te.astype(np.float32).add_prefix("C__txt_"))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=idx)
    return X


def load_flat(task: str, split_col: str, split_val: str,
              cache_dir: str | os.PathLike | None = None,
              groups: tuple[str, ...] = ("C", "J", "P", "D"),
              include_text: bool = False):
    """Returns (X DataFrame, y ndarray, ids ndarray) for the requested split."""
    stays = load_stays(cache_dir)
    mask = stays[split_col] == split_val
    sub = stays.loc[mask].reset_index(drop=True)
    text_emb = None
    if include_text:
        try:
            text_emb = load_text_emb(cache_dir)
        except Exception:
            text_emb = None
    X = build_flat_matrix(sub, groups=groups, include_text=include_text, text_emb=text_emb)
    y = sub[task_label_col(task)].to_numpy()
    return X, y, sub["stay_id"].to_numpy()


# ---------------------------------------------------------------- tiny cache
def make_tiny(cache_dir: str | os.PathLike, n: int = 300, seed: int = 0) -> Path:
    """Fabricate a minimal schema-valid cache for unit tests (not for results)."""
    rng = np.random.default_rng(seed)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    intime = pd.to_datetime("2013-01-01") + pd.to_timedelta(
        rng.integers(0, 6 * 365 * 24, n), unit="h")
    base = rng.normal(0, 1, n)
    acuity = np.clip(np.round(3 - base + rng.normal(0, 0.8, n)), 1, 5)
    acuity[rng.random(n) < 0.2] = np.nan
    y = (base + rng.normal(0, 1, n) > 0.8).astype(int)
    stays = pd.DataFrame({
        "stay_id": np.arange(n), "subject_id": np.arange(n),
        "intime": intime, "outtime": intime + pd.to_timedelta(rng.uniform(1, 30, n), unit="h"),
        "temperature": 37 + rng.normal(0, 0.6, n) + 0.3 * base,
        "heartrate": 85 + 10 * base + rng.normal(0, 10, n),
        "resprate": 17 + 2 * base + rng.normal(0, 3, n),
        "o2sat": 97 - 1.5 * base + rng.normal(0, 1.5, n),
        "sbp": 125 - 8 * base + rng.normal(0, 15, n),
        "dbp": 75 - 4 * base + rng.normal(0, 10, n),
        "pain": np.clip(rng.normal(4, 3, n), 0, 10),
        "acuity": acuity,
        "chiefcomplaint": rng.choice(["chest pain", "fever", "fall", "sob", "abd pain"], n),
        "arrival_transport": rng.choice(["WALK IN", "AMBULANCE", "OTHER"], n, p=[0.6, 0.3, 0.1]),
        "race": rng.choice(["WHITE", "BLACK", "ASIAN", "OTHER"], n),
        "gender": rng.choice(["M", "F"], n),
        "insurance": rng.choice(["Medicare", "Medicaid", "Private", "Other"], n),
        "language": rng.choice(["English", "Spanish", "Other"], n, p=[0.8, 0.1, 0.1]),
        "anchor_age": np.clip(rng.normal(55, 20, n), 18, 100),
        "n_ed_visits_365d": rng.poisson(0.8, n),
        "y_critical": y, "y_hosp": (rng.random(n) < 0.45).astype(int),
        "y_revisit72": (rng.random(n) < 0.1).astype(int),
        "y_death": ((base + rng.normal(0, 1, n) > 1.6)).astype(int),
        "y_sepsis24h": ((0.7 * base + rng.normal(0, 1, n)) > 1.0).astype(int),
        "edlos_hours": np.clip(rng.gamma(3, 3, n), 0.5, 72),
        "split_random": rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2]),
        "split_temporal": np.where(intime < pd.Timestamp("2017-01-01"), "dev", "test"),
    })
    rows = []
    for sid, hr0 in zip(stays["stay_id"], stays["heartrate"]):
        for h in range(6):
            rows.append((sid, h, hr0 + rng.normal(0, 3), 17 + rng.normal(0, 2),
                         97 + rng.normal(0, 1), 125 + rng.normal(0, 10), 75 + rng.normal(0, 8),
                         37 + rng.normal(0, 0.3), np.nan, np.nan, np.nan, np.nan))
    hourly = pd.DataFrame(rows, columns=["stay_id", "h"] + HOURLY_FEATS)
    text_emb = pd.DataFrame(
        np.hstack([stays[["stay_id"]].to_numpy(),
                   rng.normal(0, 1, (n, 768)).astype(np.float32)]),
        columns=["stay_id"] + [f"emb_{i}" for i in range(768)])
    text_emb["stay_id"] = text_emb["stay_id"].astype(int)
    stays.to_parquet(cache_dir / "stays.parquet", index=False)
    hourly.to_parquet(cache_dir / "hourly.parquet", index=False)
    text_emb.to_parquet(cache_dir / "text_emb.parquet", index=False)
    return cache_dir


# ------------------------------------------------------------------ run I/O
def write_run(run_dir: str | os.PathLike, metrics: dict, preds: pd.DataFrame,
              config: dict, model_state=None) -> None:
    """Persist a run per SPEC §4. `model_state` = optional torch state_dict."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    preds.to_parquet(run_dir / "preds.parquet", index=False)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    if model_state is not None:
        import torch
        torch.save(model_state, run_dir / "model.pt")


def run_dir_for(results_root: str | os.PathLike, task: str, model: str,
                config_tag: str, seed: int, temporal: bool = False) -> Path:
    p = Path(results_root) / "runs" / task / model / config_tag / f"seed{seed}"
    return p / "temporal" if temporal else p
