"""Scan results/runs/ into an index DataFrame (SPEC §4).

Layout: {results_root}/runs/{task}/{model}/{config_tag}/seed{k}[/temporal]/metrics.json
Directories lacking metrics.json are skipped (tolerant by contract).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def scan_runs(results_root: str | Path) -> pd.DataFrame:
    """Return DataFrame with columns:
    task, model, config_tag, seed(int), temporal(bool), auroc(float), path(str).
    `path` points at the run directory (the temporal subdir when temporal=True).
    """
    runs_dir = Path(results_root) / "runs"
    rows: list[dict] = []
    if runs_dir.is_dir():
        for metrics_path in sorted(runs_dir.rglob("metrics.json")):
            rel = metrics_path.relative_to(runs_dir)
            parts = rel.parts
            temporal = False
            if len(parts) == 5:
                task, model, config_tag, seed_tok, _ = parts
            elif len(parts) == 6 and parts[4] == "temporal":
                task, model, config_tag, seed_tok = parts[:4]
                temporal = True
            else:
                continue  # unexpected depth -> skip
            if not seed_tok.startswith("seed"):
                continue
            try:
                seed = int(seed_tok[len("seed"):])
            except ValueError:
                continue
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            rows.append({
                "task": task,
                "model": model,
                "config_tag": config_tag,
                "seed": seed,
                "temporal": temporal,
                "auroc": float(metrics.get("auroc", np.nan)),
                "path": str(metrics_path.parent),
            })
    cols = ["task", "model", "config_tag", "seed", "temporal", "auroc", "path"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["task", "model", "config_tag", "seed", "temporal"],
                            ignore_index=True)
    return df


def load_preds(run_path: str | Path) -> pd.DataFrame:
    """Load preds.parquet of a run dir; returns empty DataFrame if absent."""
    p = Path(run_path) / "preds.parquet"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_parquet(p)
