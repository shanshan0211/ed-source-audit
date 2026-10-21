"""E1 — branch masking (SPEC §5).

For every task x combo (config audit.e1_combos) x seed, ensure a training run
exists for model in {lgbm, tlqt}, then summarise ΔAUC vs the full combo.

Training integration points:
  * tlqt -> subprocess: {python} -m wjem_ed_audit.models.run_tlqt
            --config <cfg> --task <t> --groups <C,J,P,D> --seed <k>
  * lgbm -> in-process: wjem_ed_audit.models.trainer.train_baseline(
            cfg, task, "lgbm", groups, seed, cache_dir, results_root)
Existing run dirs (with metrics.json) are skipped. All model-side imports are
lazy so this module remains importable without optional model dependencies.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as astats
from .runs_index import scan_runs

E1_MODELS = ("lgbm", "tlqt")


# ---------------------------------------------------------------- job spec
def cls_tasks(cfg: dict) -> list[str]:
    """Classification tasks only (ΔAUC undefined for the edlos regression).

    audit.e1_tasks (optional) restricts the TLQ-T branch-mask grid to keep
    compute bounded (e.g. ["critical", "sepsis24h"]); lgbm runs all tasks."""
    tasks = [t for t in cfg.get("tasks", []) if t != "edlos"]
    e1t = (cfg.get("audit") or {}).get("e1_tasks")
    if e1t:
        tasks = [t for t in tasks if t in set(e1t)]
    return tasks


def build_e1_jobs(cfg: dict) -> list[dict]:
    """Pure job constructor: task x model x combo x seed."""
    combos = cfg["audit"]["e1_combos"]
    jobs = []
    for task in cls_tasks(cfg):
        for model in E1_MODELS:
            for combo, groups in combos.items():
                for seed in cfg.get("seeds", [0]):
                    jobs.append({"task": task, "model": model, "combo": combo,
                                 "groups": list(groups), "seed": int(seed)})
    return jobs


def job_run_dir(results_root: str | Path, job: dict) -> Path:
    return (Path(results_root) / "runs" / job["task"] / job["model"]
            / job["combo"] / f"seed{job['seed']}")


def resolve_python(project_root: str | Path) -> str:
    cand = Path(project_root).parent / ".venv" / "bin" / "python"
    return str(cand) if cand.is_file() else sys.executable


def build_subprocess_cmd(job: dict, python_exe: str,
                         config_path: str | Path) -> list[str]:
    """CLI argv for one TLQ-T training job (run_tlqt entry point).

    lgbm jobs go in-process via models.trainer.train_baseline (its CLI,
    run_baselines, only covers the full combo, so there is no subprocess
    route for per-combo baseline jobs).
    """
    if job["model"] != "tlqt":
        raise ValueError(f"no subprocess route for model {job['model']!r}; "
                         "baselines are trained in-process")
    groups = ",".join(job["groups"])
    return [python_exe, "-m", "wjem_ed_audit.models.run_tlqt",
            "--config", str(config_path), "--task", job["task"],
            "--groups", groups, "--seed", str(job["seed"])]


# ------------------------------------------------------------- ensure runs
def _train_lgbm_in_process(job: dict, cfg: dict, cache_dir, results_root) -> None:
    """Lazy in-process call into models.trainer (import-safe by design)."""
    from wjem_ed_audit.models.trainer import train_baseline  # lazy import
    train_baseline(cfg, job["task"], "lgbm", job["groups"], job["seed"],
                   cache_dir, results_root)


def ensure_runs(jobs: list[dict], results_root, cfg: dict, config_path,
                project_root, timeout: int | None = None) -> list[dict]:
    """Trigger missing runs; skip existing. Returns [{**job, "status", "path"}]."""
    results_root = Path(results_root)
    cache_dir = cfg.get("data", {}).get("cache_dir")
    python_exe = resolve_python(project_root)
    env = None
    src_dir = Path(project_root) / "src"
    if src_dir.is_dir():
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src_dir) + (":" + env["PYTHONPATH"]
                                            if env.get("PYTHONPATH") else "")
    out = []
    for job in jobs:
        run_dir = job_run_dir(results_root, job)
        rec = {**job, "path": str(run_dir)}
        if (run_dir / "metrics.json").is_file():
            rec["status"] = "exists"
            out.append(rec)
            continue
        try:
            if job["model"] == "lgbm":
                _train_lgbm_in_process(job, cfg, cache_dir, results_root)
                rec["status"] = "trained_in_process"
            else:
                cmd = build_subprocess_cmd(job, python_exe, config_path)
                subprocess.run(cmd, check=True, cwd=project_root, env=env,
                               timeout=timeout)
                rec["status"] = "trained_subprocess"
        except Exception as exc:  # Missing optional dependency or training failure
            rec["status"] = f"failed:{type(exc).__name__}: {exc}"
        out.append(rec)
    return out


# ---------------------------------------------------------------- collect
def collect_e1(results_root, cfg: dict) -> pd.DataFrame:
    """Build the E1 table from existing runs: per-seed ΔAUC vs full + summary.

    Columns: task, model, combo, seed, auroc, delta_vs_full, mean, lo, hi, p.
    Detail rows carry seed/auroc/delta; summary rows (seed="ALL") carry the
    cross-seed mean ΔAUC with t-based 95% CI, one-sample t-test p (SPEC §5)
    and BH-FDR adjusted p (family: non-full combos within each task x model).
    """
    idx = scan_runs(results_root)
    cols = ["task", "model", "combo", "seed", "auroc", "delta_vs_full",
            "mean", "lo", "hi", "p", "p_adj"]
    if idx.empty:
        return pd.DataFrame(columns=cols)
    idx = idx[~idx["temporal"]].copy()
    full = idx[idx["config_tag"] == "full"][
        ["task", "model", "seed", "auroc"]].rename(columns={"auroc": "auroc_full"})
    m = idx.merge(full, on=["task", "model", "seed"], how="left")
    m["delta_vs_full"] = m["auroc"] - m["auroc_full"]
    tasks = cls_tasks(cfg)
    combos = list(cfg["audit"]["e1_combos"].keys())
    rows: list[dict] = []
    for task in tasks:
        for model in E1_MODELS:
            sub = m[(m["task"] == task) & (m["model"] == model)]
            for combo in combos:
                sc = sub[sub["config_tag"] == combo]
                for _, r in sc.sort_values("seed").iterrows():
                    rows.append({"task": task, "model": model, "combo": combo,
                                 "seed": int(r["seed"]), "auroc": r["auroc"],
                                 "delta_vs_full": r["delta_vs_full"],
                                 "mean": np.nan, "lo": np.nan, "hi": np.nan,
                                 "p": np.nan, "p_adj": np.nan})
                deltas = sc["delta_vs_full"].dropna().to_numpy()
                aucs = sc["auroc"].dropna().to_numpy()
                if deltas.size:
                    agg = astats.aggregate_seeds(sc.assign(v=sc["delta_vs_full"]), "v")
                    _, p = astats.ttest_one_sample(deltas)
                    rows.append({"task": task, "model": model, "combo": combo,
                                 "seed": "ALL",
                                 "auroc": float(aucs.mean()) if aucs.size else np.nan,
                                 "delta_vs_full": agg["mean"], "mean": agg["mean"],
                                 "lo": agg["lo"], "hi": agg["hi"], "p": p,
                                 "p_adj": np.nan})
    df = pd.DataFrame(rows, columns=cols)
    # BH-FDR within each (task, model) family of non-full combo delta tests
    # (combo "full" has delta==0 by construction and is excluded).
    is_summ = (df["seed"] == "ALL") & (df["combo"] != "full")
    for _, gidx in df[is_summ].groupby(["task", "model"]).groups.items():
        df.loc[gidx, "p_adj"] = astats.bh_fdr(df.loc[gidx, "p"].to_numpy())
    return df


def run_e1(cfg: dict, results_root, config_path=None, project_root=None,
           out_dir=None, trigger_training: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """E1 driver: ensure runs (optional) -> collect -> write CSV. Returns (df, job_recs)."""
    results_root = Path(results_root)
    out_dir = Path(out_dir) if out_dir else results_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_e1_jobs(cfg)
    recs: list[dict] = []
    if trigger_training and config_path is not None and project_root is not None:
        recs = ensure_runs(jobs, results_root, cfg, config_path, project_root)
    else:
        recs = [{**j, "path": str(job_run_dir(results_root, j)),
                 "status": "exists" if (job_run_dir(results_root, j) / "metrics.json").is_file()
                          else "missing"} for j in jobs]
    df = collect_e1(results_root, cfg)
    df.to_csv(out_dir / "E1_branch_mask.csv", index=False)
    with open(out_dir / "E1_job_status.json", "w") as f:
        json.dump(recs, f, indent=2, default=str)
    return df, recs
