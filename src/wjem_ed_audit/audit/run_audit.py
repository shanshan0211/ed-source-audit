"""Audit CLI: run E1->E2->E4->E3->E5 then the SDS/CSR summary (SPEC §5).

Usage (from the project root, per SPEC):
    python -m wjem_ed_audit.audit.run_audit \
        --config configs/smoke.yaml [--only E1,E2]

Paths in the config (results_root, data.cache_dir) are resolved relative to
the project root, inferred as the config file's parent directory's parent.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import api
from . import ablate, masks, permute, repr_interv, sds, transfer
from .runs_index import load_preds, scan_runs

STEPS = ("E1", "E2", "E3", "E4", "E5")
EXEC_ORDER = ("E1", "E2", "E4", "E3", "E5")  # SPEC-mandated execution order


def _resolve(project_root: Path, p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else project_root / p


def _banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def _build_bootstrap_cache(idx: pd.DataFrame) -> dict:
    """Preserve stay identifiers in seed-sorted prediction frames for pairing."""
    cache: dict = {}
    if idx.empty:
        return cache
    sub = idx[~idx["temporal"]]
    for _, r in sub.sort_values(["task", "model", "config_tag", "seed"]).iterrows():
        preds = load_preds(r["path"])
        if preds.empty or not {"y_true", "y_score"} <= set(preds.columns):
            continue
        key = (r["task"], r["model"], r["config_tag"])
        if "stay_id" not in preds:
            raise ValueError("SDS bootstrap requires stay_id in stored predictions")
        cache.setdefault(key, []).append(preds[["stay_id", "y_true", "y_score"]])
    return cache


def run_audit(config_path: str, only: set[str] | None = None) -> dict:
    config_path = str(config_path)
    cfg = api.load_config(config_path)
    project_root = Path(config_path).resolve().parent.parent
    results_root = _resolve(project_root, cfg.get("results_root", "results"))
    cache_dir = _resolve(project_root, cfg.get("data", {}).get("cache_dir",
                                                               "data/cache"))
    # make the resolved path visible to in-process training triggers (E1 lgbm)
    cfg.setdefault("data", {})["cache_dir"] = str(cache_dir)
    audit_dir = results_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    n_boot = int(cfg.get("audit", {}).get("bootstrap", 500))
    todo = [s for s in EXEC_ORDER if only is None or s in only]
    summary: dict[str, str] = {}

    for step in todo:
        t0 = time.time()
        if step == "E1":
            _banner("E1 branch mask")
            df, recs = masks.run_e1(cfg, results_root, config_path=config_path,
                                    project_root=project_root, out_dir=audit_dir)
            n_fail = sum(1 for r in recs if str(r["status"]).startswith("failed"))
            n_exist = sum(1 for r in recs if r["status"] == "exists")
            print(f"E1: jobs={len(recs)} existing={n_exist} failed={n_fail}; "
                  f"table rows={len(df)}")
            summ = df[df["seed"] == "ALL"]
            if not summ.empty:
                show = [c for c in ["task", "model", "combo", "delta_vs_full",
                                    "lo", "hi", "p", "p_adj"] if c in summ.columns]
                print(summ[show].to_string(index=False))
            summary["E1"] = f"rows={len(df)} jobs_failed={n_fail}"
        elif step == "E2":
            _banner("E2 field ablation (lgbm, flat)")
            df = ablate.run_e2(cfg, cache_dir=cache_dir, out_dir=audit_dir)
            summ = df[df["seed"] == "ALL"]
            show = [c for c in ["task", "field", "delta_field", "delta_lo",
                                "delta_hi", "p", "p_adj"] if c in summ.columns]
            print(summ[show].to_string(index=False))
            summary["E2"] = f"rows={len(df)}"
        elif step == "E4":
            _banner("E4 group permutation")
            df = permute.run_e4(cfg, results_root, cache_dir=cache_dir,
                                out_dir=audit_dir)
            ok = df[df["status"] == "ok"]
            print(f"E4: rows={len(df)} ok={len(ok)} "
                  f"skipped/failed={len(df) - len(ok)}")
            summary["E4"] = f"rows={len(df)} ok={len(ok)}"
        elif step == "E3":
            _banner("E3 representation intervention (TLQ-T)")
            df = repr_interv.run_e3(cfg, results_root, cache_dir=cache_dir,
                                    out_dir=audit_dir)
            ok = df[df["status"] == "ok"]
            print(f"E3: rows={len(df)} ok={len(ok)} skipped={len(df) - len(ok)}")
            summary["E3"] = f"rows={len(df)} ok={len(ok)}"
        elif step == "E5":
            _banner("E5 temporal transfer")
            df = transfer.run_e5(cfg, results_root, out_dir=audit_dir,
                                 n_boot=n_boot)
            dec = df[df["kind"] == "decay_summary"]
            if not dec.empty:
                print(dec[["task", "model", "combo", "decay", "lo", "hi", "p"]]
                      .to_string(index=False))
            print("E5: eICU row status=not_available_pending_dua appended")
            summary["E5"] = f"rows={len(df)}"
        print(f"[{step}] {time.time() - t0:.1f}s -> {audit_dir}", flush=True)

    # ---------------------------------------------------------- SDS / CSR
    _banner("SDS/CSR summary")
    idx = scan_runs(results_root)
    cache = _build_bootstrap_cache(idx)
    tbl = sds.sds_table(idx, bootstrap_cache=cache, n_boot=n_boot)
    tbl.to_csv(audit_dir / "sds_csr.csv", index=False)
    if not tbl.empty:
        core = tbl[tbl["metric"].isin(["SDS_mask", "SDS_solo", "CSR", "R"])]
        print(core.to_string(index=False))
    else:
        print("sds_csr: no complete full/clin_only/meta_only triples found; "
              "wrote empty table.")
    summary["SDS"] = f"rows={len(tbl)}"
    print("\n=== audit summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WJEM ED-Audit: E1-E5 + SDS/CSR")
    ap.add_argument("--config", required=True, help="path to yaml config")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of E1,E2,E3,E4,E5")
    args = ap.parse_args(argv)
    only = None
    if args.only:
        only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
        bad = only - set(STEPS)
        if bad:
            ap.error(f"unknown step(s): {sorted(bad)}; choose from {STEPS}")
    run_audit(args.config, only=only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
