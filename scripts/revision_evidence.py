"""Recompute exploratory evidence from existing seed-level aggregate audit rows.

Does not use synthetic runs or claim to reconstruct patient-level uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wjem_ed_audit.audit.stats import aggregate_seeds, bh_fdr, ttest_one_sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results_revision/evidence_20261019")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results_real/audit/E3_repr_intervention.csv"
    df = pd.read_csv(source)
    df = df[(df["task"] == "critical") & df["seed"].ne("ALL") & df["status"].eq("ok")]
    delta = df.pivot(index="seed", columns="intervention", values="delta_vs_none")
    rows = []
    for group in ("J", "P"):
        paired = (delta[f"suppress_{group}"] - delta["suppress_random_C"]).dropna()
        summary = aggregate_seeds(pd.DataFrame({"v": paired}), "v")
        _, p = ttest_one_sample(paired.to_numpy())
        rows.append({"comparison": f"suppress_{group}_minus_random_C", "n_seeds": len(paired),
                     "mean_paired_auc_loss_difference": summary["mean"],
                     "seed_t_ci_lo": summary["lo"], "seed_t_ci_hi": summary["hi"], "p": p,
                     "scope": "exploratory historical snapshot; seed uncertainty only; mixed protocol"})
    result = pd.DataFrame(rows)
    result["p_adj_bh_two_comparisons"] = bh_fdr(result["p"].to_numpy())
    result.to_csv(args.output / "E3_paired_seed_control.csv", index=False)
    files = [source, ROOT / "results_real/audit/E1_branch_mask.csv",
             ROOT / "results_real/audit/sds_csr.csv", ROOT / "results_real/tables/table1_performance.csv"]
    manifest = {"analysis": "post-analysis seed-level diagnostic; no patient-level reanalysis",
                "inputs": [{"file": str(p.relative_to(ROOT)),
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
                "real_predictions_available": (ROOT / "results_real/runs").is_dir(),
                "real_cache_available": (ROOT / "data/cache_real/stays.parquet").is_file()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
