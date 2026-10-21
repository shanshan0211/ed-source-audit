"""CLI: build the stays/hourly parquet cache.

Usage:
    PYTHONPATH=src python -m wjem_ed_audit.data.build_cache --config configs/smoke.yaml

source=synthetic -> synth.generate; source=mimic_dir (needs DUA) -> cohort.build.
Writes {cache_dir}/stays.parquet and {cache_dir}/hourly.parquet, then prints
row counts and label rates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .api import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default=None,
                    help="override config.data.source (synthetic | mimic_dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    source = args.source or data_cfg.get("source", "synthetic")

    if source == "synthetic":
        from . import synth
        stays, hourly = synth.generate(cfg)
    elif source in ("mimic", "mimic_dir"):
        from . import cohort
        stays, hourly = cohort.build(cfg)
    else:
        raise ValueError(f"unknown data.source {source!r}")

    cache_dir = Path(data_cfg.get("cache_dir", "data/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    stays.to_parquet(cache_dir / "stays.parquet", index=False)
    hourly.to_parquet(cache_dir / "hourly.parquet", index=False)

    print(f"[build_cache] source={source} cache_dir={cache_dir}")
    print(f"[build_cache] stays rows={len(stays)}  hourly rows={len(hourly)}")
    for col in ("y_critical", "y_hosp", "y_revisit72"):
        rate = stays[col].mean()
        print(f"[build_cache] {col} rate = {rate:.4f}")
    print(f"[build_cache] edlos_hours mean = {stays['edlos_hours'].mean():.2f} h")
    print("[build_cache] split_random:",
          stays["split_random"].value_counts().to_dict())
    print("[build_cache] split_temporal:",
          stays["split_temporal"].value_counts().to_dict())


if __name__ == "__main__":
    main()
