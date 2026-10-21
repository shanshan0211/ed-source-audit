"""CLI: train TLQ-T for one (task, groups, seed) combination.

Invoked by the audit runner as a subprocess per input combination.

Usage:
    PYTHONPATH=src python -m wjem_ed_audit.models.run_tlqt \
        --config configs/smoke.yaml --task critical --groups C,J,P,D --seed 0

Defaults: --task all (every task in the config), --groups full (=C,J,P,D).
"""
from __future__ import annotations

import argparse

from wjem_ed_audit.data import api
from wjem_ed_audit.models.trainer import train_tlqt

FULL = ["C", "J", "P", "D"]


def parse_groups(s: str) -> list[str]:
    if s.lower() == "full":
        return list(FULL)
    return [g.strip() for g in s.split(",") if g.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="all")
    ap.add_argument("--groups", default="full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = api.load_config(args.config)
    cache_dir = cfg["data"]["cache_dir"]
    results_root = cfg.get("results_root", "results")
    tasks = cfg["tasks"] if args.task == "all" else [t.strip() for t in args.task.split(",")]
    groups = parse_groups(args.groups)

    from .trainer import config_tag_for
    from ..data.api import run_dir_for
    tag = config_tag_for(groups, cfg)
    for task in tasks:
        rd = run_dir_for(results_root, task, "tlqt", tag, args.seed)
        if (rd / "metrics.json").exists():
            print(f"[tlqt] {task}/{tag}/seed{args.seed}: exists, skip")
            continue
        m = train_tlqt(cfg, task, groups, args.seed, cache_dir, results_root)
        print(f"[tlqt] {task}/{'_'.join(groups)}/seed{args.seed}: {m}")


if __name__ == "__main__":
    main()
