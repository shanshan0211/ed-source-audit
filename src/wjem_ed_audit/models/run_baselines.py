"""CLI: run flat baselines + clinical score models (full group combo only).

Audit combos (noJ / clin_only / ...) are trained on demand by the audit runner
via ``trainer.train_baseline``; this entry point only covers the ``full``
combination for every task in the config (or ``--task``).

Clinical scores (news2/qsofa) need no training: the score is computed from the
stays table triage vitals and written as a run with config_tag ``full`` (the
same deterministic metrics are written for every requested seed).

Usage:
    PYTHONPATH=src python -m wjem_ed_audit.models.run_baselines \
        --config configs/smoke.yaml [--task all|critical|...] [--seeds 0,1]
"""
from __future__ import annotations

import argparse

from wjem_ed_audit.data import api
from wjem_ed_audit.models.baselines import CLINICAL_SCORES, is_classification
from wjem_ed_audit.models.trainer import (_eval_scores, _preds_frame,
                                          config_tag_for, train_baseline)

FULL = ("C", "J", "P", "D")


def run_clinical_scores(cfg: dict, task: str, seed: int, cache_dir,
                        results_root) -> None:
    """Evaluate NEWS2 / qSOFA directly as y_score (classification tasks only)."""
    if not is_classification(task):
        return
    stays = api.load_stays(cache_dir)
    y_col = api.task_label_col(task)
    for name, fn in CLINICAL_SCORES.items():
        tag = config_tag_for(FULL, cfg)
        for split_col, split_val, temporal in (
                ("split_random", "test", False),
                ("split_temporal", "test", True)):
            sub = stays.loc[stays[split_col] == split_val]
            if len(sub) == 0:
                continue
            y = sub[y_col].to_numpy()
            scores = fn(sub)
            m = _eval_scores(task, y, scores)
            m.update({"n_test": int(len(y)), "split": f"{split_col}={split_val}"})
            api.write_run(api.run_dir_for(results_root, task, name, tag, seed,
                                          temporal=temporal),
                          m, _preds_frame(sub["stay_id"].to_numpy(), split_val,
                                          y, scores, task),
                          {"model": name, "task": task, "groups": list(FULL),
                           "seed": seed, "config_tag": tag,
                           "note": "training-free clinical score; missing "
                                   "vitals scored as normal"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="all")
    ap.add_argument("--seeds", default=None, help="comma list; default config seeds")
    args = ap.parse_args(argv)

    cfg = api.load_config(args.config)
    cache_dir = cfg["data"]["cache_dir"]
    results_root = cfg.get("results_root", "results")
    tasks = cfg["tasks"] if args.task == "all" else [t.strip() for t in args.task.split(",")]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [int(s) for s in cfg["seeds"]])

    for task in tasks:
        for seed in seeds:
            for model_name in cfg["baselines"]["models"]:
                m = train_baseline(cfg, task, model_name, FULL, seed,
                                   cache_dir, results_root)
                print(f"[baselines] {task}/{model_name}/seed{seed}: {m}")
            run_clinical_scores(cfg, task, seed, cache_dir, results_root)
            for name in cfg["baselines"].get("clinical_scores", []):
                if is_classification(task):
                    print(f"[clinical ] {task}/{name}/seed{seed}: done")


if __name__ == "__main__":
    main()
