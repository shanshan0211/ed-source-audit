"""Portable historical-pipeline runner; not a triage-time validation protocol."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_plan(config_path: Path, allow_restricted_data: bool = False):
    config_path = config_path.resolve()
    if config_path.parent != ROOT / "configs":
        raise ValueError("Keep the configuration in this repository's configs/ directory.")
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Expected a YAML configuration mapping.")
    source = cfg.get("data", {}).get("source")
    if source not in ("synthetic", "mimic_dir", "mimic"):
        raise ValueError("An explicit synthetic or mimic_dir data source is required.")
    if source != "synthetic" and not allow_restricted_data:
        raise ValueError("Restricted-data execution needs --allow-restricted-data; "
                         "this flag does not grant data-use or ethics approval.")
    seeds = cfg.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds):
        raise ValueError("Configuration must contain a nonempty integer seeds list.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct.")
    paths = []
    for raw in (cfg.get("data", {}).get("cache_dir"), cfg.get("results_root")):
        if not raw:
            raise ValueError("Explicit cache_dir and results_root are required.")
        path = Path(raw)
        paths.append((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    if (paths[0] == paths[1] or paths[0] in paths[1].parents
            or paths[1] in paths[0].parents):
        raise ValueError("Cache and results must be separate, nonnested directories.")
    for path in paths:
        if path == ROOT or path in ROOT.parents:
            raise ValueError("Cache/output must not be a repository or ancestor directory.")
        if path.exists():
            raise FileExistsError(f"Refusing an existing cache/output directory: {path}. "
                                  "Choose fresh paths in a configs/*.local.yaml file.")
    prefix = [sys.executable, "-m"]
    suffix = ["--config", str(config_path)]
    modules = ["data.build_cache", "data.text_embed", "models.run_baselines"]
    commands = [prefix + ["wjem_ed_audit." + name] + suffix for name in modules]
    commands += [prefix + ["wjem_ed_audit.models.run_tlqt"] + suffix + ["--seed", str(seed)]
                 for seed in seeds]
    commands += [prefix + ["wjem_ed_audit." + name] + suffix
                 for name in ("audit.run_audit", "eval.report")]
    return cfg, commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ci.yaml")
    parser.add_argument("--allow-restricted-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg, commands = build_plan(args.config, args.allow_restricted_data)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Preflight failed: {exc}\n")
    print("Historical pipeline; outputs are not the revised manuscript.", flush=True)
    print(f"Source: {cfg['data']['source']}; dry-run: {args.dry_run}", flush=True)
    if cfg["data"]["source"] == "synthetic":
        print("Synthetic outputs have no clinical meaning.", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for number, command in enumerate(commands, 1):
        print(f"[{number}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
