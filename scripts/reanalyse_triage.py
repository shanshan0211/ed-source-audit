"""Patient-disjoint triage-record reanalysis; no hourly/post-arrival features.

Run on the original data machine. Outputs go to a NEW directory and cannot
overwrite the historical bundle. The executable core requires pandas,
pyarrow, scipy, scikit-learn, and pyyaml; XGBoost/LightGBM are optional.
This is a post-analysis amendment, not a preregistration or new clinical result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wjem_ed_audit.data import api

COMBOS = {"full": ("C", "J", "P", "D"), "clin_only": ("C", "D"),
          "meta_only": ("J", "P", "D"), "noJ": ("C", "P", "D"),
          "noP": ("C", "J", "D"), "D_only": ("D",),
          "C_only": ("C",), "J_D": ("J", "D")}


def patient_split(stays: pd.DataFrame, seed: int = 42) -> pd.Series:
    """One patient belongs to exactly one fixed 60/20/20 partition."""
    if stays["subject_id"].isna().any() or stays["stay_id"].duplicated().any():
        raise ValueError("Nonmissing subject_id and unique stay_id are required")
    subjects = np.sort(stays["subject_id"].unique())
    if len(subjects) < 10:
        raise ValueError("At least 10 patients required")
    subjects = np.random.default_rng(seed).permutation(subjects)
    a, b = int(0.6 * len(subjects)), int(0.8 * len(subjects))
    mapping = {s: part for part, ids in
               (("train", subjects[:a]), ("val", subjects[a:b]), ("test", subjects[b:]))
               for s in ids}
    return stays["subject_id"].map(mapping)


def feature_frame(stays: pd.DataFrame, groups, include_text: bool = False):
    """Explicit allowlist: never reads insurance/language, disposition, or hourly data.

    Recorded hour is retained. Shifted day-of-week is excluded because a
    per-patient date offset does not establish the real weekday of attendance.
    Chief complaint uses train-fitted TF-IDF, not the historical embedding cache.
    """
    numeric, categorical = [], []
    if "C" in groups:
        numeric += api.TRIAGE_VITALS
    if "J" in groups:
        numeric += ["acuity"]
    if "P" in groups:
        numeric += ["arrival_hour", "n_ed_visits_365d"]
        categorical += ["arrival_transport"]
    if "D" in groups:
        numeric += ["anchor_age"]
        categorical += ["gender", "race"]
    data = stays.copy()
    if "P" in groups:
        data["arrival_hour"] = pd.to_datetime(data["intime"]).dt.hour
    X = data[numeric + categorical].copy()
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        X[col] = X[col].fillna("UNKNOWN").astype(str)
    if include_text and "C" in groups:
        X["chiefcomplaint"] = data["chiefcomplaint"].fillna("").astype(str)
    return X, numeric, categorical


def make_pipeline(model: str, seed: int, numeric, categorical, with_text: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                    keep_empty_features=True)),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("category", OneHotEncoder(handle_unknown="ignore"), categorical))
    if with_text:
        transforms.append(("complaint", TfidfVectorizer(max_features=2000, min_df=2,
                                                      ngram_range=(1, 2)), "chiefcomplaint"))
    prep = ColumnTransformer(transforms, sparse_threshold=1.0)
    if model == "lr":
        estimator = LogisticRegression(max_iter=3000, random_state=seed)
    elif model == "lgbm":
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                                  subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=seed,
                                  n_jobs=4, verbosity=-1)
    elif model == "xgb":
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9,
                                 random_state=seed, n_jobs=4, eval_metric="logloss")
    else:
        raise ValueError(model)
    return Pipeline([("preprocess", prep), ("model", estimator)])


def metrics(y, score, validation_y, validation_score):
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 confusion_matrix, roc_auc_score, roc_curve)
    fpr, tpr, thresholds = roc_curve(validation_y, validation_score)
    candidates = np.flatnonzero(np.isfinite(thresholds))
    best = candidates[np.argmax((tpr - fpr)[candidates])]
    threshold = float(thresholds[best])
    tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, score)), "n_test": int(len(y)),
            "n_positive": int(np.sum(y)), "youden_threshold_from_validation": threshold,
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp))}


def preflight(cache_dir: Path) -> dict:
    paths = [cache_dir / "stays.parquet"]
    return {"ready": all(p.is_file() for p in paths),
            "required_files": [{"path": str(p.resolve()), "exists": p.is_file()} for p in paths],
            "information_time": "triage record; no hourly observations",
            "excluded": ["insurance", "language", "day_of_week", "hourly", "disposition", "hadm_id"],
            "split": "subject_id disjoint 60/20/20; fixed split_seed=42",
            "caveats": ["triage record timestamps must be verified in source extraction",
                        "anchor_age is not exact encounter age",
                        "not a reconstructed calendar-time validation",
                        "post-analysis protocol amendment"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache_real")
    parser.add_argument("--output", type=Path, default=ROOT / "results_revision/triage_primary")
    parser.add_argument("--models", nargs="+", choices=["lr", "lgbm", "xgb"], default=["lr", "lgbm"])
    parser.add_argument("--tasks", nargs="+", choices=["critical", "death"], default=["critical", "death"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    check = preflight(args.cache_dir)
    print(json.dumps(check, indent=2, ensure_ascii=True))
    if args.dry_run:
        return 0 if check["ready"] else 2
    if not check["ready"]:
        raise SystemExit("Real cache missing: no results were generated; synthetic data are not substituted.")
    if args.bootstrap < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive bootstrap count and distinct seeds required")
    # Refuse even an existing empty output folder to prevent result mixing.
    args.output.mkdir(parents=True, exist_ok=False)
    stays = api.load_stays(args.cache_dir).sort_values("stay_id").reset_index(drop=True)
    stays["split_revision"] = patient_split(stays)
    parts = {p: stays["split_revision"].eq(p).to_numpy() for p in ("train", "val", "test")}
    manifests = []
    for split, mask in parts.items():
        manifests.append({"partition": split, "stays": int(mask.sum()),
                          "patients": int(stays.loc[mask, "subject_id"].nunique())})
    manifest = {"protocol": check, "partitions": manifests, "args": vars(args),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "package_versions": {p: importlib.metadata.version(p) for p in
                                     ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")},
                "cache_size": (args.cache_dir / "stays.parquet").stat().st_size}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    stays[["stay_id", "subject_id", "split_revision"]].to_parquet(args.output / "split_manifest.parquet", index=False)
    from wjem_ed_audit.audit.runs_index import scan_runs
    from wjem_ed_audit.audit.sds import sds_table
    from sklearn.metrics import roc_auc_score
    import joblib
    summary = []
    for task in args.tasks:
        y = stays[api.task_label_col(task)].to_numpy()
        if not np.isin(y, [0, 1]).all() or any(np.unique(y[m]).size != 2 for m in parts.values()):
            raise ValueError(f"{task}: complete binary labels with events in every partition required")
        for name in args.models:
            for combo, groups in COMBOS.items():
                X, numeric, categorical = feature_frame(stays, groups, args.include_text)
                for seed in args.seeds:
                    pipe = make_pipeline(name, seed, numeric, categorical,
                                         args.include_text and "C" in groups)
                    pipe.fit(X.loc[parts["train"]], y[parts["train"]])
                    va = pipe.predict_proba(X.loc[parts["val"]])[:, 1]
                    score = pipe.predict_proba(X.loc[parts["test"]])[:, 1]
                    result = metrics(y[parts["test"]], score, y[parts["val"]], va)
                    result["validation_auroc"] = float(roc_auc_score(y[parts["val"]], va))
                    preds = stays.loc[parts["test"], ["stay_id", "subject_id"]].copy()
                    preds["y_true"], preds["y_score"] = y[parts["test"]], score
                    path = api.run_dir_for(args.output, task, name, combo, seed)
                    config = {"model": name, "task": task, "groups": groups,
                              "seed": seed, "feature_columns": list(X), "protocol": check}
                    api.write_run(path, result, preds, config)
                    joblib.dump(pipe, path / "pipeline.joblib")
                    summary.append({"task": task, "model": name, "combo": combo,
                                    "seed": seed, **result})
                    pd.DataFrame(summary).to_csv(args.output / "performance.csv", index=False)
                    print(task, name, combo, seed, result["auroc"], flush=True)
    index = scan_runs(args.output)
    patients = stays.set_index("stay_id")["subject_id"]
    sds_table(index, n_boot=args.bootstrap, patient_ids=patients).to_csv(
        args.output / "sds_patient_bootstrap.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
