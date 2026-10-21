"""Checks for paired uncertainty and triage-time feature boundaries."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wjem_ed_audit.audit.sds import compute_sds_family, sds_table

spec = importlib.util.spec_from_file_location("revision_triage", ROOT / "scripts/reanalyse_triage.py")
triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triage)


def fixture_predictions():
    rng = np.random.default_rng(9)
    y = np.tile([0, 1], 50)
    preds = pd.DataFrame({"stay_id": np.arange(100), "y_true": y,
                          "y_score": y * 0.6 + rng.normal(0, 1, 100)})
    index = pd.DataFrame([{"task": "critical", "model": "lr", "config_tag": c,
                           "seed": s, "temporal": False}
                          for c in ("full", "clin_only", "meta_only") for s in (0, 1)])
    cache = {("critical", "lr", c): [preds.sample(frac=1, random_state=s).copy() for s in (0, 1)]
             for c in ("full", "clin_only", "meta_only")}
    return index, cache


def test_identical_predictions_have_zero_removal_uncertainty():
    index, cache = fixture_predictions()
    result = sds_table(index, cache, n_boot=80).set_index("metric")
    assert result.loc["SDS_mask", ["value", "lo", "hi"]].eq(0).all()
    assert result.loc["CSR", ["value", "lo", "hi"]].eq(0).all()
    assert result.loc["auc_full", "hi"] > result.loc["auc_full", "lo"]


def test_patient_bootstrap_preserves_all_model_pairing():
    index, cache = fixture_predictions()
    subjects = pd.Series(np.repeat(np.arange(50), 2), index=np.arange(100))
    result = sds_table(index, cache, n_boot=80, patient_ids=subjects).set_index("metric")
    assert result.loc["SDS_mask", ["value", "lo", "hi"]].eq(0).all()


def test_mismatched_patients_and_labels_are_rejected():
    index, cache = fixture_predictions()
    cache[("critical", "lr", "clin_only")][0].loc[:, "stay_id"] += 1000
    with pytest.raises(ValueError, match="identical labeled"):
        sds_table(index, cache, n_boot=2)


def test_sds_algebra_is_not_four_independent_metrics():
    values = compute_sds_family(.85, .82, .77)
    assert values["CSR"] == pytest.approx(1 - values["SDS_solo"])
    assert values["R"] == pytest.approx(values["SDS_solo"] - values["SDS_mask"])


def sample_stays():
    n = 60
    df = pd.DataFrame({"stay_id": np.arange(n), "subject_id": np.repeat(np.arange(20), 3),
                       "intime": pd.Timestamp("2150-01-01"), "anchor_age": 45,
                       "gender": "F", "race": "OTHER", "acuity": 3,
                       "arrival_transport": "WALK IN", "n_ed_visits_365d": 0,
                       "insurance": "ADMISSION_MARKER", "language": "ADMISSION_MARKER",
                       "chiefcomplaint": "chest pain", "disposition": "ADMITTED"})
    for col in triage.api.TRIAGE_VITALS:
        df[col] = 1.
    return df


def test_patient_split_is_disjoint_and_deterministic():
    df = sample_stays()
    split = triage.patient_split(df)
    assert split.equals(triage.patient_split(df))
    assert df.assign(part=split).groupby("subject_id")["part"].nunique().eq(1).all()
    assert set(split) == {"train", "val", "test"}


def test_feature_allowlist_excludes_documentation_and_future_data():
    X, numeric, cats = triage.feature_frame(sample_stays(), ("C", "J", "P", "D"), True)
    assert not {"insurance", "language", "disposition", "hourly", "day_of_week"} & set(X)
    assert "chiefcomplaint" in X and "arrival_hour" in X
    pipe = triage.make_pipeline("lr", 0, numeric, cats, False)
    # A category seen only at evaluation must not enter the learned vocabulary.
    X.loc[40:, "gender"] = "HELDOUT_ONLY"
    y = np.tile([0, 1], 30)
    pipe.fit(X.iloc[:40], y[:40])
    assert "HELDOUT_ONLY" not in str(pipe.named_steps["preprocess"].named_transformers_["category"].categories_)
    assert len(pipe.predict_proba(X.iloc[40:])) == 20


def test_cli_with_fabricated_data_and_overwrite_protection(tmp_path):
    """End-to-end fixture only: fabricated results stay in pytest temporary storage."""
    cache = tmp_path / "fabricated_cache"
    cache.mkdir()
    stays = sample_stays()
    stays["y_critical"] = np.tile([0, 1], 30)
    stays.to_parquet(cache / "stays.parquet", index=False)
    output = tmp_path / "fabricated_results"
    args = ["--cache-dir", str(cache), "--output", str(output),
            "--models", "lr", "--seeds", "0", "--tasks", "critical", "--bootstrap", "10"]
    assert triage.main(args) == 0
    table = pd.read_csv(output / "performance.csv")
    assert set(table["combo"]) == set(triage.COMBOS)
    assert (output / "sds_patient_bootstrap.csv").is_file()
    manifest = pd.read_parquet(output / "split_manifest.parquet")
    assert manifest.groupby("subject_id")["split_revision"].nunique().eq(1).all()
    with pytest.raises(FileExistsError):
        triage.main(args)


def test_missing_real_cache_is_not_replaced_by_synthetic(tmp_path):
    assert triage.main(["--cache-dir", str(tmp_path / "missing"), "--dry-run"]) == 2
