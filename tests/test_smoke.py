"""Smoke tests for calibration / dca / subgroups / figures / report.

Runs under pytest or directly:
    python tests/test_smoke.py

Fixtures (all fabricated, no clinical data needed):
- tiny cache via api.make_tiny in a process-owned temporary directory
- fake run dirs per SPEC §4 (tasks critical+edlos, models lr+xgb, seeds 0/1)
- fake audit CSVs per SPEC §5 (E1/E3/E5/sds_csr)
"""
from __future__ import annotations

import os
import sys
import traceback
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wjem_ed_audit.data import api  # noqa: E402
from wjem_ed_audit.eval import calibration, dca, figures, report, subgroups  # noqa: E402

_TEST_DIRECTORY = TemporaryDirectory(prefix="wjem-ed-audit-tests-")
CACHE = Path(_TEST_DIRECTORY.name) / "cache"
RES = Path(_TEST_DIRECTORY.name) / "results"
CFG_PATH = RES / "fixture_config.yaml"
SEEDS = [0, 1]


def _fake_scores(yt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(0.12 + 0.7 * yt + rng.normal(0, 0.18, len(yt)), 0.0, 1.0)


def build_fixtures() -> None:
    api.make_tiny(CACHE, n=200, seed=0)
    stays = api.load_stays(CACHE)
    test = stays[stays["split_random"] == "test"].reset_index(drop=True)

    (RES / "audit").mkdir(parents=True)

    # ---- fake run dirs (SPEC §4)
    for task in ["critical", "edlos"]:
        yt = test[api.task_label_col(task)].to_numpy(dtype=float)
        for mi, model in enumerate(["lr", "xgb"]):
            for seed in SEEDS:
                rng = np.random.default_rng(100 * mi + seed)
                if task == "critical":
                    ys = _fake_scores(yt, rng)
                    metrics = {
                        "auroc": float(roc_auc_score(yt, ys)),
                        "auprc": float(average_precision_score(yt, ys)),
                        "brier": float(brier_score_loss(yt, ys)),
                        "ece": float(calibration.expected_calibration_error(yt, ys)),
                    }
                    preds = pd.DataFrame({"stay_id": test["stay_id"], "split": "test",
                                          "y_true": yt, "y_score": ys})
                else:
                    yp = yt + rng.normal(0, 2.0, len(yt))
                    metrics = {
                        "mae": float(mean_absolute_error(yt, yp)),
                        "rmse": float(mean_squared_error(yt, yp) ** 0.5),
                        "r2": float(r2_score(yt, yp)),
                    }
                    preds = pd.DataFrame({"stay_id": test["stay_id"], "split": "test",
                                          "y_true": yt, "y_pred": yp})
                metrics.update({"n_test": int(len(yt)), "split": "random_test"})
                cfg = {"groups": ["C", "J", "P", "D"], "model": model, "seed": seed}
                api.write_run(RES / "runs" / task / model / "full" / f"seed{seed}",
                              metrics, preds, cfg)

    # ---- fake audit CSVs (SPEC §5)
    rng = np.random.default_rng(7)
    e1_base = {"full": 0.0, "clin_only": -0.03, "meta_only": -0.10, "noJ": -0.06, "noP": -0.01}
    e1 = pd.DataFrame([
        {"task": "critical", "combo": c, "seed": s,
         "delta_auc": v + rng.normal(0, 0.005)}
        for c, v in e1_base.items() for s in SEEDS])
    e1.to_csv(RES / "audit" / "E1_branch_mask.csv", index=False)

    e3_rows = []
    for cond, d in [("suppress_J", -0.05), ("suppress_P", -0.02), ("suppress_random", -0.01)]:
        for s in SEEDS:
            e3_rows.append({"task": "critical", "condition": cond, "seed": s,
                            "delta_auc": d + rng.normal(0, 0.004), "probe_auc": np.nan})
    for s in SEEDS:
        e3_rows.append({"task": "critical", "condition": "esi_probe", "seed": s,
                        "delta_auc": np.nan, "probe_auc": 0.82 + rng.normal(0, 0.01)})
    pd.DataFrame(e3_rows).to_csv(RES / "audit" / "E3_repr_intervention.csv", index=False)

    e5_rows = []
    for combo, base in [("full", 0.88), ("noJ", 0.85)]:
        for split, drop in [("random_test", 0.0), ("temporal_test", 0.05)]:
            for s in SEEDS:
                e5_rows.append({"task": "critical", "combo": combo, "split": split,
                                "seed": s, "auc": base - drop + rng.normal(0, 0.005),
                                "status": np.nan})
    e5_rows.append({"task": "critical", "combo": "full", "split": "eicu", "seed": 0,
                    "auc": np.nan, "status": "not_available_pending_dua"})
    pd.DataFrame(e5_rows).to_csv(RES / "audit" / "E5_transfer.csv", index=False)

    sds = pd.DataFrame([
        {"task": "critical", "metric": m, "value": v, "ci_lo": v - 0.05, "ci_hi": v + 0.05}
        for m, v in [("SDS_mask", 0.25), ("SDS_solo", 0.55), ("CSR", 0.20), ("R", 0.05)]])
    sds.to_csv(RES / "audit" / "sds_csr.csv", index=False)

    # ---- fixture config
    cfg = {
        "data": {"source": "synthetic", "cache_dir": str(CACHE)},
        "tasks": ["critical", "edlos"],
        "main_task": "critical",
        "seeds": SEEDS,
        "eval": {"subgroups": ["anchor_age", "gender", "arrival_transport", "acuity"],
                 "dca_thresholds": [0.01, 0.5, 50]},
        "results_root": str(RES),
    }
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)


build_fixtures()


# ---------------------------------------------------------------------- tests
def test_calibration():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 300)
    ys = np.clip(0.1 + 0.7 * yt + rng.normal(0, 0.2, 300), 0, 1)
    df = calibration.calibration_curve_data(yt, ys, n_bins=15)
    assert len(df) == 15 and df["count"].sum() == 300
    ok = df[df["count"] > 0]
    assert ok["frac_pos"].between(0, 1).all() and ok["pred_mean"].between(0, 1).all()
    ece = calibration.expected_calibration_error(yt, ys)
    assert 0.0 <= ece <= 1.0
    out = RES / "figures" / "ut_calibration.png"
    calibration.plot_calibration({"m1": (yt, ys), "m2": (yt, 1 - ys)}, out)
    assert out.exists() and out.stat().st_size > 0


def test_dca():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 2, 300)
    ys = np.clip(0.1 + 0.7 * yt + rng.normal(0, 0.2, 300), 0, 1)
    th = np.linspace(0.01, 0.5, 50)
    df = dca.decision_curve(yt, ys, th)
    assert len(df) == 50
    assert np.isfinite(df["net_benefit"]).all()
    assert (df["treat_none"] == 0).all()
    assert np.isclose(df.loc[0, "net_benefit"], df.loc[0, "treat_all"], atol=0.05)
    out = RES / "figures" / "ut_dca.png"
    dca.plot_dca({"m1": (yt, ys)}, th, out)
    assert out.exists() and out.stat().st_size > 0


def test_subgroups():
    stays = api.load_stays(CACHE)
    test = stays[stays["split_random"] == "test"].reset_index(drop=True)
    yt = test["y_critical"].to_numpy(dtype=float)
    ys = _fake_scores(yt, np.random.default_rng(2))
    df = subgroups.subgroup_performance(
        test, yt, ys, ["anchor_age", "gender", "arrival_transport", "acuity"], min_n=5)
    assert set(df["variable"]) == {"anchor_age", "gender", "arrival_transport", "acuity"}
    assert set(df.loc[df["variable"] == "anchor_age", "level"]) == {"<65", ">=65"}
    assert set(df.loc[df["variable"] == "acuity", "level"]) <= {"1-2", "3", "4-5", "missing"}
    assert df.loc[df["variable"] == "gender", "n"].sum() == len(test)
    auc = df["auc"].dropna()
    assert len(auc) > 0 and auc.between(0, 1).all()


def test_figures():
    figs = RES / "figures"
    p2 = figures.fig2_branch_mask(RES / "audit" / "E1_branch_mask.csv", figs / "fig2.png")
    p3 = figures.fig3_repr_intervention(RES / "audit" / "E3_repr_intervention.csv", figs / "fig3.png")
    p4 = figures.fig4_transfer(RES / "audit" / "E5_transfer.csv", figs / "fig4.png")
    p5 = figures.fig5_sds_csr(RES / "audit" / "sds_csr.csv", figs / "fig5.png")
    for p in [p2, p3, p4, p5]:
        assert p is not None and Path(p).exists() and Path(p).stat().st_size > 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert figures.fig2_branch_mask(RES / "audit" / "nope.csv", figs / "x.png") is None
        assert any("missing" in str(x.message) for x in w)


def test_report():
    rc = report.main(["--config", str(CFG_PATH)])
    assert rc == 0
    t1 = RES / "tables" / "table1_performance.csv"
    t2 = RES / "tables" / "table2_benchmark.csv"
    t3 = RES / "tables" / "table3_subgroups.csv"
    md = RES / "REPORT.md"
    for p in [t1, t2, t3, md]:
        assert p.exists() and p.stat().st_size > 0, p
    df1 = pd.read_csv(t1)
    assert {"task", "model", "config_tag", "n_seeds"} <= set(df1.columns)
    assert df1["n_seeds"].eq(2).all()
    crit = df1[df1["task"] == "critical"]
    assert crit["auroc_mean"].between(0, 1).all()
    assert crit["brier_mean"].between(0, 1).all()
    assert crit["ece_mean"].between(0, 1).all()
    ed = df1[df1["task"] == "edlos"]
    assert ed["r2_mean"].le(1.0).all() and ed["mae_mean"].ge(0).all()
    df2 = pd.read_csv(t2)
    assert {"item", "ours", "published", "note"} <= set(df2.columns)
    assert (df2["published"].astype(str) == "0.894").any()
    df3 = pd.read_csv(t3)
    auc = df3["auc"].dropna()
    assert len(auc) > 0 and auc.between(0, 1).all()
    for name in ["fig2_branch_mask.png", "fig3_repr_intervention.png",
                 "fig4_transfer.png", "fig5_sds_csr.png",
                 "calibration_critical.png", "dca_critical.png"]:
        p = RES / "figures" / name
        assert p.exists() and p.stat().st_size > 0, p
    text = md.read_text(encoding="utf-8")
    assert "局限性" in text and "E1" in text and "SDS" in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} smoke tests passed")
    sys.exit(1 if failed else 0)
