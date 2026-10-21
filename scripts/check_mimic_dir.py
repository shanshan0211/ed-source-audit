#!/usr/bin/env python
"""Validate a local MIMIC-IV-ED v2.2 download before building the cohort.

Usage (from the project root wjem_ed_audit/):
    python scripts/check_mimic_dir.py /path/to/mimic-iv-ed-2.2

Checks, per table (edstays / triage / vitalsign / diagnosis / medrecon / pyxis):
  1. the file exists as <table>.csv.gz or <table>.csv;
  2. the header matches the official MIMIC-IV-ED v2.2 schema exactly;
  3. the first 100 data rows parse with the documented types
     (ids integer, vitals numeric-with-NaN, timestamps parseable);
  4. timestamps fall in the de-identification window: MIMIC-IV-ED shifts all
     dates into 2100-2200 (true coverage is 2011-2019), so any other year
     range means a wrong/corrupted file;
  5. stay_id connectivity: edstays.stay_id and triage.stay_id intersect
     (first 1000 rows of each), proving the tables are joinable.

Prints one PASS/FAIL line per check and exits 0 only if everything passes.

Schema source (hard-coded below; verified 2026-09-23 against):
  https://physionet.org/content/mimic-iv-ed/2.2/           (release notes v2.2)
  https://mimic.mit.edu/docs/iv/modules/ed/edstays.html
  https://mimic.mit.edu/docs/iv/modules/ed/triage.html     (pain is free text)
  https://mimic.mit.edu/docs/iv/modules/ed/vitalsign.html
  https://mimic.mit.edu/docs/iv/modules/ed/diagnosis.html
  https://mimic.mit.edu/docs/iv/modules/ed/medrecon.html
  https://mimic.mit.edu/docs/iv/modules/ed/pyxis.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Official MIMIC-IV-ED v2.2 columns (see docstring for sources).
SCHEMA: dict[str, list[str]] = {
    "edstays": ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
                "gender", "race", "arrival_transport", "disposition"],
    "triage": ["subject_id", "stay_id", "temperature", "heartrate", "resprate",
               "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"],
    "vitalsign": ["subject_id", "stay_id", "charttime", "temperature",
                  "heartrate", "resprate", "o2sat", "sbp", "dbp",
                  "rhythm", "pain"],
    "diagnosis": ["subject_id", "stay_id", "seq_num", "icd_code",
                  "icd_version", "icd_title"],
    "medrecon": ["subject_id", "stay_id", "charttime", "name", "gsn", "ndc",
                 "etc_rn", "etccode", "etcdescription"],
    "pyxis": ["subject_id", "stay_id", "charttime", "med_rn", "name",
              "gsn_rn", "gsn"],
}

# Columns that must parse as dates, per table (used for the 2100-2200 check).
DATE_COLS: dict[str, list[str]] = {
    "edstays": ["intime", "outtime"],
    "vitalsign": ["charttime"],
    "medrecon": ["charttime"],
    "pyxis": ["charttime"],
}

# Columns that must be numeric after coercion (NaN allowed; MIMIC inserts NaN
# for values removed during de-identification). pain is TEXT since v2.0 and is
# therefore only coerced, never required.
NUMERIC_COLS: dict[str, list[str]] = {
    "edstays": ["subject_id", "stay_id"],
    "triage": ["subject_id", "stay_id", "temperature", "heartrate", "resprate",
               "o2sat", "sbp", "dbp", "acuity"],
    "vitalsign": ["subject_id", "stay_id", "temperature", "heartrate",
                  "resprate", "o2sat", "sbp", "dbp"],
    "diagnosis": ["subject_id", "stay_id", "seq_num", "icd_version"],
    "medrecon": ["subject_id", "stay_id", "etc_rn"],
    "pyxis": ["subject_id", "stay_id", "med_rn", "gsn_rn"],
}

SAMPLE_ROWS = 100
JOIN_ROWS = 1000
# True coverage is 2011-2019; PhysioNet shifts dates into 2100-2200.
VALID_YEAR_RANGES = (range(2011, 2020), range(2100, 2201))


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.checks = 0

    def emit(self, ok: bool, label: str, detail: str = "") -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
        line = f"[{'PASS' if ok else 'FAIL'}] {label}"
        if detail:
            line += f" -- {detail}"
        print(line)


def _find_file(mimic_dir: Path, table: str) -> Path | None:
    for suffix in (".csv.gz", ".csv"):
        p = mimic_dir / f"{table}{suffix}"
        if p.exists():
            return p
    return None


def _years_ok(dates: pd.Series) -> tuple[bool, str]:
    years = dates.dt.year.dropna().unique()
    if len(years) == 0:
        return False, "no parseable dates"
    in_shifted = all(y in VALID_YEAR_RANGES[1] for y in years)
    in_true = all(y in VALID_YEAR_RANGES[0] for y in years)
    lo, hi = int(min(years)), int(max(years))
    if in_shifted:
        return True, f"years {lo}-{hi} (de-id shift window; true 2011-2019)"
    if in_true:
        return True, f"years {lo}-{hi} (true dates, unshifted)"
    return False, f"years {lo}-{hi} outside 2100-2200 (and 2011-2019)"


def check_table(rep: Report, mimic_dir: Path, table: str) -> pd.DataFrame | None:
    path = _find_file(mimic_dir, table)
    rep.emit(path is not None, f"{table}: file exists (.csv.gz or .csv)",
             str(path) if path else f"expected {mimic_dir}/{table}.csv.gz")
    if path is None:
        return None

    try:
        df = pd.read_csv(path, nrows=SAMPLE_ROWS)
    except Exception as exc:  # corrupt gzip, encoding, ...
        rep.emit(False, f"{table}: readable", f"{type(exc).__name__}: {exc}")
        return None

    expected = SCHEMA[table]
    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    rep.emit(got == expected, f"{table}: header matches v2.2 schema",
             "" if got == expected else
             f"missing={missing or '-'} extra={extra or '-'} order={'ok' if not missing and not extra else 'n/a'}")

    if len(df) == 0:
        rep.emit(False, f"{table}: sample rows", "file has a header but 0 data rows")
        return df

    # type parseability on the 100-row sample
    bad: list[str] = []
    for col in NUMERIC_COLS.get(table, []):
        if col not in df:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all() and df[col].notna().any():
            bad.append(col)
    rep.emit(not bad, f"{table}: numeric columns parse ({SAMPLE_ROWS}-row sample)",
             "" if not bad else f"non-numeric columns: {bad}")

    # date range in the de-identification window
    for col in DATE_COLS.get(table, []):
        if col not in df:
            continue
        dates = pd.to_datetime(df[col], errors="coerce")
        parseable = dates.notna().mean() >= 0.95
        rep.emit(parseable, f"{table}.{col}: >=95% of sample parses as timestamp")
        if parseable:
            ok, detail = _years_ok(dates)
            rep.emit(ok, f"{table}.{col}: year range", detail)
    return df


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    mimic_dir = Path(argv[1])
    rep = Report()
    print(f"Checking MIMIC-IV-ED v2.2 directory: {mimic_dir}")
    rep.emit(mimic_dir.is_dir(), "directory exists", str(mimic_dir))
    if not mimic_dir.is_dir():
        print(f"\nSUMMARY: 1/{rep.checks} checks passed (FAIL)")
        return 1

    samples: dict[str, pd.DataFrame] = {}
    for table in SCHEMA:
        df = check_table(rep, mimic_dir, table)
        if df is not None:
            samples[table] = df

    # stay_id connectivity: edstays ∩ triage must be non-empty.
    if "edstays" in samples and "triage" in samples:
        try:
            ed_ids = pd.read_csv(_find_file(mimic_dir, "edstays"),
                                 usecols=["stay_id"], nrows=JOIN_ROWS)["stay_id"]
            tri_ids = pd.read_csv(_find_file(mimic_dir, "triage"),
                                  usecols=["stay_id"], nrows=JOIN_ROWS)["stay_id"]
            n_common = len(set(ed_ids) & set(tri_ids))
            rep.emit(n_common > 0, "stay_id connectivity edstays∩triage",
                     f"{n_common} shared stay_id in first {JOIN_ROWS} rows")
        except Exception as exc:
            rep.emit(False, "stay_id connectivity edstays∩triage",
                     f"{type(exc).__name__}: {exc}")

    passed = rep.checks - rep.failures
    status = "PASS" if rep.failures == 0 else "FAIL"
    print(f"\nSUMMARY: {passed}/{rep.checks} checks passed ({status})")
    if rep.failures:
        print("Fix the FAIL items above (re-download or complete the DUA "
              "steps in docs/dua_runbook.md), then re-run this script.")
    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
