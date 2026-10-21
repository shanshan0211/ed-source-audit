"""Real MIMIC-IV-ED cohort builder (config.data.mimic_dir -> stays/hourly schema).

Requires TWO PhysioNet credentialed downloads (step-by-step: docs/dua_runbook.md):

  1. MIMIC-IV-ED v2.2 -> ``cfg.data.mimic_dir`` (edstays/triage/vitalsign/...).
  2. MIMIC-IV v2.2 (core) -> ``cfg.data.mimic_core_dir``: the download root that
     contains ``hosp/`` and ``icu/`` subdirectories. Optional overrides:
     ``cfg.data.mimic_hosp_dir`` / ``cfg.data.mimic_icu_dir``.
     Without the core module the builder raises NotImplementedError, because
     y_critical (main-task label), anchor_age and insurance/language all live
     there (hosp/patients.csv.gz, hosp/admissions.csv.gz, icu/icustays.csv.gz).

Column names below follow the official v2.2 documentation (verified 2026-09-23):
  https://physionet.org/content/mimic-iv-ed/2.2/
  https://mimic.mit.edu/docs/iv/modules/ed/{edstays,triage,vitalsign,diagnosis,medrecon,pyxis}.html
  https://mimic.mit.edu/docs/iv/modules/hosp/{patients,admissions}.html
  https://mimic.mit.edu/docs/iv/modules/icu/icustays.html

De-identification facts that drive the logic in this file (PhysioNet docs):

  * All dates are shifted into 2100-2200 by a PER-SUBJECT constant offset.
    => within-patient time differences (ED intime -> first ICU intime, 365-day
       lookback, EDLOS) are exact, but calendar years are NOT recoverable from
       timestamps. Never split or filter on the shifted clock.
  * The only real-world time marker is patients.anchor_year_group
    ("2011 - 2013" / "2014 - 2016" / "2017 - 2019" / ...). split_temporal is
    therefore built from anchor_year_group (dev = groups ending before the
    configured cutoff year, test = the rest), which also makes the temporal
    split patient-disjoint.
  * patients.anchor_age = age at anchor_year, capped at 91 for age > 89. The
    exact age at an ED visit is not recoverable (visit year is only known to a
    3-year bin); anchor_age is the standard approximation used in the MIMIC
    literature and is what SPEC §1.1 expects.
  * triage.pain / vitalsign.pain are FREE TEXT since MIMIC-IV-ED v2.0 (official
    release notes: the column "includes free-text entries which are not valid
    numbers"). All triage/vitalsign numerics are coerced with
    pd.to_numeric(errors="coerce"); unparseable entries become NaN.

Additional labels (SPEC §1.1)
=============================

``y_death`` (death-only sensitivity endpoint, research plan §2/§5 — ICU
transfer may be a direct downstream of ESI):
    ``admissions.hospital_expire_flag == 1`` OR ``edstays.disposition ==
    'EXPIRED'`` (death in the ED). ``y_death`` is therefore a subset of
    ``y_critical`` by construction, mirroring the main-task label definition.

``y_sepsis24h`` (secondary task, benchmarked against WJEM 2024 Xie J,
AUC 0.92): a DOCUMENTED Sepsis-3 approximation evaluated over the 24 h after
ED triage (``[intime, intime + 24h)``; per-subject constant date shift makes
cross-module time differences exact):

    suspected infection (SI) within 24 h of triage, defined as EITHER
      * antibiotic administration: ED ``pyxis.csv.gz`` dispensation or
        ``medrecon.csv.gz`` entry whose ``name`` matches a curated
        case-insensitive generic antibacterial list (_ANTIBIOTIC_RE), OR
      * blood culture order: ``hosp/microbiologyevents.csv.gz`` row with
        ``spec_type_desc`` containing "BLOOD" within the window;
    AND SOFA-proxy >= 2 points accumulated over the same 24 h window:
      * respiration  : min(o2sat) < 94 OR max(resprate) >= 22
                       (triage row + ``vitalsign.csv.gz``), proxying the
                       PaO2/FiO2 axis which does not exist in the ED module;
      * coagulation  : min(platelet) < 150   (labevents itemid 51265);
      * liver        : max(bilirubin) >= 1.2 (labevents itemid 50885);
      * cardiovascular: min(sbp) < 100 OR any vasopressor infusion in
                       ``icu/inputevents.csv.gz`` (VASOPRESSOR_ITEMIDS),
                       proxying MAP < 70 / vasopressor requirement;
                       if inputevents.csv.gz is unavailable the vasopressor
                       term degrades to 0 (SBP<100 retained) and a warning
                       is emitted — labels remain valid as a lower-bound
                       approximation;
      * renal        : max(creatinine) >= 1.2 (labevents itemid 50912);
      * neuro        : GCS is NOT recorded in the ED module -> scored 0.

    Deviations from formal Sepsis-3 (documented, by design): SpO2/RR proxy
    for the oxygenation axis; SBP < 100 + any-vasopressor proxy for the
    cardiovascular axis; neuro axis always 0 (biased toward specificity);
    baseline organ function assumed normal (no pre-admission lab lookback);
    the window is ED-relative 24 h rather than the canonical 48 h
    infection-episode window; thresholds use the lowest SOFA grade (1 point)
    per axis; medrecon includes reconciled (not necessarily administered)
    home medications, biasing suspected infection toward sensitivity. These
    choices follow the triage-24h sepsis protocol spirit of Xie J et al.,
    WJEM 2024;15(5):379-385 and are adequate for a benchmark target, not a
    clinical case definition. When any required core file
    (microbiologyevents/labevents/inputevents) is unavailable the builder
    raises NotImplementedError pointing at docs/dua_runbook.md.

Function signature mirrors synth.generate: ``build(cfg) -> (stays, hourly)``.
Output schema follows SPEC §1.1/§1.2 exactly (same columns as synth.generate).
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .api import HOURLY_FEATS
from .synth import _stratified_random_split

# Physiological validity ranges (Harutyunyan-style clipping, mirroring the
# Xie 2022 benchmark protocol): out-of-range -> NaN (treated as missing).
# triage/vitalsign temperature in MIMIC-IV-ED is Fahrenheit -> convert >60.
VITAL_RANGES = {
    "temperature": (25.0, 45.0), "heartrate": (20.0, 250.0),
    "resprate": (2.0, 60.0), "o2sat": (50.0, 100.0),
    "sbp": (40.0, 300.0), "dbp": (10.0, 200.0), "pain": (0.0, 10.0),
    "lactate": (0.0, 30.0), "wbc": (0.0, 500.0),
    "creatinine": (0.0, 20.0), "glucose": (0.0, 2000.0),
}


def _clean_vitals(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Clip `cols` of df to physiological ranges; convert F->C temperature."""
    df = df.copy()
    for col in cols:
        if col not in df.columns or col not in VITAL_RANGES:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if col == "temperature":
            v = v.where(v < 60, (v - 32.0) / 1.8)
        lo, hi = VITAL_RANGES[col]
        df[col] = v.where(v.between(lo, hi))
    return df


REQUIRED_FILES = ("edstays.csv.gz", "triage.csv.gz", "vitalsign.csv.gz")

# MIMIC-IV core (hosp/icu) files this builder consumes.
REQUIRED_HOSP_FILES = ("patients.csv.gz", "admissions.csv.gz")
REQUIRED_ICU_FILES = ("icustays.csv.gz",)

# hosp/labevents.csv.gz itemids (d_labitems, MIMIC-IV v2.x) for SPEC §1.2 labs.
LAB_ITEMIDS = {
    "lactate": (50813,),            # Lactate, blood
    "wbc": (51301,),                # White Blood Cells
    "creatinine": (50912,),         # Creatinine, blood
    "glucose": (50931, 50809),      # Glucose (chemistry) / Glucose (blood gas)
}

# Extra labevents itemids needed for the sepsis SOFA-proxy axes that are not
# part of the SPEC §1.2 hourly lab set (d_labitems, MIMIC-IV v2.x).
SOFA_LAB_ITEMIDS = {
    "platelet": (51265,),           # Platelet Count, blood
    "bilirubin": (50885,),          # Bilirubin, Total, blood
    "creatinine": (50912,),         # Creatinine, blood
}

# icu/inputevents.csv.gz itemids for vasopressor infusions (d_items).
VASOPRESSOR_ITEMIDS = (221906,      # Norepinephrine
                       221289,      # Epinephrine
                       221662,      # Dopamine
                       221749,      # Phenylephrine
                       222315)      # Vasopressin

# Curated case-insensitive generic-name antibacterial list, matched against
# pyxis.name / medrecon.name for the suspected-infection criterion.
_ANTIBIOTIC_RE = re.compile(
    r"vancomycin|piperacillin|tazobactam|cefazolin|ceftriaxone|cefepime|"
    r"ceftaroline|cefuroxime|cephalexin|meropenem|ertapenem|imipenem|"
    r"aztreonam|levofloxacin|ciprofloxacin|moxifloxacin|ofloxacin|"
    r"azithromycin|clarithromycin|erythromycin|doxycycline|minocycline|"
    r"clindamycin|metronidazole|ampicillin|amoxicillin|penicillin|nafcillin|"
    r"oxacillin|dicloxacillin|gentamicin|tobramycin|amikacin|linezolid|"
    r"daptomycin|nitrofurantoin|sulfamethoxazole|trimethoprim|fosfomycin|"
    r"colistin|tigecycline",
    flags=re.IGNORECASE)

SEPSIS_WINDOW_H = 24

TRIAGE_NUMERIC = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
                  "pain", "acuity"]
VITALSIGN_NUMERIC = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature"]


def _check_dir(mimic_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (mimic_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"MIMIC-IV-ED directory {mimic_dir} is missing {missing}. "
            "Download requires a signed PhysioNet DUA (docs/dua_runbook.md); "
            "use data.source=synthetic until then. After downloading, verify "
            "with: python scripts/check_mimic_dir.py <mimic_dir>")


def _resolve_core_dirs(data_cfg: dict) -> tuple[Path | None, Path | None]:
    """Locate the MIMIC-IV core hosp/ and icu/ module directories."""
    core = data_cfg.get("mimic_core_dir")
    core = Path(core) if core else None
    hosp = data_cfg.get("mimic_hosp_dir")
    icu = data_cfg.get("mimic_icu_dir")
    hosp_dir = Path(hosp) if hosp else (core / "hosp" if core else None)
    icu_dir = Path(icu) if icu else (core / "icu" if core else None)
    return hosp_dir, icu_dir


def _require_core(hosp_dir: Path | None, icu_dir: Path | None) -> None:
    """Fail fast with instructions when the MIMIC-IV core module is absent."""
    missing: list[str] = []
    if hosp_dir is None or not hosp_dir.is_dir():
        missing.append("hosp/ (patients.csv.gz, admissions.csv.gz)")
    elif not all((hosp_dir / f).exists() for f in REQUIRED_HOSP_FILES):
        missing.append(f"hosp/ under {hosp_dir} is missing one of {REQUIRED_HOSP_FILES}")
    if icu_dir is None or not icu_dir.is_dir():
        missing.append("icu/ (icustays.csv.gz)")
    elif not all((icu_dir / f).exists() for f in REQUIRED_ICU_FILES):
        missing.append(f"icu/ under {icu_dir} is missing one of {REQUIRED_ICU_FILES}")
    if missing:
        raise NotImplementedError(
            "Cannot build y_critical (12h ICU transfer / ED or hospital death), "
            "y_death, y_sepsis24h, anchor_age, insurance, language: these "
            "live in the MIMIC-IV v2.2 *core* module, not in MIMIC-IV-ED. "
            "Missing: "
            + "; ".join(missing)
            + ". Set data.mimic_core_dir in configs/default.yaml to the "
            "MIMIC-IV v2.2 download root containing hosp/ and icu/ "
            "(physionet.org/files/mimiciv/2.2/; DUA steps in "
            "docs/dua_runbook.md). The ED-module-only path is intentionally "
            "not supported because the main-task label would be undefined.")


# ---------------------------------------------------------------------- stays
def _build_stays(mimic_dir: Path, hosp_dir: Path, icu_dir: Path,
                 cfg: dict) -> pd.DataFrame:
    ed = pd.read_csv(mimic_dir / "edstays.csv.gz",
                     parse_dates=["intime", "outtime"])
    # hadm_id is empty for patients not admitted after the ED stay.
    ed["hadm_id"] = pd.to_numeric(ed["hadm_id"], errors="coerce").astype("Int64")
    tri = pd.read_csv(mimic_dir / "triage.csv.gz")
    # triage numerics were free text before de-identification (v2.0 release
    # notes); coerce so stray text becomes NaN instead of object dtype.
    for col in TRIAGE_NUMERIC:
        tri[col] = pd.to_numeric(tri[col], errors="coerce")
    tri = _clean_vitals(tri, TRIAGE_NUMERIC)

    stays = ed.merge(
        tri[["stay_id"] + TRIAGE_NUMERIC + ["chiefcomplaint"]],
        on="stay_id", how="left")

    stays["chiefcomplaint"] = stays["chiefcomplaint"].fillna("").str.lower().str.strip()
    stays["arrival_transport"] = stays["arrival_transport"].fillna("UNKNOWN")
    stays["edlos_hours"] = (
        (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
    ).clip(lower=0, upper=72.0)  # winsorize at 72h (boarding artifacts; documented)

    # y_hosp: admitted to the hospital from the ED.
    stays["y_hosp"] = (stays["disposition"].str.upper() == "ADMITTED").astype(np.int64)

    # y_revisit72: same subject has another ED stay starting within
    # (outtime, outtime + 72h]. Self-join on subject sorted by intime.
    ed_sorted = ed.sort_values(["subject_id", "intime"])
    nxt = ed_sorted.groupby("subject_id")["intime"].shift(-1)
    gap_h = (nxt - ed_sorted["outtime"]).dt.total_seconds() / 3600.0
    ed_sorted = ed_sorted.assign(y_revisit72=((gap_h > 0) & (gap_h <= 72)).astype(np.int64))
    stays = stays.merge(ed_sorted[["stay_id", "y_revisit72"]], on="stay_id", how="left")
    stays["y_revisit72"] = stays["y_revisit72"].fillna(0).astype(np.int64)

    # --------------------------- MIMIC-IV core (hosp) -------------------------
    # anchor_age + anchor_year_group from hosp/patients.csv.gz
    # (subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod).
    # anchor_age is the age at anchor_year, capped at 91 (>89); see module
    # docstring for the de-identification rationale.
    pat = pd.read_csv(hosp_dir / "patients.csv.gz")
    pat = pat.drop_duplicates("subject_id")
    stays = stays.merge(
        pat[["subject_id", "anchor_age", "anchor_year_group"]],
        on="subject_id", how="left")
    stays["anchor_age"] = pd.to_numeric(stays["anchor_age"], errors="coerce")
    if stays["anchor_age"].isna().any():
        warnings.warn("some subject_id missing from hosp/patients.csv.gz -> "
                      "anchor_age NaN for those stays.")

    # insurance/language/hospital_expire_flag from hosp/admissions.csv.gz
    # (subject_id, hadm_id, ..., insurance, language, ..., hospital_expire_flag).
    adm = pd.read_csv(hosp_dir / "admissions.csv.gz")
    adm["hadm_id"] = pd.to_numeric(adm["hadm_id"], errors="coerce").astype("Int64")
    adm = adm.dropna(subset=["hadm_id"]).drop_duplicates("hadm_id")
    stays = stays.merge(
        adm[["hadm_id", "insurance", "language", "hospital_expire_flag"]],
        on="hadm_id", how="left")
    for col in ("insurance", "language"):
        stays[col] = stays[col].fillna("UNKNOWN")
    stays["gender"] = stays["gender"].fillna("UNKNOWN")
    stays["race"] = stays["race"].fillna("UNKNOWN").str.upper()

    # y_critical = first ICU admission within 12h of ED *intime* OR in-hospital
    # death (incl. death in the ED). Per-subject constant date shift keeps
    # icustays.intime and edstays.intime on the same clock within a patient.
    icu = pd.read_csv(icu_dir / "icustays.csv.gz", parse_dates=["intime"])
    icu["hadm_id"] = pd.to_numeric(icu["hadm_id"], errors="coerce").astype("Int64")
    first_icu = (icu.dropna(subset=["hadm_id"])
                    .groupby("hadm_id")["intime"].min()
                    .rename("first_icu_intime"))
    stays = stays.merge(first_icu, on="hadm_id", how="left")
    icu12 = (stays["first_icu_intime"].notna()
             & (stays["first_icu_intime"] >= stays["intime"])
             & (stays["first_icu_intime"] <= stays["intime"] + pd.Timedelta(hours=12)))
    died = ((stays["hospital_expire_flag"].fillna(0) == 1)
            | (stays["disposition"].str.upper() == "EXPIRED"))
    stays["y_critical"] = (icu12 | died).astype(np.int64)
    # y_death: death during the index ED visit or associated hospital stay,
    # including ED death; by construction a subset of y_critical.
    stays["y_death"] = died.astype(np.int64)
    # y_sepsis24h: documented Sepsis-3 approximation (see module docstring).
    stays["y_sepsis24h"] = _build_sepsis24h(mimic_dir, hosp_dir, icu_dir, stays)

    # n_ed_visits_365d: count of the subject's ED stays with intime in the
    # 365 days before this stay's intime (computable from edstays alone).
    stays["n_ed_visits_365d"] = (
        stays["stay_id"].map(_count_prior_ed_visits(ed)).fillna(0).astype(np.int64))

    # ---- splits ---------------------------------------------------------------
    # Real years are unrecoverable from shifted timestamps; anchor_year_group
    # (3-year bins) is the only real-world time marker -> temporal split by
    # anchor year group (also patient-disjoint).
    cutoff = pd.Timestamp(cfg.get("splits", {}).get("temporal_cutoff", "2017-01-01"))
    stays["split_temporal"] = stays["anchor_year_group"].map(
        lambda g: _year_group_split(g, cutoff.year))
    if stays["split_temporal"].isna().any():
        warnings.warn("missing anchor_year_group -> split_temporal='dev' for "
                      "those stays.")
        stays["split_temporal"] = stays["split_temporal"].fillna("dev")
    stays["split_random"] = _stratified_random_split(
        stays["y_critical"].to_numpy(),
        seed=int(cfg.get("data", {}).get("seed", 42)) + 1)

    keep = ["stay_id", "subject_id", "intime", "outtime",
            "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain",
            "acuity", "chiefcomplaint", "arrival_transport",
            "race", "gender", "insurance", "language", "anchor_age",
            "n_ed_visits_365d", "y_critical", "y_hosp", "y_revisit72",
            "y_death", "y_sepsis24h",
            "edlos_hours", "split_random", "split_temporal"]
    stays = stays[keep].sort_values("stay_id").reset_index(drop=True)
    stays["stay_id"] = stays["stay_id"].astype(np.int64)
    stays["subject_id"] = stays["subject_id"].astype(np.int64)
    return stays


def _year_group_split(group: object, cutoff_year: int) -> str | float:
    """Map anchor_year_group (e.g. '2014 - 2016') to dev/test by cutoff year."""
    years = re.findall(r"\d{4}", str(group))
    if len(years) < 2:
        return np.nan
    return "dev" if int(years[1]) < cutoff_year else "test"


def _count_prior_ed_visits(ed: pd.DataFrame) -> pd.Series:
    """# of prior ED stays for the same subject within 365 days before intime.

    Returns a Series indexed by stay_id (safe to .map() onto any row order).
    """
    ed = ed.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    out = np.zeros(len(ed), dtype=np.int64)
    for _, grp in ed.groupby("subject_id", sort=False):
        times = grp["intime"].to_numpy()
        pos = np.arange(len(grp))
        lo = np.searchsorted(times, times - np.timedelta64(365, "D"))
        out[grp.index.to_numpy()] = pos - lo  # prior stays within the window
    return pd.Series(out, index=ed["stay_id"].to_numpy())


# ------------------------------------------------------------------ sepsis24h
def _require_sepsis_files(mimic_dir: Path, hosp_dir: Path,
                          icu_dir: Path) -> None:
    """Fail fast when files needed for the Sepsis-3 approximation are absent."""
    missing: list[str] = []
    if not (mimic_dir / "pyxis.csv.gz").exists() \
            and not (mimic_dir / "medrecon.csv.gz").exists():
        missing.append("pyxis.csv.gz or medrecon.csv.gz (MIMIC-IV-ED; antibiotic "
                       "administration for the suspected-infection criterion)")
    for f in ("microbiologyevents.csv.gz", "labevents.csv.gz"):
        if not (hosp_dir / f).exists():
            missing.append(f"hosp/{f} (MIMIC-IV core)")
    if missing:
        raise NotImplementedError(
            "Cannot build y_sepsis24h (Sepsis-3 approximation, see cohort.py "
            "docstring): missing " + "; ".join(missing) + ". These files live "
            "in the MIMIC-IV v2.2 core download / MIMIC-IV-ED v2.2 — set "
            "data.mimic_core_dir (and data.mimic_dir) per docs/dua_runbook.md.")


def _window_mask(event_time: pd.Series, ref_time: pd.Series) -> pd.Series:
    """Boolean mask: event within [ref_time, ref_time + SEPSIS_WINDOW_H)."""
    dt_h = (event_time - ref_time).dt.total_seconds() / 3600.0
    return (dt_h >= 0) & (dt_h < SEPSIS_WINDOW_H)


def _suspected_infection(mimic_dir: Path, hosp_dir: Path,
                         stays: pd.DataFrame) -> pd.Series:
    """Boolean Series indexed by stay_id: antibiotic or blood culture <= 24h."""
    intime = stays.set_index("stay_id")["intime"]
    hit = pd.Series(False, index=intime.index)

    # (a) antibiotic administration: ED pyxis dispensation / medrecon entry.
    for fname in ("pyxis.csv.gz", "medrecon.csv.gz"):
        path = mimic_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {"stay_id", "charttime", "name"},
                         parse_dates=["charttime"])
        if "stay_id" not in df.columns:        # unexpected layout: skip
            warnings.warn(f"{fname} has no stay_id column; skipped for sepsis SI.")
            continue
        df = df[df["stay_id"].isin(intime.index)]
        df = df[df["name"].astype(str).str.contains(_ANTIBIOTIC_RE)]
        m = _window_mask(df["charttime"], df["stay_id"].map(intime))
        hit.loc[df.loc[m, "stay_id"].to_numpy()] = True

    # (b) blood culture order: hosp/microbiologyevents (keyed by subject_id).
    micro = pd.read_csv(hosp_dir / "microbiologyevents.csv.gz",
                        usecols=lambda c: c in {"subject_id", "charttime",
                                                "spec_type_desc"},
                        parse_dates=["charttime"])
    micro = micro[micro["subject_id"].isin(stays["subject_id"].unique())]
    micro = micro[micro["spec_type_desc"].astype(str).str.upper()
                  .str.contains("BLOOD", na=False)]
    micro = micro.merge(stays[["subject_id", "stay_id", "intime"]],
                        on="subject_id", how="inner")
    m = _window_mask(micro["charttime"], micro["intime"])
    hit.loc[micro.loc[m, "stay_id"].to_numpy()] = True
    return hit


def _sofa_labs(hosp_dir: Path, stays: pd.DataFrame) -> pd.DataFrame:
    """Worst SOFA-proxy lab values per stay within 24h (streamed labevents)."""
    itemid_to_col = {i: c for c, ids in SOFA_LAB_ITEMIDS.items() for i in ids}
    subjects = set(stays["subject_id"].unique())
    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(
        hosp_dir / "labevents.csv.gz",
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"], chunksize=5_000_000)
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(itemid_to_col)
                      & chunk["subject_id"].isin(subjects)]
        if len(chunk):
            kept.append(chunk)
    idx = stays["stay_id"].to_numpy()
    cols = ["platelet_min", "bilirubin_max", "creatinine_max"]
    if not kept:
        warnings.warn("no SOFA labevents rows matched -> lab SOFA axes = 0.")
        return pd.DataFrame(np.nan, index=idx, columns=cols)
    lab = pd.concat(kept, ignore_index=True)
    lab["lab"] = lab["itemid"].map(itemid_to_col)
    lab = lab.merge(stays[["subject_id", "stay_id", "intime"]],
                    on="subject_id", how="inner")
    lab = lab[_window_mask(lab["charttime"], lab["intime"]).to_numpy()]
    # Worst value per axis: min for platelet, max for bilirubin/creatinine.
    present = set(lab["lab"].unique())
    agg = pd.DataFrame(np.nan, index=idx, columns=cols)
    if "platelet" in present:
        agg["platelet_min"] = (lab[lab["lab"] == "platelet"]
                               .groupby("stay_id")["valuenum"].min())
    if "bilirubin" in present:
        agg["bilirubin_max"] = (lab[lab["lab"] == "bilirubin"]
                                .groupby("stay_id")["valuenum"].max())
    if "creatinine" in present:
        agg["creatinine_max"] = (lab[lab["lab"] == "creatinine"]
                                 .groupby("stay_id")["valuenum"].max())
    return agg


def _vasopressor_stays(icu_dir: Path, stays: pd.DataFrame) -> set:
    """stay_ids with a vasopressor infusion starting within 24h of triage.

    Optional input: when icu/inputevents.csv.gz is absent the vasopressor
    axis is silently 0 (documented degradation; SOFA cardio then relies on
    SBP<100 only). Provide the file for full-fidelity sepsis24h labels."""
    if not (icu_dir / "inputevents.csv.gz").exists():
        import warnings
        warnings.warn(
            "icu/inputevents.csv.gz absent: vasopressor axis of the SOFA "
            "proxy degraded to 0 (SBP<100 criterion retained). Provide "
            "inputevents.csv.gz for full-fidelity y_sepsis24h.", stacklevel=2)
        return set()
    subjects = set(stays["subject_id"].unique())
    hit: set = set()
    reader = pd.read_csv(
        icu_dir / "inputevents.csv.gz",
        usecols=["subject_id", "itemid", "starttime"],
        parse_dates=["starttime"], chunksize=5_000_000)
    win = stays[["subject_id", "stay_id", "intime"]]
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(VASOPRESSOR_ITEMIDS)
                      & chunk["subject_id"].isin(subjects)]
        if not len(chunk):
            continue
        chunk = chunk.merge(win, on="subject_id", how="inner")
        m = _window_mask(chunk["starttime"], chunk["intime"])
        hit.update(chunk.loc[m, "stay_id"].to_numpy())
    return hit


def _sofa_proxy(mimic_dir: Path, hosp_dir: Path, icu_dir: Path,
                stays: pd.DataFrame) -> pd.Series:
    """Integer SOFA-proxy score (0..5; neuro axis omitted -> 0) per stay_id."""
    intime = stays.set_index("stay_id")["intime"]
    idx = pd.Index(stays["stay_id"].to_numpy())
    sofa = pd.Series(0, index=idx, dtype=np.int64)

    # --- respiration + SBP axis: worst vitalsign within 24h, plus the triage row
    vs = pd.read_csv(mimic_dir / "vitalsign.csv.gz", parse_dates=["charttime"])
    vs = vs[vs["stay_id"].isin(intime.index)]
    for col in ("resprate", "o2sat", "sbp"):
        vs[col] = pd.to_numeric(vs[col], errors="coerce")
    vs = vs[_window_mask(vs["charttime"], vs["stay_id"].map(intime)).to_numpy()]
    agg = vs.groupby("stay_id").agg(o2sat_min=("o2sat", "min"),
                                    rr_max=("resprate", "max"),
                                    sbp_min=("sbp", "min"))
    tri = stays.set_index("stay_id")
    tri_o2 = pd.to_numeric(tri["o2sat"], errors="coerce").reindex(idx)
    tri_rr = pd.to_numeric(tri["resprate"], errors="coerce").reindex(idx)
    tri_sbp = pd.to_numeric(tri["sbp"], errors="coerce").reindex(idx)
    o2_min = pd.concat([agg["o2sat_min"].reindex(idx), tri_o2], axis=1).min(axis=1)
    rr_max = pd.concat([agg["rr_max"].reindex(idx), tri_rr], axis=1).max(axis=1)
    sbp_min = pd.concat([agg["sbp_min"].reindex(idx), tri_sbp], axis=1).min(axis=1)

    resp = ((o2_min < 94) | (rr_max >= 22)).fillna(False)
    sofa += resp.astype(np.int64)

    vaso = pd.Series(idx.isin(_vasopressor_stays(icu_dir, stays)), index=idx)
    cardio = ((sbp_min < 100).fillna(False)) | vaso
    sofa += cardio.astype(np.int64)

    labs = _sofa_labs(hosp_dir, stays).reindex(idx)
    sofa += (labs["platelet_min"] < 150).fillna(False).astype(np.int64)
    sofa += (labs["bilirubin_max"] >= 1.2).fillna(False).astype(np.int64)
    sofa += (labs["creatinine_max"] >= 1.2).fillna(False).astype(np.int64)
    # neuro axis: GCS unavailable in the ED module -> 0 points (see docstring).
    return sofa


def _build_sepsis24h(mimic_dir: Path, hosp_dir: Path, icu_dir: Path,
                     stays: pd.DataFrame) -> pd.Series:
    """y_sepsis24h aligned with `stays` row order (int 0/1, never NaN)."""
    _require_sepsis_files(mimic_dir, hosp_dir, icu_dir)
    suspected = _suspected_infection(mimic_dir, hosp_dir, stays)
    sofa = _sofa_proxy(mimic_dir, hosp_dir, icu_dir, stays)
    label = suspected & (sofa >= 2)
    return stays["stay_id"].map(label).fillna(0).astype(np.int64)


# --------------------------------------------------------------------- hourly
def _build_hourly(mimic_dir: Path, hosp_dir: Path,
                  stays: pd.DataFrame) -> pd.DataFrame:
    vs = pd.read_csv(mimic_dir / "vitalsign.csv.gz", parse_dates=["charttime"])
    intime = stays.set_index("stay_id")["intime"]
    vs = vs[vs["stay_id"].isin(intime.index)]
    for col in VITALSIGN_NUMERIC:
        vs[col] = pd.to_numeric(vs[col], errors="coerce")
    vs = _clean_vitals(vs, VITALSIGN_NUMERIC)
    vs["h"] = ((vs["charttime"] - vs["stay_id"].map(intime)).dt.total_seconds()
               / 3600.0).floordiv(1)
    vs = vs[(vs["h"] >= 0) & (vs["h"] < 24)]
    vs["h"] = vs["h"].astype(np.int64)

    # Multiple measurements within the same hour -> mean (documented MIMIC
    # practice of ~1 row/hour after aggregation).
    agg = (vs.groupby(["stay_id", "h"])[VITALSIGN_NUMERIC]
           .mean().reset_index())

    labs = _build_hourly_labs(hosp_dir, stays)
    if labs is not None:
        agg = agg.merge(labs, on=["stay_id", "h"], how="left")
    else:
        for col in LAB_ITEMIDS:
            agg[col] = np.nan

    agg = _clean_vitals(agg, [c for c in HOURLY_FEATS if c in agg.columns])
    hourly = agg[["stay_id", "h"] + HOURLY_FEATS].sort_values(
        ["stay_id", "h"]).reset_index(drop=True)
    hourly["stay_id"] = hourly["stay_id"].astype(np.int64)
    hourly["h"] = hourly["h"].astype(np.int64)
    return hourly


def _build_hourly_labs(hosp_dir: Path, stays: pd.DataFrame) -> pd.DataFrame | None:
    """Bucket hosp/labevents.csv.gz into ED-relative hours (h=0..23).

    Joined on subject_id with charttime in [intime, intime+24h) so labs are
    captured for ED stays with or without an associated hadm_id. labevents is
    large (~1e8 rows), so it is streamed in chunks and filtered by itemid.
    """
    lab_file = hosp_dir / "labevents.csv.gz"
    if not lab_file.exists():
        warnings.warn("hosp/labevents.csv.gz not found -> hourly labs NaN.")
        return None
    itemid_to_col = {i: c for c, ids in LAB_ITEMIDS.items() for i in ids}
    subjects = set(stays["subject_id"].unique())
    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(
        lab_file,
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"], chunksize=5_000_000)
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(itemid_to_col)
                      & chunk["subject_id"].isin(subjects)]
        if len(chunk):
            kept.append(chunk)
    if not kept:
        warnings.warn("no matching labevents rows -> hourly labs NaN.")
        return None
    lab = pd.concat(kept, ignore_index=True)
    lab["lab"] = lab["itemid"].map(itemid_to_col)

    win = stays[["subject_id", "stay_id", "intime"]]
    lab = lab.merge(win, on="subject_id", how="inner")
    lab["h"] = ((lab["charttime"] - lab["intime"]).dt.total_seconds()
                / 3600.0).floordiv(1)
    lab = lab[(lab["h"] >= 0) & (lab["h"] < 24)]
    lab["h"] = lab["h"].astype(np.int64)
    wide = (lab.groupby(["stay_id", "h", "lab"])["valuenum"].mean()
            .unstack("lab").reset_index())
    wide.columns.name = None
    for col in LAB_ITEMIDS:
        if col not in wide:
            wide[col] = np.nan
    return wide[["stay_id", "h"] + list(LAB_ITEMIDS)]


# ---------------------------------------------------------------------- entry
def build(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (stays, hourly) from real MIMIC-IV-ED + MIMIC-IV core csv dirs."""
    data_cfg = cfg.get("data", {})
    mimic_dir = Path(data_cfg.get("mimic_dir") or "")
    if not mimic_dir.is_dir():
        raise FileNotFoundError(
            f"config.data.mimic_dir={mimic_dir!s} is not a directory. "
            "MIMIC-IV-ED requires a signed PhysioNet DUA "
            "(docs/dua_runbook.md); use data.source=synthetic until then.")
    _check_dir(mimic_dir)
    hosp_dir, icu_dir = _resolve_core_dirs(data_cfg)
    _require_core(hosp_dir, icu_dir)
    stays = _build_stays(mimic_dir, hosp_dir, icu_dir, cfg)
    hourly = _build_hourly(mimic_dir, hosp_dir, stays)
    return stays, hourly
