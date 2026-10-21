"""eICU-CRD demo -> ED-triage "re-anchored" flat feature matrix (research plan §3.3).

RE-ANCHORING PROTOCOL (verbatim material for the paper's Limitations section)
=============================================================================

Re-anchor assumption
--------------------
MIMIC-IV-ED features are measured at the ED triage time point; eICU-CRD has no
ED phase at all.  We approximate the triage anchor by **ICU admission + first
60 minutes** (observationoffset in [0, 60] minutes relative to unit admit) for
vital signs, and by the **first 24 h of ICU stay** (labresultoffset in
[0, 1440]) for the first-available lab value.  This treats "ICU admission" as
the observable endpoint of "ED triage -> disposition", i.e. the vitals we feed
the model are measured *after* the ED work-up has started (post-intervention,
possibly post-treatment), not at triage.  Any transfer AUC is therefore a
**lower bound** on the true cross-dataset transportability of the clinical
signal (plan §3.3: "定位为临床信号可迁移性的下限估计").

Spectrum (case-mix) bias
------------------------
The cohort is `unitadmitsource == 'Emergency Department'` ICU admissions: these
patients were sick enough to be admitted to an ICU, so the cohort is
enriched for severity relative to an all-comers ED triage population.  The
label prevalence (hospital mortality ~8%) and the feature distributions both
shift; AUC under such a shift is not directly comparable to the development
AUC even without any model deficiency.

Field mapping (eICU-CRD demo v2.0.1 -> MIMIC-side taxonomy)
-----------------------------------------------------------
All vitals pass through unit harmonisation / plausibility cleaning before
averaging: vitalPeriodic.temperature mixes Fahrenheit into an otherwise-Celsius
column in the demo (values > 45 are converted (F-32)/1.8), and artifact zeros /
negative cuff pressures are blanked (see _PLAUSIBLE).

C group (TRIAGE_VITALS, first-60-min means):
  temperature  <- vitalPeriodic.temperature            (Celsius after cleaning)
  heartrate    <- vitalPeriodic.heartrate              (bpm)
  resprate     <- vitalPeriodic.respiration            (breaths/min)
  o2sat        <- vitalPeriodic.sao2                   (%)
  sbp          <- vitalAperiodic.noninvasivesystolic   (mmHg; FALLBACK:
                    vitalPeriodic.systemicsystolic when no aperiodic reading in
                    window, flagged via flag_bp_periodic_fallback)
  dbp          <- vitalAperiodic.noninvasivediastolic  (same fallback rule)
  pain         <- NOT RECORDED in eICU -> entire column NaN (documented; the
                    LR pipeline imputes it with the *training* median, GBMs use
                    native NaN handling)
Context labs (first value within 24 h; extracted for the missingness report and
future TLQ-T-side use, NOT part of the flat clin_only matrix which per SPEC §3
contains triage vitals only):
  lactate      <- lab.labname ~ 'lactate'
  wbc          <- lab.labname ~ 'wbc x 1000'
  creatinine   <- lab.labname ~ 'creatinine'
  glucose      <- lab.labname ~ 'glucose' | 'bedside glucose'
D group:
  anchor_age   <- patient.age ('> 89' -> 91, unparseable/empty -> NaN)
  gender       <- patient.gender  (Male->M, Female->F, else UNKNOWN)
  race         <- patient.ethnicity via ETHNICITY_MAP (Caucasian->WHITE,
                    African American->BLACK, Hispanic->HISPANIC, Asian->ASIAN,
                    Native American->OTHER, Other/Unknown/NaN->UNKNOWN)
  insurance/language: NOT AVAILABLE in eICU demo -> dummy columns filled with 0
                    at alignment time (i.e. treated as "none of the training
                    levels"), documented limitation.

Differences vs the MIMIC-side definitions
-----------------------------------------
* Anchor: ED triage single measurement  vs  ICU-admission 60-min mean
  (eICU side is smoothed and later).
* SBP/DBP: ED triage cuff reading  vs  ICU non-invasive cuff (arterial-line
  values in vitalAperiodic pa* columns are NOT used; invasive monitoring would
  further shift the distribution).
* Labs: MIMIC side consumes them via the hourly sequence (TLQ-T), not the flat
  baseline; here they are context only.
* No ESI/acuity, no arrival_transport, no chiefcomplaint text, no insurance /
  language in eICU -> only the clin_only (C+D) feature path is transferable.

Outcomes
--------
y_death      <- hospitaldischargestatus == 'Expired'  (primary; NaN if status
                missing -- those stays are dropped at evaluation time)
y_death_unit <- unitdischargestatus == 'Expired'      (ICU mortality,
                sensitivity endpoint per plan §3.3)

Data: eICU-CRD demo v2.0.1, ODbL license (data/real_demo/eicu-crd-demo/).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wjem_ed_audit.data import api

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "real_demo" / "eicu-crd-demo"
CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "real_demo" / "eicu_reanchor.parquet"

COHORT = "eicu_crd_demo_v2.0.1"

# windows in minutes relative to unit admission (eICU *offset columns)
VITAL_WINDOW_MIN = (0, 60)
LAB_WINDOW_MIN = (0, 24 * 60)

# lab labname fuzzy (case-insensitive substring) mapping, first value in window
LAB_MAP: dict[str, tuple[str, ...]] = {
    "lactate": ("lactate",),
    "wbc": ("wbc x 1000",),
    "creatinine": ("creatinine",),
    "glucose": ("glucose", "bedside glucose"),
}

# eICU ethnicity -> MIMIC-side race levels (synth DGP levels: WHITE/BLACK/
# HISPANIC/ASIAN/OTHER; UNKNOWN comes from build_flat_matrix's fillna)
ETHNICITY_MAP: dict[str, str] = {
    "Caucasian": "WHITE",
    "African American": "BLACK",
    "Hispanic": "HISPANIC",
    "Asian": "ASIAN",
    "Native American": "OTHER",
    "Other/Unknown": "UNKNOWN",
}

GENDER_MAP: dict[str, str] = {"Male": "M", "Female": "F"}

# vitalPeriodic column -> MIMIC triage vital name
_PERIODIC_MAP = {"temperature": "temperature", "heartrate": "heartrate",
                 "respiration": "resprate", "sao2": "o2sat"}
_APERIODIC_MAP = {"noninvasivesystolic": "sbp", "noninvasivediastolic": "dbp"}
_PERIODIC_BP_FALLBACK = {"systemicsystolic": "sbp", "systemicdiastolic": "dbp"}

# Generous physiologic plausibility ranges (post unit harmonisation).  The demo
# mixes Fahrenheit into vitalPeriodic.temperature and contains artifact zeros /
# negative cuff pressures; out-of-range values are set to NaN before averaging.
_PLAUSIBLE = {"temperature": (25.0, 45.0), "heartrate": (20.0, 250.0),
              "respiration": (1.0, 60.0), "sao2": (50.0, 100.0),
              "noninvasivesystolic": (40.0, 300.0),
              "noninvasivediastolic": (10.0, 200.0),
              "systemicsystolic": (40.0, 300.0),
              "systemicdiastolic": (10.0, 200.0)}


def _clean_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise units and blank physiologically impossible values (in-place
    on a copy).  temperature > 45 is treated as Fahrenheit and converted."""
    df = df.copy()
    for col, (lo, hi) in _PLAUSIBLE.items():
        if col not in df:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if col == "temperature":
            f = (v - 32.0) / 1.8
            v = v.where(v <= 45.0, f)  # >45 C implausible -> assume Fahrenheit
        df[col] = v.where((v >= lo) & (v <= hi))
    return df

NOTE_PAIN = "pain 在 eICU 无对应字段 —— C__pain 整列 NaN（LR 以训练中位数填补，GBM 原生处理 NaN）"


# ------------------------------------------------------------------ pieces
def _parse_age(s: pd.Series) -> pd.Series:
    """eICU age is a string: '> 89' -> 91, numeric -> float, else NaN."""
    def one(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if v in ("", "> 89"):
            return 91.0 if v == "> 89" else np.nan
        try:
            return float(v)
        except ValueError:
            return np.nan
    return s.map(one).astype(np.float32)


def _window_mean(df: pd.DataFrame, col_map: dict[str, str],
                 stay_ids: np.ndarray, lo: int, hi: int) -> pd.DataFrame:
    """Per-stay mean of `col_map` source columns over offset in [lo, hi]."""
    w = df[(df["observationoffset"] >= lo) & (df["observationoffset"] <= hi)]
    g = w.groupby("patientunitstayid")[list(col_map)].mean()
    g = g.rename(columns=col_map).reindex(stay_ids)
    return g


def _first_lab(lab: pd.DataFrame, stay_ids: np.ndarray,
               lo: int, hi: int) -> pd.DataFrame:
    """First (smallest labresultoffset) value per stay per LAB_MAP analyte."""
    w = lab[(lab["labresultoffset"] >= lo) & (lab["labresultoffset"] <= hi)].copy()
    w["labname_lc"] = w["labname"].str.lower()
    out = pd.DataFrame(index=stay_ids)
    for feat, pats in LAB_MAP.items():
        m = w["labname_lc"].str.contains("|".join(pats), regex=True, na=False)
        sub = (w[m].sort_values("labresultoffset")
                 .drop_duplicates("patientunitstayid", keep="first")
                 .set_index("patientunitstayid")["labresult"])
        out[feat] = pd.to_numeric(sub, errors="coerce").reindex(stay_ids)
    return out


def build_eicu_reanchor(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Build the per-stay re-anchored frame from the raw eICU-CRD demo CSVs.

    Returns one row per ED-source ICU stay with: stay identifiers, flat C__/D__
    feature columns (matching api.build_flat_matrix(groups=("C","D")) naming),
    context lab columns (lab__*), flags (flag__*), and outcomes
    (y_death, y_death_unit; NaN when the discharge status is unknown).
    """
    raw_dir = Path(raw_dir or RAW_DIR)
    pat = pd.read_csv(raw_dir / "patient.csv.gz", low_memory=False)
    ed = pat[pat["unitadmitsource"] == "Emergency Department"].copy()
    ed = ed.sort_values("patientunitstayid").reset_index(drop=True)
    stay_ids = ed["patientunitstayid"].to_numpy()

    lo, hi = VITAL_WINDOW_MIN
    vp = pd.read_csv(raw_dir / "vitalPeriodic.csv.gz")
    vp = vp[vp["patientunitstayid"].isin(stay_ids)]
    vp = _clean_vitals(vp)
    vit = _window_mean(vp, _PERIODIC_MAP, stay_ids, lo, hi)

    va = pd.read_csv(raw_dir / "vitalAperiodic.csv.gz")
    va = va[va["patientunitstayid"].isin(stay_ids)]
    va = _clean_vitals(va)
    bp = _window_mean(va, _APERIODIC_MAP, stay_ids, lo, hi)
    # fallback: invasive/periodic systemic BP when no aperiodic cuff reading
    bp_fb = _window_mean(vp, _PERIODIC_BP_FALLBACK, stay_ids, lo, hi)
    flag_fallback = pd.DataFrame(index=stay_ids)
    for c in ("sbp", "dbp"):
        flag_fallback[c] = bp[c].isna() & bp_fb[c].notna()
        bp[c] = bp[c].fillna(bp_fb[c])
    vit = pd.concat([vit, bp], axis=1)

    lab = pd.read_csv(raw_dir / "lab.csv.gz")
    lab = lab[lab["patientunitstayid"].isin(stay_ids)]
    labs = _first_lab(lab, stay_ids, *LAB_WINDOW_MIN)

    out = pd.DataFrame({
        "stay_id": stay_ids,
        "subject_id": ed["patienthealthsystemstayid"].to_numpy(),
        "hospitalid": ed["hospitalid"].to_numpy(),
    })
    for v in api.TRIAGE_VITALS:
        if v == "pain":
            out["C__pain"] = np.float32(np.nan)  # not recorded in eICU
        else:
            out[f"C__{v}"] = vit[v].astype(np.float32).to_numpy()
    out["D__anchor_age"] = _parse_age(ed["age"]).to_numpy()
    gender = ed["gender"].map(GENDER_MAP).fillna("UNKNOWN")
    race = ed["ethnicity"].map(ETHNICITY_MAP).fillna("UNKNOWN")
    out = pd.concat([out.reset_index(drop=True),
                     pd.get_dummies(gender, prefix="D__gender").astype(np.float32),
                     pd.get_dummies(race, prefix="D__race").astype(np.float32)],
                    axis=1)
    for feat in LAB_MAP:
        out[f"lab__{feat}"] = labs[feat].astype(np.float32).to_numpy()
    out["flag__bp_periodic_fallback"] = (
        flag_fallback["sbp"] | flag_fallback["dbp"]).reindex(stay_ids).to_numpy()
    out["y_death"] = ed["hospitaldischargestatus"].map(
        {"Expired": 1.0, "Alive": 0.0}).to_numpy()
    out["y_death_unit"] = ed["unitdischargestatus"].map(
        {"Expired": 1.0, "Alive": 0.0}).to_numpy()
    return out


def feature_missingness(df: pd.DataFrame) -> pd.Series:
    """Missing rate per C__/D__ numeric feature + context labs (over cohort)."""
    cols = [c for c in df.columns if c.startswith(("C__", "lab__"))] + ["D__anchor_age"]
    return df[cols].isna().mean().sort_values(ascending=False)


def load_eicu_flat(raw_dir: str | Path | None = None,
                   cache_path: str | Path | None = None,
                   label_col: str = "y_death",
                   use_cache: bool = True):
    """Returns (X, y, ids) aligned with api.build_flat_matrix(groups=("C","D")).

    X: C__<7 triage vitals> (C__pain all-NaN, see NOTE_PAIN) + D__anchor_age +
    D__gender_*/D__race_* dummies.  D-group levels absent from eICU
    (insurance/language dummies) are NOT created here; the caller reindexes to
    the training columns and fills missing dummies with 0.
    y: float array (NaN where the discharge status is unknown -- drop before
    scoring).  ids: patientunitstayid.
    Side effects: writes/reads the parquet cache and prints the feature
    missingness table of the cohort.
    """
    cache_path = Path(cache_path or CACHE_PATH)
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = build_eicu_reanchor(raw_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    miss = feature_missingness(df)
    print(f"[eicu] cohort={COHORT}  ED-source stays={len(df)}  "
          f"y_death known={df['y_death'].notna().sum()} "
          f"(prevalence={df['y_death'].mean():.3f})  "
          f"y_death_unit known={df['y_death_unit'].notna().sum()} "
          f"(prevalence={df['y_death_unit'].mean():.3f})")
    print(f"[eicu] bp periodic fallback stays: "
          f"{int(df['flag__bp_periodic_fallback'].sum())}")
    print("[eicu] feature missing rate (fraction of stays):")
    print(miss.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"[eicu] note: {NOTE_PAIN}")
    feat_cols = ([f"C__{v}" for v in api.TRIAGE_VITALS] + ["D__anchor_age"]
                 + sorted(c for c in df.columns
                          if c.startswith(("D__gender_", "D__race_"))))
    X = df[feat_cols].astype(np.float32)
    y = df[label_col].to_numpy(dtype=np.float64)
    ids = df["stay_id"].to_numpy()
    return X, y, ids
