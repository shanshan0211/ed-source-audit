"""Synthetic ED data generator (default data source; PhysioNet MIMIC-IV-ED needs a DUA).

Data-generating process (DGP) — SPEC §9 discipline
==================================================
Latent clinical severity ``S`` per stay: mixture 80% N(0,1) + 20% N(0.8,1)
(a sicker subpopulation). Everything clinical flows from ``S``:

* Triage vitals (C group) are noisy linear readouts of ``S`` with realistic
  directions: heartrate/resprate/temperature rise with S; o2sat/sbp/dbp fall.
  Age adds small independent shifts to heartrate/sbp.
* Hourly trajectory: S_0 = S + eps; S_h = rho*S_{h-1} + (1-rho)*S_target + noise,
  where S_target = 0.85*S for non-critical stays (treatment works) and 1.05*S
  for critical stays (decompensation). Hourly vitals are re-generated from S_h
  with measurement noise. Monitoring intensity rises with S (more missing
  values for low-acuity patients). Labs (lactate/wbc/creatinine/glucose) are
  drawn at h=0 with probability increasing in S and repeated at most every 6h,
  so labs are massively NaN in hourly.parquet (realistic).
* Acuity / ESI (J group): a noisy rounding of a linear map of S, i.e. acuity is
  correlated with the outcomes AND partially reconstructable from the vitals
  (redundant channel):  pre-2017  esi = round(clip(3.1 - 1.15*S + N(0,0.75))).
  ~20% of acuity values are set NaN (MCAR-ish), pain ~30% NaN, each other
  triage vital ~4% NaN.

**Temporal drift (2017-01-01 onward)**: triage practice changes — the
conditional distribution p(acuity | clinical state) shifts: the slope on S is
compressed (1.15 -> 0.35), the intercept drops (3.1 -> 2.7, "acuity creep")
and the noise grows (0.75 -> 1.0). Outcomes still depend on S exactly as
before, so models leaning on J degrade under temporal extrapolation while
C-driven models transfer. Dev/test acuity means therefore differ visibly.

* Process traces (P group): arrival_transport sampled from a softmax whose
  AMBULANCE/HELICOPTER scores rise with S (mostly a proxy) and additionally
  enters outcome logits with a small independent weight; n_ed_visits_365d ~
  Poisson(0.9) independent of S, entering outcome logits with a small weight —
  both carry a *small independent* signal.
* Demographics (D group): age ~ clipped N(55,20) with small direct outcome
  effects; gender/race/language independent of S; insurance age-dependent.
* Latent infection ``I`` ~ Bern(sigmoid(-1.1 + 0.5*S)) (~26% of stays; sicker
  patients are more often infected). ``I`` manifests in the observable data:
  triage vitals are shifted (temperature +1.1, heartrate +12, resprate +2.5,
  sbp -9, dbp -5 — fever/tachycardia/tachypnea/hypotension), hourly vitals
  get the same shifts, and hourly wbc/lactate are raised (+6 / +1.5) whenever
  drawn. The triage nurse sees the manifested severity, so acuity is drawn
  from S_obs = S + 0.7*I (nurses spot the sepsis prodrome -> acuity is
  correlated with y_sepsis24h; the 2017 drift discipline is unchanged — only
  the acuity mechanism drifts, never the outcome mechanisms).
* Outcomes (driven mainly by the clinical signal; mechanisms time-invariant):
    y_death    = Bern(sigmoid(-5.9 + 2.2*S + 0.30*AMB + 0.025*(age-55)))
      (in-hospital death; rate ~2-5%; a MORE EXTREME tail of S than
      y_critical — the death-only sensitivity endpoint of research-plan §2/§5,
      since ICU transfer may be a direct downstream of ESI. By construction
      y_death is a subset of y_critical, mirroring the MIMIC label definition
      "ICU transfer within 12h OR in-hospital death": y_critical = drawn | y_death)
    y_critical = Bern(sigmoid(-2.6 + 1.5*S + 0.35*AMB + 0.10*n_ed + 0.015*(age-55)))
      OR y_death   (rate ~20%)
    y_hosp     = Bern(sigmoid(-0.9 + 1.2*S + 0.30*AMB + 0.12*n_ed + 0.020*(age-55)))
    y_revisit72 = Bern(sigmoid(-1.9 - 0.30*S + 0.30*n_ed)) AND discharged
    y_sepsis24h = Bern(sigmoid(-3.1 + 0.9*S + 1.7*I + 0.010*(age-55)))
      (Sepsis-3-style 24h label, rate ~8-15%; driven by S + the infection
      latent whose temperature/wbc/lactate signature keeps the clinical
      branch dominantly predictive — secondary task benchmarked against
      WJEM 2024 Xie J, AUC 0.92)
    edlos_hours = clip(N(3.2 + 3.5*y_hosp + 1.2*relu(S) + 0.10*n_ed, 1.6), 0.5, 72)
* chiefcomplaint: Gumbel-argmax over 20 complaints whose scores are
  base_i + gamma_i*S (severe complaints have gamma>0); ~4% empty.

Splits: split_random = 60/20/20 train/val/test stratified on y_critical;
split_temporal: dev = intime < 2017-01-01, test = 2017-01-01..2019-12-31.

Output schema matches SPEC §1.1/§1.2 exactly (stays/hourly); chiefcomplaint
strings live in the stays table — embeddings are produced by text_embed.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .api import HOURLY_FEATS

# (complaint, base_score, gamma_loading_on_S)
_COMPLAINTS: list[tuple[str, float, float]] = [
    ("chest pain", 0.2, 0.7),
    ("shortness of breath", 0.0, 0.8),
    ("altered mental status", -1.6, 1.0),
    ("overdose", -2.0, 0.9),
    ("weakness", -0.9, 0.5),
    ("fall", -0.4, 0.3),
    ("abdominal pain", -0.1, 0.1),
    ("fever", -0.3, 0.3),
    ("dizziness", -0.5, 0.2),
    ("nausea vomiting", -0.6, 0.1),
    ("palpitations", -1.0, 0.4),
    ("headache", -0.3, 0.0),
    ("back pain", -0.4, -0.1),
    ("cough", -0.3, 0.0),
    ("laceration", -0.8, -0.2),
    ("urinary symptoms", -1.0, 0.0),
    ("rash", -1.2, -0.3),
    ("sore throat", -1.0, -0.3),
    ("tooth pain", -1.6, -0.4),
    ("ankle sprain", -1.4, -0.3),
]

TEMPORAL_CUTOFF = pd.Timestamp("2017-01-01")

# Observable manifestation of the latent infection indicator I (see docstring):
# additive shifts applied to triage AND hourly vitals, and to hourly labs.
INFECT_VITAL_SHIFT = {"heartrate": 12.0, "resprate": 2.5, "o2sat": -1.2,
                      "sbp": -9.0, "dbp": -5.0, "temperature": 1.1}
INFECT_LAB_SHIFT = {"wbc": 6.0, "lactate": 1.5}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _stratified_random_split(y: np.ndarray, seed: int,
                             p_train: float = 0.6, p_val: float = 0.2) -> np.ndarray:
    """60/20/20 train/val/test split stratified on binary y."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(y), dtype=object)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * p_train))
        n_va = int(round(n * p_val))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    return split


def _vitals_from_S(S: np.ndarray, age: np.ndarray, rng: np.random.Generator,
                   noise_scale: float = 1.0) -> dict[str, np.ndarray]:
    """Map latent severity to the 6 vital signs (triage & hourly share formulas)."""
    ns = noise_scale
    return {
        "heartrate": 82 + 16 * S + 0.15 * (age - 55) + rng.normal(0, 7 * ns, len(S)),
        "resprate": 16.5 + 3.5 * S + rng.normal(0, 2.0 * ns, len(S)),
        "o2sat": np.clip(97.5 - 2.2 * S + rng.normal(0, 1.1 * ns, len(S)), 70, 100),
        "sbp": 128 - 14 * S + 0.30 * (age - 55) + rng.normal(0, 12 * ns, len(S)),
        "dbp": 76 - 7 * S + rng.normal(0, 8 * ns, len(S)),
        "temperature": np.clip(36.9 + 0.5 * S + rng.normal(0, 0.4 * ns, len(S)), 33, 42),
    }


def _labs_from_S(S: np.ndarray, age: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "lactate": np.clip(np.exp(0.35 + 0.45 * S + rng.normal(0, 0.35, len(S))), 0.3, 20),
        "wbc": np.clip(8 + 3.5 * S + rng.normal(0, 2.5, len(S)), 0.5, 60),
        "creatinine": np.clip(1.0 + 0.35 * np.maximum(S, 0) + 0.004 * (age - 50)
                              + rng.normal(0, 0.25, len(S)), 0.2, 15),
        "glucose": np.clip(105 + 18 * np.maximum(S, 0) + rng.normal(0, 25, len(S)), 40, 600),
    }


def _draw_acuity(S: np.ndarray, post_2017: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """ESI from latent severity; conditional distribution drifts after 2017-01-01."""
    latent = np.where(
        post_2017,
        2.7 - 0.35 * S + rng.normal(0, 1.00, len(S)),   # drifted triage practice
        3.1 - 1.15 * S + rng.normal(0, 0.75, len(S)),   # historical triage practice
    )
    return np.clip(np.round(latent), 1, 5).astype(float)


def _draw_arrival_transport(S: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cats = np.array(["WALK IN", "AMBULANCE", "HELICOPTER", "OTHER"], dtype=object)
    scores = np.stack([
        0.7 - 0.6 * S,
        0.0 + 0.5 * S,
        -3.5 + 1.0 * S,
        np.full_like(S, -2.5),
    ], axis=1)
    g = rng.gumbel(size=scores.shape)
    return cats[np.argmax(scores + g, axis=1)]


def _draw_complaints(S: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    names = np.array([c[0] for c in _COMPLAINTS], dtype=object)
    base = np.array([c[1] for c in _COMPLAINTS])
    gamma = np.array([c[2] for c in _COMPLAINTS])
    scores = base[None, :] + gamma[None, :] * S[:, None]
    g = rng.gumbel(size=scores.shape)
    out = names[np.argmax(scores + g, axis=1)].copy()
    out[rng.random(len(S)) < 0.04] = ""
    return out


def generate(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (stays, hourly) DataFrames per SPEC §1.1/§1.2."""
    data_cfg = cfg.get("data", {})
    n = int(data_cfg.get("n_stays", 6000))
    seed = int(data_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    # ---- latent severity & demographics -------------------------------------
    S = np.where(rng.random(n) < 0.2,
                 rng.normal(0.8, 1.0, n), rng.normal(0.0, 1.0, n))
    age = np.clip(rng.normal(55, 20, n), 18, 101)
    gender = rng.choice(["M", "F"], n)
    race = rng.choice(["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"], n,
                      p=[0.60, 0.18, 0.10, 0.06, 0.06])
    insurance = np.where(
        (age >= 65)[:, None],
        rng.choice(["Medicare", "Medicaid", "Private", "Other"], (n, 4),
                   p=[0.75, 0.05, 0.15, 0.05]),
        rng.choice(["Medicare", "Medicaid", "Private", "Other"], (n, 4),
                   p=[0.08, 0.25, 0.55, 0.12]),
    )[:, 0]
    language = rng.choice(["English", "Spanish", "Other"], n, p=[0.85, 0.08, 0.07])

    # ---- timestamps: uniform over 2011-01-01 .. 2019-12-31 -------------------
    t0 = pd.Timestamp("2011-01-01")
    n_hours = (pd.Timestamp("2020-01-01") - t0).total_seconds() / 3600.0
    intime = pd.to_datetime(t0 + pd.to_timedelta(rng.uniform(0, n_hours, n), unit="h"))
    post_2017 = np.asarray(intime >= TEMPORAL_CUTOFF)

    # ---- process traces -------------------------------------------------------
    arrival_transport = _draw_arrival_transport(S, rng)
    is_amb = (arrival_transport == "AMBULANCE").astype(float)
    n_ed_visits = np.minimum(rng.poisson(0.9, n), 30)

    # ---- latent infection (sepsis substrate; manifests in vitals/labs) --------
    infect = (rng.random(n) < _sigmoid(-1.1 + 0.5 * S)).astype(np.float64)

    # ---- outcomes (mainly clinically driven) ---------------------------------
    # y_death first: in-hospital death is by definition a subset of y_critical
    # (SPEC §1.1: y_critical = 12h ICU transfer OR in-hospital death).
    p_death = _sigmoid(-5.9 + 2.2 * S + 0.30 * is_amb + 0.025 * (age - 55))
    y_death = (rng.random(n) < p_death).astype(np.int64)
    p_crit = _sigmoid(-2.6 + 1.5 * S + 0.35 * is_amb + 0.10 * n_ed_visits
                      + 0.015 * (age - 55))
    y_critical = ((rng.random(n) < p_crit) | (y_death == 1)).astype(np.int64)
    p_hosp = _sigmoid(-0.9 + 1.2 * S + 0.30 * is_amb + 0.12 * n_ed_visits
                      + 0.020 * (age - 55))
    y_hosp = (rng.random(n) < p_hosp).astype(np.int64)
    p_rev = _sigmoid(-1.9 - 0.30 * S + 0.30 * n_ed_visits)
    y_revisit72 = ((rng.random(n) < p_rev) & (y_hosp == 0)).astype(np.int64)
    p_sep = _sigmoid(-3.1 + 0.9 * S + 1.7 * infect + 0.010 * (age - 55))
    y_sepsis24h = (rng.random(n) < p_sep).astype(np.int64)
    edlos = np.clip(rng.normal(3.2 + 3.5 * y_hosp + 1.2 * np.maximum(S, 0)
                               + 0.10 * n_ed_visits, 1.6, n), 0.5, 72.0)
    outtime = intime + pd.to_timedelta(edlos, unit="h")

    # ---- triage vitals (S_0 = S + eps) ---------------------------------------
    S0 = S + rng.normal(0, 0.05, n)
    vit = _vitals_from_S(S0, age, rng, noise_scale=1.15)
    for k, shift in INFECT_VITAL_SHIFT.items():   # infection signature
        vit[k] = vit[k] + shift * infect
    pain = np.clip(rng.normal(4 + 0.4 * S, 2.8, n), 0, 10)

    # acuity: the nurse perceives the infection-manifested severity (S + 0.7*I);
    # the 2017 drift still applies to this acuity mechanism only.
    acuity = _draw_acuity(S + 0.7 * infect, post_2017, rng)
    acuity[rng.random(n) < 0.20] = np.nan          # ~20% missing
    pain[rng.random(n) < 0.30] = np.nan            # ~30% missing
    for k in vit:                                   # ~4% missing each
        vit[k][rng.random(n) < 0.04] = np.nan

    stays = pd.DataFrame({
        "stay_id": np.arange(n, dtype=np.int64),
        "subject_id": rng.integers(0, int(0.85 * n), n).astype(np.int64),
        "intime": intime,
        "outtime": outtime,
        "temperature": vit["temperature"],
        "heartrate": vit["heartrate"],
        "resprate": vit["resprate"],
        "o2sat": vit["o2sat"],
        "sbp": vit["sbp"],
        "dbp": vit["dbp"],
        "pain": pain,
        "acuity": acuity,
        "chiefcomplaint": _draw_complaints(S, rng),
        "arrival_transport": arrival_transport,
        "race": race,
        "gender": gender,
        "insurance": insurance,
        "language": language,
        "anchor_age": age,
        "n_ed_visits_365d": n_ed_visits.astype(np.int64),
        "y_critical": y_critical,
        "y_hosp": y_hosp,
        "y_revisit72": y_revisit72,
        "y_death": y_death,
        "y_sepsis24h": y_sepsis24h,
        "edlos_hours": edlos,
        "split_random": _stratified_random_split(y_critical, seed + 1),
        "split_temporal": np.where(post_2017, "test", "dev"),
    })

    # ---- hourly table ----------------------------------------------------------
    # Row count per stay = min(24, ceil(edlos)); h=0 mirrors triage vitals.
    rho = 0.75
    rows: list[tuple] = []
    for i in range(n):
        nh = int(min(24, max(1, math.ceil(edlos[i]))))
        s_traj = np.empty(nh)
        s_traj[0] = S0[i]
        s_target = S[i] * (1.05 if y_critical[i] else 0.85)
        for h in range(1, nh):
            s_traj[h] = rho * s_traj[h - 1] + (1 - rho) * s_target + rng.normal(0, 0.15)
        mon_prob = 0.70 + 0.25 * _sigmoid(np.asarray(S[i]))  # sicker -> more monitoring
        lab0 = rng.random() < (0.22 + 0.50 * _sigmoid(np.asarray(S[i])))
        lact_prob = 0.45 + 0.30 * _sigmoid(np.asarray(S[i]))
        age_i = np.asarray(age[i])
        for h in range(nh):
            if h == 0:
                hr, rr, o2 = vit["heartrate"][i], vit["resprate"][i], vit["o2sat"][i]
                sb, db, tp = vit["sbp"][i], vit["dbp"][i], vit["temperature"][i]
            elif rng.random() < mon_prob:
                v = _vitals_from_S(s_traj[h:h + 1], age_i, rng, noise_scale=0.7)
                hr = v["heartrate"][0] + INFECT_VITAL_SHIFT["heartrate"] * infect[i]
                rr = v["resprate"][0] + INFECT_VITAL_SHIFT["resprate"] * infect[i]
                o2 = v["o2sat"][0] + INFECT_VITAL_SHIFT["o2sat"] * infect[i]
                sb = v["sbp"][0] + INFECT_VITAL_SHIFT["sbp"] * infect[i]
                db = v["dbp"][0] + INFECT_VITAL_SHIFT["dbp"] * infect[i]
                tp = v["temperature"][0] + INFECT_VITAL_SHIFT["temperature"] * infect[i]
            else:
                hr = rr = o2 = sb = db = tp = np.nan
            labs_now = (h == 0 and lab0) or (h > 0 and h % 6 == 0 and lab0
                                             and rng.random() < 0.35)
            if labs_now:
                lb = _labs_from_S(s_traj[h:h + 1], age_i, rng)
                lac = (lb["lactate"][0] + INFECT_LAB_SHIFT["lactate"] * infect[i]
                       if rng.random() < lact_prob else np.nan)
                wbc = lb["wbc"][0] + INFECT_LAB_SHIFT["wbc"] * infect[i]
                cre, glu = lb["creatinine"][0], lb["glucose"][0]
            else:
                lac = wbc = cre = glu = np.nan
            rows.append((i, h, hr, rr, o2, sb, db, tp, lac, wbc, cre, glu))
    hourly = pd.DataFrame(rows, columns=["stay_id", "h"] + HOURLY_FEATS)
    hourly["stay_id"] = hourly["stay_id"].astype(np.int64)
    hourly["h"] = hourly["h"].astype(np.int64)

    return stays, hourly
