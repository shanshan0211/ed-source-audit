# Public code-interface reference

This document describes the supplied implementation, not a preregistration or evidence that all intended experiments have been completed. Historical comments referring to SPEC sections concern engineering conventions. Scientific interpretation is governed by `docs/REPRODUCIBILITY.md` and the current manuscript, not historical automatic-report claims.

## 1. Cache interface

`wjem_ed_audit.data.api` reads `stays.parquet`, `hourly.parquet` and `text_emb.parquet` from an explicitly configured private cache. `stay_id` joins records. Stay tables contain `subject_id`, arrival/discharge times, triage vitals, ESI (`acuity`), categories and task labels. Hourly features are heart/respiratory rate, oxygen saturation, blood pressure, temperature, lactate, WBC, creatinine and glucose. Text embeddings use `stay_id` plus `emb_0` through `emb_767`. Inspect the API/extraction code for full schemas; no real cache is distributed.

## 2. Inputs

C = triage measurements, plus hourly measurements/text in TLQ-T; J = recorded ESI; P = transport, arrival hour/shifted weekday and prior ED visits; D = historical demographics including admission-linked insurance/language. Every original reduced combination retains D. These are implementation group labels, not causal categories or guaranteed triage-time availability.

## 3. Models and tasks

Flat models: LR, XGBoost, LightGBM, modified NEWS2/qSOFA. TLQ-T uses source-specific query poolers; the configured query-group sizes sum to 28. Tasks are `critical`, `hosp`, `revisit72`, `edlos`, `death`, and `sepsis24h`. Only EDLOS is regression. Some tasks/windows are unsuitable for arrival-time interpretation; see the limitations document.

## 4. Private run layout

`runs/{task}/{model}/{tag}/seed{k}/` contains `metrics.json`, identifier-bearing `preds.parquet` and `config.json`; TLQ-T can also create checkpoints. A `temporal/` subfolder denotes the historical anchor-period partition, not verified calendar-time validation. These generated run folders must not be uploaded.

## 5. Audit interfaces

E1 = retrained source combinations; E2 = field removal; E3 = fitted representation suppression and ESI probe; E4 = test-set permutation; E5 = anchor-period/ICU-anchored stress tests. Their quantities differ. Audit CSVs and `sds_csr.csv` are local outputs, not included release inputs. Patient-level uncertainty needs paired identifiers and appropriate patient/hospital clustering.

## 6. Evaluation

AUROC, AUPRC, Brier/ECE, calibration and decision curves describe different properties. Clinical raw point scores are not probabilities. Historical report/benchmark code remains an engineering artifact, not the revised paper or a matched cross-study comparison.

## 7. Reanalysis extension

`scripts/reanalyse_triage.py` uses a separate patient-disjoint protocol and explicit feature allowlist. It has fabricated-data tests; no real-data reanalysis is shipped. It deliberately does not overwrite a pre-existing output directory.

## 8. Running and release

Run from the repository root. `python scripts/run_pipeline.py --config configs/ci.yaml --dry-run` prints a historical-pipeline plan using the active interpreter. Real-data configs require `--allow-restricted-data`. No global Python path or parent `.venv` is assumed. Use `python scripts/check_release.py` before publication, together with manual privacy and license review.
