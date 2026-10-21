# Reproducibility scope and known limitations

## Three distinct scopes

| Component | What can be checked | What it does not establish |
|---|---|---|
| Fabricated-data tests and synthetic configs | Interfaces, shapes, pairing, statistical identities, pipeline execution | Clinical performance or reproduction of real-data estimates |
| Historical `src/` pipeline and `configs/default.yaml` | Supplied implementation and intended run structure | Recovery of the exact historical environment, checkpoints, or mixed training snapshots |
| `scripts/reanalyse_triage.py` | Patient-disjoint tabular workflow, exclusion allowlist, training-only preprocessing | Completed real-data reanalysis, strict source-time verification, ESI-only equivalence study, or matched-input TLQ-T comparison |

The supplied research workspace had real-data aggregate summaries but not its original real-data caches, individual predictions, or checkpoints. No such artifacts are shipped. Source availability alone cannot independently regenerate the paper's clinical numbers. The original seeds, configurations, exact input files, model/library versions, hardware, and training snapshots would need to be reconciled by the experiment owner.

## Historical analysis boundaries

- TLQ-T uses up to 24 hours of measurements and complaint text, unlike flat models' triage measurements. The main critical label includes ICU entry within 12 hours and index-encounter death without a uniform 12-hour death horizon. Input windows can overlap or extend beyond events; these are not strict triage-time predictions.
- Historical D includes insurance/language linked through hospital-admission records. These may encode admission-linked documentation rather than triage-available demographics.
- Random historical splitting is at the stay level; patients can overlap. Some TLQ-T preprocessing used held-out observations. The patient anchor-period partition does not establish a true visit-calendar split. Shifted weekday is not the original weekday.
- Performance summaries and E1–E5 summaries can represent different historical snapshots. Several TLQ-T runs used different training settings; they are not a fully matched frozen experiment.
- E1 retrains reduced-input models. E3 suppresses fitted representations; its random control uses C queries. E4 performs unconditional group permutations, using different units in flat models and TLQ-T. These interventions estimate different model-conditional quantities; neither is a causal effect or an architecture-only comparison.
- SDS/CSR/R are normalized AUROC contrasts, not information or causal shares. Only two of the four contrasts are independent. Seed variability is not patient-sampling uncertainty. The corrected paired bootstrap code does not retroactively repair saved historical intervals.
- NEWS2/qSOFA implementations are modified/incomplete and output raw points, not calibrated event probabilities. Historical report/plot code is retained for transparency, including legacy clipping/thresholding behavior. Do not interpret those clinical-score traces as valid probability-calibration or probability-threshold decision-curve comparisons.
- The eICU demonstration changes the population to ED-sourced ICU stays and measurement anchor to the first ICU hour. It is an ICU-anchored stress test for refitted flat models, not same-task ED external validation or TLQ-T validation.
- The infection label is a simplified proxy rather than validated Sepsis-3. Auxiliary dynamic outputs lack a validated outcome-free landmark risk set. Automatic report templates and cross-study benchmark comparisons are historical engineering outputs, not the current manuscript or evidence of superiority.

## Patient-disjoint reanalysis tool

`scripts/reanalyse_triage.py` needs a private `stays.parquet` conforming to the supplied data API. Run `--dry-run` first; this checks file availability, not all source timestamps or label validity. With no real cache, the script stops rather than substituting synthetic data.

The tool uses a fixed subject-disjoint 60/20/20 split; median imputation, category vocabularies and optional TF-IDF are fitted on training data only. It excludes admission-linked insurance/language, disposition, hourly records, and shifted weekday. Eight combinations are full, C+D, J+P+D, C+P+D, C+J+D, D-only, C-only, and J+D. These do **not** include ESI alone or constitute the full prespecified conditional-increment/equivalence study.

The default models are LR/LightGBM; XGBoost is optional. Thresholds are selected in validation. Paired bootstrap can cluster by patient. LR seeds need not generate different fits. The optional TF-IDF sensitivity is not historical Bio_ClinicalBERT and does not match the TLQ-T input budget. Underlying labels retain their historical definitions; the tool does not rebuild labels or verify extraction availability times.

Outputs include identifier-bearing predictions, split manifests, fitted pipelines and local paths. They must remain private. Successful fabricated-data tests are not real-data experiments.

## Existing-aggregate scripts

No aggregate CSVs are included. To use the scripts locally, supply separately reviewed aggregate files in the expected layout:

- `revision_evidence.py`: reads `results_real/audit/E3_repr_intervention.csv`; hashes that file plus `E1_branch_mask.csv`, `sds_csr.csv` and `results_real/tables/table1_performance.csv`. Recomputes only seed-level J/P-versus-control comparisons; refuses an existing output folder.
- `revision_displays.py`: reads `results_real/audit/E1_branch_mask.csv`, `E3_repr_intervention.csv`, `E5_transfer.csv` and `E5_eicu_real.csv`. Requires `pdflatex`, the `standalone` class and PGFPlots (compatibility level 1.18). Produces the source-audit display and an E5 display table; fits no models and estimates no new patient-level intervals. Its fixed `results_revision/figures_20261021` output can be overwritten; use a fresh checkout/output location when needed.

Do not populate the expected paths using synthetic outputs and then label them real clinical results.

## Changes made for this release candidate

Core scientific algorithms and original experiment results are not altered. Packaging adds `pyproject.toml`, a cross-platform pipeline runner, synthetic-only explicit hashing selection, source/file checks, CI configuration, and public-facing documentation. The smoke tests use an isolated temporary directory instead of a fixed `/tmp` directory. Examples no longer require a parent-directory virtual environment. Synthetic configuration wording is corrected and the unused `n_query` field is aligned to the implemented 16+4+4+4=28 queries.

Original local research files remain outside this release folder and are unchanged. See the manifest for release checksums and the local preparation record for original-source checksums. No public upload, clinical retraining, or real-data validation was performed during packaging.
