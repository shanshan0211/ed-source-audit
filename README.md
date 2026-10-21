# ED information-source audit

Research code accompanying the working manuscript **Information-Source Dependence in Emergency Department Prediction Models: A Retrospective Audit**.

[中文说明](README_zh.md)

This is a **code-only release candidate**, not a validated clinical tool. No patient records, real-data caches, individual predictions, embeddings, fitted models, experiment outputs, or manuscript drafts are included. This folder was prepared locally; it has not been uploaded or assigned a public repository URL.

## What this code does

The historical pipeline groups inputs as clinical measurements/text (C), recorded ESI (J), process features (P), and demographics (D). It implements flat baselines, the Triage Latent-Query Transformer (TLQ-T), retrained input ablations, representation interventions, ESI probes, test-set permutations, and stress tests. Retrained substitutability and sensitivity of a fitted model are different questions; neither estimates an input's causal effect.

Important limits of the analyses are retained, not repaired by packaging: 24-hour TLQ-T inputs overlap the 12-hour ICU outcome component; the death component has no uniform 12-hour horizon; flat models and TLQ-T have unmatched inputs; some demographics are admission-linked; the historical split/preprocessing and training snapshots have limitations. The anchor-period partition is not verified calendar-time validation, and the ICU-anchored eICU experiment is not external validation of ED triage or TLQ-T. See [reproducibility and limitations](docs/REPRODUCIBILITY.md).

The newer `scripts/reanalyse_triage.py` is a **post-analysis reanalysis tool** with patient-disjoint splitting and training-only preprocessing. It has fabricated-data tests, but has not been run on the real patient data in the supplied workspace. It is not evidence of completed new clinical experiments, a preregistration, or a matched-input TLQ-T comparison.

## Install

Use Python 3.10 or newer (3.11/3.12 recommended). From this repository root:

```bash
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The minimal installation supports the tests and LR reanalysis. For the historical model pipeline:

```bash
python -m pip install -e ".[models,text,dev]"
```

Install a suitable PyTorch build for your hardware if needed. LightGBM may need a platform OpenMP runtime. Dependency ranges are installation guidance, **not the recovered environment of the historical experiments**. No weights or third-party model files are redistributed.

## Quick checks without clinical data

```bash
python -m pytest -q
python scripts/check_release.py
python scripts/run_pipeline.py --config configs/ci.yaml --dry-run
```

Tests fabricate their own small examples. They do not validate clinical performance. Running tests requires the minimal installation above; running the full synthetic pipeline additionally requires the `models` extra.

```bash
python scripts/run_pipeline.py --config configs/ci.yaml
```

`ci.yaml` is a CPU-oriented synthetic configuration. Synthetic configurations explicitly use deterministic hashing text embeddings without downloading a language model. Synthetic scores have no clinical meaning. The runner uses the current Python interpreter, stops on errors, and refuses existing cache/output directories to reduce accidental overwriting or result mixing. A full audit still fits multiple models; it is not an instant unit test. `--dry-run` runs no pipeline steps.

## Authorized clinical-data work

Access to full MIMIC/eICU requires the applicable PhysioNet credentialing, agreements, and institutional determination. Obtain and store data yourself according to their terms; this release neither supplies data nor grants access. See the [data policy](docs/DATA_POLICY.md).

1. Copy `configs/default.yaml` to `configs/real.local.yaml`, and set your private data/cache/output paths. Keep configs inside `configs/`, because the audit CLI infers the repository root from that directory.
2. Read the known design limitations before using the historical pipeline. Running it does not automatically produce valid triage-time or matched-input results. Real-data execution requires an explicit acknowledgement:

   ```bash
   python scripts/run_pipeline.py --config configs/real.local.yaml --allow-restricted-data --dry-run
   ```

3. For the distinct patient-disjoint tabular reanalysis, check an authorized local cache first:

   ```bash
   python scripts/reanalyse_triage.py --cache-dir /path/to/private/cache --dry-run
   ```

   To run LR reanalysis, choose an output directory that does not already exist:

   ```bash
   python scripts/reanalyse_triage.py --cache-dir /path/to/private/cache --models lr --output /path/to/private/new-analysis
   ```

Paths above are examples, not included datasets. Additional source-time verification remains necessary. The default reanalysis uses critical and death outcomes, eight input combinations, five seeds, and patient-clustered bootstrap; its scope and exclusions are described in [reproducibility](docs/REPRODUCIBILITY.md). Keep all generated artifacts private pending data-use/privacy review.

## Layout

```text
configs/                 Synthetic examples and historical real-data template
src/wjem_ed_audit/
  data/                  Extraction, synthetic fixtures, embeddings, datasets
  models/                Flat baselines, TLQ-T, training, probes
  audit/                 Ablations, permutations, interventions, contrasts
  eval/                  Metrics, plots, historical report generation
scripts/
  run_pipeline.py        Portable, guarded historical-pipeline runner
  reanalyse_triage.py    Patient-disjoint tabular reanalysis tool
  revision_evidence.py   Recompute seed-level E3 comparisons from aggregates
  revision_displays.py   Draw main-paper displays from aggregate CSVs
  check_mimic_dir.py     Inspect a local authorized ED export
  check_release.py       File/secret-pattern and manifest consistency checks
tests/                   Fabricated-data regression and packaging tests
docs/                    Data policy, limitations, publication checklist
SPEC.md                  Public code-interface reference
RELEASE_MANIFEST.json    Inventory and SHA256 checksums
```

The revision display/evidence scripts need aggregate inputs that are **not included**. They do not reconstruct those inputs from the published text. The historical automatic report is not the current manuscript; its cross-study benchmark table, probability treatment of modified clinical scores, and template claims must not be used as paper-ready conclusions.

## Publishing, citation, and license

Follow the [publication checklist](docs/PUBLICATION_CHECKLIST.md). Only publish this release folder, not its parent research workspace. `.gitignore` is a guardrail, not a privacy review and not protection for files already tracked in Git. The release checker is heuristic and does not certify absence of sensitive information.

The manuscript is a working draft; no publication DOI, public repository URL, software DOI, or verified archival version is available yet. Do not invent these identifiers. Refer to the manuscript title above when describing the project until a final citation is available.

**The authors have not yet selected a software license.** Public visibility alone does not grant an open-source license. No MIT/Apache license or third-party redistribution right has been assumed. Confirm ownership/third-party permissions and choose a license before advertising the release as open source.
