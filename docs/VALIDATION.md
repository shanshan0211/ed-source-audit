# Local release validation

Checked on Windows on 2026-10-21 using Python 3.12.13. This records packaging tests, not the environment of the original clinical experiments.

- All **22 fabricated-data tests passed**: evaluation smoke tests, paired bootstrap and patient-split/feature-boundary regression tests, BH adjustment checks, guarded runner checks, explicit offline hashing, and release-checker checks.
- All 44 Python source/test/script files were parsed successfully, and `pyproject.toml` was read successfully.
- A wheel was built and installed into an isolated local verification directory with dependency resolution disabled. This checks local packaging; it is not a clean online installation of every optional dependency.
- The synthetic CI pipeline plan was checked using `--dry-run`. No actual end-to-end TLQ-T run, CUDA run, transformer download, LightGBM/XGBoost run, clinical-data extraction, or clinical model retraining was performed during packaging.
- The file inventory/SHA256 manifest and heuristic file/secret/path scan passed. No clinical result files or patient records were packaged. This does not replace human privacy/license review.
- The GitHub Actions workflow was prepared but has not run remotely. No repository, release tag, license, software DOI, or public upload was created.

Local test dependencies were NumPy 2.5.4, pandas 3.0.1, PyArrow 26.0.0, SciPy 1.18.1, scikit-learn 1.9.1, PyYAML 6.0.3, joblib 1.6.0, Matplotlib 3.11.2 and pytest 9.1.1. Other supported dependency versions/platforms have not been independently exercised here. Local verification environments and build artifacts are outside this release folder.
