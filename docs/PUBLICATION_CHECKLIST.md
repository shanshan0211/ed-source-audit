# Before publishing on GitHub

This is a local code-release candidate. Repository creation/upload and data/model publication have not been performed.

- Confirm ownership and permission to redistribute the code, including the original experiment developer's contributions and any institutional/third-party rights.
- Choose and add an appropriate software `LICENSE`. None has been selected on the authors' behalf. Do not call this an open-source release before this step.
- Confirm any required ethics/data-use wording with the institution; public code does not imply authorized source-data access or an institutional waiver.
- Upload only this folder's code/documentation. Exclude the parent workspace, clinical/synthetic result artifacts, source exports, cached records/embeddings, patient predictions, weights, local environments, and private logs.
- Read `README.md`, `docs/DATA_POLICY.md` and `docs/REPRODUCIBILITY.md`. Do not describe the code as a completed strict triage-time, causal, preregistered, matched-input, or ED externally validated pipeline.
- Install the intended dependencies, run `python -m pytest -q`, and record your actual environment. Test success is not verification of clinical results.
- Run `python scripts/check_release.py` and inspect the files manually. The checker uses filename/type checks and secret-pattern heuristics; it cannot certify privacy, license compliance, or scientific validity.
- If you intentionally change public code/docs (for example by adding `LICENSE`), rerun `python scripts/check_release.py --write-manifest`, then inspect the changes. Do not use it to approve unreviewed data files.
- If using Git, inspect `git status --short` and `git diff --cached --stat` before committing. Do not force-add ignored files. Inspect any existing history; `.gitignore` is not retrospective cleanup.
- Check GitHub Actions after upload. The included workflow has only been prepared locally; no remote CI run is claimed.
- Once a real repository URL and immutable version/tag exist, update the manuscript's code-availability statement. Add a software DOI only if actually archived. Do not fabricate URLs, release tags, or DOIs.
