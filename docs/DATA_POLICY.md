# Data and privacy policy

This repository is code only. Dataset access and software availability are separate matters.

The full MIMIC-IV-ED/MIMIC-IV resources require the applicable PhysioNet credentials and data-use agreements. eICU full and demonstration resources have their own access/license terms. Check the exact version's official page; a demo resource is not the full cohort. This package grants neither access nor redistribution rights and makes no claim that institutional approval or exemption has been obtained.

- [MIMIC-IV-ED v2.2](https://physionet.org/content/mimic-iv-ed/2.2/)
- [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/)
- [eICU-CRD](https://physionet.org/content/eicu-crd/)

Do not commit raw tables, patient/stay identifiers, shifted patient timestamps, free text, embeddings, row-level predictions, split manifests containing identifiers, fitted estimators/checkpoints, credentials, DUA/ethics documents, or private filesystem paths. De-identification does not make patient-level data unrestricted.

The scripts can create these artifacts during authorized local use. Treat all generated data, models, logs, and outputs as private until reviewed under the applicable agreement. Even an aggregate table requires disclosure review; this package intentionally includes no result CSVs.

Keep source data and generated artifacts outside the repository where practicable. Use `configs/*.local.yaml` for private paths; these files are ignored. Do not paste credentials into configs, terminal history, issues, or logs. No automatic authenticated dataset download is supplied.

The historical extraction checker may display identifiers or table samples when run against private data. Its output is not a publication artifact. Do not attach such output to public issues.

`.gitignore` does not remove tracked files or cleanse Git history. If sensitive material has already been committed, stop publication and follow the data provider's and institution's incident procedures. Do not assume deleting the latest copy is sufficient. The release checker is an additional heuristic check, not a guarantee or data-use approval.
