# Contributing

Thank you for helping improve Cas12a Activity Ranker. Contributions are welcome for bug fixes, documentation, portability, tests and clearly scoped research extensions.

## Before opening an issue

1. Search existing issues.
2. Run `python scripts/verify_repository.py`.
3. Record your operating system, Python version and the exact command that failed.
4. Do not upload unpublished biological sequences, private sample metadata or credentials.

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Pull requests

- Keep each pull request focused on one problem.
- Add or update tests when behavior changes.
- Preserve the frozen reference files; write new experiment outputs to a new directory.
- Do not change validation membership, labels or folds without describing it as a new protocol.
- Report model comparisons on the same records and metrics.
- Avoid causal or clinical claims not supported by the assay.
- Update English and Chinese documentation together when public behavior changes.

## Data and model changes

Changes to data, features, preprocessing or model artifacts must include:

- the source and transformation;
- row and feature counts;
- SHA-256 checksums;
- train/validation grouping rules;
- a regression test or independently reproducible output;
- an explicit statement of whether the change is confirmatory or exploratory.

Regenerate the public checksum manifest only after all files are final:

```bash
python scripts/generate_sha256_manifest.py
python scripts/verify_repository.py
```

## Reporting security or privacy problems

Follow [SECURITY.md](SECURITY.md). Do not post sensitive sequence or contributor information in a public issue.
