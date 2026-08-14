# Changelog

All notable public changes to this project will be documented here.

## [1.0.0] - 2026-08-14

### Added

- Public V2-2 dataset contract with 8,417 training and 2,217 fixed-validation records.
- Exact reconstruction of 188 sequence-derived features from external sequence pairs.
- Native XGBoost JSON and CatBoost CBM inference artifacts.
- Command-line prediction with traceable examples.
- Frozen per-record predictions, metric reproduction and target-cluster bootstrap.
- Unit tests, end-to-end repository verification and GitHub Actions CI.
- English and Chinese public documentation.
- Apache-2.0 software license, NOTICE and bilingual third-party/data-license attribution.
- Byte-level provenance for the four unmodified EasyDesign repository workbooks.

### Scientific status

- XGBoost designated as the primary model.
- Equal averaging retained as sensitivity analysis.
- OOF-weighted averaging retained as an exploratory comparison, not a verified improvement.
