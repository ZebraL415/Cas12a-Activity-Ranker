# Changelog

All notable public changes to this project will be documented here.

## [1.5.0] - 2026-08-25

### Added

- Deployable D ensemble: 35% XGBoost, 59% LightGBM and 6% scaled MLP.
- `cas12a-ranker predict` table-in/table-out interface for CSV, TSV and XLSX.
- Continuous activity, within-file rank, component predictions, status, route and version columns.
- Row-level validation reports and an explicit `--on-invalid keep` triage mode.
- Bundled four-row minimal dataset and `cas12a-ranker self-test`.
- Frozen OOF/fixed predictions, complete 0.01-step weight grid and model/environment metadata.
- Independent v1.5 metric and target-cluster uncertainty reproduction.

### Changed

- SCC/Spearman and PCC/Pearson are now co-primary evaluation outcomes.
- The D ensemble replaces v1.0 XGBoost as the default prediction route.
- Documentation is organized around a new user's file-based workflow.
- v1.0 model outputs remain available as deprecated compatibility columns.

### Scientific status

- v1.5 D reaches SCC 0.7709 and PCC 0.7538 on historical fixed validation.
- Weights were selected on frozen target-group OOF predictions without fixed-validation labels.
- The historical fixed split is not claimed as untouched external validation.

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
