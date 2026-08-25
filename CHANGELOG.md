# Changelog

All notable public changes to this project will be documented here.

## [2.0.0] - 2026-08-25

### Added

- Default mapping-aware D+B+C prediction system with frozen 20%/47%/33% weights.
- Automatic all-candidate exact/IUPAC mapping against 198 frozen EasyDesign Table S2 templates.
- Training-only guide and mapping history references with explicit seen/unseen support counts.
- Sequence-only D fallback for unmapped rows and a strict `--fallback-policy error` option.
- Route-aware ranking, mapping diagnostics, component outputs and machine-readable warnings.
- Complete coarse, fine and cross-fitted D/B/C weight grids, prediction oracles and uncertainty evidence.

### Changed

- `v2` is now the default; v1.5 D and v1.0 XGBoost remain explicit comparison routes.
- User-supplied mapping columns are rejected because v2 calculates mapping from aligned targets.
- A mixed full/fallback table withholds global ranking unless `--allow-mixed-ranking` is supplied.

### Scientific status

- Cross-fitted meta-OOF SCC/PCC are 0.8213/0.8133; historical fixed SCC/PCC are 0.8462/0.8355.
- Cross-fitted SCC improvement over the previous A+B+C system has a paired target-cluster interval above zero; the PCC interval slightly crosses zero.
- Fine reweighting does not beat the locked 20%/47%/33% replacement, so fixed-validation labels are not used to choose new weights.

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
