# R09 source notes

## Scope

This provenance pass starts from the published EasyDesign article and its supporting workbook, follows the local Table S3 extraction into the current canonical table, and distinguishes label construction from the public prediction interface. Only a DOI and a public repository identifier were used in the public-source checks. Private manuscript/model files and label values were not sent to an external service.

## New, directly verified findings

1. The published workbook's Table S3 has 10,634 data rows. The header row identifies separate columns for 30 min and 20 min normalized. Every Table S3 30-min value is negative, ranging from −4.20319 to −0.306893.
2. The current canonical table contains 10,634 matching Table S3 record IDs. For every record, label_raw_value and label_normalized exactly equal the source Table S3 30 min value. No numerical transformation is applied by the project extraction step.
3. Table S4's first 21,268 data rows are an exact sequential expansion of the two Table S3 labels. For each Table S3 record i, Table S4 No.=2i−1 has the same guide/target and its out_logk_measurement equals 30 min; Table S4 No.=2i has the same guide/target and equals 20 min normalized. The full check found zero mismatches for all 10,634 records in both mappings. Table S4 has 10,725 additional rows after those paired entries.
4. The archived EasyDesign prediction wrapper uses a +4 shift and a lower bound of zero on regression-model predictions returned by the inference interface. This code path does not establish that +4, a logarithm, or any other specific transform was applied to the published Table S3 labels.

## Evidence chain and boundaries

### Source assay and published processing

The article's Methods section “Quantification of Cas12a cleavage activity” reports a Varioskan LUX reader, 485-nm excitation, 535-nm emission, 37 °C, and 2-min readings during a 2-h experiment; it selects the 30-min fluorescence reading for activity assessment. “Predicting detection activity: Data processing” reports two no-template negative-control wells and two duplicate high-activity positive-control wells per 96-well plate. The stated operation is subtract the mean negative-control signal from each plate reading, then divide by the mean positive-control signal. The denominator is not the difference between positive and negative controls.

“Data augmentation” describes two labels for each guide–target pair: the fluorescence value recorded at 30 min, and a 20-min readout normalized to 30 min by

g(t1,t2) = [f(t1+2) − f(t1−2)]^((t1−t2)/4) × f(t2),

where f(x) is the fluorescence readout at x minutes. For the described 20-to-30-min conversion, t1=30 and t2=20. This time-folding rule concerns the auxiliary label. The present analysis does not apply or recompute it.

### Published workbook and canonical mapping

The article's supporting-information listing identifies the workbook as IMT2-3-e214-s001.xlsx and records a size of 2,339,574 bytes and MD5 367977dd723c8685e90d21d194f64d92. Both the published-file snapshot and the project source snapshot match that published file identity. This ties the inspected rows to the article's named supplement. Table S3 uses columns A–H, with 30 min in column E and 20 min normalized in column F; Table S4 uses columns A–D, with out_logk_measurement in column D. The local sheet has the title row in row 1, field headers in row 2, and data beginning in row 3.

The local extraction code reads Table S3 and assigns the numerical 30 min field to both label_raw_value and label_normalized. It separately reads Table S4's out_logk_measurement into a separate source role. The observed text label_scale_group=table_s3_log_or_transformed_30min_activity is a categorical grouping value; it is not a transform formula and must not be interpreted as proof of log2, log10, or another specific operation.

The canonical data file is `exports/reproduction/data/inputs/EasyDesign_2024_V2-2_core_context_feature_table.csv` relative to the complete delivery-package root. The row-level check joined on source_table_id and record_id and found exact equality between the workbook's 30 min field and both canonical label fields for all 10,634 Table S3 records.

### Official prediction repository is a separate stage

The local README instructs users to clone the author repository scRNA-Compt/EasyDesign, and the public repository landing page was verified. In the local source snapshot, easyDesign/utils/predict_activity.py sets regression_shift=4.0 and regression_lower_bound=0.0, then applies max(0, prediction+4) to regression-model predictions. The current project runner, run_easydesign.py lines 72 and 87, stores raw model.predict output for the EasyDesign-retrained evaluation and does not call this wrapper. A direct public fetch of the raw GitHub file was throttled (HTTP 429), and the local snapshot has no Git commit metadata. The +4/clipping evidence is therefore attributed to the inspected local source snapshot, not to a pinned current upstream revision. It is a prediction-output operation, not part of the local Table S3 extraction, and it does not explain the upstream numeric transformation of the source label.

### Metric interpretation and missing evidence

The current analysis uses the source Table S3 30-min values as y_true. MAE/RMSE and the other numeric errors are on that supplied label scale. The values are not described as RFU, RFU/min, a kinetic rate, or cellular editing efficiency. No plate, well, technical-replicate, or biological-replicate identity was inferred from record IDs. The inspected published tables contain sequence and label columns, not per-record raw fluorescence trajectories or plate/well mappings. The exact upstream transform after control correction and the independent experimental unit therefore remain unresolved.

## Source locators

- Article and supplementary-information entry point: https://doi.org/10.1002/imt2.214
- Wiley full-text Methods locators: “Quantification of Cas12a cleavage activity” and “Predicting detection activity: Data processing” at HTML lines 489–493; “Data augmentation” at lines 494–501.
- Supporting workbook: IMT2-3-e214-s001.xlsx, named in the article's supporting-information listing; Table S3 header row 2, 30 min column E, 20 min normalized column F; Table S4 header row 2, out_logk_measurement column D.
- Article Data Availability statement identifies the author repository https://github.com/scRNA-Compt/EasyDesign and its main-branch README.
- Author repository: https://github.com/scRNA-Compt/EasyDesign
- Local inference implementation: paper_results_discussion/01_sources/third_party/EasyDesign/easyDesign/utils/predict_activity.py, lines 81–86 and 269–275.
- Local extraction implementation: `exports/reproduction/provenance/source_snapshot/resolve_easy_design_round2.py` relative to the complete delivery-package root, function `build_primary_dataset`, lines 390–435 and 529–534.
- Current canonical data: `exports/reproduction/data/inputs/EasyDesign_2024_V2-2_core_context_feature_table.csv` relative to the complete delivery-package root.
