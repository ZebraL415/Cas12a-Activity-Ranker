# Model card: v2.0 mapping-aware D+B+C system

## Intended use

Predict a continuous fluorescence-derived Cas12a molecular-detection activity score for an already aligned 25-position crRNA–target pair, and rank candidates supplied in the same table. SCC/Spearman and PCC/Pearson are co-primary outcomes.

This model is not intended for genome-editing efficiency, patient-level diagnosis, clinical decision-making, arbitrary sequence alignment or universal genomic search.

## Model

```text
D = 0.35 × sequence XGBoost + 0.59 × sequence LightGBM + 0.06 × sequence MLP
v2 = 0.20 × D + 0.47 × mapping module B + 0.33 × guide-template module C
```

D uses 183 active sequence-derived features. B and C use a frozen 1,191-column matrix comprising those features, guide-history encodings, position-specific pair identities, mapping counts, mapping-history encodings and one-hot mapping categories. B averages five XGBoost residual models; C uses a guide/template mean anchor plus one residual XGBoost model.

## Inputs and automatic mapping

The file interface requires `record_id`, `crRNA_sequence` and `target_aligned_25`. The two sequence columns contain exactly 25 aligned positions. The user does not provide mapping columns.

The runtime removes target gaps, searches all frozen EasyDesign Table S2 templates in forward and reverse-complement orientations, retains all exact window hits, and tries IUPAC-compatible windows only if no exact hit exists. Multiple candidates remain a combined key; the first hit is never selected as truth.

If no template candidate exists, B/C are not invented: the row explicitly uses D. New guide and template keys fall back to the frozen global training mean inside the corresponding history features and receive warnings.

## Training, references and selection

- Training partition: 8,417 records.
- History references: baseline training labels only.
- OOF protocol: five frozen target-group folds.
- D inputs: 183 active sequence features.
- B/C inputs: 1,191 frozen features.
- Released D/B/C weights: 0.20/0.47/0.33.
- Weight audit: 5,151 coarse pooled, 20,301 fine pooled and 101,505 fold-specific combinations.

The fine pooled optimum is 0.225/0.470/0.305, but cross-fitted reweighting underperforms the locked replacement. Fixed-validation labels were not used to select v2 weights. The fixed split was observed earlier in project development and is not an untouched external test.

## Performance

| Protocol | SCC | PCC | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Cross-fitted meta-OOF | 0.8213 | 0.8133 | 0.3781 | 0.2907 | 0.6579 |
| Historical fixed validation | 0.8462 | 0.8355 | 0.3523 | 0.2707 | 0.6940 |

Against the previous A+B+C system, cross-fitted SCC improves by 0.00236 with a paired target-cluster 95% interval of [0.00106, 0.00365]. PCC improves by 0.00137, with an interval slightly crossing zero. MAE is slightly worse. Claims should remain metric-specific.

## Limitations

- Activity is specific to the studied assay and normalization.
- History-derived features may capture recurring guide/template or dataset context, not causal sequence biology.
- OOF isolation is by target group, not guide; evidence for completely new guides is limited.
- Mapping only covers the frozen EasyDesign template reference.
- A mixed full/fallback file does not receive a global rank unless the user explicitly opts in.
- New orthologs, assay scales and preprocessing protocols require independent validation.

## Artifacts

- `v2_model_metadata.json`: weights, metrics, selection logic and hashes;
- `mapping/b_seed_*.json`: five B models;
- `mapping/c_guide_template_anchor.json`: C model;
- `mapping/feature_manifest.csv`: ordered 1,191 inputs;
- `mapping/table_s2_template_reference.csv`: 198 frozen mapping templates;
- `mapping/guide_history_reference.csv`: training-only guide aggregates;
- `mapping/mapping_history_reference.csv`: training-only mapping-key aggregates;
- `mapping/reference_metadata.json`: reference provenance and leakage boundary;
- `d_model_metadata.json` and `primary/d_*`: retained sequence-only D route.

All 10,634 train/validation mapping records and all 2,217 fixed-validation D/B/C/final predictions are checked against frozen references by `scripts/verify_repository.py`.
