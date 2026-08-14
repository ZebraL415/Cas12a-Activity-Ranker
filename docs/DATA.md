# Data contract and lineage

## Public data layout

```text
data/
├── raw/easydesign_supplementary/   Unmodified EasyDesign repository data files
├── processed/v2_2/                 Final model-ready table and metadata
├── examples/                       Traceable input/output examples
└── metadata/                       Data roles and public SHA-256 manifest
```

Raw source files are preserved without in-place edits. The processed layer is the only formal model input. Any future transformation should write a versioned new path and document its parent file and checksum.

## Formal V2-2 profile

The final model uses:

`data/processed/v2_2/EasyDesign_2024_V2-2_core_context_feature_table.csv`

SHA-256:

`39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0`

| Record role | Rows | Final supervised use |
|---|---:|---|
| `baseline_train` | 8,417 | Training and target-grouped OOF |
| `baseline_validation` | 2,217 | Historical fixed validation |
| External scale-unconfirmed | 1,358 | Preserved for lineage; excluded from final claims |
| **Total** | **11,992** | — |

The processed directory also contains the feature dictionary, the exact manifest, record-role metadata, frozen target-grouped folds and source-provenance hashes.

## Feature contract

| Group | Count | Meaning |
|---|---:|---|
| Pair alignment | 11 | Difference, substitution and gap summaries |
| Pair position | 75 | Per-position difference, substitution and target-gap indicators |
| Substitution type | 12 | Direct A/C/G/T substitution counts |
| Sequence composition | 32 | Length, GC, entropy, homopolymer and base fractions |
| Sequence context | 58 | Guide/target GC windows and selected k-mer frequencies |
| **Total** | **188** | Candidate model inputs |

Five columns are constant in the training partition and are removed, leaving 183 active deployed inputs. IDs, labels, split fields and source mapping metadata are not model features.

## Sequence representation

Each model input represents 25 aligned guide–target positions. In this project's data representation, positions 1–4 contain the PAM block and positions 5–25 contain the spacer block.

A gap (`-`) is an alignment gap/bulge, not a frameshift. The repository accepts an already aligned 25-position target and preserves gap events as separate features. It intentionally does not infer an alignment from a variable-length external sequence.

## Validation protocol

Five OOF folds are frozen by target sequence. A target never occurs in more than one training fold. The historical fixed validation set contains 1,796 distinct target sequences and is not used to select the final 59/41 ensemble weight.

Because that validation set was inspected during earlier model development, it is not described as an untouched test set. Broad claims should be reserved for a future independently collected assay cohort.

## Source lineage and redistribution

Two upstream publication channels are intentionally distinguished:

1. `data/raw/easydesign_supplementary/Table S1.xlsx` through `Table S4.xlsx` are unmodified files from the official [EasyDesign GitHub repository](https://github.com/scRNA-Compt/EasyDesign). All four were verified byte-for-byte against upstream commit `5c06a30d0a43be28a958831587f6ab706c2d4876` on 2026-08-14. They are covered by EasyDesign's repository-level Apache License 2.0.
2. The V2-2 table was built using Tables S2, S3 and S5 from the publisher's combined supporting workbook `IMT2-3-e214-s001.xlsx`. That workbook appears in the Supporting Information section of Huang et al. (2024), whose machine-readable article record specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The complete combined workbook is not redistributed here; the audited source checksum is MD5 `367977dd723c8685e90d21d194f64d92` and SHA-256 `3edd9372ccfbca56aba22dae8687898ba482d1654ae07196d07b99cb6deeb27e`.

The repository version of `Table S2.xlsx` independently preserves the corresponding `Training data` (10,634 rows), `Augment data` (31,993 rows) and `Test data` (1,358 rows) exports, but it is not the same workbook layout as the publisher's Tables S1-S9 file.

The processed V2-2 data are explicitly marked as an adaptation: selected source records were assigned stable IDs and roles, guide-target alignments and gaps were reconstructed, frozen target-group folds were added, and 188 sequence-derived features were calculated. Labels were preserved; the 1,358 scale-unconfirmed records remain outside final supervised claims.

The source study is:

Huang B, Guo L, Yin H, et al. *Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics*. **iMeta**. 2024;3(4):e214. <https://doi.org/10.1002/imt2.214>

See the root [third-party notice](../THIRD_PARTY_NOTICES.md) for the exact license scope, upstream checksums and attribution.
