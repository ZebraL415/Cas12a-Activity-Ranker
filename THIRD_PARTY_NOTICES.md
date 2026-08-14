# Third-party notices and data licensing

This document separates the license for project-owned software from the terms that apply to upstream and data-derived research artifacts. Nothing in the root [Apache-2.0 license](LICENSE) replaces a third party's copyright or license.

## License scope

| Material | Location | Terms |
|---|---|---|
| Project-owned source code, tests, workflows and software documentation | `src/`, `scripts/`, `tests/`, `.github/`, packaging/configuration files and project-authored Markdown unless stated otherwise | Apache License 2.0 |
| Unmodified files distributed by the EasyDesign GitHub repository | `data/raw/easydesign_supplementary/` | EasyDesign upstream Apache License 2.0 |
| Data-derived research artifacts | `data/processed/`, `data/examples/`, `models/` and `results/` | CC BY 4.0, to the extent copyright or database rights apply, with the upstream attribution and change notice below |
| Presentation materials | `reports/` | Project-created content is CC BY 4.0; any cited or reproduced third-party material remains subject to its source terms |

The Apache-2.0 license text is provided in `LICENSE`. The CC BY 4.0 license is available at <https://creativecommons.org/licenses/by/4.0/>.

## EasyDesign repository files

The following files are byte-identical to the files published in the official [scRNA-Compt/EasyDesign](https://github.com/scRNA-Compt/EasyDesign) repository at commit `5c06a30d0a43be28a958831587f6ab706c2d4876`, verified on 2026-08-14:

| File | SHA-256 |
|---|---|
| `Table S1.xlsx` | `2437cd51c33500be7c777e13f935e1c09cbbdbb79b3134f2fbafb1ac611246cc` |
| `Table S2.xlsx` | `1938131f3cb1522477352d74e52b5bb17c239c500a34b6105efa070e0aef7b2b` |
| `Table S3.xlsx` | `447136b8390c67e7cdb231222d0f0696f117a13d52c639e945cf5fd9a23f6848` |
| `Table S4.xlsx` | `44c42a997bfc076c090e5c33d4b9854e53577fe431f306a646ea5e9f96a931e8` |

These four workbooks were not modified. They are repository-distributed data files, not four extracted sheets from the publisher's combined Tables S1-S9 workbook. EasyDesign's repository-level Apache-2.0 license applies to them; the upstream repository does not contain a separate `NOTICE` file.

## Published supporting information

The final processed table also adapts data from Tables S2, S3 and S5 of the publisher's combined supporting workbook `IMT2-3-e214-s001.xlsx`:

> Huang B, Guo L, Yin H, et al. Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics. *iMeta*. 2024;3(4):e214. <https://doi.org/10.1002/imt2.214>

Copyright © 2024 The Authors. The article's machine-readable full-text record places the supporting workbook in the article's Supporting Information section and identifies the work as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The complete publisher workbook is not redistributed in this repository.

For source verification, the audited publisher workbook has MD5 `367977dd723c8685e90d21d194f64d92` (matching the attachment checksum in the full-text record) and SHA-256 `3edd9372ccfbca56aba22dae8687898ba482d1654ae07196d07b99cb6deeb27e`. It can be obtained from the article DOI rather than from this repository.

### Changes made by this project

The processed V2-2 data are an adaptation, not an unchanged copy. The project:

- selected the activity records used for the development and external-lineage roles;
- reconstructed 25-position guide-target alignments and recorded gaps explicitly;
- added stable record identifiers, role labels and frozen target-grouped folds;
- derived 188 auditable sequence/alignment/context features;
- preserved the source label fields while keeping the scale-unconfirmed external records out of the final supervised claims; and
- trained model artifacts and generated predictions, metrics and uncertainty summaries from those processed data.

No endorsement by the EasyDesign authors, iMeta or John Wiley & Sons is implied. See [the data contract](docs/DATA.md) for scientific lineage and boundaries.
