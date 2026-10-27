# Citation audit for the R03/R09 manuscript insert

Search date: 2026-10-13. The methodological search was limited to Nature Portfolio journals; the EasyDesign source article is retained separately as the indispensable primary citation for its data and endpoint.

| Segment | Claim to cite | Source and support | Manuscript use |
|---|---|---|---|
| C01 | In grouped biological data, row-wise separation may not reflect the intended prediction setting when related entities occur across folds. | Joeres, Blumenthal & Kalinina, *Nature Communications* (2025), DataSAIL. Strong methodological context: the paper studies identity- and similarity-aware splits for one- and two-entity biomolecular data and demonstrates that split design changes measured generalization. | Cite after the rationale for target-, guide-, template- and joint-identity protocols; do not imply DataSAIL generated our folds. |
| C02 | When weights or model families are chosen, a separate outer loop can estimate the performance of the complete selection procedure. | Cearns, Hahn & Baune, *Translational Psychiatry* (2019). Background/method support: the article explains the inner-selection / outer-evaluation logic. | Cite once in nested-validation Methods; the exact folds, candidates, and ranking objective remain project-specific. |
| C03 | The inherited response and EasyDesign comparator originate from the published 30-min Cas12a diagnostic activity table. | Huang et al., *iMeta* (2024). Primary-source support: paper Methods and Tables S3/S4. This is outside the Nature Portfolio search filter, but is the required source paper for the dataset and its processing description. | Cite the response-provenance paragraph and the EasyDesign comparison. |

## Checked metadata

- Joeres R, Blumenthal DB, Kalinina OV. “Data splitting to avoid information leakage with DataSAIL.” *Nature Communications* 16, 3337 (2025). DOI: 10.1038/s41467-025-58606-8.
- Cearns M, Hahn T, Baune BT. “Recommendations and future directions for supervised machine learning in psychiatry.” *Translational Psychiatry* 9, 271 (2019). DOI: 10.1038/s41398-019-0607-2.
- Huang B, Guo L, Yin H, Wu Y, Zeng Z, Xu S, Lou Y, Ai Z, Zhang W, Kan X, Yu Q, Du S, Li C, Wu L, Huang X, Wang S, Wang X. “Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics.” *iMeta* 3, e214 (2024). DOI: 10.1002/imt2.214.

The attached `nature_citations.ris` contains these three complete records. Nature-family methods articles provide context, while the project’s own split assignment and the EasyDesign paper remain the direct evidence for project-specific statements.
