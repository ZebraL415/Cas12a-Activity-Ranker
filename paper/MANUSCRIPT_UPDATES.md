# Manuscript updates for the corresponding paper release

The supplied Word files remain unchanged. Use the following replacements when
integrating the public materials into the manuscript and Supplementary Material.

## 1. Fig.1: seven models and one statistical stream

Replace the current figure with `results/recalculated/figures/Fig1_corrected`.
Panels b–d now display all seven Table 3 models, including the history-prior
reference and sequence-plus-mapping XGBoost. Their predictions come from the
same 8,417 records and target folds. The stronger contextual single learner
remains visible alongside TriCasRank.

For the main TriCasRank–D SCC comparison, the corrected fixed evaluator gives
ΔSCC = 0.0803924717 and a 95% interval that rounds to **0.072–0.090**, rather
than 0.072–0.089 in the current abstract/figure. The release retains both
the original archived source and the new formal result, and explicitly uses
seed 20261012 for the corrected figure and formal comparisons.

Suggested abstract replacement:

> In five-fold target-grouped evaluation of 8,417 records, TriCasRank increased
> Spearman correlation from 0.741 for the sequence-only ensemble to 0.821
> (paired difference, 0.080; 95% target-group bootstrap interval, 0.072–0.090).

中文：摘要和 Fig.1 的区间上界同步改为 0.090。新增两项对照后，图中的模型名单与 Table 3 一致；强 mapping 模型的结果也直接展示出来。

### Replacement Fig.1 caption

**Fig.1. Context-aware prediction under target-grouped evaluation.** a, Shared
8,417-record, five-fold protocol with 6,825 target groups. b, SCC for seven
models; points and bars show pooled OOF estimates and 95% target-group bootstrap
intervals. c, Five-metric profiles; numbers are raw values and colour is the
direction-corrected within-metric rank. d, Outer-fold SCC, with the same colour
key as b: TriCasRank blue, sequence-plus-mapping XGBoost orange, other controls
grey. e, Paired bootstrap distribution for TriCasRank − D. f, Prediction
density using all records; darker cells contain more records. g, Calibration
in ten observed-activity bins. Dashed lines in f,g denote identity. Bootstrap
analyses use 2,000 target-group samples, seed 20261012.

## 2. Fig.4: corrected identity results and full within-guide range

Replace the current figure with `results/recalculated/figures/Fig4_corrected`.
The c–g values are regenerated from the revised fixed four-protocol predictions.
Template/Joint SCC for TriCasRank is **0.609/0.609**, not the older plotted
0.621/0.628. TriCasRank − D MAE is **+0.004114/−0.005786**, rather than the
older +0.064/+0.060. The corresponding 95% paired intervals cross zero.
Panel h now includes the negative within-guide values on a full −1 to 1 scale;
the eligibility rule remains at least five records, giving 1,107 defined guides.
Panels i–j remain within the shared 5,995-record, 5,923-target-group comparison.

中文：这次修图主要让已经更新的正文数字与图面一致。Template/Joint 的排序增益较小，而误差差值接近零；负的 guide 内排序结果也完整保留。EasyDesign 比较继续使用 5,995 条共同记录。

### Replacement Fig.4 caption

**Fig.4. Identity-dependent transfer and the matched EasyDesign comparison.**
a,b, Four separate identity-isolation protocols and their group counts, using
the same 8,417 records. c,d, Pooled OOF SCC and RMSE for D and TriCasRank.
e,f, Paired differences, TriCasRank − D, with 95% intervals from 2,000 samples
of the corresponding isolation groups. g, Calibration in ten prediction-ordered
bins. h, Guide-isolated within-guide SCC for 1,107 guides with at least five
records; boxes show median and interquartile range, whiskers use 1.5× IQR,
and points show outliers. i, SCC and RMSE with 95% target-group bootstrap
intervals on the matched 5,995 records. j, MAE by observed-activity quintile;
TriCasRank has lower absolute error on 67.4% of paired records. EasyDesign-adapted
is a local retraining of the released regressor under the shared protocol.

Fig.2 and Fig.3 retain their accepted scientific content. Their vector files
and complete panel sources are provided without substituting the newer
protocol-group analyses for their original controlled target-fold experiments.

## 3. S6: the electronic file is now supplied

Use `results/recalculated/tables/Table_S6_fold_subgroup_audit.xlsx` and copy its
actual row counts from `Table_S6_contents.csv` into the supplementary directory:

| Sheet | Data rows |
|---|---:|
| Fold metrics | 35 |
| History support | 4 |
| Template support | 5 |
| Activity bins | 80 |
| Guide ranking | 4 |
| Provenance | 6 |

The six sheets retain their original purposes. The source registry is now one
row per sheet, and its paths refer to the public assets. Candidate-template
counts and training-history support counts are labelled separately. Attach
the workbook itself, rather than only the table describing it.

## 4. Table 4: separate development observations from the formal result

The original +0.00236 row concerns **complete DBC replacing legacy ABC** in
the earlier cross-fitted meta evaluation. Its public source is
`results/v2_dbc/v2_vs_v1_target_cluster_bootstrap.csv` in the Git repository,
copied to the results archive as `development/legacy_DBC_vs_ABC_bootstrap.csv`.
That source gives ΔSCC 0.002357503 and CI [0.001061233, 0.003650242]. Module D
itself has SCC 0.740980940 under the current fixed target OOF evaluation.

Suggested corrected rows:

| Development observation | Evaluation | Verified result |
|---|---|---|
| Sequence-only Module D | Current fixed target-grouped OOF | SCC 0.741 |
| Complete DBC replacing legacy ABC | Earlier cross-fitted meta OOF | ΔSCC +0.00236, 95% CI [+0.00106, +0.00365] |

The exact sources for the three original MLP/RFF numeric observations have
not been recovered. The available older 2,217-record prediction file produces
different values and therefore does not validate those rows. The recommended
Table 4 uses the two verified rows above; the original table remains a labelled
snapshot in `tables/as_submitted/`, with row-level status in
`tables/Table4_evidence_registry.csv`.

中文：建议把 Table 4 收束为这两项可核对的开发期观察。前三项旧 MLP/RFF 数字暂缺精确来源，因此不把其他旧文件的成绩填进去，也不与正式 OOF 结果混写。

## 5. Short Methods clarifications

**Routing:** “When label-free reference mapping is unavailable, the prediction
uses the sequence-only Module D. An unseen historical key uses the fitted
history fallback within the contextual feature calculation.”

**Nested selection:** “For each of five outer folds, the other four folds
provided four training-only inner validation folds. D-component weights were
selected first, followed by D+B and D/B/C weights, then the model family.
The step-0.05 grids included the original fixed weights. We evaluated the
selected rule only on its outer query fold. The resulting SELECTED_POLICY was
the primary nested comparison; nested TriCasRank architecture effects were
reported separately.”

**Source scale:** Adopt the source-supported paragraph in
`sources/labels/Methods_en.md`. It distinguishes inherited published labels
from newly executed processing. Plate identifiers and an exact raw-to-negative-
label transformation have not been recovered from public files; no new
physical fluorescence unit should be assigned to MAE/RMSE.

**Historical coverage:** Where 95.4% appears, it denotes records
(2,116/2,217), not unique guides. The corresponding unique-guide coverage is
992/1,008 (98.4%).

## 6. Declarations

Use `AVAILABILITY_DRAFT.md`. Its publication-ready paragraph becomes accurate
after the approved release and its four downloads are public. The team still
confirms Acknowledgements and funding facts. No Word source was changed by
this materials update.

## 7. Reference [1]

The current DOI 10.1016/j.bios.2019.111421 identifies a different paper. If
reference [1] supports the Cas12a collateral-cleavage mechanism, replace the
entry with Chen JS et al. *CRISPR-Cas12a target binding unleashes indiscriminate
single-stranded DNase activity*. Science. 2018;360:436–439.
https://doi.org/10.1126/science.aar6245.
