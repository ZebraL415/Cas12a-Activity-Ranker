# TriCasRank paper reproducibility

This directory connects the manuscript to its inputs, predictions, statistical
analyses and figure source data. It adds a research workflow alongside the
original `cas12a-ranker` tool; the default tool and its model files are unchanged.

## Review status

This branch prepares the paper release `paper-2026-10-27`. Its four archives are
review artifacts until the PR has been approved and merged. The release URL
becomes a download route only after publication. The maintainer can install the
local review archives with `fetch --asset-dir` in the meantime. The review
branch is pushed; PR creation is deferred until the maintainer is ready to
sign in. [Open the branch comparison](https://github.com/ZebraL415/Cas12a-Activity-Ranker/compare/main...codex/paper-reproducibility?expand=1).

## Start with saved results

Use Python 3.12 on macOS, Linux or WSL. Create an environment from
[`config/requirements-paper.txt`](config/requirements-paper.txt).

```bash
python -m venv .venv-paper
source .venv-paper/bin/activate
python -m pip install -r paper/config/requirements-paper.txt

# After the approved paper release is published:
python paper/reproduce.py fetch --data-dir ./paper-data

# During PR review, use the supplied local archives instead:
python paper/reproduce.py fetch --asset-dir /path/to/review-assets --data-dir ./paper-data

python paper/reproduce.py results --data-dir ./paper-data --output-dir ./paper-results
```

`results` checks record/label/fold/group correspondence, recalculates five
metrics pooled and by fold, reconstructs all 40 nested selections from the 160
inner component tables, repeats the formal statistical analyses, and builds
the corrected Fig.1, Fig.4 and six-sheet S6 workbook. `--points-only` gives a
quick numerical and table preview. The full command uses 2,000 resamples; there
is no weight search outside the accepted nested grid.

All input/output directories are explicit. Run from another current directory
by giving the absolute path to `paper/reproduce.py`. Outputs never overwrite
the supplied manuscript or the original release models.

## Four separately downloadable archives

| Archive | Extracted directory | Use |
|---|---|---|
| `tricasrank-paper-inputs.tar.gz` | `inputs/` | Processed input table, formal identity folds, matching roster, contexts and released EasyDesign architecture |
| `tricasrank-paper-results.tar.gz` | `results/` | Fixed and nested OOF predictions, historical predictions, classic baselines, information-block controls, perturbations, accepted/recalculated statistics and figure sources |
| `tricasrank-paper-nested.tar.gz` | `nested/` | All 160 inner fits and 40 outer fits: component predictions, fit/query IDs, candidate scores and selected policies |
| `tricasrank-paper-candidate.tar.gz` | `candidate/` | Additional all-8,417 refit, loader and 27-row regression fixture |

Each archive contains a README, field definitions, provenance and a file
manifest. The canonical table keeps all 11,992 source-lineage records: the
8,417 development records and 2,217 historical records form the 10,634-record
formal analysis; 1,358 other source records retain their original role and are
not included in these supervised comparisons.

## Three distinct evaluation objects

| Stored model / evaluation | Meaning |
|---|---|
| `TriCasRank`, `fixed_recipe_grouped_cv` | Fixed D weights .35/.59/.06 and top weights .20/.47/.33; training-fold histories are rebuilt |
| `TriCasRank_nested`, `nested_grouped_selection` | Same learner families; both weight layers chosen using only the four inner folds |
| `SELECTED_POLICY`, `nested_grouped_selection` | Inner-selected family among D, sequence + mapping XGBoost, D+B and TriCasRank |

The nested primary comparison families are 12 core and 4 common-context
comparisons for **SELECTED_POLICY**, as defined in the accepted selection
configuration. Nested TriCasRank architecture contrasts are secondary.
The fixed primary comparison families contain 16 core and 4 common-context
contrasts. Holm adjustment is applied separately to each family. Isolation
groups are resampling clusters, rather than claimed biological replicates.

The historical 2,217-record partition is non-independent historical fixed
validation. Its predictions are useful for continuity, separate from the
identity-grouped OOF analyses. Historical development observations are listed
with their actual evaluation scope in the source index.

## Retrain an outer fold

The accepted components are fitted once per fit/query split and reused across
the finite candidate grid. The full supported training entry can run any of
the four protocols and either cohort.

```bash
python paper/reproduce.py train --data-dir ./paper-data --output-dir ./refit-core \
  --cohort core_dev --protocol target --outer-fold 0 --kind nested

# The common cohort additionally needs an isolated TensorFlow environment:
python -m venv .venv-easydesign
.venv-easydesign/bin/python -m pip install -r paper/config/requirements-easydesign.txt
python paper/reproduce.py train --data-dir ./paper-data --output-dir ./refit-common \
  --cohort common_dev --protocol target --outer-fold 0 --kind nested \
  --tf-python .venv-easydesign/bin/python

python paper/reproduce.py train --data-dir ./paper-data --output-dir ./l02-history \
  --cohort core_dev --protocol target --outer-fold 0 --kind perturbation \
  --perturbation history_shuffle --seed 137
python paper/reproduce.py train --data-dir ./paper-data --output-dir ./l02-mapping \
  --cohort core_dev --protocol target --outer-fold 0 --kind perturbation \
  --perturbation mapping_permutation --seed 137
```

Omit `--outer-fold` to run five folds. `--kind fixed` keeps the original
weights. For perturbations, use the core target protocol and seeds 137, 271 or
409. History-label shuffling refits B/C in TriCasRank, whereas mapping-block
permutation refits the single sequence-plus-mapping XGBoost control. Each uses
its own archived intact comparator. Run complete fits serially.

The exact three classic baselines and the four information-block arms have
their own replay entry. They reuse the archived constructors and matrix order:

```bash
python paper/train_controls.py --data-dir ./paper-data --output-dir ./classic-refit --task classic --outer-fold 0
python paper/train_controls.py --data-dir ./paper-data --output-dir ./block-refit --task blocks --outer-fold 0
```

EasyDesign is a local retraining of the released model on the shared 5,995
records. The input is 45×10. A training-only stopping subset chooses the epoch;
weights are then reset before refitting on the whole outer training set.
The matched comparison has four protocols; published author test scores are
documented separately as literature context.

## Use the additional candidate

```bash
python paper/reproduce.py predict-candidate --data-dir ./paper-data --demo --output ./demo.csv
python paper/reproduce.py predict-candidate --data-dir ./paper-data --input ./pairs.csv --output ./scores.csv
```

Input columns are `record_id`, `crRNA_sequence`, `target_aligned_25`. The
candidate archive retains the original Python module paths needed by its
serialized feature state. An unseen history key still uses contextual
prediction; only a missing reference mapping triggers the D fallback.

## Find a paper result

See [`result_index.csv`](result_index.csv), the [data dictionary](DATA_DICTIONARY.md),
the [numerical validation report](validation/README.md) and
the [manuscript replacement guide](MANUSCRIPT_UPDATES.md).
The [`tables/as_submitted/`](tables/as_submitted/) tables are read-only snapshots
for the version crosswalk; recalculated tables are generated under the chosen
output directory. In particular, Table 4 is a development-history table, not a
uniform OOF scoreboard.

## Sources and licences

Processed responses inherit the EasyDesign Table S3 30-minute labels. Source
article: Huang et al. (2024), *iMeta* 3:e214,
<https://doi.org/10.1002/imt2.214>. The original code and repository-distributed
files are at <https://github.com/scRNA-Compt/EasyDesign>. The full publisher
workbook is obtained from the article, not repackaged here.

Project-owned code is Apache-2.0. Derived data, predictions, model artifacts and
figure/table outputs follow the existing CC BY 4.0 scope; upstream notices
remain in force. See [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).
