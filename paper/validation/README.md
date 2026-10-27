# Publication-workflow validation

The original Word files and default deployment weights were kept unchanged.
The public workflow was checked on macOS with Python 3.12.14, the pinned core
dependencies, and a separate TensorFlow 2.18 / tf-keras 2.18 environment.

| Check | Result |
|---|---|
| Original tool regression tests | 18 passed |
| Paper entry, feature, model, inference, protocol, statistical and EasyDesign adapter tests | 39 passed with the asset-backed SavedModel check enabled |
| Fixed prediction coverage | 240,280 long rows across both cohorts and four protocols |
| Nested prediction coverage | 485,164 long rows across both cohorts and four protocols |
| Fixed point metrics | All 192 pooled/fold rows reconstructed; largest five-metric difference below 8×10⁻¹⁶ |
| Nested selection replay | 40/40 policies, all weight layers and candidate-score rows reconstructed from 160 inner tables |
| Fixed formal inference | 20 contrasts, including intervals and raw/Holm p values, numerically identical to accepted outputs |
| Nested inference | 340 pooled metric/interval rows, 1,700 fold metric rows, 16 primary and 32 secondary contrast rows identical |
| Perturbation statistics | 30 metric contrasts reproduced; maximum discrepancy below 9×10⁻¹⁵ |
| Core first outer fold | Four inner component refits and the outer component fit agree with accepted predictions |
| Common first outer fold | Four inner component refits and the outer component fit agree with accepted predictions |
| EasyDesign first outer fold | Stopping subset, epoch 117, weight reset and full outer refit completed; all 1,199 predictions identical |
| Perturbation fit smoke | Seed 137, outer fold 0 for both controls agrees with archived predictions |
| Classic and 2×2 replay smoke | Target outer fold 0 agrees for Ridge, SVR, LightGBM and all four information-block arms |
| Additional candidate | All 27 fixtures match; reversed input order, unseen keys and no-mapping fallback checked |
| Fig.1/Fig.4 | Editable SVG/PDF and real 600-dpi PNG; physical alignment, 5-pt glyph floor and final PDF collision checks pass |
| S6 | Six sheets; 35/4/5/80/4/6 data rows; source counts generated from the actual workbook |
| Moved-directory reproduction | Four archives installed into renamed input/code directories; full results, 40 selections and candidate prediction reproduced outside the original working directory |

The detailed machine-readable reports accompany this directory. Archived
experiment identifiers are retained in the data, while host-specific paths
are replaced by package-relative references.

The source preflight has two reviewed advisories: PNG is the requested raster
format rather than TIFF, and the log-axis group counts are asserted positive
before transformation. Final rendered PDFs have no collision or clipping
findings, and all glyphs exceed 5 pt.

## Numerical details worth preserving

Nested component CSVs require `float_precision='round_trip'` to preserve exact
prediction ties in the candidate grid. The mapping-permutation experiment
originally returned float32 predictions; restoring this dtype before float64
statistics recovers the original archived values. Neither adjustment changes
the experiment, its random stream or the chosen model.

Fig.1/Fig.4 statistics use the documented formal seed 20261012. The old Fig.1
bootstrap source used 20260906, so its rounded interval upper bound differed.
Both source contexts remain identifiable. The corrected figure uses the same
stream as the revised formal comparison.

Fig.2/Fig.3 remain accepted archived figures with their original panel sources.
Their scientific content was not recomputed under a different control design.
The complete publisher workbook is not redistributed.

## Re-run the unit checks

```bash
python -m pip install -r paper/config/requirements-paper.txt pytest==8.3.5
PYTHONPATH=paper/engines/fixed/src TRICASRANK_PAPER_DATA=/path/to/paper-data \
  python -m pytest paper/tests paper/engines/fixed/tests -q
```

When the downloaded assets are absent, only the real EasyDesign SavedModel
file check is skipped; the remaining tests still run without TensorFlow.
