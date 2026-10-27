# Methods supplement: outcome provenance

## Manuscript-ready text

We used the published 30-min activity values in EasyDesign Table S3 as the response, preserving the source study's plate-level control normalization: the mean of two no-template negative-control wells was subtracted from each plate reading, and the corrected value was divided by the mean of two high-activity positive-control wells. We inherited these processed labels rather than repeating the plate correction. A record-level audit confirmed exact agreement between all 10,634 Table S3 values and the responses used here; the separate 20-min normalized field and Table S4 augmentation were not used in the primary evaluation. Target-, guide-, atomic-template- and joint-identity partitions changed which records were used for fitting and evaluation while retaining the same response values. Accordingly, Spearman correlation measures agreement in guide ranking, whereas MAE and RMSE quantify deviations on the published 30-min activity-label scale (−4.20319 to −0.306893). (Huang et al., 2024; citation number to be assigned in the manuscript.)

## Integration notes

- The source paper's time-folding rule applies to the auxiliary 20 min normalized label, not to the primary 30 min column used here.
- The archived EasyDesign predictor source adds 4 to regression-model predictions and clips the inference output at zero. This prediction-output convention is separate from label generation. In the present EasyDesign-retrained evaluation, the project runner saves raw model.predict output directly; it does not call that inference wrapper or add 4.
- Questions about any transformation beyond the published plate correction and about replicate/plate identifiers are listed separately in Questions_for_A.md; they do not alter the verified record-level mapping reported above.
