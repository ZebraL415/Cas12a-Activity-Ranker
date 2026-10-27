# References for R09

1. Huang B, Guo L, Yin H, Wu Y, Zeng Z, Xu S, Lou Y, Ai Z, Zhang W, Kan X, Yu Q, Du S, Li C, Wu L, Huang X, Wang S, Wang X. Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics. *iMeta*. 2024;3(4):e214. [https://doi.org/10.1002/imt2.214](https://doi.org/10.1002/imt2.214). See Methods sections “Quantification of Cas12a cleavage activity,” “Predicting detection activity: Data processing,” and “Data augmentation.”

2. Huang et al. Supporting information for the article above. Workbook IMT2-3-e214-s001.xlsx (2,339,574 bytes; published MD5 367977dd723c8685e90d21d194f64d92). Relevant sheets are Table S3 (training data, columns “30 min” and “20 min normalized”) and Table S4 (augmentation, column “out_logk_measurement”). The article DOI page is the stable public entry point for the supporting files: [Wiley article and supporting information](https://onlinelibrary.wiley.com/doi/10.1002/imt2.214).

3. EasyDesign author repository, scRNA-Compt/EasyDesign. [https://github.com/scRNA-Compt/EasyDesign](https://github.com/scRNA-Compt/EasyDesign). Relevant source file: easyDesign/utils/predict_activity.py, which transforms released regression-model predictions for the inference interface. This repository is not the source of the present study's Table S3 labels.

4. Cearns M, Hahn T, Baune BT. Recommendations and future directions for supervised machine learning in psychiatry. *Translational Psychiatry*. 2019;9(1):271. [https://doi.org/10.1038/s41398-019-0607-2](https://doi.org/10.1038/s41398-019-0607-2). Methodological background for separating inner model/weight selection from outer performance estimation.

5. Joeres R, Blumenthal DB, Kalinina OV. Data splitting to avoid information leakage with DataSAIL. *Nature Communications*. 2025;16:3337. [https://doi.org/10.1038/s41467-025-58606-8](https://doi.org/10.1038/s41467-025-58606-8). Methodological context for identity-aware evaluation of dependent biomolecular data; this study's folds were inherited and not generated with DataSAIL.
