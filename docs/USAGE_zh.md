# 使用指南

准备 CSV：

```csv
candidate_id,crRNA_sequence,target_aligned_25
candidate_01,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGACGCCA
candidate_02,TTTGGGCTAGGTGTGATAGGAATGG,TTTGGGCTAGGAGTGATAGGAATGG
```

两列序列均表示 25 个已对齐位置。`crRNA_sequence` 接受 A/C/G/T 或 U；`target_aligned_25` 还允许用 `-` 表示已经确定的 target gap。其他 ID 或元数据列会原样保留。

运行：

```bash
python scripts/predict_external.py --input candidates.csv --output predictions.csv
```

正式预测列是 `prediction_xgboost_primary`，`rank_xgboost_descending` 是当前文件内的降序排名。CatBoost、等权和加权列用于对照与敏感性分析。分数不是概率，也不能解释成患者层面的诊断准确率。

无 gap 且恰好 25 nt 时，可以用 `target_sequence` 代替 `target_aligned_25`；含 gap 时必须显式提供已对齐序列。本仓库在 alignment 之后提取特征，本身不是序列对齐工具。

安装验证：

```bash
python scripts/predict_external.py \
  --input data/examples/external_sequence_pairs.csv \
  --output /tmp/cas12a_predictions.csv
python scripts/verify_repository.py
```

预期结果保存在 `data/examples/expected_predictions.csv`。
