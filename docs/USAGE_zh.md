# 使用指南

## 输入表要求

支持 CSV、TSV 和 XLSX；每行是一对已经对齐的 crRNA–target 序列。

```csv
record_id,crRNA_sequence,target_aligned_25,sample_group
candidate_01,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGACGCCA,screen_1
candidate_02,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGTCGCCA,screen_1
```

必须包含：

- `record_id`：非空，且文件内不能重复；
- `crRNA_sequence`：恰好 25 个对齐位置，允许 A/C/G/T，U 会按 T 处理；
- `target_aligned_25`：恰好 25 个对齐位置，允许 A/C/G/T 和 `-`。

列名区分大小写。程序保留其他列、行数和原有顺序。v1.5 不接收用户预先计算的 mapping 列，也不猜测生物学 alignment。

## 运行

```bash
cas12a-ranker predict --input candidates.csv --output predictions.csv
cas12a-ranker predict --input candidates.xlsx --sheet Sheet1 --output predictions.xlsx
```

主要结果列：

| 列名 | 含义 |
|---|---|
| `cas12a_activity_score` | v1.5 D 融合连续活性预测 |
| `cas12a_activity_rank` | 本文件内降序名次，1 为最高 |
| `cas12a_prediction_status` | 正常行为 `success` |
| `cas12a_model_route` | 默认行为 `sequence_d_v1_5` |
| `cas12a_model_version` | 冻结版本号 |
| `cas12a_prediction_xgboost` | XGBoost 分支结果 |
| `cas12a_prediction_lightgbm` | LightGBM 分支结果 |
| `cas12a_prediction_mlp` | MLP 分支结果 |
| `cas12a_warning_codes` | 正常为空；保留错误行时写明原因 |

活性是回归数值，不是概率；排名只比较同一个输入文件中的行。

## 输入错误

默认情况下，只要有一行不合格，程序就停止预测并生成 `<输入文件名>_input_errors.csv`，明确指出记录、原表行号、字段和原因。

数据清理时也可以保留错误行，只预测其余有效行：

```bash
cas12a-ranker predict --input candidates.csv --on-invalid keep
```

错误行会得到 `cas12a_prediction_status=invalid_input`，活性留空。若输入已经含有 `cas12a_` 结果列，必须明确加 `--overwrite-results` 才会替换。

## 最小验证

```bash
cas12a-ranker self-test
cas12a-ranker predict \
  --input data/examples/minimal_input.csv \
  --output minimal_output.csv
```

冻结预期结果位于 `data/examples/minimal_expected_output.csv`。
