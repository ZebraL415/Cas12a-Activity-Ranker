# 使用说明

## 输入表

CSV、TSV 或 XLSX 每行放一对已经对齐的 crRNA–target：

```csv
record_id,crRNA_sequence,target_aligned_25,sample_group
candidate_01,TTTGTTGGGGCGTCCTTAGACGCCA,TTTGTTGGGGCGTCCTTAGACGCCA,screen_1
candidate_02,TTTGGGCTAGGTGTGATAGGAATGG,TTTGGGCTAGGAGTGATAGGAATGG,screen_1
```

必需列名区分大小写：

- `record_id`：非空且文件内唯一；
- `crRNA_sequence`：25 位 A/C/G/T，U 自动转成 T；
- `target_aligned_25`：25 个已经对齐的位置，可含 `-`。

程序不负责从两条任意长度序列生成生物学 alignment；但会根据用户提供的对齐 target 自动查找冻结的 EasyDesign template。因此不要提交 `mapping_*` 列，程序会拒绝这些列，避免用户信息与内置 mapping 冲突。

## 运行

```bash
cas12a-ranker predict --input candidates.csv --output predictions.csv
cas12a-ranker predict --input candidates.tsv --output predictions.tsv
cas12a-ranker predict --input candidates.xlsx --sheet Sheet1 --output predictions.xlsx
```

默认运行 v2；需要对照旧路线时显式指定 `--model d` 或 `--model xgboost-legacy`。

## 主要输出

| 列名 | 含义 |
|---|---|
| `cas12a_activity_score` | 该行最终采用的连续活性分数 |
| `cas12a_activity_rank` | 全局降序名次；混合完整/回退路线时默认留空 |
| `cas12a_rank_within_route` | 相同模型路线内的名次 |
| `cas12a_prediction_d/b/c` | 三个模块的预测 |
| `cas12a_prediction_full_dbc` | 完整 `20% D + 47% B + 33% C`；mapping 失败时留空 |
| `cas12a_prediction_status` | `success`、`fallback_success` 或 `invalid_input` |
| `cas12a_model_route` | 完整 v2、D 回退或显式指定的旧模型 |
| `cas12a_mapping_*` | mapping 状态、置信度、数量和所有候选键 |
| `cas12a_guide_seen` / `cas12a_template_key_seen` | 训练历史里是否出现过 |
| `cas12a_*_reference_count` | 支持相应历史键的训练记录数 |
| `cas12a_warning_codes` | 分号分隔的逐行提示 |

这些分数是本实验体系下的回归结果，不是概率或临床阈值。

## 回退和混合排名

默认情况下，无法 mapping 的行用 D 回退并明确标记。若不允许任何回退：

```bash
cas12a-ranker predict --input candidates.csv --fallback-policy error
```

完整 v2 与 D 回退行同时存在时，全局名次默认留空，但路线内名次仍然输出。确实需要跨路线硬排时：

```bash
cas12a-ranker predict --input candidates.csv --allow-mixed-ranking
```

## 输入错误

默认发现错误就停止，并生成 `<stem>_input_errors.csv`。只为整理数据时，可以保留错误行、同时预测正确行：

```bash
cas12a-ranker predict --input candidates.csv --on-invalid keep
```

错误行的状态为 `invalid_input`，没有活性分数。若输入含旧 `cas12a_` 输出列，可用 `--overwrite-results` 明确替换；用户 mapping 列不能通过这个参数绕过。

## 最小验证

```bash
cas12a-ranker self-test
cas12a-ranker predict --input data/examples/minimal_input.csv --output minimal_output.csv
```
