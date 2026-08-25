# Cas12a Activity Ranker

**读取 CSV、TSV 或 XLSX 中已经对齐的 crRNA–target 配对，预测连续 Cas12a 荧光反应活性并在表内排序。**

[English README](README.md) · [详细使用说明](docs/USAGE_zh.md) · [模型卡](models/MODEL_CARD_zh.md) · [结果](docs/RESULTS_zh.md) · [复现](docs/REPRODUCIBILITY_zh.md)

> 本工具预测当前 Cas12a 分子检测实验体系中的活性，不预测基因编辑效率、患者诊断结果或临床准确率。

## 三步开始使用

### 1. 安装并自检

```bash
git clone https://github.com/ZebraL415/Cas12a-Activity-Ranker.git
cd Cas12a-Activity-Ranker
git lfs pull

python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .

cas12a-ranker self-test
```

看到 `PASS Cas12a Activity Ranker self-test` 即说明四条内置样本已经完整跑通。

### 2. 按固定列名准备表格

每行是一对已经对齐的序列，必须有三列：

| 列名 | 要求 |
|---|---|
| `record_id` | 非空，并且在文件内唯一 |
| `crRNA_sequence` | 恰好 25 个对齐位置；A/C/G/T，U 会按 T 处理 |
| `target_aligned_25` | 恰好 25 个对齐位置；A/C/G/T 或 `-` |

其他实验编号、分组或备注列都可以保留，程序不会删除或重排。

### 3. 预测

```bash
cas12a-ranker predict --input candidates.csv --output predictions.csv
```

输出与输入逐行对应，原有列不变，并在右侧增加：

| 主要结果列 | 含义 |
|---|---|
| `cas12a_activity_score` | v1.5 连续活性预测值 |
| `cas12a_activity_rank` | 本次输入文件内的降序名次，1 为最高 |
| `cas12a_prediction_status` | 是否成功预测 |
| `cas12a_model_route` | 本行走过的模型路径 |
| `cas12a_model_version` | 冻结版本号 |

程序支持 CSV、TSV 和 XLSX。若不写 `--output`，会自动生成 `<输入文件名>_cas12a_predictions.<扩展名>`。输入不合格时会生成逐行错误表，而不是悄悄修改序列。

## v1.5 实际运行什么？

v1.5 的 D 模型从同一套 183 项有效序列特征得到三路预测，再按冻结权重合并：

```text
最终活性 = 0.35 × XGBoost + 0.59 × LightGBM + 0.06 × MLP
```

权重只根据 8,417 条训练数据的 5 折 target-group OOF 预测选择，没有使用历史固定验证集的标签来挑权重。

## 表现

SCC/Spearman 表示候选排序是否正确；PCC/Pearson 表示预测活性数值与真实值是否同步变化。本版本把两者都作为第一梯队指标。

| 模型 | SCC / Spearman ↑ | PCC / Pearson ↑ | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---:|---:|---:|---:|---:|
| XGBoost 分支 | 0.7652 | 0.7476 | 0.4247 | 0.3312 | 0.5550 |
| LightGBM 分支 | 0.7690 | 0.7527 | 0.4207 | **0.3257** | 0.5635 |
| MLP 分支 | 0.6347 | 0.6147 | 0.5175 | 0.4099 | 0.3395 |
| **v1.5 D 融合** | **0.7709** | **0.7538** | **0.4205** | 0.3260 | **0.5639** |

以上数字来自同一 2,217 条历史固定验证集。它在项目早期已被查看，因此不能称作完全未见的外部测试集。完整逐样本预测、OOF 结果、全部权重枚举和运行记录位于 [`results/v1_5_d_ensemble`](results/v1_5_d_ensemble)。

与 v1.0 XGBoost 相比，D 的 SCC 数值提高 `0.0027`、PCC 提高 `0.0040`、RMSE 降低 `0.0027`；三项 target-cluster bootstrap 区间均跨 0。因此 v1.5 的发布理由是工具链完整和各指标方向一致的数值改善，不把它夸大成已经证明对所有新数据都更优。

## 复核发布内容

```bash
cas12a-ranker self-test
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/reproduce_metrics.py
```

这些检查覆盖模型文件哈希、序列特征重建、2,217 条逐样本预测以及 SCC、PCC、RMSE、MAE 和 R²。

## 使用边界

- 活性数值依赖当前荧光实验体系和归一化方式。
- 验证集中多数记录的 guide 在训练中出现过，对完全新 guide 的证据有限。
- 排名只表示同一次输入文件内的相对顺序，不是概率或临床阈值。
- 新 Cas12a ortholog、实验量纲或序列处理流程需要独立验证。
- v1.5 要求用户提供已经对齐的 25 位序列，不负责自动完成生物学 alignment/mapping。

## 引用与许可

源任务和数据来自 Huang 等的 EasyDesign 研究：*Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics*, **iMeta** (2024), [doi:10.1002/imt2.214](https://doi.org/10.1002/imt2.214)。使用时请同时引用该研究和实际使用的仓库 commit。

项目软件采用 [Apache License 2.0](LICENSE)，加工数据与数据衍生产物采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。具体范围和来源见 [THIRD_PARTY_NOTICES_zh.md](THIRD_PARTY_NOTICES_zh.md)。
