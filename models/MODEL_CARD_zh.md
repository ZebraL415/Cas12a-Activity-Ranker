# 模型卡：v2.0 mapping-aware D+B+C 系统

## 用途

对已经对齐的 25 位 crRNA–target 序列对预测 Cas12a 分子检测实验中由荧光得到的连续活性分数，并对同一输入表的候选排序。SCC/Spearman 和 PCC/Pearson 都是一级指标。

它不用于预测基因编辑效率、患者诊断、临床决策、任意序列 alignment 或通用基因组搜索。

## 模型结构

```text
D = 35% 序列 XGBoost + 59% 序列 LightGBM + 6% 序列 MLP
v2 = 20% D + 47% mapping 模块 B + 33% guide-template 模块 C
```

D 使用 183 项有效序列特征。B/C 使用冻结的 1,191 项输入，包括序列特征、guide 历史、逐位置碱基对、mapping 数量、mapping 历史和 mapping 类别。B 平均 5 个 XGBoost 残差模型；C 使用 guide/template 平均锚点和一个 XGBoost 残差模型。

## 输入与自动 mapping

输入表只要求 `record_id`、`crRNA_sequence`、`target_aligned_25`；用户不提交 mapping 列。

程序去除 target gap 后，在 198 条冻结的 EasyDesign Table S2 template 中搜索正向和反向互补窗口；先保留全部精确候选，只有完全没有精确结果时才查 IUPAC 兼容候选。多候选保持为组合键，不把第一条当成真值。

找不到 template 时不伪造 B/C，而是明确回退到 D。新 guide 或新 template-key 在相应历史特征中回到训练总体均值，并输出 warning。

## 训练与选择

- 训练数据：8,417 行；
- 历史参考：只使用 baseline_train 标签；
- OOF：5 个冻结 target-group folds；
- D 输入：183 项；B/C 输入：1,191 项；
- 正式 D/B/C 权重：20%/47%/33%；
- 权重审计：5,151 个粗网格、20,301 个细网格、101,505 个分折组合。

细网格总体最高点是 22.5%/47%/30.5%，但跨折重新选权重的表现低于锁定替换方案。因此固定验证标签没有参与 v2 权重选择，正式版本保留 20%/47%/33%。固定验证数据在项目早期曾被查看，不是 untouched test。

## 表现

| 验证方式 | SCC | PCC | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| cross-fitted meta-OOF | 0.8213 | 0.8133 | 0.3781 | 0.2907 | 0.6579 |
| 历史固定验证 | 0.8462 | 0.8355 | 0.3523 | 0.2707 | 0.6940 |

相对旧 A+B+C，cross-fitted SCC 提高 0.00236，配对 target-cluster 95% 区间为 [0.00106, 0.00365]。PCC 数值提高 0.00137，但区间轻微跨 0；MAE 略差。因此结论必须按指标分别描述。

## 局限

- 活性分数只对应本实验和归一化尺度。
- 历史特征可能包含重复 guide/template 或数据背景信息，不代表因果生物机制。
- OOF 隔离的是 target group，不是 guide；全新 guide 证据有限。
- mapping 只覆盖冻结的 EasyDesign template。
- 完整路线和 D 回退混合时，默认不提供跨路线全局名次。
- 新 Cas12a 类型、实验尺度或预处理流程需要独立验证。

## 文件

- `v2_model_metadata.json`：权重、指标、选择逻辑和哈希；
- `mapping/b_seed_*.json`：5 个 B 模型；
- `mapping/c_guide_template_anchor.json`：C 模型；
- `mapping/feature_manifest.csv`：1,191 项有序输入；
- `mapping/table_s2_template_reference.csv`：198 条 template；
- `mapping/guide_history_reference.csv`：训练集 guide 汇总；
- `mapping/mapping_history_reference.csv`：训练集 mapping-key 汇总；
- `mapping/reference_metadata.json`：来源和防泄漏边界；
- `d_model_metadata.json` 与 `primary/d_*`：保留的 D 回退路线。
