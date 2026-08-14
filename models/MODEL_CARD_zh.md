# 模型卡

## 预期用途

根据 25 位 Cas12a guide–target 序列对，对荧光实验派生的连续诊断反应活性进行排序，使更有希望的候选优先进入实验。正式部署模型为 `primary/xgboost_final.json`。

## 输入与预处理

外部接口读取 `crRNA_sequence` 和已经对齐的 target，生成 188 项对齐、位置、替换类型、组成和上下文特征。训练中 5 项为常量，部署矩阵使用 183 项；顺序和缺失值中位数分别冻结于 `model_input_metadata.json` 与 `training_medians.csv`。

含 gap 的 target 必须由调用者提供 25 位 alignment。U 会转成 T；模糊碱基将直接报错，不会静默填补。

## 训练与评估

- 训练：8,417 条。
- 评估：历史固定验证 2,217 条，对应 1,796 个不同 target sequence。
- 最终 OOF：按 target 分组的冻结五折，仅用于组合权重选择。
- XGBoost：1,100 棵树，learning rate 0.02，max depth 7，hist，seed 42。
- CatBoost：1,000 轮，learning rate 0.02，depth 9，seed 42。

固定验证集没有参与最终 59/41 权重搜索，但在项目早期已经被查看，因此不是 untouched test。

## 模型选择

XGBoost 的 RMSE、MAE 与 R² 最好；加权组合 Spearman 仅高 `0.001111`，target-cluster bootstrap 区间跨 0，因此 XGBoost 被定为正式主模型。CatBoost、等权和 OOF 加权只作敏感性对照。

## 局限

- 预测值是实验体系相关的活性分数，不是编辑效率、患者层面的诊断准确率或因果机制。
- 固定验证集中 95.4% 记录的 guide 在训练中见过，完全新 guide 证据有限。
- 极端活性预测有范围压缩。
- XGBoost 与 CatBoost 的预测和残差高度相关，互补性弱。
- 新实验量纲、Cas12a ortholog 或序列处理流程需要独立校准和验证。

正式推理使用原生 JSON/CBM。两个原生模型已与 2,217 条冻结预测逐条核对，最大绝对差小于 `1e-6`。
