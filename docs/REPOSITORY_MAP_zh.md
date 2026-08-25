# 仓库结构

```text
README*.md                 面向新用户的入口
data/processed/v2_2/      权威数据表、manifest 和 folds
data/examples/            最小输入和冻结 v2 输出
src/cas12a_ml/features.py 序列特征
src/cas12a_ml/mapping.py  自动保留全部 template 候选
src/cas12a_ml/v2_features.py 训练历史与 1,191 项 B/C 输入
src/cas12a_ml/predict.py  v2、D 回退和旧模型推理
models/primary/           D 模型
models/mapping/           B/C 模型、template、历史表和 manifest
results/v2_dbc/           OOF、固定预测和完整权重审计
scripts/                  参考表生成、预测、复算和完整校验
tests/                    数据、mapping、接口和推理测试
docs/                     使用、数据、结果和复现说明
```

正式执行链是：用户提交已对齐表格；`io.py` 检查；`features.py` 建立 D 输入；`mapping.py` 查找全部冻结 template 候选；`v2_features.py` 合并序列、mapping 和只来自训练集的历史；`predict.py` 计算 D、B、C 和 20%/47%/33% 最终分数。mapping 失败时，程序明确记录 D 回退和 warning，不制造 B/C 输入。

`results/v2_dbc/fixed_validation_predictions.csv` 是 2,217 行逐样本回归基准；`results/v2_dbc/weight_audit/` 保存粗网格、细网格和跨折网格。本机 `_local_only/` 只用于保留内部原始材料，不进入 GitHub 或正式运行链。
