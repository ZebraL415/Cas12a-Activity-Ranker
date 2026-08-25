# 仓库结构

公开仓库主目录包括：

```text
LICENSE / NOTICE            Apache-2.0 与归属摘要
THIRD_PARTY_NOTICES*.md     数据与第三方许可边界
data/raw/                 上游官方仓库原样数据文件
data/processed/v2_2/      正式特征表、manifest、字典和 folds
data/examples/            可追溯示例输入与预期结果
data/metadata/            公开文件 SHA-256 和数据角色
src/cas12a_ml/            特征重建、D 融合推理、表格接口与命令行
scripts/                  预测、训练、指标复算和完整性校验
models/                   v1.5 三路模型、旧模型与预处理信息
results/v1_5_d_ensemble/  D 的 OOF、固定验证、5,151 组权重和指标
tests/                    数据、特征和推理测试
docs/                     使用、数据、结果和复现说明
.github/                  CI、Issue 和 Pull Request 模板
```

正式执行链为：`data/processed/v2_2/feature_manifest.csv` 定义 188 项候选输入；过滤 5 项训练常量后，`models/model_input_metadata.json` 冻结 183 项特征顺序；`features.py` 从序列对重建矩阵，`predict.py` 分别调用 XGBoost、LightGBM 和 MLP，再按 0.35/0.59/0.06 合并；`io.py` 保证输入输出逐行对应。

本机工作副本使用 `_local_only/` 统一保存原始提交、内部日志、旧 joblib 和旧数据包布局。整个目录由 `.gitignore` 排除，不进入公开 GitHub 或正式运行链。
