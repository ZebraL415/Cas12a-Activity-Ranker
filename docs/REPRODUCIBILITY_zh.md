# 复现说明

## 环境

v2.0 使用 Python 3.12，依赖版本冻结在 `requirements.txt`，大模型文件通过 Git LFS 保存。

```bash
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

## 发布验收

```bash
cas12a-ranker self-test
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/reproduce_v2_metrics.py
```

检查范围包括：

- CSV、TSV、XLSX 的行数、顺序和原列保持；
- 输入错误、禁止用户 mapping 列和明确回退；
- 权威数据、模型和参考表 SHA-256；
- 10,634 行训练/验证数据的自动 mapping 完全一致；
- 188 项序列重建、183 项 D 输入和 1,191 项 B/C 有序输入；
- guide/mapping 历史只含训练标签；
- target 不跨 OOF fold；
- 2,217 行 D、B、C 和最终预测误差均小于 `1e-6`；
- SCC、PCC、RMSE、MAE、R² 复算误差小于 `1e-8`；
- 20,301 个总体细权重和 101,505 个分折权重，以及保留 20%/47%/33% 的决策。

## 参考表怎样生成

`scripts/build_v2_references.py` 从权威 V2-2 表和 EasyDesign 合并来源工作簿生成部署参考。脚本强制只取 8,417 行 `baseline_train`，冻结 guide 分类汇总、6 类 mapping-key 汇总、198 条 template 和全部来源哈希。固定验证标签不进入参考表。

## 复现边界

仓库可复现发布模型推理、自动 mapping、特征、历史查表、逐行预测、指标、不确定性和权重枚举；不承诺不同平台重新训练能得到逐 bit 相同的树和神经网络。正式复现主张是“冻结模型的精确推理 + 完整 OOF/固定证据 + 锁定环境”。

v2 的选择依据是按 target group 划分的 cross-fitted meta-OOF。2,217 行固定验证没有参与本次 D/B/C 权重选择，但早期曾被查看，因此是历史固定验证，不是 untouched external validation。分折隔离 target，不隔离 guide，全新 guide 的泛用性仍然有限。
