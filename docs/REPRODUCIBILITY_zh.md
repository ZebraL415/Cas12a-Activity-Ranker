# 复现说明

v1.5 支持 Python 3.12，依赖版本冻结在 `requirements.txt`，模型文件通过 Git LFS 保存。

```bash
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .

cas12a-ranker self-test
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/reproduce_metrics.py
```

发布验收会检查 CSV/TSV/XLSX 输入输出、行数与顺序保留、错误报告、数据哈希、188 项特征重建、183 项模型输入顺序、5 折 target 隔离、三份模型哈希、2,217 条逐样本预测、SCC/PCC/误差指标以及 0.35/0.59/0.06 的 OOF 权重最优行。

本版本可以严格复核“冻结模型如何预测、保存的指标如何得到、权重如何从 OOF 枚举中选出”。它不声称在任意操作系统重新训练都会得到逐字节相同的树和神经网络权重；跨平台数值库可能造成差异。

权重只用 8,417 条训练记录的 target-group OOF 结果选择。2,217 条固定验证记录没有参与此次选权，但在项目早期已经被查看，因此称为“历史固定验证”，不称为完全未见的外部测试集。
