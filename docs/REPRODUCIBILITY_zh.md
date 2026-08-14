# 可复现说明

支持环境为 Python 3.12，依赖版本锁定在 `requirements.txt`。macOS 还需安装 `libomp`。

验收顺序：

```bash
pip install -r requirements.txt
pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/train_final_four.py --verify-only
python scripts/train_final_four.py --smoke-test
python scripts/reproduce_metrics.py
```

校验覆盖正式数据 SHA 和行数、188/183 特征契约、OOF 中 target 不跨 fold、外部特征精确重建、四种模型输出与全部 2,217 条冻结预测一致、五项指标复算和公开文件校验清单。

完整训练命令：

```bash
python scripts/train_final_four.py --output reproduced_run
```

该流程训练两个基础学习器的五折 OOF，搜索 101 组组合权重，然后拟合两个最终模型并在历史固定验证集上比较四种输出。逐样本冻结预测使他人无需先完整重训也能核验最终指标。

固定验证集是历史固定验证，不是独立外部队列；加权组合增益很小且 target-cluster 区间跨 0，因此只作探索性结果。
