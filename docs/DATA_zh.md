# 数据契约与血缘

公开数据目录分为：`data/raw/` 上游官方仓库原样数据文件、`data/processed/v2_2/` 正式模型输入、`data/examples/` 可追溯示例和 `data/metadata/` 数据角色与公开 SHA-256 清单。源文件不做原地修改，未来变换必须写入新的版本化路径。

正式表为：

`data/processed/v2_2/EasyDesign_2024_V2-2_core_context_feature_table.csv`

SHA-256：`39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0`

| 数据角色 | 行数 | 用途 |
|---|---:|---|
| `baseline_train` | 8,417 | 训练与 target-grouped OOF |
| `baseline_validation` | 2,217 | 历史固定验证 |
| 外部量纲未确认 | 1,358 | 仅保留血缘，不进入最终声明 |
| **总计** | **11,992** | — |

188 项候选输入包括：pair alignment 11、pair position 75、substitution type 12、sequence composition 32、sequence context 58。训练中 5 项为常量，部署使用 183 项；ID、标签、split 和来源映射不进入模型。

每个输入表示 25 个已对齐位置。本项目数据中 1–4 位为 PAM block，5–25 位为 spacer block。`-` 是 alignment gap/bulge，不是 frameshift。外部含 gap 的 target 必须预先对齐，本库不会从变长序列自行猜测 alignment。

OOF 五折按 target sequence 冻结，同一 target 不跨 fold。固定验证有 1,796 个不同 target，v1.5 D 的 0.35/0.59/0.06 权重选择没有读取该验证集；但该验证集在早期已经被查看，因此不能称为 untouched test。

## 来源血缘与再分发

本仓库明确区分两条上游发布渠道：

1. `data/raw/easydesign_supplementary/Table S1.xlsx` 至 `Table S4.xlsx` 是 [EasyDesign 官方 GitHub 仓库](https://github.com/scRNA-Compt/EasyDesign)原样发布的文件。2026-08-14 已与上游提交 `5c06a30d0a43be28a958831587f6ab706c2d4876` 逐字节核验，适用上游仓库的 Apache License 2.0。
2. V2-2 的构建使用了期刊合并补充工作簿 `IMT2-3-e214-s001.xlsx` 中的 Table S2、S3 和 S5。该工作簿位于 Huang 等 2024 年论文的 Supporting Information 节，文章机器可读记录标明 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本仓库不再分发完整期刊合并工作簿；已核验源文件的 MD5 为 `367977dd723c8685e90d21d194f64d92`，SHA-256 为 `3edd9372ccfbca56aba22dae8687898ba482d1654ae07196d07b99cb6deeb27e`。

GitHub 版 `Table S2.xlsx` 独立保存了对应的 `Training data`（10,634 条）、`Augment data`（31,993 条）和 `Test data`（1,358 条）导出，但其工作簿结构并不等同于期刊 Tables S1-S9 合并文件。

V2-2 明确属于加工版本：项目选择源记录并增加稳定 ID 和数据角色，重建 guide-target 对齐及 gap，加入冻结的 target-group folds，并计算 188 项序列特征。原标签被保留，1,358 条量纲未确认记录不进入最终监督声明。

来源论文：Huang B, Guo L, Yin H, et al. *Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics*. **iMeta**. 2024;3(4):e214. <https://doi.org/10.1002/imt2.214>

精确许可范围、上游 SHA-256 和归属见根目录[第三方声明](../THIRD_PARTY_NOTICES_zh.md)。
