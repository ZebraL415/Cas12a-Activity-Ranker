# 第三方声明与数据许可

本文件用于区分“本项目原创软件的许可证”和“上游材料及数据衍生产物的使用条件”。根目录的 [Apache-2.0 许可证](LICENSE) 不会覆盖或替换第三方原有版权与许可。

## 许可范围

| 内容 | 路径 | 条款 |
|---|---|---|
| 本项目原创代码、测试、工作流和软件文档 | `src/`、`scripts/`、`tests/`、`.github/`、打包/配置文件及未另行说明的项目 Markdown | Apache License 2.0 |
| EasyDesign 官方 GitHub 原样发布的数据文件 | `data/raw/easydesign_supplementary/` | EasyDesign 上游 Apache License 2.0 |
| 数据衍生研究产物 | `data/processed/`、`data/examples/`、`models/`、`results/` | 在版权或数据库权利适用的范围内按 CC BY 4.0 发布，并保留下述来源与改动说明 |
| 汇报材料 | `reports/` | 项目原创部分按 CC BY 4.0；其中引用或再现的第三方材料仍遵循各自来源条款 |

Apache-2.0 全文见 `LICENSE`；CC BY 4.0 全文见 <https://creativecommons.org/licenses/by/4.0/>。

## EasyDesign 官方仓库文件

以下四个文件已在 2026-08-14 与 [scRNA-Compt/EasyDesign](https://github.com/scRNA-Compt/EasyDesign) 官方仓库提交 `5c06a30d0a43be28a958831587f6ab706c2d4876` 逐字节核验：

| 文件 | SHA-256 |
|---|---|
| `Table S1.xlsx` | `2437cd51c33500be7c777e13f935e1c09cbbdbb79b3134f2fbafb1ac611246cc` |
| `Table S2.xlsx` | `1938131f3cb1522477352d74e52b5bb17c239c500a34b6105efa070e0aef7b2b` |
| `Table S3.xlsx` | `447136b8390c67e7cdb231222d0f0696f117a13d52c639e945cf5fd9a23f6848` |
| `Table S4.xlsx` | `44c42a997bfc076c090e5c33d4b9854e53577fe431f306a646ea5e9f96a931e8` |

四个工作簿均未修改。它们是 EasyDesign GitHub 仓库发布的数据文件，并不是从期刊 Tables S1-S9 合并工作簿拆出的四张工作表；它们适用上游仓库的 Apache-2.0。上游仓库没有另附 `NOTICE` 文件。

## 期刊补充材料

最终加工表还使用了期刊合并补充工作簿 `IMT2-3-e214-s001.xlsx` 的 Table S2、S3 和 S5：

> Huang B, Guo L, Yin H, et al. Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics. *iMeta*. 2024;3(4):e214. <https://doi.org/10.1002/imt2.214>

版权 © 2024 The Authors。文章的机器可读全文记录把该工作簿列在同一文章的 Supporting Information 中，并标明 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本仓库不再分发完整的期刊合并工作簿。

用于来源核验的期刊工作簿 MD5 为 `367977dd723c8685e90d21d194f64d92`（与全文记录中的附件校验值一致），SHA-256 为 `3edd9372ccfbca56aba22dae8687898ba482d1654ae07196d07b99cb6deeb27e`。使用者应从论文 DOI 获取，而不是从本仓库下载该完整工作簿。

### 本项目进行的改动

V2-2 是加工版本而非原样副本。本项目选择了开发与外部血缘记录，重建了 25 位 guide-target 对齐并显式记录 gap，增加稳定 ID、数据角色和冻结的 target-grouped folds，派生 188 项可审计特征，保留原始标签字段但排除量纲未确认记录的最终监督声明，并由此训练模型、生成预测、指标和不确定性分析。

本仓库不代表 EasyDesign 作者、iMeta 或 John Wiley & Sons 对本项目的认可。科学血缘和边界见[数据契约](docs/DATA_zh.md)。
