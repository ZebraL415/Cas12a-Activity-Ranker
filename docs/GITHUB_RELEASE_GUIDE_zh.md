# GitHub 上传与发布指南

以下步骤以公开目录 `Cas12a-Activity-Ranker` 和 GitHub 仓库 `ZebraL415/Cas12a-Activity-Ranker` 为例。

## 1. 仍需完成的人工决定

1. GitHub 用户名已确认为 `ZebraL415`，README 和远端示例已经更新。
2. 确认贡献者姓名、顺序、项目标题和引用方式，然后添加 `CITATION.cff`。

许可证核验已完成：原创软件使用 Apache-2.0；EasyDesign GitHub 原样数据沿用 Apache-2.0；加工数据和数据衍生研究产物按 CC BY 4.0 发布。精确边界见 `THIRD_PARTY_NOTICES.md`。

## 2. 安装并启用 Git LFS

```bash
brew install git-lfs
git lfs install
```

仓库的 `.gitattributes` 已配置大数据表、模型、XLSX 和 PPTX 使用 LFS。首次提交前确认：

```bash
git lfs track
```

## 3. 最后一次本地验收

```bash
cd /path/to/Cas12a-Activity-Ranker

python scripts/generate_sha256_manifest.py
python -m unittest discover -s tests -v
python scripts/verify_repository.py
python scripts/train_final_four.py --verify-only
python scripts/train_final_four.py --smoke-test
```

检查不会上传内部归档：

```bash
git status --short --ignored
git check-ignore -v _local_only/
```

## 4. 在 GitHub 创建空仓库

在 GitHub 新建 `Cas12a-Activity-Ranker`：

- Visibility 选择 Public；
- 不要让网页自动创建 README、`.gitignore` 或 LICENSE，避免与本地已完成的文件冲突。

## 5. 初始化并首次推送

HTTPS：

```bash
git init -b main
git lfs install
git add .
git lfs ls-files
git status --short
git commit -m "Release Cas12a Activity Ranker v1.0.0"
git remote add origin https://github.com/ZebraL415/Cas12a-Activity-Ranker.git
git push -u origin main
```

SSH 用户可把 remote 改为：

```bash
git remote add origin git@github.com:ZebraL415/Cas12a-Activity-Ranker.git
```

不要在 `git add .` 后直接推送；先检查 `git status` 和 `git lfs ls-files`，确认内部压缩包、旧 joblib 和日志没有进入 staged changes。

## 6. 推送后的 GitHub 设置

- Description：`Machine-learning toolkit for ranking Cas12a crRNA–target pairs by fluorescence-derived diagnostic activity.`
- Topics：`cas12a`, `crispr-diagnostics`, `machine-learning`, `bioinformatics`, `xgboost`, `catboost`, `crrna-design`, `reproducible-research`
- 开启 Issues 和 Private vulnerability reporting；Discussions 可选。
- 在 Branch protection 中要求 `main` 通过 Pull Request 和 `test` CI 后才能合并。
- 检查 Actions 页面，确认第一次 `test` 工作流为绿色。

## 7. 创建 v1.0.0 Release

```bash
git tag -a v1.0.0 -m "Cas12a Activity Ranker v1.0.0"
git push origin v1.0.0
```

在 GitHub Releases 中用该 tag 创建 Release，说明：

- 正式数据 SHA-256；
- XGBoost 主模型和 CatBoost 配套模型版本；
- 加权组合是探索性结果；
- 固定验证集不是 untouched external test；
- 已知的 unseen-guide 和量纲外推限制。
