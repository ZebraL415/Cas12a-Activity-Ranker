# GitHub 公开发布检查清单

## 发布前必须由项目负责人决定

- [x] README 中的 GitHub 用户名已替换为 `ZebraL415`。
- [ ] 确认仓库正式名称，推荐 `Cas12a-Activity-Ranker`。
- [ ] 确认作者顺序、项目名称和推荐引用方式，并补充 `CITATION.cff`。
- [x] 已选择 Apache-2.0 作为原创软件许可证，并添加 `LICENSE` 与 `NOTICE`。
- [x] 已核对 EasyDesign 官方仓库文件（Apache-2.0）及期刊补充材料（CC BY 4.0）的再分发条件；范围和改动见 `THIRD_PARTY_NOTICES.md`。
- [ ] 确认公开 PPT、示例和模型中不含个人信息、私有序列或未公开数据。

## Git 与 Git LFS

- [ ] 安装 Git LFS：`brew install git-lfs`，然后运行 `git lfs install`。
- [ ] 运行 `git lfs track` 检查 `.gitattributes` 已生效。
- [ ] 确认 `_local_only/` 整体被 `.gitignore` 排除。
- [ ] 首次提交前运行 `git status --short --ignored` 检查上传边界。

## 测试与 GitHub 设置

- [ ] 本地运行单元测试和 `scripts/verify_repository.py`。
- [ ] 推送后确认 GitHub Actions 的 `test` 工作流通过。
- [ ] 开启 Issues、Discussions（可选）和 Private vulnerability reporting。
- [ ] 设置仓库 Description、Topics 和主页链接。
- [ ] 保护 `main` 分支，要求 Pull Request 和 CI 通过后合并。
- [ ] 创建 `v1.0.0` Release，并记录数据 SHA、模型版本和已知限制。
