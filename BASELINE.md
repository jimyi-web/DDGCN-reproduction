# PHEME 完整复现基线

基线 tag：`reproduction-pheme-complete`；分支：`master`。

本版本冻结已经完成的 PHEME 复现实验代码和结果。此后模型变更应在新分支进行，使用独立输出目录，不移动本 tag，不覆盖基线 checkpoint、预测或审计证据。Weibo 仍受外部知识资源阻塞，不属于“完整完成”的声明范围。

## 保存与验证

- 现有代码、README.md、REPRODUCTION.md、PAPER_PROGRESS.md、WEIBO_RESOURCE_STATUS.md 已纳入版本控制。本次归档未修改模型代码或结构。
- `runs/pheme_early_5fold_metrics.csv`、环境记录、完整性/训练审计 JSON 与审计日志、产物 SHA-256 清单均保留。
- `runs/baseline_artifacts/` 保存服务器外部实验目录的 183 份文件副本，包括训练与 early summary、25 折结果和预测、65 份 early 预测、重载报告、划分、配置、环境和训练源码快照。
- `runs/baseline_archive_manifest.json` 记录副本的服务器来源和 SHA-256；对原产物清单中的非 checkpoint 文件已重新核验哈希。checkpoint 保留原完整性审计结论，本次不重新训练或加载权重。
- 提交前进行了文件大小和常见密钥模式检查，未发现匹配；最大文件约 1.55 MiB。历史源码及原始证据中的换行/空白保持原样，避免破坏快照哈希。

## 不上传的本地文件

`.gitignore` 排除数据集 `data/`、下载包 `downloads/`、`bert-base-*` 预训练资源（含本地软链接）、checkpoint/权重、图和数组缓存、Python 缓存、编辑器配置、临时日志与备份、原始 observations.jsonl、密钥和私有配置。CSV、summary、预测、epoch 指标和审计记录不按目录整体忽略。

`data/pheme/` 原已跟踪的七个文件仅从 Git 索引移除，服务器文件未删除。它们仍存在于旧提交历史；本次不重写历史。

服务器上的 `/root/autodl-tmp/ddgcn-runs/`、下载数据、权重及工作区已有 checkpoint、预测、观察记录和审计文件均保留。GitHub 基线不包含运行所需的大型数据和权重，恢复实验时须结合 REPRODUCTION.md 的数据路径与哈希清单。
