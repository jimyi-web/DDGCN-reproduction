# DDGCN 论文复现进度

更新时间：2026-09-10T02:18:18.095789+00:00。PHEME 指定实验已完成；Weibo 忠实复现仍受外部知识资源阻塞。

| 实验 | 已完成折数 | Accuracy 均值 ± 样本标准差 | Macro-F1 均值 ± 样本标准差 | 论文 Acc / F1 |
|---|---:|---:|---:|---:|
| PHEME DDGCN | 5/5 | 0.852298 ± 0.008771 | 0.840072 ± 0.011072 | 0.855 / 0.844 |
| PHEME w/o TF | 5/5 | 0.851415 ± 0.021519 | 0.837996 ± 0.023245 | 0.842 / 0.833 |
| PHEME w/o dynamic | 5/5 | 0.852637 ± 0.017245 | 0.840309 ± 0.018020 | 0.841 / 0.835 |
| PHEME w/o knowledge | 5/5 | 0.852471 ± 0.010640 | 0.839754 ± 0.013154 | 0.833 / 0.819 |
| PHEME w/o propagation | 5/5 | 0.854902 ± 0.012635 | 0.842737 ± 0.015417 | 0.832 / 0.821 |
| Weibo 忠实复现 | 0/5，知识资源阻塞 | — | — | 0.948 / 0.950 |

- 主实验目录 `/root/autodl-tmp/ddgcn-runs/pheme_cv_checkpointed` 保持只读。
- C 重载核验：5/5通过，5748条预测完全一致。
- B early detection：13个截止点×5折已完成；不重新生成 observation。
- D 四组消融：20/20 folds 完成；运行中的 epoch 不计入完成数。
- E Weibo：原始数据完整，entity linking / Probase 依赖未恢复；没有替代方案的忠实复现结果。
- F 文档由已落盘产物生成。若仍有任务未完成，这份表不是最终完成证明。

日志目录 `/root/autodl-tmp/ddgcn-runs`：pheme_remaining_queue.log、pheme_ablations_checkpointed.log、pheme_early_graphs_seed2026.log、pheme_early_evaluation.log。

命令、环境、哈希路径、逐截止点表、修改说明与差异见 [REPRODUCTION.md](REPRODUCTION.md)；资源证据见 [WEIBO_RESOURCE_STATUS.md](WEIBO_RESOURCE_STATUS.md)。
