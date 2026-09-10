# Weibo 忠实复现资源核查

核查日期：2026-09-09 UTC。结论：**原始数据完整；原论文的实体链接和概念输入仍不可恢复，未产出 Weibo 忠实复现指标。** 这表示当前证据范围内存在外部依赖阻塞，不表示这些资源永久不存在。

## 已具备

- 原始包 `/root/autodl-tmp/ddgcn-downloads/rumdect.zip`，完整 ZIP CRC 通过；SHA-256 `eadb8d0dd53885a7c8caac7818e7f1c13b4a67e7b5fc2112fa6fa375a88b1c5f`。
- `/root/autodl-tmp/DDGCN-data/weibo_raw`：4664 个事件，3,805,656 条帖子，原始标签与事件 ID 完整对应。
- 按作者代码排除 7 个指定事件后为 4657。原始标签计数 0=2345、1=2312；作者代码反转标签后计数相反。论文表格的类别命名与该反转需要进一步核对，不能只按类别数量推断标签含义。
- 中文 BERT：`/root/autodl-tmp/bert-base-chinese`；权重 SHA-256 `8a693db616eaf647ed2bfe531e1fa446637358fc108a8bf04e8d4db17e837ee9`；已验证离线加载。

## 检索证据

| 资源 | 检查结果 |
|---|---|
| [原仓库](https://github.com/MengzSun/DDGCN) | README 只提供 PHEME 已处理包、PHEME 原始包、Weibo 原始包；未发现 Weibo entity/concept 成品 |
| [Issues](https://github.com/MengzSun/DDGCN/issues) | 已保存先前成功获取的 5 个 issue 和 3 条评论；#2 回复原始包，#3 报告实体 API 批量调用失败，#5 请求共享 weibo_entity；未发现成品链接 |
| [Fork network](https://github.com/MengzSun/DDGCN/network/members) | 列出 5 个可见 fork：Dante-cjh、Peng057、Zhoujie-SONG、anshiquanshu66、lumen2018。逐个核对根目录、data、preprocess、releases；数据目录均仅 PHEME，未发现新 Weibo 资源链接。页面计数为 6，但公开列表只返回 5 个；未声称核查了不可见 fork |
| [Releases](https://github.com/MengzSun/DDGCN/releases) | 原仓库与可见 fork 未发现可下载的处理后 Weibo release 附件 |
| [作者实验室工具说明](https://www.atailab.cn/topics/pdf/5%20%E6%8F%90%E5%8F%96%E5%AE%9E%E4%BD%93%E5%B7%A5%E5%85%B7.pdf) | 仍指向 Shuyantech 实体链接和 CN-Probase；没有附带模型或知识快照 |
| [官方实体 API 文档](https://kw.fudan.edu.cn/apis/qa/) | 文档 HTTP 200；旧 `entitylinkingapi` 和文档 `api/entitylinking/cutsegment` 公共示例均 HTTP 404 |
| [官方 CN-Probase 文档](https://kw.fudan.edu.cn/apis/cnprobase/) | 文档 HTTP 200；概念 API 公共示例 HTTP 404，HTTPS 主机名证书不匹配 |
| [官方查询页](https://kw.fudan.edu.cn/cnprobase/search/) | 页面和 JavaScript 可访问；按网页实际使用的 `/cnprobase/api/getconcepts?q=复旦大学&start=0` 查询返回 HTTP 200、空 JSON `{}`，并未获取可用概念数据 |

原始 HTTP 响应、网页、fork 目录记录、先前 issue 记录保存在 `/root/projects/DDGCN/runs/weibo_resource_search/`。本轮 GitHub REST API 返回公共 IP 的 rate-limit 403，因此改用公开网页完成可见 fork 核查；这不是仓库不存在的证据。未使用源码中的作者 API key，未向作者发送消息，未通过忽略 TLS 验证伪装接口正常。

## 忠实复现还需获得什么

优先需要作者原始 `weibo_entity`、`weibo_concept`（或相同处理结果的完整图数组、节点编号、文本与标签映射）。否则至少需要论文时期可恢复的 Shuyantech 实体链接实现/权重及其 CN-DBpedia 实体规范化版本，以及 CN-Probase 概念关系与权重快照。只恢复现代可用 API，也仍要记录知识库版本漂移、链接结果变化与批量接口限制。

官方文档说明大规模使用需要联系知识工场申请 key。提供可用授权或作者处理后缓存是解除当前阻塞的实际条件；仅再次下载 Weibo 原始包不能补齐这些输入。

`preprocess/build_weibo_open.py` 的 jieba/POS、粗粒度概念和不同 PMI 窗口是**实验性替代方案**，不属于忠实复现；本轮没有运行该方案训练，也没有把替代结果填入论文对比表。
