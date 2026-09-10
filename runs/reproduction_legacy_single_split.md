# DDGCN PHEME 复现说明

本工作区运行的是完整 `dual_DynamicGCN_nBatch_completed` 模型。使用作者预处理数据和当前兼容环境，完成真实数据训练、验证及测试；不将运行基线等同于论文数值的精确复现。论文的所有数据集、消融和早期检测实验不在本次 PHEME 基线范围内。

## 直接运行

在当前服务器任意目录执行：

```bash
bash /root/projects/DDGCN/run_pheme.sh
```

默认参数：5 轮、batch 32、seed 2026、学习率 0.001。脚本直接使用 `/root/miniconda3/envs/ddgcn/bin/python`，不依赖当前终端是否激活 Conda。自动生成 `runs/pheme_completed_<UTC时间>_<进程号>/` 并保存同名 `.log`。换环境时通过 `DDGCN_PYTHON` 指定 Python 的绝对路径。

显式指定新输出目录及参数：

```bash
bash /root/projects/DDGCN/run_pheme.sh runs/my_pheme_run --epochs 5 --batch 32 --lr 0.001 --seed 2026
```

第一个参数是输出目录，后面是 Python 入口参数。相对路径以项目根目录为基准。目录或同名日志存在时拒绝覆盖。

只做真实批次检查：

```bash
bash /root/projects/DDGCN/run_pheme.sh runs/my_smoke --smoke-only
```

不训练，直接加载本次最佳权重测试：

```bash
bash /root/projects/DDGCN/run_pheme.sh runs/my_evaluation --evaluate-run runs/pheme_completed_seed2026_reproduced
```

测试模式继承原实验的 seed 和 batch，校验全部输入 SHA-256 和完整数据划分，加载 `best.pt` 后输出测试指标及每条预测。需要保留对应实验的 `config.json`、`input_manifest.json`、`split.json` 和 `best.pt`，以及相同数据、BERT 和模型源码。测试输出目录必须是新目录。当前没有断点续训参数。

## 数据清单和缺少的资源

PHEME 训练必需的 11 个文件均已具备：

```text
data/pheme/node2idx_mid.txt
data/pheme/mid2text.txt
data/pheme/pheme_temporal_data/propagation_node_idx.npy
data/pheme/pheme_temporal_data/propagation_node.npy
data/pheme/pheme_temporal_data/knowledge_node_idx.npy
data/pheme/pheme_temporal_data/knowledge_node.npy
data/pheme/pheme_temporal_data/label.npy
data/pheme/pheme_temporal_data/propagation_root_index.npy
bert-base-uncased/config.json
bert-base-uncased/vocab.txt
bert-base-uncased/pytorch_model.bin
```

入口启动时先检查缺失文件，并对这些文件计算 SHA-256，记录为 `input_manifest.json`。全部训练使用本地文件；一键入口强制启用 Hugging Face 离线模式。

现有 `pheme_clean/`、`pheme_entity/` 和 `pheme_concept_yago/` 是中间结果，训练入口不直接读取它们。原始 PHEME 线程目录 `all-rnr-annotated-threads/` 不在工作区，但已有时序图，因此不影响训练。

若要复现 Weibo，仍缺 `data/weibo/` 中的文本、编号映射、六个时序图数组和 `bert-base-chinese/`。若从原始数据重新构图，还需实体识别服务和 YAGO 知识库文件；预处理脚本有作者硬编码路径和默认执行流程，不能直接当成通用一键管线。原始数据来源保留在 README 中。本次没有下载这些非 PHEME 训练必需的资源，也没有执行预处理联网脚本。

加载器读取 6425 条 PHEME 样本，过滤无回复的单节点样本后剩余 5748 条。按标签分层随机划分为训练 4022、验证 576、测试 1150，比例 70%/10%/20%。标签 0 为非谣言、1 为谣言；这不是事件留一划分。

## 环境与参数

当前已安装环境：Python 3.8.20、PyTorch 2.0.1+cu118、PyG 2.3.1、torch-scatter 2.1.2+pt20cu118、torch-sparse 0.6.18+pt20cu118、torch-cluster 1.6.3+pt20cu118、Transformers 4.30.2、NumPy 1.24.4、SciPy 1.10.1、pandas 1.5.3、scikit-learn 1.3.2。每次运行保存完整 `requirements.freeze.txt`。

作者原始环境是 Python 3.6 / PyTorch 1.4.0+cu100。原 `requirement.txt` 含作者机器的本地 wheel 路径，不能在新机器直接安装；迁移时须使用对应 Python/PyTorch/CUDA 的 wheel 来源。当前服务器无需重装。

使用 RTX 4090 D（约 24 GB 显存）。CUDA 不可用时入口会选择 CPU，但本次验证环境是 GPU。容器内存限制为 80 GiB；知识图文件约 4.53 GB，整体加载与转换会额外占用内存。`best.pt` 和 `last.pt` 合计约 2.7 GiB，原子替换写入还需为一个临时检查点预留空间。

模型使用三个图阶段、隐藏维度 128、文本长度 30、dropout 0.2；Adam 优化器，BERT 学习率 5e-5，其余参数 0.001；保留原 sigmoid + BCE 分类方式。按验证准确率最大化进行调度和最佳权重选择，最终只加载最佳权重做测试。

## 修改清单

### 本次新增及完善

- `run_pheme.sh`：一键启动，固定使用现有环境、离线加载、自动命名输出、拒绝覆盖旧实验、保存完整日志，并通过 pipefail 传递训练失败退出码。
- `model/run_baseline.py`：新增 11 个输入文件的预检查和 SHA-256 清单；新增 `--evaluate-run` 独立测试模式及输入/划分一致性校验；新增参数检查；权重先写临时文件再替换，避免中断破坏已有检查点；最终测试前释放优化器引用，并从 CPU 加载检查点以减少 GPU 内存重复占用。
- `README.md`：增加当前服务器的一键运行说明，保留作者原始说明。
- `REPRODUCTION.md`：重写为实际运行、缺失数据、环境、修改和结果说明。

### 之前已经存在，本次沿用并验证

- `model/data.py`：SciPy 改用公开 eigsh 导入；缺失的 `path` 导入改成 `path_zh`。
- `model/path_zh.py`：数据、BERT、保存路径根据源码目录定位，移除作者机器绝对路径。
- `model/data_process.py`：SciPy 公开导入；不等长数组使用 `dtype=object`；新增 `DualGraphData`，使知识图边按知识图节点数进行组批偏移。
- `model/run_baseline.py` 已有功能：种子固定、划分/源码/依赖记录，真实 batch 2/32 的图边与梯度检查，按样本数统计指标、验证集不打乱、最佳和最后权重保存、最终测试及逐条预测。

未改动 `model/models.py` 的模型公式，也未将原入口 `model/train_dynamic.py` 的结果冒充本次结果。原入口存在调度方向和未实际保存权重等历史问题；当前文档统一使用新的入口。

## 指标与复现限制

`macro_f1` 是两类 F1 的算术平均，`legacy_f1` 是宏平均 Precision/Recall 的调和平均，两者分别记录。不要混用。

保留原完整模型的 `embedding.detach()` 行为，节点 embedding 不更新；未使用的 GRU、BERT pooler、mean.linear 也没有梯度。`smoke.json` 记录了这些参数。BERT 编码器、双图 GCN、时序融合和分类层保持原计算方式。

固定 seed 能复用划分；CUDA 图聚合仍可能有非确定性，重复训练的浮点结果不保证逐位一致。精确比较论文数值仍需对齐作者数据划分、环境、指标口径和实验协议。

## 本次实验

正式输出目录：`runs/pheme_completed_seed2026_reproduced/`。
完整控制台日志：`runs/pheme_completed_seed2026_reproduced.log`。

本次训练已完成，最终结果见该目录中的 `result.json` 和日志中的 `COMPLETE`。旧目录 `runs/pheme_completed_seed2026/` 只有第一轮完整验证记录，缺少最终测试结果；它被保留作历史记录。

预期产物：`config.json`、`input_manifest.json`、`requirements.freeze.txt`、`source/`、`source.diff`、`split.json`、`smoke.json`、`epochs.jsonl`、`best.pt`、`last.pt`、`result.json`、`test_predictions.json`。

## 本次实测结果（2026-09-08）

五轮训练、验证和最终测试均正常完成（退出码 0），最佳权重来自第 5 轮。数据加载至最终测试约 24.80 分钟。测试集共 1150 条，其中预测正确 984 条。

| 测试指标 | 数值 |
|---|---:|
| Accuracy | 85.5652% |
| Macro-Precision | 84.2955% |
| Macro-Recall | 84.8258% |
| Macro-F1 | 84.5411% |
| Legacy-F1 | 84.5598% |

混淆矩阵（行为真实类别，列为预测类别）：

| | 预测非谣言 0 | 预测谣言 1 |
|---|---:|---:|
| 真实非谣言 0 | 640 | 91 |
| 真实谣言 1 | 75 | 344 |

| 轮次 | 训练 Accuracy | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|---:|
| 1 | 77.37% | 84.38% | 82.77% |
| 2 | 88.34% | 85.07% | 83.16% |
| 3 | 93.91% | 85.24% | 84.30% |
| 4 | 97.17% | 84.20% | 83.13% |
| 5 | 98.16% | 86.63% | 85.47% |

实际产物：`runs/pheme_completed_seed2026_reproduced/result.json`、`best.pt`、`last.pt`、`test_predictions.json` 及完整日志。指标已从逐条预测独立重算，五轮记录、最佳轮次选择、1150 个测试帖子 ID 的唯一性均已检查。

`verification.json` 记录缺失输入、哈希变化、非法参数、模拟中断保存保护等检查结果。本次输入哈希和划分与旧实验完全一致；当前模型源码与本次训练快照一致。原始数据、旧实验和模型计算公式均保留。

独立进程重载验证已通过：`runs/pheme_completed_seed2026_recheck/` 正常结束，1150 条预测与训练结束时完全一致，Accuracy、Precision、Recall 和 F1 全部一致；loss 浮点差约 1.24e-9。仅测试总用时约 3.03 分钟（包含重新加载和转换全部图数据）。
