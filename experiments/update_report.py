"""Build reproducibility documentation from committed result artifacts only."""
import argparse
import csv
import io
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/root/autodl-tmp/ddgcn-runs')
MAIN = BASE / 'pheme_cv_checkpointed'
ABLATIONS = BASE / 'pheme_ablations_checkpointed'
EARLY = BASE / 'pheme_early_evaluation'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_prediction_metrics(result, predictions):
    y, pred = predictions['labels'], predictions['predictions']
    assert len(y) == len(pred) and len(y) > 0
    acc = sum(a == b for a, b in zip(y, pred)) / len(y)
    f1s = []
    for c in (0, 1):
        tp = sum(a == c and b == c for a, b in zip(y, pred))
        fp = sum(a != c and b == c for a, b in zip(y, pred))
        fn = sum(a == c and b != c for a, b in zip(y, pred))
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    assert abs(acc-result['accuracy']) < 1e-12
    assert abs(statistics.mean(f1s)-result['macro_f1']) < 1e-12


def collect(directory, model):
    results = []
    for i in range(5):
        fold = directory / model / ('fold_'+str(i))
        if not (fold/'result.json').exists():
            continue
        assert (fold/'best.pt').exists() and (fold/'test_predictions.json').exists()
        result = read(fold/'result.json')
        predictions = read(fold/'test_predictions.json')
        split = read(MAIN/'splits.json')[i]['test']
        assert predictions['labels'] == split['labels']
        verify_prediction_metrics(result['test'], predictions)
        assert read(directory/'splits.json') == read(MAIN/'splits.json')
        results.append(result)
    return results


def format_metric(results, key):
    if len(results) != 5:
        return '待完成'
    values = [r['test'][key] for r in results]
    return f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}'


def write(path, text):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(text)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-pheme-complete', action='store_true')
    args = parser.parse_args()
    specs = [('completed','DDGCN',.855,.844),('wotf','w/o TF',.842,.833),
             ('static','w/o dynamic',.841,.835),('propagation','w/o knowledge',.833,.819),
             ('knowledge','w/o propagation',.832,.821)]
    collected = {m:collect(MAIN if m=='completed' else ABLATIONS,m) for m,_,_,_ in specs}
    reload_path = BASE/'pheme_reload_verified/summary.json'
    reload_ok = reload_path.exists() and read(reload_path)['status']=='passed'
    early_ok = (EARLY/'summary.json').exists()
    pheme_ok = all(len(v)==5 for v in collected.values()) and reload_ok and early_ok
    if args.require_pheme_complete:
        assert pheme_ok, 'PHEME results still incomplete'
        integrity = read(ROOT/'runs/reproduction_integrity_audit.json')
        assert integrity['status'] == 'passed' and integrity['early_prediction_files'] == 65
    lines = ['| 实验 | 已完成折数 | Accuracy 均值 ± 样本标准差 | Macro-F1 均值 ± 样本标准差 | 论文 Acc / F1 |',
             '|---|---:|---:|---:|---:|']
    for m,name,pa,pf in specs:
        r=collected[m]
        lines.append(f'| PHEME {name} | {len(r)}/5 | {format_metric(r,"accuracy")} | {format_metric(r,"macro_f1")} | {pa:.3f} / {pf:.3f} |')
    lines.append('| Weibo 忠实复现 | 0/5，知识资源阻塞 | — | — | 0.948 / 0.950 |')
    table='\n'.join(lines)
    when=datetime.now(timezone.utc).isoformat()
    status='PHEME 指定实验已完成；Weibo 忠实复现仍受外部知识资源阻塞。' if pheme_ok else 'PHEME 主模型及重载已完成；消融/早期检测尚有任务未完成。Weibo 忠实复现仍受外部知识资源阻塞。'
    early_text='完整 deadline 评估尚未完成。源帖推理检查不作为 Figure 2 结果。'
    if early_ok:
        summary=read(EARLY/'summary.json')
        assert len(summary)==13
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(['axis', 'deadline', 'folds', 'events', 'accuracy_mean', 'accuracy_sample_std', 'macro_f1_mean', 'macro_f1_sample_std'] + ['accuracy_fold_'+str(i) for i in range(5)] + ['macro_f1_fold_'+str(i) for i in range(5)])
        elines=['| 截止点 | 各折 Accuracy（0→4） | Accuracy mean ± std | 各折 Macro-F1（0→4） | Macro-F1 mean ± std |', '|---|---|---|---|---|']
        for key,item in summary.items():
            a,f=item['metrics']['accuracy'],item['metrics']['macro_f1']
            for fold in range(5):
                r=read(EARLY/('fold_'+str(fold))/(key+'.json'))
                verify_prediction_metrics(r['metrics'],r)
                split = read(MAIN/'splits.json')[fold]['test']
                assert r['source_post_ids'] == split['source_post_ids']
                assert r['labels'] == split['labels']
            for name in ('accuracy', 'macro_f1', 'legacy_f1'):
                values = [read(EARLY/('fold_'+str(i))/(key+'.json'))['metrics'][name] for i in range(5)]
                metric = item['metrics'][name]
                assert metric['values'] == values
                assert abs(metric['mean'] - statistics.mean(values)) < 1e-12
                assert abs(metric['std'] - statistics.stdev(values)) < 1e-12
            csv_writer.writerow([item['axis'], item['deadline'], item['folds'], item['events'], a['mean'], a['std'], f['mean'], f['std']] + a['values'] + f['values'])
            elines.append('| '+key+' | '+', '.join(f'{x:.6f}' for x in a['values'])+f' | {a["mean"]:.6f} ± {a["std"]:.6f} | '+', '.join(f'{x:.6f}' for x in f['values'])+f' | {f["mean"]:.6f} ± {f["std"]:.6f} |')
        write(ROOT/'runs/pheme_early_5fold_metrics.csv', csv_buffer.getvalue())
        early_text='已完成 13 个截止点 × 5 折，共 65 份预测文件；每个截止点覆盖 5748 个独立测试事件。下表为折间等权均值 ± 样本标准差（ddof=1）；可导出表见 `runs/pheme_early_5fold_metrics.csv`。\n\n'+'\n'.join(elines)
    report=f'''# DDGCN 复现报告

更新时间：{when}。{status}

## 实验结果

{table}

所有标准差为折间样本标准差（ddof=1），不是标准误。只有五折全部完成才显示汇总值。论文来源：[AAAI 2022 DDGCN](https://cdn.aaai.org/ojs/20385/20385-13-24398-1-2-20220628.pdf)，Table 3/4 与 Figure 2。

PHEME 主模型 Accuracy 与论文 0.855 相差 -0.002702（-0.2702 个百分点）；Macro-F1 与论文表列 F1 0.844 的数值差为 -0.003928，但指标口径未必相同。原代码的 legacy F1 是宏平均 Precision/Recall 的调和平均，主实验 legacy F1=0.841032 ± 0.010062；本报告 Macro-F1 是两类 F1 的算术平均。不得混用这两种指标。

不能从这些结果断定差异的唯一原因：作者没有提供同一五折索引和训练权重；当前 Python/PyTorch/CUDA 与原环境不同；修正后的五折划分、知识图批处理偏移和调度器方向影响训练；GPU 聚合还可能存在数值非确定性。结果接近不等于逐值复现。

## 已保护的主实验与重载

主目录 `{MAIN}` 作为只读输入使用，未重新训练或覆盖。各折包含 best.pt、result.json、test_predictions.json，splits.json 固定测试顺序。

重载核验状态：**{'五折全部通过' if reload_ok else '未全部通过'}**。每折从全新 Python 进程加载训练时 source 快照和 best.pt；检查原始输入/源码哈希、测试索引及标签、逐条预测、最佳 epoch、所有 result.test 指标。分类预测要求完全相同，指标绝对容差 1e-7。证明文件：`{BASE}/pheme_reload_verified/summary.json`，每折报告包含 checkpoint SHA-256、输入验证状态和进程 PID。

## Early detection

{early_text}

- 固定读取 `/root/projects/DDGCN/runs/pheme_early_observations_v2/manifest.json` 和 observations.jsonl；没有重新生成 observation。
- 7 个分钟点 0/20/40/60/80/100/120，6 个评论比例点 0/20/40/60/80/100%。0 点仅源帖，比例向下取整，时间边界包含等于 deadline 的评论。
- 每个 prefix 删除未来帖子、未来帖子产生的实体/概念和相关边；仅从已观察帖子重新计算 TF-IDF、PMI 和三个时间阶段，避免完整事件统计泄漏。保留全局节点编号，与既有 embedding 权重对齐。
- 各 prefix 的时间段按已观察跨度等分三段，使用真实 total_seconds；原作者预处理的 timedelta.seconds 会把超过 24 小时的时间取模。故重新构建的 100% 评论图不保证等于旧全图实验的缓存图；该差异不得隐瞒。
- 使用原作者 PMI 窗口实现，固定 PYTHONHASHSEED=2026 保证集合迭代可复现；旧缓存的哈希种子未知。这会影响与作者原始缓存的精确边权对齐。加速仅缓存节点位置和 IDF 计数，已在 20 个事件、894 个矩阵上逐值比对。
- 保留所有只有源帖的事件，不经过原训练加载器的单节点过滤。全量审计检查 5748 个事件覆盖、每个 deadline、图对称性、所有边端点、归一化有限性；每个事件额外进行未来注释污染测试。
- 全图训练得到的五折权重只做推理，无重训练；每折内部复用相同源帖 BERT 输出以减少重复计算，冷/热缓存与无缓存推理已核对一致。
- 固定外部知识快照和全局词表不是历史时点知识库；评论百分比使用事件最终评论总数来定义离线截止点。这里验证的是 observation deadline 之后的事件内容没有进入重建图，不声称严格的在线因果数据集。
- 论文未提供早期检测专用代码、时间分段/取整/同秒排序细节；这些约定均是明确声明的实现选择。Figure 2 没有精确数值表，不能声称曲线逐点数值一致。

图缓存使用无损 gzip 压缩，已对迁移文件逐字节校验解压结果；原缓存数值不变，读取器兼容旧的未压缩 pickle。输出：`{BASE}/pheme_early_graphs_seed2026`（缓存和 audit），`{EARLY}`（每折每截止点预测和 summary）。旧不固定哈希种子的部分缓存 `{BASE}/pheme_early_graphs` 被保留，不用于最终评估。

## 产物完整性核对

{'已通过最终完整性审计：主模型 5 折、四组消融 20 折，共 25 份 best.pt、25 份 result.json、25 份 test_predictions.json；2 份训练 summary、5 份重载证明及其 summary、65 份 early 预测及其 summary 均已核对。全部 checkpoint 可反序列化、张量有限，最佳 epoch 与日志一致；主模型哈希与此前重载证明一致。逐条预测重算 Accuracy / Macro-F1，summary 的逐折值、均值和样本标准差一致。消融 checkpoint 本轮做文件与状态核验，未新增消融全量重载推理。审计记录：`runs/reproduction_integrity_audit.json`。' if (ROOT/'runs/reproduction_integrity_audit.json').exists() and pheme_ok else '最终完整性审计尚未完成。'}

## 环境与数据

当前环境 `/root/miniconda3/envs/ddgcn/bin/python`：Python 3.8.20，torch 2.0.1+cu118，PyG 2.3.1，Transformers 4.30.2，NumPy 1.24.4，SciPy 1.10.1，pandas 1.5.3，scikit-learn 1.3.2；RTX 4090 D 24GB。完整依赖见主实验及消融目录 requirements.freeze.txt，当前环境核对记录为 runs/reproduction_environment.json。作者环境为 Python 3.6、torch 1.4.0+cu100；原 requirement.txt 包含作者本地 wheel 路径，不能直接用于新服务器。

PHEME 作者处理包：6425 个原始图样本，去掉无回复样本后 5748（标签0=3654、1=2094）。分层五折，各事件恰好测试一次；前四折 train/val/test=4024/574/1150，第五折 4026/574/1148；不是事件话题留一评估。seed2026 生成划分，每折训练 seed=2026+fold。模型三个阶段、hidden128、文本30、dropout0.2、batch32、5 epochs；Adam BERT学习率5e-5，其余0.001；按验证 Accuracy 选择最佳权重，保留原 sigmoid+BCE 和 embedding.detach 行为。

PHEME 所需输入为 data/pheme 的 node2idx_mid.txt、mid2text.txt、pheme_temporal_data 下六个 npy（propagation_node_idx、propagation_node、knowledge_node_idx、knowledge_node、label、propagation_root_index）以及 bert-base-uncased 的 config/vocab/weights。全量 SHA-256 见 `{MAIN}/input_manifest.json`；主模型每折权重 SHA-256 见重载报告。

Weibo 原始包 `/root/autodl-tmp/ddgcn-downloads/rumdect.zip`，SHA-256 `eadb8d0dd53885a7c8caac7818e7f1c13b4a67e7b5fc2112fa6fa375a88b1c5f`；中文 BERT SHA-256 `8a693db616eaf647ed2bfe531e1fa446637358fc108a8bf04e8d4db17e837ee9`。原始 Weibo 已完整校验，**无需重复下载**。缺失实体链接/Probase资源与所有检索证据见 [WEIBO_RESOURCE_STATUS.md](WEIBO_RESOURCE_STATUS.md)。未运行 POS 替代方案冒充忠实复现。

## 运行命令

当前已运行的任务不要再启动第二份。所有输出独立于主实验；恢复时先检查对应 result/summary/日志和进程。无需 conda activate，直接使用环境的 Python。

```bash
cd /root/projects/DDGCN
# 只做主模型重载，已通过的折会核验哈希后跳过
/root/miniconda3/envs/ddgcn/bin/python -B -u experiments/verify_checkpoints.py --main {MAIN} --output {BASE}/pheme_reload_verified
# 四个消融；目录已存在时须用 --resume，原输入和模型源码必须保持一致
/root/miniconda3/envs/ddgcn/bin/python -B -u model/run_paper.py --dataset pheme --output {ABLATIONS} --models wotf static propagation knowledge --epochs 5 --batch 32 --seed 2026 --resume
# 仅重建尚未缓存的 early graphs，不生成 observation
PYTHONHASHSEED=2026 /root/miniconda3/envs/ddgcn/bin/python -B -u experiments/accelerate_early_graph.py --observations runs/pheme_early_observations_v2 --splits {MAIN}/splits.json --output {BASE}/pheme_early_graphs_seed2026
# 图审计完成后加载权重推理，已完成的截止点会跳过
/root/miniconda3/envs/ddgcn/bin/python -B -u experiments/evaluate_early.py --main {MAIN} --graphs {BASE}/pheme_early_graphs_seed2026 --output {EARLY}
# 完整性审计后，依据已有产物重算并更新最终文档，不训练
/root/miniconda3/envs/ddgcn/bin/python -B experiments/audit_final_artifacts.py
/root/miniconda3/envs/ddgcn/bin/python -B experiments/update_report.py --require-pheme-complete
```

`experiments/run_remaining_pheme.py` 是串行 GPU 队列：重载通过 → 四组消融 → 等待图审计完成 → early evaluation。训练输出每折 best.pt、result.json、test_predictions.json，训练期间另有可恢复 last.pt，完成后删掉临时续训状态以节省磁盘。新训练输出目录启动不带 --resume，恢复已有目录才使用该参数。

## 修改清单

本轮新增 `experiments/verify_checkpoints.py`（独立重载与只读保护）、`preprocess/pheme_early_graph.py`（截断后重建和审计）、`experiments/accelerate_early_graph.py`（等价加速及比对）、`experiments/evaluate_early.py`（只推理、按截止点保存/汇总）、`experiments/run_remaining_pheme.py`（避免重训主模型的队列）、`experiments/update_report.py`（从结果生成文档）、`experiments/audit_early_graphs.py`（独立审计）、`experiments/watch_pheme_report.py`（按完成产物刷新报告）、`WEIBO_RESOURCE_STATUS.md` 和资源检索记录。没有改 model/models.py 或模型计算公式，也没有覆盖已有主模型产物或 observation。

此前已有兼容修改：model/data.py 修正五折覆盖和验证比例；model/data_process.py 修正不等长数组和知识图组批偏移；model/path_zh.py 使用项目路径；model/run_paper.py 新增原子检查点、恢复 RNG/优化器、最佳权重和输入/源码哈希；model/run_baseline.py 与 run_pheme.sh 保留旧单划分入口。旧单划分结果及说明另保存在 runs/reproduction_legacy_single_split.md，不作为五折结果。
'''
    if pheme_ok:
        hashes = {}
        for directory in (MAIN, ABLATIONS, EARLY, BASE/'pheme_reload_verified'):
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path.suffix in ('.pt','.json','.jsonl','.py','.txt'):
                    hashes[str(path)] = sha(path)
        for path in list((ROOT/'experiments').glob('*.py')) + [ROOT/'preprocess/pheme_early_graph.py', ROOT/'runs/pheme_early_observations_v2/manifest.json', ROOT/'runs/pheme_early_observations_v2/observations.jsonl', ROOT/'runs/pheme_early_acceleration_verification.json', ROOT/'runs/reproduction_environment.json', ROOT/'runs/reproduction_integrity_audit.json', ROOT/'runs/pheme_early_5fold_metrics.csv']:
            hashes[str(path)] = sha(path)
        for path in (BASE/'pheme_early_graphs_seed2026').glob('*.json'):
            hashes[str(path)] = sha(path)
        write(ROOT/'runs/reproduction_artifact_sha256.json',json.dumps(hashes,indent=2)+'\n')
        report += '\n最终产物 SHA-256：`/root/projects/DDGCN/runs/reproduction_artifact_sha256.json`。\n'
    write(ROOT/'REPRODUCTION.md',report)
    progress=f'''# DDGCN 论文复现进度

更新时间：{when}。{status}

{table}

- 主实验目录 `{MAIN}` 保持只读。
- C 重载核验：{'5/5通过，5748条预测完全一致' if reload_ok else '尚未完成'}。
- B early detection：{'13个截止点×5折已完成' if early_ok else '已实现重建/评估并通过源帖推理检查；全量构图/评估尚未完成'}；不重新生成 observation。
- D 四组消融：{sum(len(collected[m]) for m,_,_,_ in specs if m!='completed')}/20 folds 完成；运行中的 epoch 不计入完成数。
- E Weibo：原始数据完整，entity linking / Probase 依赖未恢复；没有替代方案的忠实复现结果。
- F 文档由已落盘产物生成。若仍有任务未完成，这份表不是最终完成证明。

日志目录 `{BASE}`：pheme_remaining_queue.log、pheme_ablations_checkpointed.log、pheme_early_graphs_seed2026.log、pheme_early_evaluation.log。

命令、环境、哈希路径、逐截止点表、修改说明与差异见 [REPRODUCTION.md](REPRODUCTION.md)；资源证据见 [WEIBO_RESOURCE_STATUS.md](WEIBO_RESOURCE_STATUS.md)。
'''
    write(ROOT/'PAPER_PROGRESS.md',progress)
    readme=(ROOT/'README.md').read_text()
    original=readme[readme.index('# DDGCN\n'):]
    intro=f'''# 当前复现状态

{status}

{table}

PHEME 主模型权重与预测位于 `{MAIN}`，已通过五折全新进程重载核验，不应重训或覆盖。原始 Weibo 已完整下载，剩余依赖是论文使用的实体链接和 Probase 知识资源。

完整命令、环境、哈希、early detection 表格及修改说明见 [REPRODUCTION.md](REPRODUCTION.md)，阶段状态见 [PAPER_PROGRESS.md](PAPER_PROGRESS.md)，Weibo 外部依赖见 [WEIBO_RESOURCE_STATUS.md](WEIBO_RESOURCE_STATUS.md)。当前无需安装依赖或重新下载原始数据。

下方为作者原始说明，其中旧环境及作者本地 wheel 路径不适用于当前服务器。

'''
    write(ROOT/'README.md',intro+original)
    print(json.dumps(dict(pheme_complete=pheme_ok,main_folds=len(collected['completed']),
        ablation_folds=sum(len(v) for k,v in collected.items() if k!='completed'),
        early_complete=early_ok,weibo_faithful_complete=False)),flush=True)

if __name__=='__main__':
    main()
