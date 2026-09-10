"""Evaluate saved five-fold DDGCN weights on reconstructed observation graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import gzip
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
from run_paper import (MODELS, seed_all, evaluate, save_json, DataLoader, torch,
    dual_dynamic_graph_Config, path_Set_BERT, data_process_nn_nBatch_BERT_nfold, dataset_from_dict)
from data import normalize_adj, sparse_mx_to_torch


class CachedRootBert(torch.nn.Module):
    """Memoize deterministic source-text BERT outputs for inference only."""
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.cache = {}

    def forward(self, input_ids, attention_mask):
        if self.training or self.bert.training or torch.is_grad_enabled():
            raise RuntimeError('BERT cache is restricted to no-grad evaluation')
        key = (tuple(input_ids.flatten().tolist()), tuple(attention_mask.flatten().tolist()))
        if key not in self.cache:
            self.cache[key] = self.bert(input_ids, attention_mask=attention_mask)
        return self.cache[key]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample(graph, label):
    data = dict(idx_p=torch.tensor(graph['p_ids']).long(), idx_k=torch.tensor(graph['k_ids']).long(),
                y=torch.tensor([label]).long(), root_idx_p=torch.tensor([graph['root']]).long())
    for kind, short in [('propagation', 'p'), ('knowledge', 'k')]:
        indices, values = [], []
        for matrix in graph[kind]:
            idx, val, shape = sparse_mx_to_torch(normalize_adj(matrix.copy()))
            assert torch.isfinite(val).all()
            assert idx.shape[0] == 2 and int(idx.max()) < matrix.shape[0] and int(idx.min()) >= 0
            indices.append(idx)
            values.append(val)
        data['x_' + short + '_indices'] = indices
        data['x_' + short + '_values'] = values
    return dataset_from_dict({k: [v] for k, v in data.items()})[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.main, args.graphs, args.output = (x.resolve() for x in (args.main, args.graphs, args.output))
    if args.output == args.main or args.main in args.output.parents or args.output == args.graphs:
        p.error('output must be independent')
    audit = json.loads((args.graphs / 'audit.json').read_text())
    assert audit['status'] == 'passed' and audit['events'] == 5748
    provenance = json.loads((args.graphs / 'provenance.json').read_text())
    assert provenance['splits_sha256'] == sha(args.main / 'splits.json')
    main_manifest = json.loads((args.main / 'input_manifest.json').read_text())
    for name, digest in main_manifest['source_sha256'].items():
        # New runners may be added; actual model, layers, data, config must match.
        if name in ('models.py', 'layers.py', 'data.py', 'data_process.py', 'config.py', 'path_zh.py', 'util.py'):
            assert sha(ROOT / 'model' / name) == digest, name
    for name, info in main_manifest.items():
        if name != 'source_sha256':
            assert sha(ROOT / name) == info['sha256'], name
    from audit_early_graphs import audit as independent_audit
    graph_verification = independent_audit(args.graphs, args.main)
    cfg = json.loads((args.main / 'config.json').read_text())
    splits = json.loads((args.main / 'splits.json').read_text())
    settings = dict(main=str(args.main), graphs=str(args.graphs),
        graph_provenance_sha256=sha(args.graphs / 'provenance.json'),
        graph_cache_sha256=graph_verification['cache_sha256'],
        auditor_sha256=sha(Path(__file__).with_name('audit_early_graphs.py')),
        evaluator_sha256=sha(__file__), graph_audit=audit,
        checkpoints={str(i): sha(args.main / 'completed' / ('fold_' + str(i)) / 'best.pt') for i in range(5)},
        protocol='inference only; fixed full-event-trained checkpoints; sample standard deviation (ddof=1)',
        root_bert_cache='identical source token sequences within each frozen checkpoint',
        status_note='observation-only graph inference; full-event training and static global vocabulary retained')
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'graph_verification.json', graph_verification)
    if (args.output / 'config.json').exists():
        assert json.loads((args.output / 'config.json').read_text()) == settings
    else:
        save_json(args.output / 'config.json', settings)
    torch.set_num_threads(cfg['threads'])
    seed_all(cfg['seed'])
    config = dual_dynamic_graph_Config()
    paths = path_Set_BERT('pheme')
    processor = data_process_nn_nBatch_BERT_nfold(config.text_max_length, paths, 'pheme')
    processor.get_node2id()
    processor.get_mid2text()
    processor.get_token()
    tokens, texts = processor.node2idx_dict, processor.mid2bert_tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    windows = [('minutes', d) for d in (0, 20, 40, 60, 80, 100, 120)] + [('comments_percent', d) for d in (0, 20, 40, 60, 80, 100)]
    for fold in range(5):
        split = splits[fold]['test']
        out = args.output / ('fold_' + str(fold))
        out.mkdir(exist_ok=True)
        if all((out / (axis + '_' + str(d) + '.json')).exists() for axis, d in windows):
            print('SKIP completed early fold', fold, flush=True)
            continue
        seed_all(cfg['seed'] + fold)
        model = MODELS['completed'](tokens, texts, paths.path_bert, config.n_class, config, device,
            n_hidden=3, dropout=0.2, instance_norm=False).to(device)
        checkpoint = torch.load(args.main / 'completed' / ('fold_' + str(fold)) / 'best.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        del checkpoint
        model.bert = CachedRootBert(model.bert)
        bundles = []
        for eid, label in zip(split['source_post_ids'], split['labels']):
            with (args.graphs / 'events' / (eid + '.pkl')).open('rb') as f:
                magic = f.read(2)
                f.seek(0)
                bundle = pickle.load(gzip.GzipFile(fileobj=f) if magic == b"\x1f\x8b" else f)
            assert bundle['observation']['event_id'] == eid
            bundle['samples'] = {n: sample(g, label) for n, g in bundle['graphs'].items()}
            bundles.append(bundle)
        for axis, deadline in windows:
            target = out / (axis + '_' + str(deadline) + '.json')
            if target.exists():
                saved = json.loads(target.read_text())
                assert saved['source_post_ids'] == split['source_post_ids']
                assert saved['labels'] == split['labels']
                continue
            if axis == 'comments_percent' and deadline == 0:
                result = json.loads((out / 'minutes_0.json').read_text())
                result.update(axis=axis, deadline=deadline, reused_identical_observation='minutes_0')
            else:
                data = [b['samples'][b['observation'][axis + '_prefix_lengths'][str(deadline)]] for b in bundles]
                metrics, labels, predictions = evaluate(model, DataLoader(data, batch_size=cfg['batch'],
                    shuffle=False, num_workers=0), device)
                assert labels == split['labels'] and len(predictions) == len(split['source_post_ids'])
                result = dict(fold=fold, axis=axis, deadline=deadline, metrics=metrics,
                    source_post_ids=split['source_post_ids'], labels=labels, predictions=predictions)
            save_json(target, result)
            print('EARLY', fold, axis, deadline, result['metrics']['accuracy'], result['metrics']['macro_f1'], flush=True)
        del model, bundles
        torch.cuda.empty_cache()
    summary = {}
    for axis, deadline in windows:
        reports = [json.loads((args.output / ('fold_' + str(i)) / (axis + '_' + str(deadline) + '.json')).read_text()) for i in range(5)]
        all_ids = [eid for r in reports for eid in r['source_post_ids']]
        assert len(all_ids) == len(set(all_ids)) == 5748
        metrics = {}
        for name in ('accuracy', 'macro_f1', 'legacy_f1'):
            values = [r['metrics'][name] for r in reports]
            metrics[name] = dict(values=values, mean=statistics.mean(values), std=statistics.stdev(values))
        summary[axis + '_' + str(deadline)] = dict(axis=axis, deadline=deadline, folds=5, events=5748, metrics=metrics)
    save_json(args.output / 'summary.json', summary)
    print('EARLY EVALUATION COMPLETE', flush=True)

if __name__ == '__main__':
    main()
