"""Recorded PHEME completed-model baseline: smoke test, train, validate and test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.loader import DataLoader
from config import dual_dynamic_graph_Config
from data_process import data_process_nn_nBatch_BERT, loadData_bert
from models import dual_DynamicGCN_nBatch_completed
from path_zh import path_Set_BERT, PROJECT_ROOT


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def input_manifest(paths):
    files = [Path(paths.path_node2idx_mid), Path(paths.path_mid2text)]
    files += [Path(paths.path_temporal) / name for name in (
        'propagation_node_idx.npy', 'propagation_node.npy', 'knowledge_node_idx.npy',
        'knowledge_node.npy', 'label.npy', 'propagation_root_index.npy')]
    files += [Path(paths.path_bert) / name for name in ('config.json', 'vocab.txt', 'pytorch_model.bin')]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required local inputs are missing:\n' + '\n'.join(missing))
    manifest = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        manifest[str(path.relative_to(PROJECT_ROOT))] = {
            'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    return manifest


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def test_checkpoint(model, checkpoint_path, dataset, split, device, out, batch_size, started, split_sizes):
    # Load on CPU so optimizer tensors do not occupy additional GPU memory.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    del checkpoint
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics, labels, predictions = evaluate(model, loader, device)
    result = {'best_epoch': best_epoch, 'test': metrics, 'total_seconds': time.time()-started,
              'split_sizes': split_sizes}
    save_json(out/'result.json', result)
    save_json(out/'test_predictions.json', {'source_post_ids': split['test']['source_post_ids'],
              'labels': labels, 'predictions': predictions})
    print('COMPLETE', json.dumps(result), flush=True)


def dataset_from_dict(d):
    return loadData_bert(d['idx_p'], d['x_p_indices'], d['x_p_values'],
                         d['idx_k'], d['x_k_indices'], d['x_k_values'], d['y'], d['root_idx_p'])


def check_batch(batch):
    """No edge may leave its own sample in either graph family."""
    for family in ('propagation', 'knowledge'):
        counts = batch['x_' + family + '_node_num'].tolist()
        for stage in range(3):
            if family == 'propagation' and stage == 0:
                edges, nums = batch.edge_index, batch.edge_num
            else:
                edges = batch[f'x_{family}_node_indices_{stage}_edge_index']
                nums = batch[f'x_{family}_edge_num_{stage}']
            edge_offset = node_offset = 0
            for nodes, n_edges in zip(counts, nums.tolist()):
                part = edges[:, edge_offset:edge_offset+n_edges]
                assert part.numel() == 0 or (int(part.min()) >= node_offset and int(part.max()) < node_offset+nodes)
                edge_offset += n_edges
                node_offset += nodes
            assert edge_offset == edges.shape[1]


def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            assert torch.isfinite(output).all(), 'Nonfinite evaluation output'
            loss_sum += torch.nn.functional.binary_cross_entropy(output, batch.target).item() * batch.num_graphs
            labels.extend(batch.target.argmax(1).cpu().tolist())
            predictions.extend(output.argmax(1).cpu().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    p, r = float(precision.mean()), float(recall.mean())
    return {'loss': loss_sum / len(labels), 'accuracy': accuracy_score(labels, predictions),
            'macro_precision': p, 'macro_recall': r, 'macro_f1': float(f1.mean()),
            'legacy_f1': 2*p*r/(p+r) if p+r else 0.0,
            'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(), 'support': support.tolist()}, labels, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--evaluate-run', help='Evaluate best.pt from a completed run; verify its inputs and split.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 2 or args.threads < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs/threads must be positive, batch >= 2, and lr finite and positive')
    if args.evaluate_run and args.smoke_only:
        parser.error('--evaluate-run and --smoke-only cannot be combined')
    reference = None
    if args.evaluate_run:
        reference = Path(args.evaluate_run).resolve()
        for name in ('config.json', 'input_manifest.json', 'split.json', 'best.pt'):
            if not (reference/name).is_file():
                parser.error('Evaluation requires ' + str(reference/name))
        previous = json.loads((reference/'config.json').read_text())
        args.seed = previous['seed']
        args.batch = previous['batch']
    paths = path_Set_BERT('pheme')
    print('Checking local data and BERT inputs...', flush=True)
    manifest = input_manifest(paths)
    if reference and manifest != json.loads((reference/'input_manifest.json').read_text()):
        raise ValueError('Input hashes differ from the saved run; refusing to evaluate a different dataset.')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'input_manifest.json', manifest)
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    started = time.time()
    save_json(out/'config.json', dict(vars(args), device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              torch=torch.__version__, cuda=torch.version.cuda, split=[0.7, 0.1, 0.2], model='completed', bert_lr=5e-5))
    (out/'requirements.freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (out/'source.diff').write_text(subprocess.check_output(['git', 'diff', '--', 'model'], cwd=PROJECT_ROOT, text=True))
    source_dir = out/'source'; source_dir.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        (source_dir/source.name).write_bytes(source.read_bytes())
    config = dual_dynamic_graph_Config()
    processor = data_process_nn_nBatch_BERT(config.text_max_length, paths, 'pheme')
    print('Loading all PHEME data...', flush=True)
    train, val, test, tokens, texts = processor.load_sparse_temporal_data(config.train, config.val, config.test)
    split = {}
    id2node = {idx: node for node, idx in tokens.items()}
    for name, d in [('train', train), ('val', val), ('test', test)]:
        roots = [int(x.item()) for x in d['root_idx_p']]
        split[name] = {'root_indices': roots, 'source_post_ids': [id2node[x] for x in roots], 'labels': [int(x.item()) for x in d['y']]}
    save_json(out/'split.json', split)
    if reference and split != json.loads((reference/'split.json').read_text()):
        raise ValueError('Regenerated split differs from the saved run; refusing to evaluate a different test set.')
    sets = [set(split[x]['root_indices']) for x in ['train', 'val', 'test']]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i+1,3))
    datasets = [dataset_from_dict(d) for d in [train, val, test]]
    print('Data loaded:', [len(d) for d in datasets], 'seconds:', time.time()-started, flush=True)
    model = dual_DynamicGCN_nBatch_completed(tokens, texts, paths.path_bert, config.n_class, config, device,
                                            n_hidden=3, dropout=0.2, instance_norm=False).to(device)
    if reference:
        test_checkpoint(model, reference/'best.pt', datasets[2], split, device, out,
                        args.batch, started, [len(d) for d in datasets])
        return
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    smoke_results = []
    for size in sorted(set([2, args.batch])):
        batch = next(iter(DataLoader(datasets[0], batch_size=size, shuffle=False, num_workers=0)))
        check_batch(batch)
        batch = batch.to(device)
        model.train(); model.zero_grad(set_to_none=True)
        if device.type == 'cuda': torch.cuda.synchronize()
        tick = time.time()
        output = model(batch)
        assert tuple(output.shape) == (size, 2)
        loss = torch.nn.functional.binary_cross_entropy(output, batch.target)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {name: float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        assert gradients and all(np.isfinite(x) for x in gradients.values())
        for prefix in ['bert.', 'layer_stack_propagation.', 'layer_stack_knowledge.', 'fc_rumor_2.']:
            assert any(name.startswith(prefix) and value > 0 for name, value in gradients.items()), prefix
        if device.type == 'cuda': torch.cuda.synchronize()
        result = {'batch': size, 'loss': loss.item(), 'seconds': time.time()-tick,
                  'gradient_norms': gradients, 'missing_gradients': [name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]}
        smoke_results.append(result)
        print('SMOKE PASS batch', size, 'loss', result['loss'], 'seconds', result['seconds'], flush=True)
        del output, loss, batch
    save_json(out/'smoke.json', smoke_results)
    model.load_state_dict(initial); del initial
    model.zero_grad(set_to_none=True)
    seed_all(args.seed)
    if args.smoke_only: return
    loaders = [DataLoader(d, batch_size=args.batch, shuffle=(i==0), num_workers=0) for i,d in enumerate(datasets)]
    bert_ids = {id(p) for p in model.bert.parameters()}
    optimizer = torch.optim.Adam([{'params': model.bert.parameters(), 'lr': 5e-5},
        {'params': [p for p in model.parameters() if id(p) not in bert_ids], 'lr': args.lr}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    best = -1.0
    for epoch in range(1, args.epochs+1):
        model.train(); total_loss = correct = count = 0
        tick = time.time()
        for step, batch in enumerate(loaders[0], 1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = torch.nn.functional.binary_cross_entropy(output, batch.target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            n = batch.num_graphs
            total_loss += loss.item()*n
            correct += int((output.argmax(1)==batch.target.argmax(1)).sum())
            count += n
            if step == 1 or step % 10 == 0 or step == len(loaders[0]):
                print(f'Epoch {epoch}/{args.epochs} batch {step}/{len(loaders[0])} loss={total_loss/count:.5f} acc={correct/count:.4f} elapsed={time.time()-tick:.1f}s', flush=True)
        metrics, _, _ = evaluate(model, loaders[1], device)
        scheduler.step(metrics['accuracy'])
        record = {'epoch': epoch, 'train_loss': total_loss/count, 'train_accuracy': correct/count,
                  'validation': metrics, 'seconds': time.time()-tick}
        with (out/'epochs.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print('EPOCH RESULT', json.dumps(record), flush=True)
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(), 'args': vars(args), 'validation': metrics}
        save_checkpoint(out/'last.pt', checkpoint)
        if metrics['accuracy'] > best:
            best = metrics['accuracy']; save_checkpoint(out/'best.pt', checkpoint)
    model.zero_grad(set_to_none=True)
    del checkpoint, optimizer, scheduler
    test_checkpoint(model, out/'best.pt', datasets[2], split, device, out,
                    args.batch, started, [len(d) for d in datasets])


if __name__ == '__main__':
    main()
