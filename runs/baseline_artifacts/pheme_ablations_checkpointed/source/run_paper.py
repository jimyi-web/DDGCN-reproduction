"""Run DDGCN's five-fold main and ablation experiments reproducibly."""
import argparse
import copy
import hashlib
import subprocess
import sys
import json
import os
from pathlib import Path
import random
import statistics
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.loader import DataLoader

from config import dual_dynamic_graph_Config, dual_dynamic_graph_Config_zh
from data_process import data_process_nn_nBatch_BERT_nfold, loadData_bert
from models import (
    dual_DynamicGCN_nBatch_completed,
    dual_DynamicGCN_nBatch_completed_static,
    dual_DynamicGCN_nBatch_completed_wotf,
    dual_DynamicGCN_nBatch_knowledge,
    dual_DynamicGCN_nBatch_propagation,
)
from path_zh import path_Set_BERT

MODELS = {
    'completed': dual_DynamicGCN_nBatch_completed,
    'wotf': dual_DynamicGCN_nBatch_completed_wotf,
    'static': dual_DynamicGCN_nBatch_completed_static,
    'propagation': dual_DynamicGCN_nBatch_propagation,
    'knowledge': dual_DynamicGCN_nBatch_knowledge,
}
PAPER_NAMES = {
    'completed': 'DDGCN',
    'wotf': 'DDGCN w/o TF',
    'static': 'DDGCN w/o dynamic',
    'propagation': 'DDGCN w/o knowledge',
    'knowledge': 'DDGCN w/o propagation',
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def dataset_from_dict(data):
    return loadData_bert(
        data['idx_p'], data['x_p_indices'], data['x_p_values'],
        data['idx_k'], data['x_k_indices'], data['x_k_values'],
        data['y'], data['root_idx_p'])


def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            if not torch.isfinite(output).all():
                raise RuntimeError('nonfinite evaluation output')
            loss_sum += torch.nn.functional.binary_cross_entropy(
                output, batch.target).item() * batch.num_graphs
            labels.extend(batch.target.argmax(1).cpu().tolist())
            predictions.extend(output.argmax(1).cpu().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0)
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    legacy_f1 = (2 * macro_precision * macro_recall /
                 (macro_precision + macro_recall)
                 if macro_precision + macro_recall else 0.0)
    return {
        'loss': loss_sum / len(labels),
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': float(f1.mean()),
        'legacy_f1': legacy_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'support': support.tolist(),
    }, labels, predictions


def train_one(model_name, fold, datasets, tokens, texts, paths, config,
              args, device, output):
    run_dir = output / model_name / ('fold_' + str(fold))
    result_path = run_dir / 'result.json'
    if result_path.is_file() and args.resume:
        if not (run_dir / 'best.pt').is_file():
            raise FileNotFoundError('Completed legacy run has no saved weights: ' + str(run_dir))
        print('SKIP completed', model_name, fold, flush=True)
        return json.loads(result_path.read_text())
    if run_dir.exists() and not (args.resume and (run_dir / 'last.pt').is_file()):
        raise FileExistsError(str(run_dir) + ' has no resumable checkpoint; preserve it and choose a new output')
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_seed = args.seed + fold
    seed_all(fold_seed)
    model = MODELS[model_name](
        tokens, texts, paths.path_bert, config.n_class, config, device,
        n_hidden=3, dropout=0.2, instance_norm=False).to(device)
    loaders = [DataLoader(dataset, batch_size=args.batch, shuffle=(i == 0),
                          num_workers=0) for i, dataset in enumerate(datasets)]
    bert_parameters = set(map(id, model.bert.parameters()))
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': args.bert_lr},
        {'params': [p for p in model.parameters() if id(p) not in bert_parameters],
         'lr': args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2)
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    start_epoch = 1
    elapsed = 0.0
    if args.resume and (run_dir / 'last.pt').is_file():
        checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_state = checkpoint['best_state']
        best_epoch = checkpoint['best_epoch']
        best_accuracy = checkpoint['best_accuracy']
        start_epoch = checkpoint['epoch'] + 1
        elapsed = checkpoint['elapsed']
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        del checkpoint
        # A crash after writing the epoch log but before checkpoint commit
        # must not leave duplicate records when the epoch is rerun.
        history = run_dir / 'epochs.jsonl'
        if history.exists():
            records = [json.loads(line) for line in history.read_text().splitlines()]
            history.write_text(''.join(json.dumps(record) + '\n' for record in records
                                      if record['epoch'] < start_epoch))
    started = time.time() - elapsed
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch in loaders[0]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output_value = model(batch)
            loss = torch.nn.functional.binary_cross_entropy(output_value, batch.target)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch.num_graphs
            correct += int((output_value.argmax(1) == batch.target.argmax(1)).sum())
            count += batch.num_graphs
        validation, _, _ = evaluate(model, loaders[1], device)
        scheduler.step(validation['accuracy'])
        record = {
            'epoch': epoch,
            'train_loss': loss_sum / count,
            'train_accuracy': correct / count,
            'validation': validation,
            'seconds': time.time() - started,
        }
        with (run_dir / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('EPOCH', model_name, fold, json.dumps(record), flush=True)
        if validation['accuracy'] > best_accuracy:
            best_accuracy = validation['accuracy']
            best_epoch = epoch
            best_state = copy.deepcopy({
                name: value.detach().cpu() for name, value in model.state_dict().items()})
            temporary = run_dir / 'best.pt.tmp'
            torch.save({'model_state_dict': best_state, 'epoch': best_epoch}, temporary)
            temporary.replace(run_dir / 'best.pt')
        temporary = run_dir / 'last.pt.tmp'
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_state': best_state, 'best_epoch': best_epoch,
            'best_accuracy': best_accuracy, 'elapsed': time.time() - started,
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        }, temporary)
        temporary.replace(run_dir / 'last.pt')

    model.load_state_dict(best_state)
    test, labels, predictions = evaluate(model, loaders[2], device)
    result = {
        'dataset': args.dataset,
        'model': model_name,
        'paper_name': PAPER_NAMES[model_name],
        'fold': fold,
        'seed': fold_seed,
        'best_epoch': best_epoch,
        'validation_accuracy': best_accuracy,
        'test': test,
        'split_sizes': [len(dataset) for dataset in datasets],
        'seconds': time.time() - started,
    }
    save_json(run_dir / 'test_predictions.json', {'labels': labels, 'predictions': predictions})
    save_json(result_path, result)
    (run_dir / 'last.pt').unlink(missing_ok=True)
    del model, optimizer, scheduler, best_state
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('FOLD COMPLETE', json.dumps(result), flush=True)
    return result


def aggregate(output, selected_models, selected_folds):
    summary = {}
    metric_names = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'legacy_f1')
    for model_name in selected_models:
        results = []
        for fold in selected_folds:
            path = output / model_name / ('fold_' + str(fold)) / 'result.json'
            if path.is_file():
                results.append(json.loads(path.read_text()))
        if not results:
            continue
        metrics = {}
        for metric in metric_names:
            values = [item['test'][metric] for item in results]
            metrics[metric] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'values': values,
            }
        summary[model_name] = {
            'paper_name': PAPER_NAMES[model_name],
            'folds': len(results),
            'metrics': metrics,
        }
    save_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('pheme', 'weibo'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', nargs='+', choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bert-lr', type=float, default=0.00005)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='Compatibility flag; best weights are always saved')
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds):
        parser.error('--folds must contain values from 0 through 4')
    if args.epochs < 1 or args.batch < 2 or args.threads < 1:
        parser.error('epochs/threads must be positive and batch must be at least 2')

    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        parser.error(str(output) + ' exists; use --resume or a new output')
    if args.resume and (output / 'config.json').is_file():
        previous = json.loads((output / 'config.json').read_text())
        for key in ('dataset', 'epochs', 'batch', 'lr', 'bert_lr', 'seed', 'threads'):
            if previous.get(key) != getattr(args, key):
                parser.error('Resume configuration mismatch: ' + key)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = dual_dynamic_graph_Config_zh() if args.dataset == 'weibo' else dual_dynamic_graph_Config()
    paths = path_Set_BERT(args.dataset)
    from run_baseline import input_manifest
    manifest = input_manifest(paths)
    sources = sorted(Path(__file__).parent.glob('*.py'))
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest['source_sha256'] = source_hashes
    manifest_path = output / 'input_manifest.json'
    if args.resume and (output / 'config.json').is_file():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume input/source manifest missing or mismatched')
    else:
        save_json(manifest_path, manifest)
        snapshot = output / 'source'
        snapshot.mkdir(exist_ok=True)
        for path in sources:
            (snapshot / path.name).write_bytes(path.read_bytes())
        (output / 'requirements.freeze.txt').write_text(subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True))
    save_json(output / 'config.json', dict(
        vars(args), device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
        split=[0.7, 0.1, 0.2], protocol='stratified five-fold cross-validation'))

    processor = data_process_nn_nBatch_BERT_nfold(config.text_max_length, paths, args.dataset)
    print('Loading and constructing all graph tensors...', flush=True)
    loaded = processor.load_sparse_temporal_data(config.train, config.val, config.test)
    fold_dicts = loaded[:15]
    tokens, texts = loaded[15:]
    all_datasets = []
    split_manifest = []
    inverse_tokens = {value: key for key, value in tokens.items()}
    for fold in range(5):
        dictionaries = fold_dicts[fold * 3:fold * 3 + 3]
        datasets = [dataset_from_dict(value) for value in dictionaries]
        all_datasets.append(datasets)
        split_item = {'fold': fold}
        for name, value in zip(('train', 'validation', 'test'), dictionaries):
            roots = [int(root.item()) for root in value['root_idx_p']]
            split_item[name] = {
                'root_indices': roots,
                'source_post_ids': [inverse_tokens[root] for root in roots],
                'labels': [int(label.item()) for label in value['y']],
            }
        split_manifest.append(split_item)
    split_path = output / 'splits.json'
    if args.resume and split_path.is_file():
        if json.loads(split_path.read_text()) != split_manifest:
            raise ValueError('Resume split mismatch')
    else:
        save_json(split_path, split_manifest)

    for model_name in args.models:
        for fold in args.folds:
            train_one(model_name, fold, all_datasets[fold], tokens, texts,
                      paths, config, args, device, output)
            aggregate(output, args.models, args.folds)
    summary = aggregate(output, args.models, args.folds)
    print('PAPER RUN COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
