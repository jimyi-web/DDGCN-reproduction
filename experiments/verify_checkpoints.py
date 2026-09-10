"""Fresh-process verification of completed DDGCN folds; main run is read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def worker(args):
    # Reuse the exact implementation saved with the trained weights.
    sys.path.insert(0, str(args.main / 'source'))
    import torch
    from run_paper import (seed_all, MODELS, evaluate, dataset_from_dict,
                           data_process_nn_nBatch_BERT_nfold, DataLoader)
    from config import dual_dynamic_graph_Config
    from path_zh import path_Set_BERT
    # Snapshot path_Set_BERT derives ROOT from __file__; correct only its paths.
    import path_zh
    path_zh.PROJECT_ROOT = str(ROOT)
    cfg = json.loads((args.main / 'config.json').read_text())
    manifest = json.loads((args.main / 'input_manifest.json').read_text())
    for name, info in manifest.items():
        if name == 'source_sha256':
            for source, digest in info.items():
                assert sha(args.main / 'source' / source) == digest, source
        else:
            assert sha(ROOT / name) == info['sha256'], name
    torch.set_num_threads(cfg['threads'])
    seed_all(cfg['seed'])
    config = dual_dynamic_graph_Config()
    paths = path_Set_BERT('pheme')
    processor = data_process_nn_nBatch_BERT_nfold(config.text_max_length, paths, 'pheme')
    loaded = processor.load_sparse_temporal_data(config.train, config.val, config.test)
    tokens, texts = loaded[15:]
    data = loaded[3 * args.fold + 2]
    split = json.loads((args.main / 'splits.json').read_text())[args.fold]['test']
    inverse = {v: k for k, v in tokens.items()}
    roots = [int(r.item()) for r in data['root_idx_p']]
    assert roots == split['root_indices']
    assert [inverse[r] for r in roots] == split['source_post_ids']
    assert [int(y.item()) for y in data['y']] == split['labels']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_all(cfg['seed'] + args.fold)
    model = MODELS['completed'](tokens, texts, paths.path_bert, config.n_class,
        config, device, n_hidden=3, dropout=0.2, instance_norm=False).to(device)
    original = args.main / 'completed' / ('fold_' + str(args.fold))
    checkpoint_hash = sha(original / 'best.pt')
    checkpoint = torch.load(original / 'best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    metrics, labels, predictions = evaluate(model, DataLoader(dataset_from_dict(data),
        batch_size=cfg['batch'], shuffle=False, num_workers=0), device)
    expected = json.loads((original / 'test_predictions.json').read_text())
    result = json.loads((original / 'result.json').read_text())
    assert labels == expected['labels'], 'label order mismatch'
    assert predictions == expected['predictions'], 'prediction mismatch'
    import numpy as np
    for key, value in metrics.items():
        assert np.allclose(value, result['test'][key], rtol=0, atol=1e-7), (key, value, result['test'][key])
    assert checkpoint['epoch'] == result['best_epoch']
    assert sha(original / 'best.pt') == checkpoint_hash
    report = dict(status='passed', fold=args.fold, pid=os.getpid(), checkpoint_sha256=checkpoint_hash,
        predictions_sha256=sha(original / 'test_predictions.json'),
        result_sha256=sha(original / 'result.json'), splits_sha256=sha(args.main / 'splits.json'),
        source_manifest_verified=True, input_manifest_verified=True, prediction_mismatches=0,
        metric_absolute_tolerance=1e-7, device=str(device), test=metrics)
    (args.output / ('fold_' + str(args.fold) + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print('VERIFIED', args.fold, json.dumps(metrics), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fold', type=int, choices=range(5))
    args = p.parse_args()
    args.main = args.main.resolve()
    args.output = args.output.resolve()
    if args.output == args.main or args.main in args.output.parents:
        p.error('output must be outside the completed main run')
    if args.fold is not None:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for fold in range(5):
        path = args.output / ('fold_' + str(fold) + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            original = args.main / 'completed' / ('fold_' + str(fold))
            assert saved['status'] == 'passed'
            for name, key in [('best.pt', 'checkpoint_sha256'), ('test_predictions.json', 'predictions_sha256'), ('result.json', 'result_sha256')]:
                assert saved[key] == sha(original / name)
            assert saved['splits_sha256'] == sha(args.main / 'splits.json')
        else:
            with (args.output / ('fold_' + str(fold) + '.log')).open('a') as log:
                subprocess.run([sys.executable, '-B', '-u', __file__, '--main', str(args.main),
                    '--output', str(args.output), '--fold', str(fold)], stdout=log, stderr=subprocess.STDOUT, check=True)
        reports.append(json.loads(path.read_text()))
        print('RELOAD VERIFIED fold', fold, flush=True)
    (args.output / 'summary.json').write_text(json.dumps(dict(status='passed', folds=reports,
        separate_process_per_fold=True, predictions_exact=True), indent=2) + '\n')

if __name__ == '__main__':
    main()
