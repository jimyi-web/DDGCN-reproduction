"""Read-only integrity audit of the selected PHEME reproduction artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import torch
from update_report import ROOT, BASE, MAIN, ABLATIONS, EARLY, read, sha, write, verify_prediction_metrics


def check_summary(item, reports, field):
    assert item['folds'] == 5
    for name, metric in item['metrics'].items():
        values = [(r[field] if field else r)[name] for r in reports]
        assert metric['values'] == values, name
        assert abs(metric['mean'] - statistics.mean(values)) < 1e-12, name
        assert abs(metric['std'] - statistics.stdev(values)) < 1e-12, name


def audit(training_only=False):
    torch.set_num_threads(2)
    splits = read(MAIN/'splits.json')
    ids = [eid for s in splits for eid in s['test']['source_post_ids']]
    assert len(ids) == len(set(ids)) == 5748
    reload = read(BASE/'pheme_reload_verified/summary.json')
    assert reload['status'] == 'passed' and len(reload['folds']) == 5
    checks, hashes = [], {}
    for directory, models in ((MAIN, ['completed']), (ABLATIONS, ['wotf', 'static', 'propagation', 'knowledge'])):
        assert read(directory/'splits.json') == splits
        for name in ('config.json', 'input_manifest.json', 'requirements.freeze.txt', 'summary.json'):
            assert (directory/name).stat().st_size > 0, str(directory/name)
        manifest = read(directory/'input_manifest.json')
        for name, value in manifest.items():
            if name == 'source_sha256':
                for source, digest in value.items():
                    assert sha(directory/'source'/source) == digest, source
            else:
                path = ROOT/name
                if str(path) not in hashes:
                    hashes[str(path)] = sha(path)
                assert hashes[str(path)] == value['sha256'], name
                assert path.stat().st_size == value['bytes'], name
        summary = read(directory/'summary.json')
        assert set(summary) == set(models)
        for model in models:
            reports, schema = [], None
            for fold in range(5):
                folder = directory/model/('fold_'+str(fold))
                result, pred = read(folder/'result.json'), read(folder/'test_predictions.json')
                assert (result['model'], result['fold'], result['seed']) == (model, fold, 2026+fold)
                assert pred['labels'] == splits[fold]['test']['labels']
                assert set(pred['predictions']) <= {0, 1}
                verify_prediction_metrics(result['test'], pred)
                epochs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
                assert [e['epoch'] for e in epochs] == list(range(1, 6))
                best = max(epochs, key=lambda e: e['validation']['accuracy'])
                assert result['best_epoch'] == best['epoch']
                assert result['validation_accuracy'] == best['validation']['accuracy']
                checkpoint = torch.load(folder/'best.pt', map_location='cpu')
                assert checkpoint['epoch'] == result['best_epoch']
                state = checkpoint['model_state_dict']
                current = {k: (list(v.shape), str(v.dtype)) for k, v in state.items()}
                assert current and (schema is None or current == schema)
                schema = current
                assert all(torch.isfinite(v).all().item() for v in state.values())
                del checkpoint, state
                digest = sha(folder/'best.pt')
                hashes[str(folder/'best.pt')] = digest
                if model == 'completed':
                    old = reload['folds'][fold]
                    assert old == read(BASE/'pheme_reload_verified'/('fold_'+str(fold)+'.json'))
                    assert old['status'] == 'passed' and old['prediction_mismatches'] == 0
                    for filename, key in [('best.pt','checkpoint_sha256'), ('result.json','result_sha256'), ('test_predictions.json','predictions_sha256')]:
                        assert sha(folder/filename) == old[key]
                    assert sha(MAIN/'splits.json') == old['splits_sha256']
                assert not list(folder.glob('*.tmp')) and not (folder/'last.pt').exists()
                checks.append(dict(model=model, fold=fold, checkpoint_sha256=digest, best_epoch=result['best_epoch'], test_events=len(pred['labels'])))
                reports.append(result)
            check_summary(summary[model], reports, 'test')
            print('AUDITED', model, '5/5', flush=True)
    early_count = 0
    if not training_only:
        summary, config = read(EARLY/'summary.json'), read(EARLY/'config.json')
        windows = [('minutes', n) for n in (0,20,40,60,80,100,120)] + [('comments_percent', n) for n in (0,20,40,60,80,100)]
        assert set(summary) == {axis+'_'+str(n) for axis,n in windows}
        verification = read(EARLY/'graph_verification.json')
        assert verification['status'] == 'passed' and verification['events'] == 5748
        assert verification['cache_sha256'] == config['graph_cache_sha256']
        graphs = BASE/'pheme_early_graphs_seed2026'
        assert config['graph_provenance_sha256'] == sha(graphs/'provenance.json')
        assert set(verification['cache_sha256']) == set(ids)
        for eid, digest in verification['cache_sha256'].items():
            assert sha(graphs/'events'/(eid+'.pkl')) == digest, eid
        assert verification['observation_sha256'] == sha(ROOT/'runs/pheme_early_observations_v2/observations.jsonl')
        assert config['evaluator_sha256'] == sha(ROOT/'experiments/evaluate_early.py')
        assert config['auditor_sha256'] == sha(ROOT/'experiments/audit_early_graphs.py')
        for fold in range(5):
            assert config['checkpoints'][str(fold)] == hashes[str(MAIN/'completed'/('fold_'+str(fold))/'best.pt')]
        for axis, n in windows:
            key = axis+'_'+str(n)
            reports, population = [], []
            for fold in range(5):
                r = read(EARLY/('fold_'+str(fold))/(key+'.json'))
                assert (r['fold'], r['axis'], r['deadline']) == (fold, axis, n)
                assert r['source_post_ids'] == splits[fold]['test']['source_post_ids']
                assert r['labels'] == splits[fold]['test']['labels']
                assert set(r['predictions']) <= {0,1}
                verify_prediction_metrics(r['metrics'], r)
                if axis == 'comments_percent' and n == 0:
                    zero = read(EARLY/('fold_'+str(fold))/'minutes_0.json')
                    assert r['predictions'] == zero['predictions'] and r['metrics'] == zero['metrics']
                reports.append(r)
                population.extend(r['source_post_ids'])
                early_count += 1
            assert len(population) == len(set(population)) == summary[key]['events'] == 5748
            assert (summary[key]['axis'], summary[key]['deadline']) == (axis,n)
            check_summary(summary[key], reports, 'metrics')
        assert not list(EARLY.rglob('*.tmp'))
    result = dict(status='training_passed' if training_only else 'passed', checked_at=datetime.now(timezone.utc).isoformat(),
        training_folds=len(checks), checkpoints_readable_and_finite=len(checks), training_results=25,
        training_predictions=25, training_summaries=2, reload_folds=5,
        early_prediction_files=early_count, early_windows=early_count//5, events_per_window=5748,
        summary_values_means_sample_std_recomputed=True, main_unchanged_since_reload=True,
        checkpoint_validation='CPU deserialization, finite tensors, matching per-model fold schemas and best epoch; main hashes match previous independent inference reload',
        weibo_faithful_complete=False, weibo_blocker='External entity linking / Probase resources unavailable; no approximation counted as faithful reproduction',
        folds=checks, input_and_checkpoint_sha256=hashes)
    target = ROOT/'runs'/('reproduction_training_audit.json' if training_only else 'reproduction_integrity_audit.json')
    write(target, json.dumps(result, indent=2)+'\n')
    print('AUDIT SAVED', target, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-only', action='store_true')
    audit(p.parse_args().training_only)
