"""Reconstruct observation-only PHEME graphs using the released graph equations.

Consumes existing observation manifests without rewriting them. Global node IDs
and cached per-post entity/concept annotations are held fixed for checkpoint
compatibility; event graph weights and temporal segments use observed posts only.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import gzip
import os
from types import SimpleNamespace
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('released_pheme_graph', ROOT / 'preprocess/getTwittergraph.py')
released = importlib.util.module_from_spec(spec)
spec.loader.exec_module(released)
released.np = SimpleNamespace(zeros=np.zeros, float=float)
released.tqdm = lambda iterable, **kwargs: iterable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotations(event_id):
    records = {}
    with (ROOT / 'data/pheme/pheme_entity' / (event_id + '.csv')).open() as f:
        for row in csv.DictReader(f):
            mid = row['mid'].strip()
            if mid in records:
                continue
            entities = [v.strip(' ').strip("'").strip(' ') for v in
                        row['entity'].strip('\t').strip('[').strip(']').split(',')]
            records[mid] = dict(parent=row['parent'].strip(), entity=[v for v in entities if v])
    with (ROOT / 'data/pheme/pheme_concept_yago' / (event_id + '.csv')).open() as f:
        for row in csv.DictReader(f):
            concepts = [' '.join(v.strip(' ').strip("'").strip('<').strip('>').split('_')).strip()
                        for v in row['concept'].strip('\t').strip('[').strip(']').split(',')]
            records[row['mid'].strip()]['concept'] = [v for v in concepts if v]
    return records


def construct(records, observation, count, mapping):
    ids = observation['ordered_post_ids'][:count]
    selected = set(ids)
    times = dict(zip(ids, observation['elapsed_seconds'][:count]))
    # Preserve released CSV ordering for PMI windows, but only pass observed rows.
    tree = {mid: dict(parent=(row['parent'] if row['parent'] in selected else 'None'),
                     entity=list(row['entity']), concept=list(row['concept']), t=times[mid])
            for mid, row in records.items() if mid in selected}
    assert set(tree) == selected and observation['event_id'] in tree
    assert all(row['t'] >= 0 for row in tree.values())
    local_names = set(tree)
    for row in tree.values():
        local_names.update(row['entity'])
        local_names.update(row['concept'])
    local_mapping = {name: mapping[name] for name in local_names}
    p_ids = [mapping[mid] for mid in tree]
    k_ids = sorted(local_mapping.values())
    assert len(set(p_ids)) == len(p_ids) and len(set(k_ids)) == len(k_ids)
    propagation = released.build_temporal_propagation_graph(p_ids, local_mapping, tree)
    knowledge = released.build_temporal_knowledge_graph(k_ids, local_mapping, tree)
    matrices = []
    min_degree = float('inf')
    for values in (propagation, knowledge):
        stages = []
        for dense in values:
            graph = sparse.csr_matrix(dense)
            assert np.isfinite(graph.data).all()
            assert (graph != graph.T).nnz == 0
            # The released normalization adds self loops. Check its domain here.
            degree = np.asarray(graph.sum(1)).ravel() + 1
            min_degree = min(min_degree, float(degree.min()))
            if not (degree > 0).all():
                raise ValueError(('nonpositive normalization degree', observation['event_id'], count))
            coo = graph.tocoo()
            assert all(0 <= v < graph.shape[0] for v in coo.row)
            assert all(0 <= v < graph.shape[1] for v in coo.col)
            stages.append(graph)
        matrices.append(stages)
    assert set(p_ids).issubset(k_ids)
    return dict(p_ids=p_ids, k_ids=k_ids, propagation=matrices[0], knowledge=matrices[1],
                observed_ids=ids, root=mapping[observation['event_id']],
                max_elapsed=max(times.values()), min_normalization_degree=min_degree)


def same_graph(a, b):
    assert a['p_ids'] == b['p_ids'] and a['k_ids'] == b['k_ids']
    for kind in ('propagation', 'knowledge'):
        for x, y in zip(a[kind], b[kind]):
            assert (x != y).nnz == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "2026":
        p.error("set PYTHONHASHSEED=2026 for reproducible released PMI set iteration")
    args.output = args.output.resolve()
    if args.output == args.observations.resolve() or args.observations.resolve() in args.output.parents:
        p.error('use an independent output directory')
    manifest = json.loads((args.observations / 'manifest.json').read_text())
    assert sha(args.splits) == manifest['splits_sha256']
    records = [json.loads(s) for s in (args.observations / 'observations.jsonl').read_text().splitlines()]
    splits = json.loads(args.splits.read_text())
    ids = [eid for fold in splits for eid in fold['test']['source_post_ids']]
    assert len(ids) == len(set(ids)) == len(records) == 5748
    assert set(ids) == {r['event_id'] for r in records}
    mapping = dict((line.rstrip('\n').split('\t')[0], int(line.rstrip('\n').split('\t')[1]))
                   for line in (ROOT / 'data/pheme/node2idx_mid.txt').read_text().splitlines())
    provenance = dict(storage="lossless gzip pickle; legacy raw pickle readable", python_hash_seed=os.environ["PYTHONHASHSEED"], observations_sha256=sha(args.observations / 'observations.jsonl'),
        observation_manifest_sha256=sha(args.observations / 'manifest.json'),
        splits_sha256=sha(args.splits), mapping_sha256=sha(ROOT / 'data/pheme/node2idx_mid.txt'),
        released_builder_sha256=sha(ROOT / 'preprocess/getTwittergraph.py'), builder_sha256=sha(__file__),
        time_policy='total_seconds; thirds of observed event duration',
        pmi_order='released CSV row order restricted to observed posts',
        knowledge_policy='cached per-post entities/concepts; recompute TF-IDF/PMI from observed rows only',
        singleton_policy='keep all source-only events',
        limitations=['fixed released global vocabulary and external knowledge snapshot; not historically dated knowledge',
                    'percentage deadlines use eventual comment count to define an offline benchmark',
                    'paper does not specify prefix rounding, zero cutoff, or within-prefix stage boundaries'])
    args.output.mkdir(parents=True, exist_ok=True)
    prov_path = args.output / 'provenance.json'
    if prov_path.exists():
        assert json.loads(prov_path.read_text()) == provenance, 'cache provenance changed'
    else:
        prov_path.write_text(json.dumps(provenance, indent=2) + '\n')
    graphdir = args.output / 'events'
    graphdir.mkdir(exist_ok=True)
    audits = []
    for number, observation in enumerate(records):
        eid = observation['event_id']
        entity_path = ROOT / 'data/pheme/pheme_entity' / (eid + '.csv')
        concept_path = ROOT / 'data/pheme/pheme_concept_yago' / (eid + '.csv')
        assert sha(entity_path) == manifest['input_sha256'][entity_path.name]
        annotation_hashes = [sha(entity_path), sha(concept_path)]
        target = graphdir / (eid + '.pkl')
        if target.exists():
            with target.open('rb') as f:
                magic = f.read(2)
                f.seek(0)
                cached = pickle.load(gzip.GzipFile(fileobj=f) if magic == b"\x1f\x8b" else f)
            assert cached['annotation_hashes'] == annotation_hashes
            audits.append(cached['audit'])
            continue
        rows = annotations(eid)
        counts = sorted(set(observation['minutes_prefix_lengths'].values()) |
                        set(observation['comments_percent_prefix_lengths'].values()))
        graphs = {}
        for count in counts:
            graphs[count] = construct(rows, observation, count, mapping)
        for deadline, count in observation['minutes_prefix_lengths'].items():
            cutoff = int(deadline) * 60
            assert graphs[count]['max_elapsed'] <= cutoff
            expected = 1 if cutoff == 0 else 1 + sum(t <= cutoff for t in observation['elapsed_seconds'][1:])
            assert count == expected
        for percent, count in observation['comments_percent_prefix_lengths'].items():
            assert count == 1 + (len(observation['ordered_post_ids']) - 1) * int(percent) // 100
        # Counterfactual test on every event: poisoning all future annotations and
        # parent links cannot affect a source-only graph or introduce unknown IDs.
        poisoned = {mid: row if mid == eid else dict(parent='FUTURE', entity=['UNSEEN_FUTURE'], concept=['FUTURE'])
                    for mid, row in rows.items()}
        same_graph(graphs[1], construct(poisoned, observation, 1, mapping))
        audit = dict(event_id=eid, prefixes=len(graphs), future_perturbation_passed=True,
                     edge_endpoints_valid=True, deadlines_valid=True,
                     min_normalization_degree=min(g['min_normalization_degree'] for g in graphs.values()))
        bundle = dict(graphs=graphs, observation=observation, annotation_hashes=annotation_hashes, audit=audit)
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as f:
            with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=1, mtime=0) as compressed:
                pickle.dump(bundle, compressed, protocol=4)
        temporary.replace(target)
        audits.append(audit)
        if number % 100 == 0:
            print('GRAPH EVENTS', number + 1, '/', len(records), flush=True)
    assert {a['event_id'] for a in audits} == set(ids)
    report = dict(status='passed', events=len(audits), unique_prefix_graphs=sum(a['prefixes'] for a in audits),
                  all_events_covered=True, all_edge_endpoints_valid=True, all_deadlines_valid=True,
                  future_annotation_perturbation_events=len(audits),
                  min_normalization_degree=min(a['min_normalization_degree'] for a in audits))
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
