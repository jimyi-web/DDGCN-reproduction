"""Independent validation of cached early graphs against immutable observations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import gzip
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(graphs, main, limit=None):
    observations_dir = ROOT/'runs/pheme_early_observations_v2'
    observations = {r['event_id']:r for r in (json.loads(s) for s in (observations_dir/'observations.jsonl').read_text().splitlines())}
    provenance = json.loads((graphs/'provenance.json').read_text())
    assert provenance['observations_sha256'] == sha(observations_dir/'observations.jsonl')
    assert provenance['observation_manifest_sha256'] == sha(observations_dir/'manifest.json')
    assert provenance['splits_sha256'] == sha(main/'splits.json')
    splits = json.loads((main/'splits.json').read_text())
    population = [eid for fold in splits for eid in fold['test']['source_post_ids']]
    assert len(population)==len(set(population))==len(observations)==5748
    mapping = {line.split('\t')[0]:int(line.split('\t')[1]) for line in (ROOT/'data/pheme/node2idx_mid.txt').read_text().splitlines()}
    all_post_ids = {mapping[line.split('\t')[0]] for line in (ROOT/'data/pheme/mid2text.txt').read_text().splitlines()}
    if limit is not None:
        population = [eid for eid in population if (graphs/'events'/(eid+'.pkl')).exists()][:limit]
        assert len(population)==limit
    matrix_count = 0
    cache_hashes = {}
    for eid in population:
        path = graphs/'events'/(eid+'.pkl')
        cache_hashes[eid] = sha(path)
        with path.open('rb') as f:
            magic=f.read(2)
            f.seek(0)
            bundle=pickle.load(gzip.GzipFile(fileobj=f) if magic==b"\x1f\x8b" else f)
        obs=observations[eid]
        assert bundle['observation']==obs
        assert len(obs['ordered_post_ids'])==len(set(obs['ordered_post_ids']))
        assert obs['elapsed_seconds'][0]==0 and min(obs['elapsed_seconds'])>=0
        assert obs['elapsed_seconds']==sorted(obs['elapsed_seconds'])
        entity_path=ROOT/'data/pheme/pheme_entity'/(eid+'.csv')
        concept_path=ROOT/'data/pheme/pheme_concept_yago'/(eid+'.csv')
        assert bundle['annotation_hashes']==[sha(entity_path),sha(concept_path)]
        annotations={}
        with entity_path.open() as f:
            for row in csv.DictReader(f):
                mid=row['mid'].strip()
                if mid not in annotations:
                    annotations[mid]=[v.strip(' ').strip("'").strip(' ') for v in row['entity'].strip('\t').strip('[').strip(']').split(',') if v.strip(' ').strip("'").strip(' ')]
        concepts={}
        with concept_path.open() as f:
            for row in csv.DictReader(f):
                values=[' '.join(v.strip(' ').strip("'").strip('<').strip('>').split('_')).strip() for v in row['concept'].strip('\t').strip('[').strip(']').split(',')]
                concepts[row['mid'].strip()]=[v for v in values if v]
        for count,graph in bundle['graphs'].items():
            ids=obs['ordered_post_ids'][:count]
            observed_post_indices={mapping[mid] for mid in ids}
            assert set(graph['p_ids'])==observed_post_indices
            assert graph['observed_ids']==ids and graph['root']==mapping[eid]
            allowed=set(observed_post_indices)
            for mid in ids:
                allowed.update(mapping[v] for v in annotations[mid]+concepts[mid])
            assert set(graph['k_ids'])==allowed
            assert set(graph['k_ids']) & all_post_ids == observed_post_indices, 'future/cross-event post in knowledge graph'
            for kind,indices in [('propagation',graph['p_ids']),('knowledge',graph['k_ids'])]:
                assert len(indices)==len(set(indices))
                for matrix in graph[kind]:
                    assert matrix.shape==(len(indices),len(indices))
                    coo=matrix.tocoo()
                    assert np.isfinite(coo.data).all()
                    if coo.nnz:
                        assert min(coo.row.min(),coo.col.min())>=0
                        assert max(coo.row.max(),coo.col.max())<len(indices)
                    assert (matrix!=matrix.T).nnz==0
                    assert (np.asarray(matrix.sum(1)).ravel()+1>0).all()
                    matrix_count+=1
        for deadline,count in obs['minutes_prefix_lengths'].items():
            selected_times=obs['elapsed_seconds'][:count]
            cutoff=int(deadline)*60
            assert max(selected_times)<=cutoff
            assert count==(1 if cutoff==0 else 1+sum(t<=cutoff for t in obs['elapsed_seconds'][1:]))
        for percentage,count in obs['comments_percent_prefix_lengths'].items():
            assert count==1+(len(obs['ordered_post_ids'])-1)*int(percentage)//100
    return dict(status='passed' if limit is None else 'partial_check_passed',events=len(population),matrices_checked=matrix_count,
        all_semantic_nodes_derived_from_observed_posts=True,no_future_or_cross_event_post_nodes=True,
        all_edge_endpoints_legal=True,all_deadlines_verified=True,full_test_population_covered=(limit is None),
        cache_sha256=cache_hashes,observation_sha256=provenance['observations_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--main',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.graphs,a.main)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('INDEPENDENT GRAPH AUDIT PASSED',result['events'],result['matrices_checked'])
