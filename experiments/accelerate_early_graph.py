"""Cache repeated node lookups / IDF counts without changing released equations."""
import collections
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocess'))
import pheme_early_graph as graph

class IndexedNodes(list):
    def __init__(self, values):
        super().__init__(values)
        self.positions = {v: i for i, v in enumerate(values)}
        assert len(self.positions) == len(self)

    def index(self, value):
        return self.positions[value]


def accelerate():
    propagation = graph.released.build_temporal_propagation_graph
    knowledge = graph.released.build_temporal_knowledge_graph
    original_tfidf = graph.released.get_tfidf_edge

    def build_knowledge(ids, mapping, tree):
        df = collections.Counter(entity for row in tree.values() for entity in set(row['entity']))
        counts = {id(row['entity']): collections.Counter(row['entity']) for row in tree.values()}
        def tfidf(entity, entities, subtree):
            assert subtree is tree
            return (counts[id(entities)][entity] / len(entities)) * math.log(len(tree) / (df[entity] + 1))
        graph.released.get_tfidf_edge = tfidf
        try:
            return knowledge(IndexedNodes(ids), mapping, tree)
        finally:
            graph.released.get_tfidf_edge = original_tfidf

    graph.released.build_temporal_propagation_graph = lambda ids, mapping, tree: propagation(IndexedNodes(ids), mapping, tree)
    graph.released.build_temporal_knowledge_graph = build_knowledge


def verify():
    import pickle
    root = graph.ROOT
    observations = [json.loads(line) for line in (root / 'runs/pheme_early_observations_v2/observations.jsonl').read_text().splitlines()]
    mapping = {line.split('\t')[0]: int(line.split('\t')[1]) for line in (root / 'data/pheme/node2idx_mid.txt').read_text().splitlines()}
    original_p = graph.released.build_temporal_propagation_graph
    original_k = graph.released.build_temporal_knowledge_graph
    accelerate()
    fast_p = graph.released.build_temporal_propagation_graph
    fast_k = graph.released.build_temporal_knowledge_graph
    comparisons = 0
    for obs in observations[:20]:
        rows = graph.annotations(obs['event_id'])
        counts = sorted(set(obs['minutes_prefix_lengths'].values()) | set(obs['comments_percent_prefix_lengths'].values()))
        for count in counts:
            graph.released.build_temporal_propagation_graph = original_p
            graph.released.build_temporal_knowledge_graph = original_k
            expected = graph.construct(rows, obs, count, mapping)
            graph.released.build_temporal_propagation_graph = fast_p
            graph.released.build_temporal_knowledge_graph = fast_k
            actual = graph.construct(rows, obs, count, mapping)
            graph.same_graph(expected, actual)
            comparisons += 6
    report = dict(status='passed', events=20, matrices_exactly_equal=comparisons,
                  acceleration_sha256=graph.sha(__file__), reference_builder_sha256=graph.sha(root / 'preprocess/pheme_early_graph.py'))
    (root / 'runs/pheme_early_acceleration_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    if sys.argv[1:] == ['--verify']:
        verify()
    else:
        proof = json.loads((graph.ROOT / 'runs/pheme_early_acceleration_verification.json').read_text())
        assert proof['acceleration_sha256'] == graph.sha(__file__)
        assert proof['reference_builder_sha256'] == graph.sha(graph.__file__)
        accelerate()
        graph.main()
