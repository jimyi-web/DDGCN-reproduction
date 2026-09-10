"""Build DDGCN-compatible Weibo graphs without unavailable web APIs.

The paper repository's Chinese entity linker and concept service are no longer
an auditable bulk preprocessing dependency. This replacement keeps the graph
format and temporal construction, while extracting named terms locally with
jieba POS tags. Its manifest records that semantic substitution explicitly.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

import jieba.posseg as pseg
import numpy as np
from scipy import sparse


EXCLUDED_EVENTS = {
    '3495745049431351', '3843025795130981', '3926120686649225',
    '3925816377308372', '3853172374694439', '3852804358130490',
    '3852846782594165',
}
POS_CONCEPT = {
    'nr': 'person', 'ns': 'place', 'nt': 'organization', 'nz': 'proper_noun',
    'n': 'noun', 'vn': 'event_or_action',
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def clean_text(value):
    return re.sub(r'[\r\n\t]+', ' ', str(value or '')).strip()


def semantic_terms(text, limit):
    """Return deterministic local entity and coarse concept terms."""
    entities = []
    concepts = []
    seen = set()
    for token in pseg.cut(text):
        word = token.word.strip()
        category = next((value for prefix, value in POS_CONCEPT.items()
                         if token.flag.startswith(prefix)), None)
        if category is None or len(word) < 2 or word in seen:
            continue
        seen.add(word)
        entities.append('entity:' + word)
        concepts.append('concept:' + category)
        if len(entities) >= limit:
            break
    return entities, list(dict.fromkeys(concepts))


def event_files(raw_dir):
    return [path for path in sorted(raw_dir.glob('*.json'))
            if path.stem not in EXCLUDED_EVENTS]


def root_post(posts):
    for post in posts:
        if post.get('parent') in (None, 'None', '', 'null'):
            return post
    return posts[0]


def read_metadata_labels(raw_dir):
    labels = {}
    for path in (raw_dir / 'Weibo.txt', raw_dir.parent / 'Weibo.txt'):
        if not path.is_file():
            continue
        for line in path.read_text(errors='ignore').splitlines():
            fields = dict(part.split(':', 1) for part in line.split() if ':' in part)
            event_id = fields.get('eid') or fields.get('eip')
            if event_id is not None and 'label' in fields:
                labels[event_id] = int(fields['label'])
        break
    return labels


def prepare_clean_data(raw_dir, output, term_limit):
    clean_dir = output / 'weibo_clean_open'
    clean_dir.mkdir(parents=True, exist_ok=True)
    files = event_files(raw_dir)
    metadata_labels = read_metadata_labels(raw_dir)
    missing_labels = [path.stem for path in files if path.stem not in metadata_labels]
    if missing_labels:
        raise ValueError('Missing Weibo.txt ground-truth labels for ' + str(len(missing_labels)) + ' events')
    post_ids = []
    semantic_nodes = set()
    event_labels = {}
    with (output / 'mid2text.txt').open('w', encoding='utf-8') as text_stream:
        for number, path in enumerate(files, 1):
            posts = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
            if not posts:
                continue
            root = root_post(posts)
            source_label = metadata_labels.get(path.stem)
            if source_label is None:
                raise ValueError('Missing ground-truth event label: ' + path.stem + '; user verified status is not a rumor label')
            else:
                # This is the exact inversion in DDGCN's load_mid2label_re.
                label = 1 - int(source_label)
            event_labels[path.stem] = int(label)
            with (clean_dir / (path.stem + '.csv')).open(
                    'w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('mid', 'parent', 'text', 't', 'entity', 'concept'))
                for post in posts:
                    mid = str(post.get('mid', '')).strip()
                    if not mid:
                        continue
                    parent = post.get('parent')
                    parent = 'None' if parent in (None, 'None', '', 'null') else str(parent)
                    text = clean_text(post.get('original_text', post.get('text', '')))
                    entities, concepts = semantic_terms(text, term_limit)
                    writer.writerow((mid, parent, text, post.get('t', 0),
                                     json.dumps(entities, ensure_ascii=False),
                                     json.dumps(concepts, ensure_ascii=False)))
                    post_ids.append(mid)
                    semantic_nodes.update(entities)
                    semantic_nodes.update(concepts)
                    text_stream.write(mid + '\t' + text + '\n')
            if number % 100 == 0:
                print('cleaned events', number, '/', len(files), flush=True)

    ordered_posts = list(dict.fromkeys(post_ids))
    nodes = ordered_posts + sorted(semantic_nodes)
    node2idx = {node: index for index, node in enumerate(nodes)}
    with (output / 'node2idx_mid.txt').open('w', encoding='utf-8') as stream:
        for node, index in node2idx.items():
            stream.write(node + '\t' + str(index) + '\n')
    with (output / 'weibo_id_label.txt').open('w', encoding='utf-8') as stream:
        for event_id in sorted(event_labels):
            stream.write(event_id + '\t' + str(event_labels[event_id]) + '\n')
    return clean_dir, node2idx, event_labels


def load_event(path):
    posts = []
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            posts.append({
                'mid': row['mid'], 'parent': row['parent'],
                't': float(row['t'] or 0),
                'entity': json.loads(row['entity']),
                'concept': json.loads(row['concept']),
            })
    return posts


def stage_number(elapsed, boundary):
    if boundary > 0 and elapsed < boundary:
        return 0
    if boundary > 0 and elapsed < 2 * boundary:
        return 1
    return 2


def sparse_stages(edge_maps, node_count):
    matrices = []
    for edges in edge_maps:
        if edges:
            rows, cols = zip(*edges)
            values = [edges[key] for key in edges]
            matrix = sparse.csr_matrix((values, (rows, cols)),
                                       shape=(node_count, node_count),
                                       dtype=np.float32)
        else:
            matrix = sparse.csr_matrix((node_count, node_count), dtype=np.float32)
        matrices.append(matrix)
    return matrices


def add_symmetric(edges, left, right, weight):
    if left == right:
        return
    edges[(left, right)] = float(weight)
    edges[(right, left)] = float(weight)


def build_event_graph(posts, node2idx):
    mids = list(dict.fromkeys(post['mid'] for post in posts))
    mid_position = {mid: index for index, mid in enumerate(mids)}
    root = root_post(posts)
    root_time = float(root['t'])
    max_elapsed = max(max(float(post['t']) - root_time, 0.0) for post in posts)
    boundary = int(max_elapsed / 3)

    propagation_edges = [dict(), dict(), dict()]
    for post in posts:
        parent = post['parent']
        if parent not in mid_position or post['mid'] == parent:
            continue
        start = stage_number(max(float(post['t']) - root_time, 0.0), boundary)
        child_idx = mid_position[post['mid']]
        parent_idx = mid_position[parent]
        for stage in range(start, 3):
            add_symmetric(propagation_edges[stage], child_idx, parent_idx, 1.0)

    semantics = []
    for post in posts:
        semantics.extend(post['entity'])
        semantics.extend(post['concept'])
    local_nodes = mids + sorted(set(semantics))
    local_position = {node: index for index, node in enumerate(local_nodes)}
    document_frequency = Counter()
    for post in posts:
        document_frequency.update(set(post['entity']))

    knowledge_edges = [dict(), dict(), dict()]
    semantic_by_stage = [[], [], []]
    for post in posts:
        start = stage_number(max(float(post['t']) - root_time, 0.0), boundary)
        entities = post['entity']
        counts = Counter(entities)
        for entity, count in counts.items():
            tf = count / len(entities)
            idf = math.log(len(posts) / (document_frequency[entity] + 1))
            weight = tf * idf
            for stage in range(start, 3):
                add_symmetric(knowledge_edges[stage], local_position[post['mid']],
                              local_position[entity], weight)
        terms = list(dict.fromkeys(entities + post['concept']))
        semantic_by_stage[start].append(terms)

    # Local co-occurrence is deterministic and avoids the retired Probase API.
    for start, documents in enumerate(semantic_by_stage):
        term_frequency = Counter(term for document in documents for term in set(document))
        pair_frequency = Counter()
        for document in documents:
            unique = sorted(set(document))
            for left_index, left in enumerate(unique):
                for right in unique[left_index + 1:]:
                    pair_frequency[(left, right)] += 1
        total = max(len(documents), 1)
        for (left, right), count in pair_frequency.items():
            pmi = math.log((count * total) /
                           (term_frequency[left] * term_frequency[right]))
            if pmi <= 0:
                continue
            for stage in range(start, 3):
                add_symmetric(knowledge_edges[stage], local_position[left],
                              local_position[right], pmi)

    return (
        np.asarray([node2idx[mid] for mid in mids], dtype=np.int64),
        sparse_stages(propagation_edges, len(mids)),
        np.asarray([node2idx[node] for node in local_nodes], dtype=np.int64),
        sparse_stages(knowledge_edges, len(local_nodes)),
        np.int64(node2idx[root['mid']]),
    )


def load_mapping(path):
    mapping = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            node, index = line.rstrip('\n').split('\t')
            mapping[node] = int(index)
    return mapping


def load_labels(path):
    labels = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            event_id, label = line.rstrip('\n').split('\t')
            labels[event_id] = int(label)
    return labels


def save_graphs(clean_dir, output, node2idx, event_labels):
    temporal = output / 'weibo_temporal_data'
    temporal.mkdir(parents=True, exist_ok=True)
    propagation_indices = []
    propagation_graphs = []
    knowledge_indices = []
    knowledge_graphs = []
    labels = []
    roots = []
    clean_files = sorted(clean_dir.glob('*.csv'))
    for number, path in enumerate(clean_files, 1):
        if path.stem not in event_labels:
            continue
        posts = load_event(path)
        if not posts:
            continue
        p_idx, p_graph, k_idx, k_graph, root = build_event_graph(posts, node2idx)
        propagation_indices.append(p_idx)
        propagation_graphs.append(p_graph)
        knowledge_indices.append(k_idx)
        knowledge_graphs.append(k_graph)
        labels.append(event_labels[path.stem])
        roots.append(root)
        if number % 100 == 0:
            print('built graphs', number, '/', len(clean_files), flush=True)

    np.save(temporal / 'propagation_node_idx.npy',
            np.asarray(propagation_indices, dtype=object), allow_pickle=True)
    np.save(temporal / 'propagation_node.npy',
            np.asarray(propagation_graphs, dtype=object), allow_pickle=True)
    np.save(temporal / 'knowledge_node_idx.npy',
            np.asarray(knowledge_indices, dtype=object), allow_pickle=True)
    np.save(temporal / 'knowledge_node.npy',
            np.asarray(knowledge_graphs, dtype=object), allow_pickle=True)
    np.save(temporal / 'label.npy', np.asarray(labels, dtype=np.int64))
    np.save(temporal / 'propagation_root_index.npy', np.asarray(roots, dtype=np.int64))
    return {
        'events': len(labels),
        'label_0': labels.count(0),
        'label_1': labels.count(1),
        'unique_global_nodes': len(node2idx),
        'posts_min': min(map(len, propagation_indices)),
        'posts_max': max(map(len, propagation_indices)),
        'posts_mean': sum(map(len, propagation_indices)) / len(propagation_indices),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, required=True,
                        help='directory containing one raw JSON file per event')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--term-limit', type=int, default=5)
    parser.add_argument('--reuse-clean', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    args = parser.parse_args()
    args.raw_dir = args.raw_dir.resolve()
    args.output = args.output.resolve()
    if args.term_limit < 1:
        parser.error('--term-limit must be positive')
    if not args.raw_dir.is_dir():
        parser.error('raw JSON directory does not exist: ' + str(args.raw_dir))
    args.output.mkdir(parents=True, exist_ok=True)

    if args.reuse_clean:
        clean_dir = args.output / 'weibo_clean_open'
        node2idx = load_mapping(args.output / 'node2idx_mid.txt')
        event_labels = load_labels(args.output / 'weibo_id_label.txt')
    else:
        clean_dir, node2idx, event_labels = prepare_clean_data(
            args.raw_dir, args.output, args.term_limit)
    if args.clean_only:
        return
    statistics = save_graphs(clean_dir, args.output, node2idx, event_labels)
    manifest = {
        'format': 'DDGCN six-array temporal format',
        'paper_faithful': False,
        'status': 'experimental POS-based substitute; not a paper reproduction',
        'time_stages': 3,
        'excluded_original_events': sorted(EXCLUDED_EVENTS),
        'entity_source': 'local jieba POS extraction',
        'concept_source': 'local coarse POS categories',
        'paper_difference': ('The authors used Shuyan entity linking and Chinese '
                             'Probase; those bulk API intermediates were not released.'),
        'term_limit_per_post': args.term_limit,
        **statistics,
    }
    save_json(args.output / 'preprocess_manifest.json', manifest)
    print('WEIBO BUILD COMPLETE', json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
