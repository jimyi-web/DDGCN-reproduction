"""Audit original Weibo event labels and propagation structure without modifying inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

EXCLUDED = {'3495745049431351', '3843025795130981', '3926120686649225',
            '3925816377308372', '3853172374694439', '3852804358130490',
            '3852846782594165'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    labels = {}
    for line in (args.raw_directory / 'Weibo.txt').read_text().splitlines():
        fields = dict(item.split(':', 1) for item in line.split()[:2])
        eid, label = fields['eid'], int(fields['label'])
        if eid in labels or label not in (0, 1):
            raise ValueError('Invalid event label metadata')
        labels[eid] = label
    files = sorted((args.raw_directory / 'Weibo').glob('*.json'))
    if {path.stem for path in files} != set(labels):
        raise ValueError('Event IDs and labels differ')
    anomalies = []
    sizes = []
    for number, path in enumerate(files, 1):
        posts = json.loads(path.read_text())
        ids = {str(post['mid']) for post in posts}
        roots = [post for post in posts if post['parent'] is None]
        missing = sum(post['parent'] is not None and str(post['parent']) not in ids
                      for post in posts)
        sizes.append(len(posts))
        if len(roots) != 1 or missing or len(ids) != len(posts):
            anomalies.append({'event_id': path.stem, 'posts': len(posts),
                              'roots': len(roots), 'missing_parent_edges': missing,
                              'duplicate_ids': len(posts) - len(ids),
                              'excluded_by_author': path.stem in EXCLUDED})
        if number % 1000 == 0:
            print('Audited', number, '/', len(files), flush=True)
    remaining = [label for eid, label in labels.items() if eid not in EXCLUDED]
    report = {
        'events': len(labels), 'source_label_counts': dict(Counter(labels.values())),
        'posts': sum(sizes), 'min_posts': min(sizes), 'max_posts': max(sizes),
        'author_excluded_ids': sorted(EXCLUDED),
        'after_author_exclusions': len(remaining),
        'source_label_counts_after_exclusions': dict(Counter(remaining)),
        'original_code_inverted_label_counts_after_exclusions': dict(Counter(1-x for x in remaining)),
        'structural_anomalies': anomalies,
        'warning': 'Label semantics must follow dataset documentation; verified is a user attribute.',
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('AUDIT COMPLETE', len(files), 'events;', len(anomalies), 'structural anomalies', flush=True)


if __name__ == '__main__':
    main()
