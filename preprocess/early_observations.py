"""Build auditable PHEME observation windows for Figure 2; no model evaluation."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

MINUTES = (0, 20, 40, 60, 80, 100, 120)
PERCENTAGES = (0, 20, 40, 60, 80, 100)


def elapsed_seconds(timestamp, root_timestamp):
    fmt = '%a %b %d %H:%M:%S %z %Y'
    return (datetime.strptime(timestamp.strip(), fmt) -
            datetime.strptime(root_timestamp.strip(), fmt)).total_seconds()


def observation_record(event_id, rows):
    by_id = {}
    duplicate_rows = 0
    for row in rows:
        mid = row['mid'].strip()
        if mid in by_id:
            if row['t'].strip() != by_id[mid]['t'].strip():
                raise ValueError('Conflicting timestamps for repeated message in ' + event_id)
            duplicate_rows += 1
            continue
        by_id[mid] = row
    if event_id not in by_id:
        raise ValueError('Missing source post ' + event_id)
    root = by_id[event_id]
    ordered = sorted(((elapsed_seconds(row['t'], root['t']), mid)
                      for mid, row in by_id.items() if mid != event_id),
                     key=lambda item: (item[0], item[1]))
    negative = [mid for seconds, mid in ordered if seconds < 0]
    ordered = [(seconds, mid) for seconds, mid in ordered if seconds >= 0]
    # Source stays first even when several comments share the root timestamp.
    ids = [event_id] + [mid for _, mid in ordered]
    seconds = [0.0] + [value for value, _ in ordered]
    minute_counts = {str(m): (1 if m == 0 else
                              1 + sum(value <= m * 60 for value, _ in ordered))
                     for m in MINUTES}
    percentage_counts = {str(p): 1 + math.floor(len(ordered) * p / 100)
                         for p in PERCENTAGES}
    assert percentage_counts['0'] == 1
    assert percentage_counts['100'] == len(ids)
    return {'event_id': event_id, 'duplicate_rows_ignored': duplicate_rows, 'ordered_post_ids': ids, 'elapsed_seconds': seconds,
            'minutes_prefix_lengths': minute_counts,
            'comments_percent_prefix_lengths': percentage_counts,
            'excluded_predating_comments': negative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity-directory', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text())
    # Reuse the exact experimental population, including all five test folds.
    event_ids = sorted({eid for fold in splits
                        for eid in fold['test']['source_post_ids']})
    seen = []
    for fold in splits:
        seen.extend(fold['test']['source_post_ids'])
    if len(seen) != len(event_ids):
        raise ValueError('Repeated test events across folds')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    stats = {'events': len(event_ids), 'predating_comments': 0,
             'comments_after_24h': 0, 'posts': 0, 'duplicate_rows_ignored': 0}
    with (args.output / 'observations.jsonl').open('w') as output:
        for event_id in event_ids:
            source = args.entity_directory / (event_id + '.csv')
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            with source.open(newline='') as stream:
                record = observation_record(event_id, list(csv.DictReader(stream)))
            stats['duplicate_rows_ignored'] += record['duplicate_rows_ignored']
            stats['predating_comments'] += len(record['excluded_predating_comments'])
            stats['comments_after_24h'] += sum(t >= 86400 for t in record['elapsed_seconds'])
            stats['posts'] += len(record['ordered_post_ids'])
            output.write(json.dumps(record) + '\n')
    manifest = {'status': 'observation windows only; graphs and evaluation still required',
                'minutes': MINUTES, 'comment_percentages': PERCENTAGES,
                'zero_delay_policy': 'source only',
                'time_boundary': 'include comments at or before delay',
                'count_policy': 'floor(percentage * number of nonnegative-time comments / 100)',
                'ties': 'timestamp then message ID',
                'paper_ambiguity': 'rounding, equal-time ties and zero-delay handling are not specified',
                'time_fix': 'total_seconds rather than modulo-day timedelta.seconds',
                'splits_sha256': hashlib.sha256(args.splits.read_bytes()).hexdigest(),
                'input_sha256': hashes, 'statistics': stats}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('OBSERVATIONS COMPLETE', json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
