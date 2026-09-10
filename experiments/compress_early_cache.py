"""Losslessly compress already-built graph bundles, without graph recomputation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--graphs',type=Path,required=True)
    a=p.parse_args()
    records=[]
    for path in sorted((a.graphs/'events').glob('*.pkl')):
        raw=path.read_bytes()
        if raw[:2]==b'\x1f\x8b':
            continue
        compressed=gzip.compress(raw,compresslevel=1,mtime=0)
        assert gzip.decompress(compressed)==raw
        temp=path.with_suffix('.gzip.tmp')
        temp.write_bytes(compressed)
        temp.replace(path)
        records.append(dict(event=path.stem,raw_bytes=len(raw),compressed_bytes=len(compressed),
            raw_sha256=hashlib.sha256(raw).hexdigest(),compressed_sha256=hashlib.sha256(compressed).hexdigest()))
    report=dict(status='passed',all_decompressed_bytes_equal_original=True,files=len(records),
        raw_bytes=sum(r['raw_bytes'] for r in records),compressed_bytes=sum(r['compressed_bytes'] for r in records),records=records)
    target=a.graphs/'lossless_compression.json'
    if target.exists():
        raise FileExistsError('Preserve previous compression report: '+str(target))
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':
    main()
