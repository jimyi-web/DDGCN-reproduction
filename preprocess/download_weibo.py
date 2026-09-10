"""Resume the README's raw archive, checking HTTP ranges and final ZIP CRCs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import time
import zipfile

import requests

URL = 'https://dl.dropboxusercontent.com/s/46r50ctrfa0ur1o/rumdect.zip'
TOTAL = 607201899
COUNT = 16
CHUNK = (TOTAL + COUNT - 1) // COUNT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    parts = directory / 'rumdect.parts'
    parts.mkdir(parents=True, exist_ok=True)

    def download(index):
        path = parts / ('part.%02d' % index)
        start = index * CHUNK
        end = min(start + CHUNK, TOTAL) - 1
        expected = end - start + 1
        for attempt in range(60):
            current = path.stat().st_size if path.exists() else 0
            if current == expected:
                return path
            if current > expected:
                raise ValueError('Oversized existing part: ' + str(path))
            offset = start + current
            try:
                with requests.get(URL, headers={'Range': 'bytes=%d-%d' % (offset, end),
                                  'Accept-Encoding': 'identity'}, stream=True,
                                  timeout=(20, 40)) as response:
                    response.raise_for_status()
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                         response.headers.get('Content-Range', ''))
                    if response.status_code != 206 or not match or tuple(map(int, match.groups())) != (offset, end, TOTAL):
                        raise ValueError('Server returned wrong HTTP range')
                    with path.open('ab') as stream:
                        for block in response.iter_content(65536):
                            if current + len(block) > expected:
                                raise ValueError('Response exceeds requested range')
                            stream.write(block)
                            current += len(block)
                print('part', index, current, '/', expected, flush=True)
            except (requests.RequestException, ValueError) as error:
                print('retry part', index, str(error)[:160], flush=True)
                time.sleep(min(attempt + 1, 10))
        raise RuntimeError('Retry limit reached for ' + str(path))

    with ThreadPoolExecutor(max_workers=8) as pool:
        ordered = list(pool.map(download, range(COUNT)))
    temporary = directory / 'rumdect.zip.validating'
    digest = hashlib.sha256()
    with temporary.open('wb') as output:
        for path in ordered:
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    output.write(block)
    if temporary.stat().st_size != TOTAL:
        raise ValueError('Archive size mismatch')
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError('ZIP CRC failed for ' + bad)
        print('ZIP verified:', len(archive.infolist()), 'members', flush=True)
    destination = directory / 'rumdect.zip'
    temporary.replace(destination)
    (directory / 'rumdect.zip.sha256').write_text(digest.hexdigest() + '  rumdect.zip\n')
    print('DOWNLOAD COMPLETE', digest.hexdigest(), flush=True)


if __name__ == '__main__':
    main()
