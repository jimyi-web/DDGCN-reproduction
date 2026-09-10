"""Refresh documentation on committed fold results; finalize after queue exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/autodl-tmp/ddgcn-runs')


def active():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            value=path.read_bytes()
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
        if b'run_remaining_pheme.py' in value and int(path.parent.name)!=os.getpid():
            return True
    return False


def main():
    previous=None
    while True:
        paths=sorted((BASE/'pheme_ablations_checkpointed').glob('*/fold_*/result.json'))
        early=BASE/'pheme_early_evaluation/summary.json'
        state=(len(paths),early.exists())
        if state!=previous:
            subprocess.run([sys.executable,'-B',str(ROOT/'experiments/update_report.py')],cwd=ROOT,check=True)
            previous=state
        if len(paths)==20 and early.exists():
            subprocess.run([sys.executable,'-B',str(ROOT/'experiments/update_report.py'),'--require-pheme-complete'],cwd=ROOT,check=True)
            print('PHEME FINAL REPORT WRITTEN; WEIBO KNOWLEDGE RESOURCE BLOCK REMAINS',flush=True)
            return
        if not active():
            print('QUEUE EXITED BEFORE ALL PHEME RESULTS; inspect pheme_remaining_queue.log',flush=True)
            return
        time.sleep(45)

if __name__=='__main__':
    main()
