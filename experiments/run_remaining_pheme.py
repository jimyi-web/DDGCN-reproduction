"""Resume-safe sequential GPU queue. Never trains or writes the main run."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/root/autodl-tmp/ddgcn-runs')
MAIN = BASE / 'pheme_cv_checkpointed'
PYTHON = sys.executable

def run(command, log):
    print('START', ' '.join(map(str, command)), flush=True)
    with log.open('a') as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    while not (BASE / 'pheme_reload_verified/summary.json').exists():
        # If the independent verifier exits unsuccessfully, stop instead of waiting forever.
        active = any(b'verify_checkpoints.py' in p.read_bytes() for p in Path('/proc').glob('[0-9]*/cmdline')
                     if p.exists())
        if not active:
            raise RuntimeError('Reload verification not complete; inspect verifier logs')
        time.sleep(10)
    assert json.loads((BASE / 'pheme_reload_verified/summary.json').read_text())['status'] == 'passed'
    output = BASE / 'pheme_ablations_checkpointed'
    command = [PYTHON, '-B', '-u', ROOT / 'model/run_paper.py', '--dataset', 'pheme',
        '--output', output, '--models', 'wotf', 'static', 'propagation', 'knowledge',
        '--epochs', '5', '--batch', '32', '--seed', '2026']
    if output.exists():
        command.append('--resume')
    run(command, BASE / 'pheme_ablations_checkpointed.log')
    graphs = BASE / 'pheme_early_graphs_seed2026'
    while not (graphs / 'audit.json').exists():
        active = any(b'accelerate_early_graph.py' in p.read_bytes() for p in Path('/proc').glob('[0-9]*/cmdline')
                     if p.exists())
        if not active:
            raise RuntimeError('Early graph construction not complete; inspect graph log')
        time.sleep(10)
    run([PYTHON, '-B', '-u', ROOT / 'experiments/evaluate_early.py', '--main', MAIN,
         '--graphs', graphs, '--output', BASE / 'pheme_early_evaluation'], BASE / 'pheme_early_evaluation.log')
    print('PHEME QUEUE COMPLETE', flush=True)

if __name__ == '__main__':
    main()
