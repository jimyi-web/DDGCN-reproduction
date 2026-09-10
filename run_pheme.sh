#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
python_bin="${DDGCN_PYTHON:-/root/miniconda3/envs/ddgcn/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "ddgcn Python not found: $python_bin. Set DDGCN_PYTHON to the environment's Python." >&2
    exit 1
fi
output="${1:-runs/pheme_completed_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ -e "$output" || -e "$output.log" ]]; then
    echo "Choose a new output directory; existing experiment: $output" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$output")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Python: $python_bin"
echo "Output: $output"
"$python_bin" -B -u model/run_baseline.py "$@" --output "$output" 2>&1 | tee -- "$output.log"
