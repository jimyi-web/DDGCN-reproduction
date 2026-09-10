#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/projects/DDGCN
PYTHON=${DDGCN_PYTHON:-/root/miniconda3/envs/ddgcn/bin/python}
DATASET=${1:?usage: run_paper.sh DATASET OUTPUT [run_paper.py options...]}
OUTPUT=${2:?usage: run_paper.sh DATASET OUTPUT [run_paper.py options...]}
shift 2

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$PYTHON" -B -u model/run_paper.py \
  --dataset "$DATASET" --output "$OUTPUT" "$@"
