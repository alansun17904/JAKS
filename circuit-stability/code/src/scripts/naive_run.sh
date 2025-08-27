#!/usr/bin/env bash

# Point Python at src so `eap`, `experiments`, `utils` can be imported
export PYTHONPATH="/workspace/JAKS/circuit-stability/code/src:${PYTHONPATH:-}"

SRC="/workspace/JAKS/circuit-stability/code/src/experiments/circuit_discovery.py"

python "$SRC" \
  "gpt2" \
  "output1" \
  --batch_size 1 \
  --ndevices 1 \
  --device "cuda" \
  --seed 42 \
  --dataset custom \
  --format zero-shot \
  --extraction tail \
  "$@"
