#!/bin/bash
# Train and evaluate Clipzyme on every retrieval split on disk with seeds 666 777 888.
# Usage: bash queue_multiple_seeds.sh --gpus 0 1 2 [--runs_per_gpu N] [--amp]
set -e
cd "$(dirname "$0")"
python -m emulator_bench.run_all "$@"
