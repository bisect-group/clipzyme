#!/usr/bin/env bash
# Checks that the CLIPZyme wrapper matches the original code. Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: clipzyme (torch, pytorch-lightning, torch-geometric, rdkit, pytest and fair-esm 2.0.0, not the ESM3 `esm` package).
# The script does not create envs. It needs no downloaded weights and no private data.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: clipzyme (with fair-esm 2.0.0)"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
files=(tests/test_fast_esm.py tests/test_frozen_esm.py tests/test_portable_equivalence.py)
marker=(-m "not gpu and not real_weights")
allow_skip=
if [[ ${1:-} == --negative-control ]]; then
    python -c "import torch,sys; sys.exit(not torch.cuda.is_available())" || { echo "FAIL: the negative control needs a CUDA device."; exit 1; }
    echo "Negative control: the production run starts from weights times 1.05. Every GPU test must FAIL."
    if out=$(EQUIV_NEGATIVE_CONTROL=1 python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no -s -m gpu 2>&1); then
        echo "$out"; echo "FAIL: a GPU test passed with the negative control (the bound is vacuous)."; exit 1
    fi
    grep -q "ratio=" <<<"$out" || { echo "$out"; echo "FAIL: the GPU test crashed before it measured a ratio."; exit 1; }
    grep "ratio=" <<<"$out"
    echo "Negative control failed as expected."
    exit 0
fi
if [[ ${1:-} == --gpu ]]; then
    marker=(-s -m "not real_weights")
    if ! python -c "import torch,sys; sys.exit(not torch.cuda.is_available())"; then
        echo "No CUDA device visible: gpu tests will be skipped."
        allow_skip=1
    fi
else
    export CUDA_VISIBLE_DEVICES=""
fi
out=$(python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no "${marker[@]}" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if [[ -z ${allow_skip:-} ]] && grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
