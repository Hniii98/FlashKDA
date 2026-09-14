#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p profile/05_bf16_state_accuracy/analysis
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
python -B profile/05_bf16_state_accuracy/harness/build.py
srun -G 1 python -B -u profile/05_bf16_state_accuracy/harness/bench.py 2>&1 | tee profile/05_bf16_state_accuracy/analysis/run_8192.log
python -B profile/05_bf16_state_accuracy/harness/analyze.py
