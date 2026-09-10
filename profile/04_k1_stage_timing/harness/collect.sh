#!/usr/bin/env bash
set -euo pipefail
cd /home/lcpu/60990375/kda-chunk32-baseline
r=profile/04_k1_stage_timing
mkdir -p "$r/analysis" "$r/reports"
py=/home/lcpu/60990375/topic7-envs/candidate-venv/bin/python
ref=/home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref
for v in C16 C32; do
 "$py" "$r/harness/compare.py" "$r/build_$v/flash_kda_C.so" --ref-dir "$ref" --output "$r/analysis/naive_$v.json" > "$r/analysis/naive_$v.log" 2>&1
done
"$py" "$r/harness/run.py" --mode timing > "$r/analysis/timing.log" 2>&1
