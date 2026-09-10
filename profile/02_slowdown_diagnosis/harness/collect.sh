#!/usr/bin/env bash
set -euo pipefail
cd /home/lcpu/60990375/kda-chunk32-baseline
run_dir=profile/02_slowdown_diagnosis
mkdir -p "$run_dir/analysis" "$run_dir/reports"
python_bin=/home/lcpu/60990375/topic7-envs/candidate-venv/bin/python
"$python_bin" "$run_dir/harness/run.py" --mode timing > "$run_dir/analysis/runtime_ablation.log" 2>&1
for variant in C16 C32; do
 ncu --profile-from-start off --clock-control none --set full --section PmSampling --section PmSampling_WarpStates -k 'regex:_flash_kda_fwd_(prepare|recurrence)' -c 2 -o "$run_dir/reports/full_$variant" "$python_bin" "$run_dir/harness/run.py" --mode profile --variant "$variant" > "$run_dir/analysis/full_$variant.log" 2>&1
 ncu --profile-from-start off --clock-control none --set source --section SourceCounters -k 'regex:_flash_kda_fwd_(prepare|recurrence)' -c 2 -o "$run_dir/reports/source_$variant" "$python_bin" "$run_dir/harness/run.py" --mode profile --variant "$variant" > "$run_dir/analysis/source_$variant.log" 2>&1
done
