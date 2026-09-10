#!/usr/bin/env bash
set -euo pipefail
cd /home/lcpu/60990375/kda-chunk32-baseline
r=profile/03_launch_bounds_tuning
mkdir -p "$r/analysis" "$r/reports"
py=/home/lcpu/60990375/topic7-envs/candidate-venv/bin/python
ref=/home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref
for n in 5 4 2; do
 "$py" "$r/harness/compare.py" "$r/build_$n/flash_kda_C.so" --ref-dir "$ref" --output "$r/analysis/naive_$n.json" > "$r/analysis/naive_$n.log" 2>&1
done
"$py" "$r/harness/run.py" --mode timing > "$r/analysis/timing.log" 2>&1
for n in 8 5 4 2; do
 ncu --profile-from-start off --clock-control none --set full --section PmSampling --section PmSampling_WarpStates -k 'regex:_flash_kda_fwd_(prepare|recurrence)' -c 2 -o "$r/reports/full_LB$n" "$py" "$r/harness/run.py" --mode profile --variant "LB$n" > "$r/analysis/full_LB$n.log" 2>&1
 ncu --profile-from-start off --clock-control none --set source --section SourceCounters -k 'regex:_flash_kda_fwd_(prepare|recurrence)' -c 2 -o "$r/reports/source_LB$n" "$py" "$r/harness/run.py" --mode profile --variant "LB$n" > "$r/analysis/source_LB$n.log" 2>&1
done
for n in 5 4 2; do
 compute-sanitizer --tool memcheck --error-exitcode 99 "$py" "$r/harness/compare.py" "$r/build_$n/flash_kda_C.so" --ref-dir "$ref" --heads 96 --lengths 1300 547 2048 963 271 3063 --output "$r/analysis/memcheck_$n.json" > "$r/analysis/memcheck_$n.log" 2>&1
done
