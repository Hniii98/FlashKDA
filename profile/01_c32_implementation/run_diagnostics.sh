#!/usr/bin/env bash
set -euo pipefail
cd /home/lcpu/60990375/kda-chunk32-baseline
python_bin=/home/lcpu/60990375/topic7-envs/candidate-venv/bin/python
ref_dir=/home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref
"$python_bin" tests/compare_naive.py --ref-dir "$ref_dir" --diagnostics --lower-bound -1 --rescale 1 --inverse-rescale 1 --output profile/01_c32_implementation/naive_gate1.json > profile/01_c32_implementation/naive_gate1.log 2>&1
"$python_bin" tests/compare_naive.py --ref-dir "$ref_dir" --diagnostics --lower-bound -1 --rescale 0.5 --inverse-rescale 1.01 --heads 96 --lengths 8192 --output profile/01_c32_implementation/naive_scale_smoke.json > profile/01_c32_implementation/naive_scale_smoke.log 2>&1
