#!/usr/bin/env bash
set -euo pipefail
cd /home/lcpu/60990375/kda-chunk32-baseline
/home/lcpu/60990375/topic7-envs/candidate-venv/bin/python tests/compare_naive.py \
  --ref-dir /home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref \
  --diagnostics --rescale-log2 -80 -81 -82 -83 -84 -85 -86 -87 -88 -89 -90 -91 -92 -93 -94 -95 -96 -97 -98 -99 -100 -101 -102 -103 -104 -105 -106 -107 -108 -109 -110 -111 -112 \
  --output profile/01_c32_implementation/sweep_fine.json > profile/01_c32_implementation/sweep_fine.log 2>&1
