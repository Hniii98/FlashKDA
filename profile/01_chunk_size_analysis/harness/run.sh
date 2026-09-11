#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
# One command reproduces the baseline and both chunk sizes. Archive results first.
for c in 16 32 64; do
  if [[ -e "profile/01_chunk_size_analysis/chunk${c}/reports/full.ncu-rep" || -e "profile/01_chunk_size_analysis/chunk${c}/RESULTS.md" || -e "profile/01_chunk_size_analysis/chunk${c}/analysis/inverse.csv" ]]; then
    echo "Archive existing chunk results before reproducing; evidence will not be overwritten." >&2
    exit 1
  fi
  mkdir -p "profile/01_chunk_size_analysis/chunk${c}/analysis" "profile/01_chunk_size_analysis/chunk${c}/reports"
done
mkdir -p profile/01_chunk_size_analysis/harness/build
trap 'rm -rf profile/01_chunk_size_analysis/harness/build profile/01_chunk_size_analysis/harness/__pycache__' EXIT
nvcc -O3 -std=c++17 -lineinfo -arch=sm_103 -Xcompiler=-fPIC -shared \
  profile/01_chunk_size_analysis/harness/kernels.cu -o profile/01_chunk_size_analysis/harness/build/kernels.so
srun -G 1 bash -c '
set -euo pipefail
for c in 16 32 64; do
  .venv/bin/python -u profile/01_chunk_size_analysis/harness/bench.py run --chunk "$c"
  for tag in full source; do
    if [[ "$tag" == full ]]; then
      sections=(--set full --section PmSampling --section PmSampling_WarpStates)
    else
      sections=(--set source --section SourceCounters)
    fi
    ncu "${sections[@]}" --profile-from-start off --clock-control none \
      --import-source yes --source-folders profile/01_chunk_size_analysis/harness --kernel-name-base function \
      -k "regex:(range_probe|neumann_kernel|triangular_kernel|prepare_kernel|recurrence_kernel)" \
      --export "profile/01_chunk_size_analysis/chunk${c}/reports/${tag}.ncu-rep" \
      .venv/bin/python -u profile/01_chunk_size_analysis/harness/bench.py profile --chunk "$c"
  done
done
'
for c in 16 32 64; do
  for tag in full source; do
    .venv/bin/python profile/01_chunk_size_analysis/harness/bench.py analyze --chunk "$c" --tag "$tag"
  done
done
.venv/bin/python profile/01_chunk_size_analysis/harness/bench.py tidy
.venv/bin/python profile/01_chunk_size_analysis/harness/bench.py report
