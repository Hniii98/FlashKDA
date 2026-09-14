#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p profile/04_compute_memory_bottlenecks/analysis profile/04_compute_memory_bottlenecks/reports
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:/opt/nvidia/nsight-compute/2025.3.1:$PATH"
for rep in profile/04_compute_memory_bottlenecks/reports/*.ncu-rep; do
  if [[ -e "$rep" ]]; then echo "Archive existing reports before rerunning: $rep" >&2; exit 1; fi
done
python -B profile/04_compute_memory_bottlenecks/harness/build.py
srun -G 1 python -B profile/04_compute_memory_bottlenecks/harness/bench.py
for kind in full source; do
  sections=(--set source --section SourceCounters)
  if [[ "$kind" == full ]]; then sections=(--set full --section PmSampling --section PmSampling_WarpStates); fi
  srun -G 1 ncu --replay-mode application --cache-control none --clock-control none \
    "${sections[@]}" --profile-from-start off --nvtx --import-source yes \
    -k 'regex:(_flash_kda_fwd_prepare|_flash_kda_fwd_recurrence)' \
    --export "profile/04_compute_memory_bottlenecks/reports/$kind" python -B profile/04_compute_memory_bottlenecks/harness/bench.py --profile
done
python -B profile/04_compute_memory_bottlenecks/harness/analyze.py
