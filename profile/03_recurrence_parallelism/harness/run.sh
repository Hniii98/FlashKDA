#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p profile/03_recurrence_parallelism/column_split/reports profile/03_recurrence_parallelism/multi_head/reports profile/03_recurrence_parallelism/persistent/reports
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:/opt/nvidia/nsight-compute/2025.3.1:$PATH"
selection=${1:-all}
case "$selection" in
  all) groups=(column_split multi_head persistent cooperative_2cta) ;;
  column_split|multi_head|persistent|cooperative_2cta) groups=("$selection") ;;
  --heads) groups=(multi_head) ;;
  --persistent) groups=(persistent) ;;
  *) echo 'Usage: run.sh [all|column_split|multi_head|persistent|cooperative_2cta]' >&2; exit 2 ;;
esac
for group in "${groups[@]}"; do
  if [[ "$group" == cooperative_2cta ]]; then
    coop_root="profile/03_recurrence_parallelism/cooperative_2cta"
    # Preserve earlier measurements instead of replacing evidence on a rerun.
    if [[ -f "$coop_root/RESULTS.md" ]]; then
      archive_dir="$coop_root/archive/$(date -u +%Y%m%dT%H%M%S)_$$"
      mkdir -p "$archive_dir"
      cp "$coop_root/RESULTS.md" "$archive_dir/"
      if [[ -f "$coop_root/SOURCE.json" ]]; then cp "$coop_root/SOURCE.json" "$archive_dir/"; fi
      if [[ -d "$coop_root/validation" ]]; then cp -r "$coop_root/validation" "$archive_dir/"; fi
    fi
    mkdir -p "$coop_root/validation"
    .venv/bin/python -B profile/03_recurrence_parallelism/harness/cooperative_2cta.py --build
    for checker in memcheck racecheck synccheck; do
      srun -G 1 compute-sanitizer --tool "$checker" --error-exitcode 1 \
        .venv/bin/python -B -u profile/03_recurrence_parallelism/harness/cooperative_2cta.py --check-only \
        > "$coop_root/validation/$checker.log" 2>&1
    done
    srun -G 1 .venv/bin/python -B -u profile/03_recurrence_parallelism/harness/cooperative_2cta.py
    continue
  fi
  args=()
  if [[ "$group" == multi_head ]]; then args=(--heads); fi
  if [[ "$group" == persistent ]]; then args=(--persistent); fi
  build_dir="profile/03_recurrence_parallelism/harness/build_${group}"
  .venv/bin/python -B profile/03_recurrence_parallelism/harness/build.py "${args[@]}"
  srun -G 1 .venv/bin/python -B -u profile/03_recurrence_parallelism/harness/bench.py "${args[@]}"
  for tag in full source; do
    if [[ "$tag" == full ]]; then
      sections=(--set full --section PmSampling --section PmSampling_WarpStates)
    else
      sections=(--set source --section SourceCounters)
    fi
    srun -G 1 ncu "${sections[@]}" --profile-from-start off --clock-control none --import-source yes --source-folders "$build_dir" -f --export "profile/03_recurrence_parallelism/${group}/reports/${tag}.ncu-rep" .venv/bin/python -B profile/03_recurrence_parallelism/harness/bench.py "${args[@]}" --profile
  done
  .venv/bin/python -B profile/03_recurrence_parallelism/harness/analyze.py "${args[@]}"
  python3 - "$build_dir" <<'PY'
from pathlib import Path
import sys
b=Path(sys.argv[1])
for p in sorted(b.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    if p.is_file():p.unlink()
    elif p.is_dir():p.rmdir()
b.rmdir()
PY
done
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('profile/03_recurrence_parallelism')
manifest={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(r.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
(r/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
