#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p profile/02_tcgen05_evaluation/reports
# Run from the repository root.
root=profile/02_tcgen05_evaluation
nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_103a,code=sm_103a -Xcompiler=-fPIC -shared "$root/harness/kernels.cu" -o "$root/harness/kernels.so"
srun -G 1 .venv/bin/python -B -u "$root/harness/bench.py"
srun -G 1 ncu --set basic --profile-from-start off --clock-control none --import-source yes --source-folders "$root/harness" -f --export "$root/reports/basic.ncu-rep" .venv/bin/python -B "$root/harness/bench.py" --profile
python3 - <<'PY'
from pathlib import Path
import hashlib, json
Path('profile/02_tcgen05_evaluation/harness/kernels.so').unlink()
root=Path('profile/02_tcgen05_evaluation')
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
(root/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
