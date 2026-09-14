from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parents[3];O=R/'profile/05_bf16_state_accuracy';B=O/'harness/build';B.mkdir(exist_ok=True)
(O/'analysis').mkdir(parents=True,exist_ok=True)
flags=['nvcc','-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','--threads','4','--ptxas-options=--register-usage-level=10','-lineinfo','-gencode','arch=compute_103a,code=sm_103a','-Xcompiler=-fPIC','-shared','-I'+str(R/'csrc'),'-I'+str(R/'cutlass/include'),'-I'+str(R/'cutlass/tools/util/include'),'-I'+str(R/'cutlass/examples/common')]
cmd=flags+[str(R/'csrc/smxx/fwd_launch.cu'),str(O/'harness/bridge.cu'),'-o',str(B/'original.so')]
subprocess.run(cmd,check=True)
files=list((R/'csrc').rglob('*'))+[R/'tests/torch_ref.py']
(O/'analysis/source.json').write_text(json.dumps(dict(source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},command=cmd,binary_sha256=hashlib.sha256((B/'original.so').read_bytes()).hexdigest()),indent=2)+'\n')
