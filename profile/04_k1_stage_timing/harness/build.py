import os,sys,subprocess,hashlib,json,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=Path(__file__).resolve().parents[1]
manifest={}
for name,source,lb in [('C16',Path('/home/lcpu/60990375/topic7-envs/baseline'),8),('C32',ROOT,4)]:
 dst=RUN/'harness'/name;dst.mkdir(exist_ok=True)
 for f in (source/'csrc').rglob('*'):
  target=dst/'csrc'/f.relative_to(source/'csrc')
  target.parent.mkdir(parents=True,exist_ok=True)
  if f.is_dir():target.mkdir(parents=True,exist_ok=True)
  elif not target.exists():target.symlink_to(f)
 replacements={}
 p=source/'csrc/smxx/fwd_kernel1.cuh';s=p.read_text();original=s
 s=s.replace('__launch_bounds__(NumThreads, 8)',f'__launch_bounds__(NumThreads, {lb})')
 prefix='''
__device__ unsigned long long k1_stage_marks[2048 * 9];
__device__ __forceinline__ void k1_stamp(int phase) {
    if (threadIdx.x == 0 && blockIdx.x % 32 == 0) {
        unsigned long long now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
        int slot = blockIdx.y * ((gridDim.x + 31) / 32) + blockIdx.x / 32;
        if (slot < 2048) reinterpret_cast<volatile unsigned long long*>(k1_stage_marks)[slot * 9 + phase] = now;
    }
}
'''
 s=prefix+s
 anchors=[('    // --- TMA load inputs',0),('    // --- QK L2 Normalization',1),('    // --- Fused gate activation',2),('// decay_apply',3),('    Tensor L =',4),('    Tensor INV =',5),('// inv (Neumann',6)]
 for anchor,phase in anchors:
  assert s.count(anchor)==1,(name,anchor)
  s=s.replace(anchor,f'    k1_stamp({phase});\n'+anchor)
 # Both sources have the same sync immediately before workspace TMA stores.
 anchor='    __syncthreads();\n    if (threadIdx.x == 0) {\n        int ws_idx'
 assert s.count(anchor)==1
 s=s.replace(anchor,'    __syncthreads();\n    k1_stamp(7);\n    if (threadIdx.x == 0) {\n        int ws_idx')
 anchor='    tma_store_wait<0>();\n    __syncthreads();'
 assert s.count(anchor)==1
 s=s.replace(anchor,anchor+'\n    k1_stamp(8);')
 replacements['smxx/fwd_kernel1.cuh']=s
 cpp=(source/'csrc/flash_kda.cpp').read_text()
 cpp='#include <vector>\nstd::vector<unsigned long long> get_k1_stage_marks();\n'+cpp
 cpp=cpp.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {','PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def("get_k1_stage_marks", &get_k1_stage_marks);')
 replacements['flash_kda.cpp']=cpp
 cu=(source/'csrc/smxx/fwd_launch.cu').read_text()
 cu+='''
#include <vector>
std::vector<unsigned long long> get_k1_stage_marks() {
    std::vector<unsigned long long> marks(2048 * 9);
    auto err = cudaMemcpyFromSymbol(marks.data(), k1_stage_marks, marks.size()*sizeof(unsigned long long));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return marks;
}
'''
 replacements['smxx/fwd_launch.cu']=cu
 for rel,code in replacements.items():
  target=dst/'csrc'/rel;target.unlink();target.write_text(code)
  old=(source/'csrc'/rel).read_text()
  (RUN/'analysis'/f'{name}_{Path(rel).name}.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),code.splitlines(True),fromfile=rel,tofile=rel)))
 code=(ROOT/'profile/01_c32_implementation/build.py').read_text()
 code=code.replace('root = Path(__file__).resolve().parents[2]',f'root = Path({str(dst)!r})')
 code=code.replace("build = root / 'profile/01_c32_implementation/build'",f'build = Path({str(RUN/('build_'+name))!r})')
 with (RUN/'analysis'/f'build_{name}.log').open('w') as log:
  subprocess.run([sys.executable,'-c',code],stdout=log,stderr=subprocess.STDOUT,check=True)
 manifest[name]={'source':str(source),'launch_bounds':lb,'binary_sha256':hashlib.sha256((RUN/f'build_{name}/flash_kda_C.so').read_bytes()).hexdigest()}
 (RUN/'analysis/build_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('built',name,flush=True)
