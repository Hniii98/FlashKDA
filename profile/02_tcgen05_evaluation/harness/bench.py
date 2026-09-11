"""B300 shared-B/transpose microbench. python -B bench.py [--profile]."""
import sys,ctypes,json,statistics,hashlib,subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
assert 'B300' in torch.cuda.get_device_name()
torch.set_grad_enabled(False);torch.set_num_threads(1)
lib=ctypes.CDLL(str(ROOT/'harness/kernels.so'))
SHAPES=[(16,16,16),(16,16,128),(16,128,16),(16,128,128),(128,128,16),(32,16,128),(32,128,128),(128,16,16),(128,16,128),(32,16,16)]
def pack(x,rows):
 b,r,k=x.shape;y=torch.zeros((b,rows,k),device='cuda',dtype=x.dtype);y[:,:r].copy_(x)
 return y.reshape(b,rows//8,8,k//8,8).permute(0,1,3,2,4).contiguous()
def prepare(s,mode,a,b):
 m,n,k=SHAPES[s];pm=max(m,64 if mode==1 else 32) if mode else m;pn=max(n,64) if mode==2 else n
 return pack(a,pm),pack(b.transpose(1,2),pn),torch.empty((len(a),m,n),device='cuda')
def call(s,mode,d):
 a,b,c=d
 r=lib.run_mma(ctypes.c_int(s),ctypes.c_int(mode),ctypes.c_int(len(a)),*[ctypes.c_void_p(t.data_ptr()) for t in d],ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
 assert r==0,r
# Shapes 5/6 are two same-chunk left operands stacked, sharing B.
# Matched baseline stacks too, ensuring equal CTA count and common-B memory reuse.
CASES=[('Neumann square',0,None),('Neumann shared-power pair',9,None),('K1 shared-B pair',5,None),('K2 shared-state pair',6,None),('INV@residual / Mqk@U',2,7),('single K/Q@state',3,8),('state update',4,None)]
def make(s,batch,seed):
 m,n,k=SHAPES[s];g=torch.Generator(device='cuda').manual_seed(seed)
 typ=torch.float16 if s in (0,9) else torch.bfloat16
 return (torch.randn(batch,m,k,device='cuda',generator=g)*.125).to(typ),(torch.randn(batch,k,n,device='cuda',generator=g)*.125).to(typ)
def variants(s,ts,a,b):
 result=[]
 for mode in range(3):
  d=prepare(s,mode,a,b)
  result.append((['mma_sync','tcgen05','tcgen05_ws'][mode],lambda d=d,mode=mode:call(s,mode,d),lambda d=d:d[2]))
 if ts is not None:
  # Both operands transpose, then output transpose: exactly AB=(B^T A^T)^T.
  d=prepare(ts,1,b.transpose(1,2),a.transpose(1,2));out=d[2].view(len(a),a.shape[1],b.shape[2])
  def trans():
   call(ts,1,d)
  result.append(('tcgen05_transpose',trans,lambda:out))
 return result
records=[];checks=[]
profile='--profile' in sys.argv
for batch in ([16384] if profile else [3,1,16384]):
 for name,s,ts in CASES:
  seeds=range(3) if batch==3 else [42]
  for seed in seeds:
   a,b=make(s,batch,seed);vs=variants(s,ts,a,b)
   if batch==3:
    ref=a.double()@b.double()
    for mode,fn,out in vs:
     fn();torch.cuda.synchronize();v=out();err=float((v.double()-ref).norm()/ref.norm())
     assert bool(v.isfinite().all()) and err<(0.003 if s in (0,9) else 2e-5),(name,mode,err)
     checks.append(dict(case=name,mode=mode,seed=seed,relative_l2=err,finite=True))
   elif profile:
    for mode,fn,out in vs:
     fn();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();fn();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
   else:
    graphs=[];inner=32 if batch==1 else 4
    for mode,fn,out in vs:
     for _ in range(5):fn()
     torch.cuda.synchronize();gr=torch.cuda.CUDAGraph()
     with torch.cuda.graph(gr):
      for _ in range(inner):fn()
     graphs.append(gr)
     for _ in range(3):gr.replay()
    torch.cuda.synchronize();samples=[[] for v in vs]
    for r in range(12):
     for pos in range(len(vs)):
      j=(r+pos)%len(vs);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
      start.record();graphs[j].replay();end.record();end.synchronize();samples[j].append(start.elapsed_time(end)*1000/inner)
    base=statistics.median(samples[0])
    for (mode,_,_),vals in zip(vs,samples):
     rec=dict(case=name,shape=SHAPES[s],batch=batch,mode=mode,median_us=statistics.median(vals),speedup=base/statistics.median(vals),samples_us=vals)
     records.append(rec);print(name,batch,mode,rec['median_us'],rec['speedup'],flush=True)
    del graphs
   del vs,a,b
   torch.cuda.empty_cache()
if not profile:
 data=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,SMs=torch.cuda.get_device_properties(0).multi_processor_count,checks=checks,timing=records,source_sha256=hashlib.sha256((ROOT/'harness/kernels.cu').read_bytes()).hexdigest(),nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
 lines=['# Reorganized GEMM: B300 results','','Shared-B pairs use the same stacked layout in all three paths: same CTA count, effective outputs and B reuse. Transpose swaps both operands and writes the original output orientation directly inside the CUDA kernel. Input packing is excluded for all paths. These are GEMM microbenchmarks, not end-to-end KDA.','','| Case | Batch | Path | µs | Speedup old/new |','|---|---|---|---|---|']
 for r in records:lines.append(f"| {r['case']} | {r['batch']} | {r['mode']} | {r['median_us']:.4f} | {r['speedup']:.3f} |")
 lines+=['',f'Correctness: {len(checks)} checks passed; maximum relative L2 {max(x["relative_l2"] for x in checks):.6g}.','','<details><summary>Raw measurements and environment</summary>','','```json',json.dumps(data,ensure_ascii=False),'```','','</details>']
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
