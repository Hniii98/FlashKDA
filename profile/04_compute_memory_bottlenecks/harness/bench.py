"""Original csrc K1->K2. 20 warmups; profile only step 21 in each case."""
import ctypes,sys,json,statistics,hashlib,subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'profile/04_compute_memory_bottlenecks'
assert torch.cuda.is_available() and 'B300' in torch.cuda.get_device_name()
torch.set_grad_enabled(False);torch.set_num_threads(1)
lib=ctypes.CDLL(str(R/'harness/build/original.so'))
CASES=[('task_n1_h96_t8192',1,96,8192)]
def inputs(N,H,T):
 torch.manual_seed(42);shape=(N,T,H,128)
 q=(torch.randn(shape,device='cuda')*.125).bfloat16();k=(torch.randn(shape,device='cuda')*.125).bfloat16();v=(torch.randn(shape,device='cuda')*.125).bfloat16();g=torch.randn(shape,device='cuda').bfloat16()
 beta=torch.randn(N,T,H,device='cuda').bfloat16();bt=beta.reshape(N*T,H).t().contiguous()
 si=(torch.randn(N,H,128,128,device='cuda')*.02).bfloat16();so=torch.empty_like(si);out=torch.empty_like(q)
 ws=torch.empty(H*N*((T+15)//16)*13824+128,device='cuda',dtype=torch.uint8);al=torch.zeros(H,device='cuda');dt=torch.zeros(H,128,device='cuda')
 return [q,k,v,g,bt,si,so,out,ws,al,dt],beta

def call(d,N,H,T):
 rc=lib.forward(*[ctypes.c_void_p(x.data_ptr()) for x in d],ctypes.c_int(N),ctypes.c_int(T),ctypes.c_int(H),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream));assert rc==0,rc

def check():
 sys.path.insert(0,str(ROOT/'tests'));from torch_ref import torch_ref
 rows=[]
 for N,H,T in [(1,2,16),(2,2,31)]:
  d,beta=inputs(N,H,T);call(d,N,H,T);torch.cuda.synchronize();refout=torch.empty_like(d[7]);refstate=torch.empty_like(d[6])
  torch_ref(d[0],d[1],d[2],d[3],beta,128**-.5,refout,d[9],d[10],-5.,initial_state=d[5],final_state=refstate)
  for label,x,y in [('output',d[7],refout),('state',d[6],refstate)]:
   err=float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-20));finite=bool(x.isfinite().all());assert finite and err<.02,(label,err)
   rows.append(dict(N=N,H=H,T=T,tensor=label,relative_l2=err,finite=finite));print('CHECK',rows[-1],flush=True)
 return rows
profile='--profile' in sys.argv
checks=[] if profile else check()
rows=[];timings=[]
for name,N,H,T in CASES:
 d,beta=inputs(N,H,T)
 # Allocation, beta layout preparation and reference work finish before warmup.
 torch.cuda.synchronize()
 for step in range(1,21):call(d,N,H,T)
 torch.cuda.synchronize()
 if profile:
  torch.cuda.nvtx.range_push(name+'/step21')
  torch.cuda.cudart().cudaProfilerStart();call(d,N,H,T);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
  torch.cuda.nvtx.range_pop()
 else:
  # Step 21 is the first measured execution; 9 blocks of 20 consecutive forwards.
  samples=[]
  for rep in range(9):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record()
   for _ in range(20):call(d,N,H,T)
   b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/20)
  assert bool(d[7].isfinite().all()) and bool(d[6].isfinite().all())
  timings.append(dict(case=name,N=N,H=H,T=T,median_us=statistics.median(samples),samples_us=samples,finite_output=True,finite_state=True))
  print('TIME',name,statistics.median(samples),flush=True)
 for kernel in ['K1','K2']:rows.append(dict(case=name,N=N,H=H,T=T,kernel=kernel,warmup=20,step=21,expected_ctas=(N*((T+15)//16)*H if kernel=='K1' else N*H)))
 del d,beta;torch.cuda.empty_cache()
if profile:(R/'analysis/cases.json').write_text(json.dumps(rows,indent=2)+'\n')
else:
 data=dict(gpu=torch.cuda.get_device_name(),SMs=torch.cuda.get_device_properties(0).multi_processor_count,torch=torch.__version__,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),warmup=20,checks=checks,timing=timings,source=json.loads((R/'analysis/source.json').read_text()))
 lines=['# Topic4 完整 forward 独立计时','','20 次 warmup 后，9 组×20 次完整 K1→K2；以下为每次 forward 的 µs。输入及 beta 转置预先准备，不包含 Python 公开接口的分配/布局转换。分 kernel 的 NCU 时间与这里的口径分开。','','| 场景 | N | H | T | K1+K2 µs |','|---|---|---|---|---|']
 for x in timings:lines.append(f"| {x['case']} | {x['N']} | {x['H']} | {x['T']} | {x['median_us']:.3f} |")
 lines+=['','<details><summary>正确性、环境与原始样本</summary>','','```json',json.dumps(data,ensure_ascii=False),'```','','</details>']
 (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
