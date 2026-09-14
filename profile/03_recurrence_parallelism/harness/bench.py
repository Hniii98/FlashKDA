"""Real K1 workspace + full sequential K2, B300 only. --profile records selected K2 cases."""
import ctypes,json,statistics,sys,hashlib,subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];PERSISTENT='--persistent' in sys.argv
HEADS='--heads' in sys.argv
GROUP='persistent' if PERSISTENT else 'multi_head' if HEADS else 'column_split'
R=ROOT/GROUP
B=ROOT/'harness'/('build_'+GROUP)
PREFIX=''
assert torch.cuda.is_available() and 'B300' in torch.cuda.get_device_name()
torch.set_grad_enabled(False);torch.set_num_threads(1)
MODES=['original','static148','dynamic148','static296','dynamic296'] if PERSISTENT else ['original','heads2'] if HEADS else ['original','split1','split2','split4']
LIBNAMES={m:('static' if m.startswith('static') else 'dynamic' if m.startswith('dynamic') else m) for m in MODES}
libs={m:ctypes.CDLL(str(B/(LIBNAMES[m]+'.so'))) for m in MODES}
queue=torch.zeros(1,device='cuda',dtype=torch.int32) if PERSISTENT else None
PROFILE='--profile' in sys.argv
CHECK='--check' in sys.argv
# N sequences, H heads, sequence lengths; P=N*H independent chains.
CASES=[('one_long',1,[4096]),('few_long',8,[4096]),('64_chains',64,[1024]),('148_chains',148,[1024]),('256_chains',256,[1024]),('many_short',256,[32]),('equal_64',8,[512]*8),('skew_64',8,[16]*7+[3984])]
if HEADS:CASES=[('two_long',2,[4096])]+CASES[1:]
if PERSISTENT:CASES += [('many_equal',4,[32]*148),('long_first',4,[2048]+[16]*147),('long_last',4,[16]*147+[2048])]
def inputs(H,lens,seed):
 torch.manual_seed(seed);N=len(lens);T=sum(lens)
 cu=torch.tensor([0]+list(__import__('itertools').accumulate(lens)),device='cuda',dtype=torch.int64)
 shape=(T,H,128)
 q=(torch.randn(shape,device='cuda')*.125).bfloat16();k=(torch.randn(shape,device='cuda')*.125).bfloat16();v=(torch.randn(shape,device='cuda')*.125).bfloat16()
 g=torch.randn(shape,device='cuda').bfloat16();beta=torch.randn(H,T,device='cuda').bfloat16()
 si=(torch.randn(N,H,128,128,device='cuda')*.02).bfloat16();so=torch.empty_like(si);out=torch.empty_like(q)
 tiles=sum((t+15)//16 for t in lens)
 ws=torch.empty(H*tiles*13824+((N+1)*4+127)//128*128,device='cuda',dtype=torch.uint8)
 al=torch.zeros(H,device='cuda');dt=torch.zeros(H,128,device='cuda')
 return [q,k,v,g,beta,si,so,out,ws,cu,al,dt],(tiles,T,H,N)
def call(mode,phase,d,meta):
 if PERSISTENT:
  workers=296 if mode.endswith('296') else 148
  libs[mode].configure(ctypes.c_int(workers),ctypes.c_void_p(queue.data_ptr()))
 rc=libs[mode].run(ctypes.c_int(phase),*[ctypes.c_void_p(x.data_ptr()) for x in d],*[ctypes.c_int(x) for x in meta],ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
 assert rc==0,(mode,rc)
def check(H,lens,seed):
 d,meta=inputs(H,lens,seed);call('original',0,d,meta);torch.cuda.synchronize()
 refs=[d[7].clone(),d[6].clone()];rows=[]
 assert all(bool(x.isfinite().all()) for x in refs),'baseline nonfinite'
 for mode in MODES[1:]:
  d[7].fill_(float('nan'));d[6].fill_(float('nan'));call(mode,2,d,meta);torch.cuda.synchronize()
  for label,value,ref in [('output',d[7],refs[0]),('state',d[6],refs[1])]:
   finite=bool(value.isfinite().all());eq=bool(torch.equal(value,ref));err=float((value.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-20))
   row=dict(H=H,lens=lens,seed=seed,mode=mode,tensor=label,finite=finite,bitwise_equal=eq,relative_l2=err);rows.append(row)
   print('CHECK',row,flush=True);assert finite and eq,row
 return rows
def check_gate_scale():
 # Check K1's stored chunk retention against the public natural-log convention.
 # With g=A_log=dt_bias=0, sigmoid=1/2 and retention=exp(-5*16/2).
 d,meta=inputs(2,[16],123);d[3].zero_();rows=[]
 offset=3*2*16*128*2
 expected=__import__('math').exp(-5.*16/2)
 for mode in MODES:
  call(mode,0,d,meta);torch.cuda.synchronize()
  actual=d[8][offset:offset+2*128*4].view(torch.float32)
  error=float((actual.double()/expected-1).abs().max())
  assert bool(actual.isfinite().all()) and error<1e-4,(mode,'gate scale',error)
  rows.append(dict(mode=mode,lower_bound=-5.,expected_retention=expected,max_relative_error=error))
 return rows

gate_checks=[] if PROFILE else check_gate_scale()
checks=[]
if not PROFILE:
 check_cases=([(1,[16]),(3,[1,17,31]),(2,[4096]),(2,[17]*300),(4,[512]+[16]*147)] if PERSISTENT else [(2,[16]),(4,[1,17,31]),(8,[512]),(2,[4096])] if HEADS else [(1,[16]),(3,[1,17,31]),(8,[512]),(2,[4096])])
 for seed in [0,1]:
  for H,lens in check_cases:checks+=check(H,lens,seed)
 if CHECK:sys.exit(0)
records=[];profile_cases=[]
for name,H,lens in CASES:
 if PROFILE and name not in (['few_long','many_equal','long_first'] if PERSISTENT else ['few_long','64_chains','256_chains','skew_64']):continue
 d,meta=inputs(H,lens,42);call('original',1,d,meta);torch.cuda.synchronize()
 if PROFILE:
  for mode in MODES:
   call(mode,2,d,meta);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();call(mode,2,d,meta);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
   profile_cases.append(dict(case=name,mode=mode,H=H,lens=lens))
 else:
  for phase in [2,0]:
   graphs=[];inner=3
   for mode in MODES:
    for _ in range(3):call(mode,phase,d,meta)
    torch.cuda.synchronize();gr=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
     for _ in range(inner):call(mode,phase,d,meta)
    graphs.append(gr)
   vals=[[] for m in MODES]
   for rnd in range(9):
    for pos in range(len(MODES)):
     j=(rnd+pos)%len(MODES);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
     a.record();graphs[j].replay();b.record();b.synchronize();vals[j].append(a.elapsed_time(b)*1000/inner)
   base=statistics.median(vals[0]);control=statistics.median(vals[1])
   for mode,samples in zip(MODES,vals):
    med=statistics.median(samples);row=dict(case=name,H=H,lens=lens,chains=H*len(lens),phase='K2' if phase==2 else 'K1+K2',mode=mode,median_us=med,speedup=base/med,vs_control=control/med,samples_us=samples);records.append(row);print('TIME',name,row['phase'],mode,round(med,3),round(base/med,3),flush=True)
   del graphs
 del d;torch.cuda.empty_cache()
if PROFILE:(R/'reports'/(PREFIX+'cases.json')).write_text(json.dumps(profile_cases,indent=2)+'\n')
else:
 data=dict(lower_bound=-5.,gate_scale=-5.*1.4426950408889634,gate_checks=gate_checks,shared_memory_per_block_optin=torch.cuda.get_device_properties(0).shared_memory_per_block_optin,gpu=torch.cuda.get_device_name(),SMs=torch.cuda.get_device_properties(0).multi_processor_count,torch=torch.__version__,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),shared_bytes={m:libs[m].shared_bytes() for m in MODES},checks=checks,timing=records,binary_sha256={m:hashlib.sha256((B/(LIBNAMES[m]+'.so')).read_bytes()).hexdigest() for m in MODES})
 lines=['# Topic3 B300 实测结果'+('：跨链 persistent 调度' if PERSISTENT else '：多 head 共 CTA' if HEADS else ''),'',('original：普通 grid；static/dynamic：固定 148/296 worker，静态步进/动态原子队列，每个任务为完整递推链。动态队列初始化计入 K2。' if PERSISTENT else 'original：每 CTA 一个 head；heads2：每 CTA 两个 head，独立 state 和流水线，保留 TMA 写出。加速比=原始耗时/新耗时。' if HEADS else 'original：原 K2；split1：单 CTA、与分片相同的协作写出；split2/4：每个 head 的 value 列拆为 2/4 个 CTA。加速比=原始耗时/新耗时。'),'','| 场景 | 阶段 | 路径 | µs | 对原版加速比 | 对 split1 加速比 |','|---|---|---|---|---|---|']
 for x in records:lines.append(f"| {x['case']} | {x['phase']} | {x['mode']} | {x['median_us']:.3f} | {x['speedup']:.3f} | {x['vs_control']:.3f} |")
 lines+=['',f'正确性：{len(checks)} 个 output/final-state 比较全部与原 K2 逐位一致。','','<details><summary>原始样本、输入形状、正确性和环境</summary>','','```json',json.dumps(data,ensure_ascii=False),'```','','</details>']
 if HEADS or PERSISTENT:
  lines=[line.rsplit('|',2)[0]+'|' if line.startswith('|') else line for line in lines]
 (R/(PREFIX+'RESULTS.md')).write_text('\n'.join(lines)+'\n')
