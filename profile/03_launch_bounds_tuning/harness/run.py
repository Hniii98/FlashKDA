"""Profile existing binaries and measure same-binary rescale ablations."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bench',ROOT/'benchmarks/bench_c32_vs_c16.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['timing','profile'],required=True)
p.add_argument('--variant',default='C32');p.add_argument('--binary',type=Path)
p.add_argument('--lower-bound',type=float,default=-5);p.add_argument('--rescale',type=float,default=2.0**-96)
p.add_argument('--output',type=Path);args=p.parse_args()
torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(0)
H,D,T,N=96,128,8192,1
shape=(1,T,H,D)
q,k=[F.normalize(torch.randn(shape,device='cuda'),dim=-1).bfloat16() for _ in range(2)]
v,g=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
beta=torch.randn(1,T,H,device='cuda',dtype=torch.bfloat16)
al,bias=torch.rand(H,device='cuda'),torch.rand(H,D,device='cuda')
h0=torch.arange(H*D*D,device='cuda',dtype=torch.float32).reshape(N,H,D,D).bfloat16().float()
paths={'C16':bench.BASELINE,'LB8':RUN/'build_8/flash_kda_C.so', **{f'LB{n}':RUN/f'build_{n}/flash_kda_C.so' for n in [5,4,2]}}
if args.binary:paths[args.variant]=args.binary.resolve()
mods={key:bench.load(key,path) for key,path in paths.items()}
out={key:torch.empty_like(q) for key in mods};state={key:torch.empty_like(h0) for key in mods}
def make_fn(key,lower_bound,rescale):
 def fn():
  m=mods[key];workspace=torch.empty(m.get_workspace_size(T,H,N),device='cuda',dtype=torch.uint8)
  extra={} if key=='C16' else {'rescale':rescale,'inverse_rescale':1.0}
  m.fwd(q,k,v,g,beta,D**-0.5,out[key],workspace,al,bias,lower_bound,initial_state=h0,final_state=state[key],**extra)
 return fn
if args.mode=='profile':
 fn=make_fn(args.variant,args.lower_bound,args.rescale)
 for _ in range(5):fn()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();fn();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
else:
 labels={f'{key}_g{abs(lb):g}_s{e}':(key,lb,2.0**e) for lb in [-5.] for key in mods for e in ([0] if key=='C16' else [-96])}
 fns={label:make_fn(*cfg) for label,cfg in labels.items()}
 for fn in fns.values():
  for _ in range(30):fn()
 torch.cuda.synchronize();samples={k:[] for k in fns};rounds=[]
 for rep in range(5):
  order=list(fns) if rep%2==0 else list(reversed(fns));record={}
  for key in order:
   vals=bench.timed(fns[key],200);samples[key]+=vals;record[key]=bench.stats(vals)
  rounds.append(record)
 rows={}
 for label,fn in fns.items():
  key,lb,s=labels[label];fn();torch.cuda.synchronize()
  rows[label]=dict(variant=key,lower_bound=lb,rescale=s,full=bench.stats(samples[label]),
   finite=bool(out[key].isfinite().all() and state[key].isfinite().all()),
   kernels=bench.kernel_times(fn,20,RUN/'analysis'/f'{label}_trace.json'))
 result=dict(rows=rows,rounds=rounds,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
  binary_sha256={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in paths.items()})
 dest=args.output or RUN/'analysis/runtime_ablation.json';dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
