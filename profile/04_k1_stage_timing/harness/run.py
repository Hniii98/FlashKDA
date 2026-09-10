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
paths={'C16':bench.BASELINE,'C32':ROOT/'profile/03_launch_bounds_tuning/build_4/flash_kda_C.so','C16T':RUN/'build_C16/flash_kda_C.so','C32T':RUN/'build_C32/flash_kda_C.so'}
if args.binary:paths[args.variant]=args.binary.resolve()
mods={key:bench.load(key,path) for key,path in paths.items()}
out={key:torch.empty_like(q) for key in mods};state={key:torch.empty_like(h0) for key in mods}
def make_fn(key,lower_bound,rescale):
 def fn():
  m=mods[key];workspace=torch.empty(m.get_workspace_size(T,H,N),device='cuda',dtype=torch.uint8)
  extra={} if key.startswith('C16') else {'rescale':rescale,'inverse_rescale':1.0}
  m.fwd(q,k,v,g,beta,D**-0.5,out[key],workspace,al,bias,lower_bound,initial_state=h0,final_state=state[key],**extra)
 return fn
if args.mode=='profile':
 fn=make_fn(args.variant,args.lower_bound,args.rescale)
 for _ in range(5):fn()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();fn();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
else:
 labels={f'{key}_g{abs(lb):g}_s{e}':(key,lb,2.0**e) for lb in [-5.] for key in mods for e in ([0] if key.startswith('C16') else [-96])}
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

if args.mode=='timing':
 phase_names=['input_load','qk_normalize','gate_cumsum_gt','decay','L_Mqk','tril_beta_init','neumann','workspace_store']
 stage_result={}
 for key in ['C16T','C32T']:
  fn=make_fn(key,-5.,1.0 if key.startswith('C16') else 2.0**-96)
  nsamples=H*((T//(16 if key.startswith('C16') else 32)+31)//32)
  samples=[]
  for _ in range(10):
   fn();torch.cuda.synchronize()
   marks=torch.tensor(mods[key].get_k1_stage_marks(),dtype=torch.int64).reshape(-1,9)[:nsamples]
   delta=marks[:,1:]-marks[:,:-1]
   assert bool((delta>=0).all()),'nonmonotonic timestamps'
   samples.append(delta)
  delta=torch.cat(samples).double()
  stage_result[key]={'sampled_ctas_per_launch':nsamples,'launches':10,'unit':'ns',
    'phases':{name:{'mean':float(delta[:,i].mean()),'median':float(delta[:,i].median()),'p95':float(delta[:,i].quantile(.95))} for i,name in enumerate(phase_names)},
    'mean_total':float(delta.sum(1).mean())}
  torch.save(delta,RUN/'analysis'/f'{key}_stage_samples.pt')
 # Check the timing binaries against their unchanged counterpart on this input.
 equality={}
 for base,timed in [('C16','C16T'),('C32','C32T')]:
  for key in [base,timed]:make_fn(key,-5.,1.0 if key.startswith('C16') else 2.0**-96)()
  torch.cuda.synchronize()
  equality[base]={'output_bitwise_equal':bool(torch.equal(out[base],out[timed])), 'state_bitwise_equal':bool(torch.equal(state[base],state[timed]))}
 (RUN/'analysis/stages.json').write_text(json.dumps({'variants':stage_result,'equality':equality},indent=2)+'\n')
