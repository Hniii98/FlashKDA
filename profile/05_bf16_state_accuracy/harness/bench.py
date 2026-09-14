"""Original K1 workspace + paired K2 reference; only persistent state dtype differs."""
import ctypes,json,math,sys,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from torch_ref import sigmoid_ext,fp32_fma
assert torch.cuda.is_available() and 'B300' in torch.cuda.get_device_name()
torch.set_grad_enabled(False);torch.set_num_threads(1)
lib=ctypes.CDLL(str(R/'harness/build/original.so'));H=96;D=128;C=16
CHECKS=[8192]
def original(q,k,v,g,beta,si):
 T=q.shape[1];bt=beta.reshape(T,H).t().contiguous();o=torch.empty_like(q);so=torch.empty_like(si)
 ws=torch.empty(H*(T//16)*13824+128,dtype=torch.uint8,device='cuda');al=torch.zeros(H,device='cuda');dt=torch.zeros(H,D,device='cuda')
 args=[q,k,v,g,bt,si,so,o,ws,al,dt]
 rc=lib.forward(*[ctypes.c_void_p(x.data_ptr()) for x in args],ctypes.c_int(1),ctypes.c_int(T),ctypes.c_int(H),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream));assert rc==0
 return o,so,ws

def input_case(kind,seed,param,T):
 # Always draw all random tensors in the same order before applying overrides.
 torch.manual_seed(seed);shape=(1,T,H,D)
 q=torch.randn(shape,device='cuda').bfloat16();k=torch.randn(shape,device='cuda').bfloat16();v=(torch.randn(shape,device='cuda')*.125).bfloat16()
 beta=torch.randn(1,T,H,device='cuda').bfloat16()
 g=torch.randn(shape,device='cuda').bfloat16()
 si=(torch.randn(1,H,D,D,device='cuda')*.02).bfloat16()
 if kind in ('beta','gate'):
  retention=.9999 if kind=='beta' else param
  p=-math.log(retention)/(5*C);g.fill_(math.log(p/(1-p)))
 if kind=='beta':beta.fill_(param)
 return q,k,v,g,beta,si

def metric(a,b):
 a=a.double();b=b.double();diff=a-b;den=b.norm();absnorm=diff.norm()
 return dict(relative_l2=float(absnorm/den.clamp_min(1e-30)),max_abs=float(diff.abs().max()),reference_norm=float(den),finite=bool(a.isfinite().all() and b.isfinite().all()))

def run(kind,seed,param,T):
 q,k,v,g,beta,si=input_case(kind,seed,param,T);aout,astate,ws=original(q,k,v,g,beta,si)
 L=T//C;offset=0
 def take(n,dtype,shape):
  nonlocal offset
  nbytes=n*torch.empty((),dtype=dtype).element_size();x=ws[offset:offset+nbytes].view(dtype).view(shape);offset+=nbytes;return x
 kd=take(H*L*C*D,torch.bfloat16,(H,L,C,D));qd=take(H*L*C*D,torch.bfloat16,(H,L,C,D));kr=take(H*L*C*D,torch.bfloat16,(H,L,C,D));gt=take(H*L*D,torch.float32,(H,L,D));inv=take(H*L*C*C,torch.bfloat16,(H,L,C,C));mqk=take(H*L*C*C,torch.bfloat16,(H,L,C,C))
 vv=v[0].permute(1,0,2).reshape(H,L,C,D);bv=sigmoid_ext.sigmoid_tanh_fp32(beta.float()).bfloat16()[0].t().reshape(H,L,C,1)
 # Canonical state is [K,V]; CUDA's external state is [V,K].
 s=si[0].transpose(-1,-2).float().repeat(2,1,1);out=torch.empty(2,H,T,D,dtype=torch.bfloat16,device='cuda')
 lost=torch.zeros((),dtype=torch.float64,device='cuda');changed=lost.clone();ratio_sum=lost.clone();ratio_min=torch.full((),float('inf'),device='cuda',dtype=torch.float64);ratio_max=torch.zeros_like(ratio_min);round_sumsq=lost.clone();rows=[]
 def both(x):return x.repeat(2,1,1)
 for j in range(L):
  sb=s.bfloat16();res=both(vv[:,j])-torch.bmm(both(kd[:,j]),sb);res=res*both(bv[:,j]);u=torch.bmm(both(inv[:,j]),res)
  y=torch.bmm(both(qd[:,j]),sb)+torch.bmm(both(mqk[:,j]),u);out[:,:,j*C:(j+1)*C]=y.reshape(2,H,C,D)
  delta=torch.bmm(both(kr[:,j].transpose(-1,-2)),u,out_dtype=torch.float32)
  nxt=fp32_fma(delta,s,gt[:,j,:,None].repeat(2,1,1))
  rounded=nxt[:H].bfloat16().float();nonzero=nxt[:H]!=s[:H]
  lost+=((rounded==s[:H])&nonzero).sum();changed+=nonzero.sum();round_sumsq+=(rounded.double()-nxt[:H].double()).square().sum()
  ratio=(nxt[:H]-s[:H]).norm()/s[:H].norm().clamp_min(1e-30)
  # Zero-initialized first updates have no meaningful relative-to-old-state ratio.
  if j>0 or bool(si.abs().max()>0):
   ratio_sum+=ratio;ratio_min=torch.minimum(ratio_min,ratio);ratio_max=torch.maximum(ratio_max,ratio)
  s=torch.cat([rounded,nxt[H:]],dim=0)
  t=(j+1)*C
  if t in CHECKS or t==T:
   if t==T:ao,ass=aout,astate
   else:ao,ass,_=original(*[x[:,:t].contiguous() for x in [q,k,v,g,beta]],si)
   ob=out[0,:,:t];oc=out[1,:,:t];ar=ao[0].permute(1,0,2)
   align_o=metric(ob,ar);align_s=metric(s[:H],ass[0].transpose(-1,-2))
   assert align_o['max_abs']==0 and align_s['max_abs']==0,('alignment',kind,seed,param,t,align_o,align_s)
   row=dict(kind=kind,seed=seed,param=param,T=t,alignment_output=align_o,alignment_state=align_s,output=metric(ob,oc),state=metric(s[:H],s[H:]),state_same_storage=metric(s[:H],s[H:].bfloat16()),actual_output_vs_fp32=metric(ar,oc),actual_state_vs_fp32=metric(ass[0].transpose(-1,-2),s[H:]),output_tail=metric(ob[:,-1024:],oc[:,-1024:]),beta_mean=float(bv.float().mean()),retention_mean=float(gt.mean()),update_ratio_mean=float(ratio_sum/max(1,j+1-(kind=='normal' and param=='zero'))),update_ratio_min=float(ratio_min),update_ratio_max=float(ratio_max),lost_update_fraction=float(lost/changed.clamp_min(1)),rounding_rms=float((round_sumsq/((j+1)*H*D*D)).sqrt()))
   assert row['output']['finite'] and row['state']['finite'];rows.append(row)
 return rows

quick='--check' in sys.argv
specs=[('random',0,'random',8192)] if quick else ([('random',seed,'random',8192) for seed in range(5)]+[('beta',seed,p,8192) for p in [-9.,-7.,-5.,-3.,-1.,0.,2.] for seed in range(5)]+[('gate',seed,p,8192) for p in [.5,.9,.99,.999,.9999] for seed in range(5)])
metadata=dict(design='controlled_abc_v2',gpu=torch.cuda.get_device_name(),SMs=torch.cuda.get_device_properties(0).multi_processor_count,torch=torch.__version__,cases=len(specs),N=1,H=H,D=D,CHUNK=C,source=json.loads((R/'analysis/source.json').read_text()))
allrows=[]
for i,(kind,seed,param,T) in enumerate(specs):
 rows=run(kind,seed,param,T);allrows+=rows
 print(f'{i+1}/{len(specs)} {kind} seed={seed} param={param} T={T}: output={100*rows[-1]["output"]["relative_l2"]:.4f}% state={100*rows[-1]["state_same_storage"]["relative_l2"]:.4f}%',flush=True)
 target=R/'analysis'/('check.md' if quick else 'DATA.md')
 target.write_text('# 原始数值证据\n\n```json\n'+json.dumps(dict(metadata=metadata,rows=allrows),ensure_ascii=False)+'\n```\n')
 torch.cuda.empty_cache()
