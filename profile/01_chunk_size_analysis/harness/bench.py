"""B300 CHUNK experiment: one entry for run, profile, analysis, and reporting.

CPU-only: report / pack / tidy. GPU commands require B300.
CUDA algorithms live in kernels.cu; Python owns inputs, reference checks and results.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = None
torch = None


def initialize_gpu():
    global torch, LIB
    import torch as gpu_torch
    torch = gpu_torch
    assert torch.cuda.is_available() and 'B300' in torch.cuda.get_device_name(), 'B300 execution is mandatory'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    LIB = ctypes.CDLL(str(ROOT / 'harness/build/kernels.so'))



def read_tables(path):
    tables = {}
    if not path.exists():
        return tables
    marker = '<!-- raw-evidence -->'
    text = path.read_text()
    if marker in text:
        payload = text.split(marker, 1)[1].split('```json\n', 1)[1].split('\n```', 1)[0]
        return json.loads(payload)
    name = None
    fields = None
    for line in path.read_text().splitlines():
        if line.startswith('## '):
            name = line[3:].strip()
            tables[name] = []
            fields = None
        elif name and line.startswith('| '):
            cells = [x.strip() for x in line[1:-1].split('|')]
            if fields is None:
                fields = cells
            elif all(x == '---' for x in cells):
                continue
            else:
                assert len(cells) == len(fields), (path, name)
                tables[name].append(dict(zip(fields, cells)))
    return tables


def write_tables(path, tables):
    lines = ['# ' + path.parent.name.upper() + ' 数值异常范围', '',
             'B300 实测。正文按区间和输入条件汇总；“正常”仅指未出现归零/Inf/NaN，不代表误差为零。', '']
    if tables.get('range'):
        rr = sorted(tables['range'], key=lambda x: float(x['a']))
        groups = []
        for row in rr:
            state = (bool(row['first_exp_zero_token']), bool(row['first_inv_nonfinite_token']))
            if not groups or groups[-1][0] != state:
                groups.append((state, []))
            groups[-1][1].append(row)
        lines += ['## range', '', '每个 token 使用相同衰减 a，扫描范围为 [0,5]。下表端点是实际采样点，未采样的小间隙不是已测结论。', '',
                  table(['已测 a 区间（含端点）', '采样点数', '负指数', '逆指数'],
                        [[f"{float(g[0]['a']):.9g} ～ {float(g[-1]['a']):.9g}", len(g),
                          '出现归零' if state[0] else '未归零', '出现非有限值' if state[1] else '有限'] for state,g in groups]), '']
        for key,label in [('first_exp_zero_token','负指数归零'),('first_inv_nonfinite_token','逆指数非有限')]:
            first = next((i for i,x in enumerate(rr) if x[key]), None)
            if first is None:
                lines.append(f'- {label}：a≤5 的本次扫描未出现。')
            elif first:
                left,right = rr[first-1],rr[first]
                lines.append(f"- {label}边界：最后正常采样 a={float(left['a']):.9g}，首次异常 a={float(right['a']):.9g}；首次异常发生于第 {right[key]} 个 token。临界位置在这两个采样值之间，精确阈值未测定。")
        lines.append('')
    if tables.get('inverse'):
        records = tables['inverse']
        # Merge patterns with identical outcomes, then combine tested a values
        # sharing the same beta/seed outcome; never interpolate discrete inputs.
        signatures = {}
        for pattern in sorted({x['pattern'] for x in records}):
            sig = tuple((float(x['a']),float(x['beta']),int(x['seed']),x['finite'])
                        for x in sorted([v for v in records if v['pattern']==pattern],key=lambda v:(float(v['a']),float(v['beta']),int(v['seed']))))
            signatures.setdefault(sig, []).append(pattern)
        summary = []
        for sig,patterns in signatures.items():
            cells = {}
            for av in sorted({x[0] for x in sig}):
                for bv in sorted({x[1] for x in sig}):
                    subset = [x for x in sig if x[0]==av and x[1]==bv]
                    bad = sum(x[3]=='False' for x in subset)
                    cells.setdefault((bv,bad,len(subset)), []).append(av)
            merged = {}
            for (bv,bad,n),aa in cells.items():
                merged.setdefault((tuple(aa),bad,n), []).append(bv)
            for (aa,bad,n),bb in merged.items():
                total = n*len(aa)*len(bb)*len(patterns)
                failures = bad*len(aa)*len(bb)*len(patterns)
                summary.append([', '.join(patterns), '{'+', '.join(map(str,aa))+'}',
                                '{'+', '.join(map(str,bb))+'}', f'{failures}/{total}',
                                '全部有限' if failures==0 else '全部 Inf/NaN' if failures==total else '部分 Inf/NaN'])
        bad = sum(x['finite']=='False' for x in records)
        lines += ['## inverse', '',
                  f'完整 FP16 Neumann：{bad}/{len(records)} 个 case 出现 Inf/NaN。3 个 seed 均纳入统计。',
                  'a 与 beta 只测试表中列出的离散值；不能将集合解释为连续异常区间。', '',
                  table(['k 类型', '已测 a', '已测 beta', '异常/总数', '结果'], summary), '',
                  'random=随机；repeated=相同；alternating=交替方向；correlated=高相关。',
                  '有限结果最大 inverse 相对 L2 = '+f"{max(float(x['inverse_rel_l2']) for x in records if x['finite']=='True'):.6g}"+'，参考为同一输入的 fp64 三角求解。', '']
    if tables.get('inverse_stages'):
        stages = tables['inverse_stages']
        partial = [x for x in stages if x['part']=='inverse_partial']
        first = next((i for i,x in enumerate(partial) if int(x['inf_elements'])+int(x['nan_elements'])>0),None)
        indices = sorted({0, len(partial)-1} if first is None else {max(0,first-1),first,len(partial)-1})
        lines += ['## inverse_stages', '',
                  '代表输入：seed=0，所有 k 相同，a=0，beta=0.9。' + ('各阶段均无 Inf/NaN，列出首次更新及最终部分和。' if first is None else '只列异常前、首次异常及最终部分和。'), '',
                  table(['部分和最高次数', '有限元素最大绝对值', 'Inf 数', 'NaN 数'],
                        [[2*int(partial[i]['power'])-1, partial[i]['finite_max_abs'] or '无有限元素',
                          partial[i]['inf_elements'],partial[i]['nan_elements']] for i in indices]), '']
    lines += ['<details>', '<summary>原始测量记录（复核及重新生成用，默认折叠）</summary>', '',
              '<!-- raw-evidence -->', '```json', json.dumps(tables,ensure_ascii=False,separators=(',', ':')), '```', '', '</details>', '']
    path.write_text('\n'.join(lines))


def pack_results(c):
    folder = ROOT / f'chunk{c}'
    target = folder / 'RESULTS.md'
    data = read_tables(target)
    sources = sorted((folder / 'analysis').glob('*.csv'))
    for p in sources:
        with p.open() as f:
            data[p.stem] = list(csv.DictReader(f))
    write_tables(target, data)
    assert read_tables(target) == data, 'Markdown round-trip changed data'
    for p in sources:
        p.unlink()


def save(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.parent.parent / 'RESULTS.md'
    data = read_tables(target)
    selection = {
        'range': lambda x: x['pattern'] == 'constant',
        'inverse': lambda x: x['method'] == 'mma_fp16acc_resident',
        'inverse_stages': lambda x: x['method'] == 'fp16acc' and int(x['seed']) == 0 and x['pattern'] == 'repeated' and float(x['a']) == 0 and float(x['beta']) == .9,
    }
    fields = {'range': ['chunk', 'pattern', 'a', 'total_max', 'first_exp_zero_token', 'first_inv_nonfinite_token', 'k_restored_nonfinite'], 'inverse': ['chunk', 'method', 'seed', 'pattern', 'a', 'beta', 'finite', 'inverse_rel_l2', 'intermediate_max', 'first_bad_stage_power'], 'inverse_stages': ['chunk', 'method', 'seed', 'pattern', 'a', 'beta', 'power', 'part', 'finite_max_abs', 'inf_elements', 'nan_elements']}
    if path.stem not in selection:
        raise ValueError('Only numerical-anomaly evidence is retained: ' + path.stem)
    data[path.stem] = [{k: str(row[k]) for k in fields[path.stem]} for row in records if selection[path.stem](row)]
    write_tables(target, data)


def rows(c, name):
    return read_tables(ROOT / f'chunk{c}/RESULTS.md')[name]


# common

def call(name,*args):
    argv=[ctypes.c_void_p(x.data_ptr()) if isinstance(x,torch.Tensor) else ctypes.c_void_p(0) if x is None else ctypes.c_int(x) for x in args]
    argv.append(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    code=getattr(LIB,name)(*argv)
    if code:raise RuntimeError(f'{name}: CUDA status {code}')

def rel(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30)).item()


def inverse(L,method=0,resident=1,trace=False):
    C=L.shape[-1];B=L.numel()//(C*C)
    output=torch.empty_like(L);bf=torch.empty_like(L,dtype=torch.bfloat16)
    tr=torch.zeros(B,5,2,C,C,device='cuda') if trace else None
    call('inverse',C,B,method,resident,L,output,bf,tr)
    return output,bf,tr

class Pipeline:
    """Diagnostic three-launch implementation, pre-normalized/pre-activated inputs."""
    def __init__(self,C,q,k,v,a,beta,state,stable=True,method=2):
        self.C=C;self.q=q;self.k=k;self.v=v;self.a=a;self.beta=beta;self.state=state
        self.H,self.T,_=q.shape;self.stable=int(stable);self.method=method
        self.kd=torch.empty_like(k);self.qd=torch.empty_like(q);self.kr=torch.empty_like(k)
        B=self.H*self.T//C
        self.gt=torch.empty(B,128,device='cuda');self.L=torch.empty(B,C,C,device='cuda',dtype=torch.float16)
        self.M=torch.empty_like(self.L,dtype=torch.bfloat16)
        self.inv=torch.empty_like(self.L);self.invbf=torch.empty_like(self.M)
        self.out=torch.empty_like(v);self.final=torch.empty_like(state)
    def prep(self):
        call('prepare',self.C,self.H,self.T,self.stable,self.q,self.k,self.a,self.beta,self.kd,self.qd,self.kr,self.gt,self.L,self.M)
    def inversion(self):
        call('inverse',self.C,self.H*self.T//self.C,self.method,1,self.L,self.inv,self.invbf,None)
    def k1(self):self.prep();self.inversion()
    def k2(self):
        call('recur',self.C,self.H,self.T,self.kd,self.qd,self.kr,self.gt,self.invbf,self.M,self.v,self.beta,self.state,self.final,self.out)
    def run(self):self.k1();self.k2()

def inputs(H,T,seed=0,pattern='random'):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    k=torch.randn(H,T,128,generator=gen,device='cuda')
    if pattern!='random':
        base=torch.randn(H,1,128,generator=gen,device='cuda')
        k=base.expand(-1,T,-1).clone()+(k*0.01 if pattern=='correlated' else 0)
        if pattern=='alternating':k[:,1::2]*=-1
    k=torch.nn.functional.normalize(k,dim=-1).bfloat16()
    q=torch.nn.functional.normalize(torch.randn(H,T,128,generator=gen,device='cuda'),dim=-1).bfloat16()
    v=torch.randn(H,T,128,generator=gen,device='cuda').bfloat16()
    state=(torch.randn(H,128,128,generator=gen,device='cuda')*.01).bfloat16()
    return q,k,v,state






# numerics

def range_test(C,folder):
    values=sorted(set([i/20 for i in range(101)]+[i/20/C for i in range(1600,1901) if i/20/C<=5]))
    rows=[]
    for pattern in ('constant',):
        a=torch.tensor(values,device='cuda',dtype=torch.float32)[:,None,None].expand(-1,C,128).contiguous()
        if pattern in ('front','back'):
            a[:,:C//2]=(a[:,:C//2]*2).clamp_max(5)
            a[:,C//2:]=(a[:,C//2:]*2-5).clamp_min(0)
            if pattern=='back':a=a.flip(1).contiguous()
        if pattern=='one_channel':a[:,:,1:]=.05
        k=torch.full_like(a,.125,dtype=torch.bfloat16);o=torch.empty(len(values),C,128,5,device='cuda')
        call('probe_range',C,len(values),a,k,o);torch.cuda.synchronize()
        cpu=o.cpu();ad=a.cpu()
        for idx,av in enumerate(values):
            x=cpu[idx];zeros=(x[:,:,0]==0).any(1).nonzero();inf=(~x[:,:,1].isfinite()).any(1).nonzero()
            rows.append(dict(chunk=C,pattern=pattern,a=av,total_max=ad[idx].sum(0).max().item(),
                first_exp_zero_token=int(zeros[0])+1 if len(zeros) else '',
                first_inv_nonfinite_token=int(inf[0])+1 if len(inf) else '',
                k_restored_nonfinite=int((~x[:,:,3].isfinite()).sum()),
                k_restored_rel_l2=rel(x[:,:,3],x[:,:,4]),
                k_restored_last_channel0=float(x[-1,0,3]),stable_last_channel0=float(x[-1,0,4])))
    save(rows,folder/'analysis/range.csv')
    print(f'RANGE: {len(rows)} cases. CUDA ex2.approx.ftz and BF16 arithmetic.',flush=True)

def inverse_test(C,folder):
    matrices=[];metadata=[]
    delta=(torch.arange(C,device='cuda')[:,None]-torch.arange(C,device='cuda')[None,:]).clamp_min(0)
    for seed in range(3):
        for pattern in ('random','repeated','alternating','correlated'):
            _,k,_,_=inputs(1,C,seed,pattern);gram=k[0].double()@k[0].double().T
            for a in (0.,.05,1.):
                for beta in (.1,.9,.999):
                    L=torch.tril(gram*torch.exp(-a*delta.double()),diagonal=-1).half()*torch.tensor(beta,device='cuda').half()
                    matrices.append(L);metadata.append(dict(seed=seed,pattern=pattern,a=a,beta=beta))
    L=torch.stack(matrices);A=torch.eye(C,device='cuda',dtype=torch.float64)+L.double()
    reference=torch.linalg.solve_triangular(A,torch.eye(C,device='cuda',dtype=torch.float64).expand_as(A),upper=False)
    cond=torch.linalg.cond(A).cpu().tolist()
    rows=[]
    candidates=[('mma_fp16acc_resident',lambda:inverse(L,0,1,True))]
    torch.save({'L':L.cpu(),'metadata':metadata},folder/'analysis/inverse_inputs.pt')
    for name,fn in candidates:
        inv,_,trace=fn();torch.cuda.synchronize()
        res=A@inv.double()-torch.eye(C,device='cuda',dtype=torch.float64)
        rhs=torch.ones(len(L),C,8,device='cuda',dtype=torch.float64)
        for i,meta in enumerate(metadata):
            st=[] if trace is None else trace[i,:int(math.log2(C))-1]
            bad=''
            for j,t in enumerate(st):
                if not bool(t.isfinite().all()):bad=2**(j+1);break
            rows.append(dict(chunk=C,method=name,**meta,finite=bool(inv[i].isfinite().all()),
                inverse_rel_l2=rel(inv[i],reference[i]),residual_fro=res[i].norm().item(),
                normalized_residual=(res[i].norm()/(A[i].norm()*inv[i].double().norm())).item(),
                rhs_rel_l2=rel(inv[i].double()@rhs[i],reference[i]@rhs[i]),condition_2=cond[i],
                intermediate_max=trace[i,:int(math.log2(C))-1].abs().max().item() if trace is not None else '',
                first_bad_stage_power=bad))
        print(f'INVERSE {name}: {sum(not bool(x.isfinite().all()) for x in inv)}/{len(L)} nonfinite',flush=True)
    save(rows,folder/'analysis/inverse.csv')





# trace_inverse

def trace_inverse(C):
    folder=ROOT/f'chunk{C}'
    data=torch.load(folder/'analysis/inverse_inputs.pt',weights_only=True)
    L=data['L'].cuda();records=[]
    for method,name in ((0,'fp16acc'),):
        inv,_,trace=inverse(L,method,1,True)
        trace=trace.cpu()
        for i,meta in enumerate(data['metadata']):
            for st in range(int(math.log2(C))-1):
                for part,label in enumerate(('power','inverse_partial')):
                    x=trace[i,st,part];finite=x[x.isfinite()]
                    records.append(dict(chunk=C,method=name,**meta,power=2**(st+1),part=label,
                        finite_max_abs=finite.abs().max().item() if finite.numel() else '',
                        inf_elements=int(x.isinf().sum()),nan_elements=int(x.isnan().sum())))
    save(records,folder/'analysis/inverse_stages.csv')
    print('B300 inverse stage traces saved',C,len(records),flush=True)



# performance




def profile(C,folder):
    # Exactly one launch of each case is profiled; data initialization is excluded.
    B=64*4096//C;manifest=[]
    def capture(name,fn):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(name)
        torch.cuda.cudart().cudaProfilerStart()
        fn();torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
        manifest.append(dict(index=len(manifest),name=name,chunk=C,H=64,T=4096))
        print('PROFILE_CASE',name,flush=True)
    decay=torch.full((B,C,128),.25,device='cuda');k=torch.full_like(decay,.125,dtype=torch.bfloat16);probe=torch.empty(B,C,128,5,device='cuda')
    capture('range',lambda:call('probe_range',C,B,decay,k,probe))
    _,key,_,_=inputs(1,C)
    L=(torch.tril(key[0].float()@key[0].float().T,-1)*.5).half().expand(B,-1,-1).contiguous()
    io=torch.empty_like(L);ib=torch.empty_like(L,dtype=torch.bfloat16)
    for name,method,resident in [('inverse_fp16_stream',0,0),('inverse_fp16_resident',0,1),('inverse_fp32acc',1,1),('inverse_triangular',2,1)]:
        capture(name,lambda:call('inverse',C,B,method,resident,L,io,ib,None))
    q,k,v,state=inputs(64,4096);a=torch.full(q.shape,.25,device='cuda');beta=torch.full((64,4096),.9,device='cuda')
    for name,stable,method in [('original',False,0),('stable',True,2)]:
        p=Pipeline(C,q,k,v,a,beta,state,stable,method)
        capture(f'{name}_prepare',p.prep);capture(f'{name}_inverse',p.inversion);capture(f'{name}_K2',p.k2)
    (folder/'analysis/profile_cases.json').write_text(json.dumps(manifest,indent=2)+'\n')



# controls





# gate_distributions


def extract(C,tag):
    sys.path.insert(0,"/opt/nvidia/nsight-compute/2025.3.1/extras/python")
    import ncu_report
    folder=ROOT/f'chunk{C}'
    rep=ncu_report.load_report(str(folder/f'reports/{tag}.ncu-rep'))
    cases=json.loads((folder/'analysis/profile_cases.json').read_text())
    actions=[]
    for r in range(rep.num_ranges()):
        rg=rep.range_by_idx(r)
        for j in range(rg.num_actions()):actions.append(rg.action_by_idx(j))
    if len(actions)!=len(cases):raise RuntimeError((C,tag,len(actions),len(cases)))
    summary=[]
    for i,action in enumerate(actions):
        allmetrics={};instances={}
        for name in action.metric_names():
            try:
                m=action[name];val=m.value()
                allmetrics[name]={'value':val,'unit':m.unit()}
                if name.startswith('pmsampling:') or name=='sm__cycles_active.sum':
                    values=[]
                    for t in range(m.num_instances()):
                        try:values.append(m.as_double(t))
                        except Exception:
                            try:values.append(m.as_uint64(t))
                            except Exception:pass
                    if values:instances[name]=values
            except Exception as ex:allmetrics[name]={'error':str(ex)}
        record=dict(index=i,case=cases[i]['name'],kernel=action.name(),metrics=allmetrics,instances=instances)
        try:
            metric=action['sass__inst_executed_per_opcode'];names=metric.correlation_ids()
            record['opcode_counts']={names.as_string(j):metric.as_uint64(j) for j in range(metric.num_instances())}
        except Exception as ex:record['opcode_error']=str(ex)
        try:record['rules']=action.rule_results_as_dicts()
        except Exception as ex:record['rules_error']=str(ex)
        if tag=='source':
            try:
                pcs=action['smsp__pcsamp_sample_count'].correlation_ids()
                record['sass']={str(pcs.as_uint64(j)):str(action.sass_by_pc(pcs.as_uint64(j))) for j in range(pcs.num_instances())}
            except Exception as ex:record['sass_error']=str(ex)
            hotspots=[]
            for name in action.metric_names():
                if 'pcsamp_warps_issue_stalled' not in name:continue
                try:
                    metric=action[name]
                    if not metric.has_correlation_ids():continue
                    corr=metric.correlation_ids();byline={}
                    for j in range(metric.num_instances()):
                        count=metric.as_uint64(j)
                        if not count:continue
                        info=action.source_info(corr.as_uint64(j))
                        key=f'{info.file_name()}:{info.line()}' if info else 'unmapped'
                        byline[key]=byline.get(key,0)+count
                    hotspots.extend(dict(metric=name,source=key,samples=count) for key,count in byline.items())
                except Exception:pass
            record['hotspots']=sorted(hotspots,key=lambda x:x['samples'],reverse=True)[:30]
        output=folder/f'analysis/{tag}_{i:02d}_{cases[i]["name"]}.json'
        output.write_text(json.dumps(record,indent=1,default=str)+'\n')
        keys=['gpu__time_duration.sum','launch__registers_per_thread','launch__shared_mem_per_block',
              'launch__shared_mem_per_block_dynamic','launch__shared_mem_per_block_static',
              'launch__grid_size','launch__block_size','launch__waves_per_multiprocessor',
              'launch__occupancy_limit_shared_mem','launch__occupancy_limit_registers','launch__occupancy_limit_blocks',
              'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
              'sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed',
              'dram__throughput.avg.pct_of_peak_sustained_elapsed','l1tex__t_sector_hit_rate.pct','lts__t_sector_hit_rate.pct',
              'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
              'smsp__sass_inst_executed_op_local_ld.sum','smsp__sass_inst_executed_op_local_st.sum']
        keys.extend(n for n in allmetrics if ('hmma' in n.lower() and ('inst' in n or 'sass' in n)) or
                    ('warps_issue_stalled' in n and n.endswith('pct')) or
                    ('mem_local' in n and n.endswith('.sum')))
        summary.append(dict(index=i,case=cases[i]['name'],kernel=action.name(),
            metrics={k:allmetrics.get(k) for k in keys},pm_series=len(instances),opcode_counts=record.get('opcode_counts',{})))
    (folder/f'analysis/{tag}_summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(f'CHUNK={C} {tag}: parsed {len(actions)} actions via ncu_report; raw metrics retained.',flush=True)

def pick(items,**keys):
    return next(x for x in items if all(str(x[k])==str(v) for k,v in keys.items()))

def fmt(x): return f'{float(x):.5g}'

def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in records])

def metrics(c,case):
    return pick(json.loads((ROOT/f'chunk{c}/analysis/full_summary.json').read_text()),case=case)

def val(a,k): return a['metrics'][k]['value']

def generate_report():
    bound=[];inv=[]
    for c in (16,32,64):
        limits=[]
        for key in ('first_exp_zero_token','first_inv_nonfinite_token'):
            hit=[x for x in rows(c,'range') if x['pattern']=='constant' and x[key]]
            limits.append(fmt(hit[0]['a']) if hit else 'a≤5 未出现')
        bound.append([c]+limits)
        iv=[x for x in rows(c,'inverse') if x['method']=='mma_fp16acc_resident']
        inv.append([c,f'{sum(x["finite"]=="False" for x in iv)}/{len(iv)}',
                    fmt(max(float(x['inverse_rel_l2']) for x in iv if x['finite']=='True'))])

    report='''# CHUNK=16/32/64：B300 实验结论汇总
    
    日期：2026-09-05。NVIDIA B300 SXM6 AC，CC 10.3，148 SM；CUDA 13.0，Nsight Compute 2025.3.1。
    
    **结论：**
    
    - **C32 已出现指数归零、溢出，C64 在更小衰减下就会出现。** 每个 token 使用相同衰减 a∈[0,5] 时，B300 扫描中 C32 首次负指数归零的 a≈2.72969，逆指数首次非有限的 a≈2.77344；C64 分别为 1.36484 和 1.38672。负指数归零会丢失原本非零的衰减贡献，逆指数溢出会使后续计算产生 Inf/NaN。C16 在本次 a≤5 扫描中没有出现这两种异常。
    - **C32/C64 的 FP16 Neumann 求逆都已算出 Inf/NaN。** 把整个 32×32 或 64×64 矩阵直接做 Neumann 展开，并用 FP16 保存、计算中间矩阵时，各有 36/108 个测试 case 得到非有限结果。例如所有 k 相同、a=0、beta=0.9 时，指数没有问题，但计算部分和 I−L+L²−…−L⁷ 时，中间数值超过 FP16 范围，C32 出现 15 个 Inf，C64 出现 703 个 Inf，无法正常完成求逆。对应理想输入的正确逆矩阵元素绝对值不超过 1，异常来自展开过程。
    - **C32/C64 都匹配 m16n8k16 MMA 形状，可以分块发射。** C32 沿 M 维拆成 2 个 M16 tile，C64 拆成 4 个；相关 N、K 维也能分别被 8、16 整除，无需 padding。因此，MMA 形状匹配不能作为 CHUNK 必须选 16 的理由。
    
    本报告只验证三件事：指数是否归零或溢出、FP16 Neumann 求逆是否产生数值异常，以及 MMA 形状能否分块发射。

    **实验范围：使用独立 CUDA microbench，在 B300 上复现所讨论的指数与求逆计算；并非修改后的生产 FlashKDA kernel。C16 是相同测试的尺寸对照。**

    ## 1. bf16 动态范围：C32 已出现指数归零和溢出，C64 的衰减阈值更低
    
    ### 1.1 纸面推导：累计衰减边界
    
    令 S_i=Σ(t=1…i) a_t，a_t∈[0,5]。以下用自然指数表示；代码中的 base-2 累加与之等价。原路径显式构造 exp(-S_i) 和 exp(+S_i)。
    
    - bf16 最大有限值为 (2−2⁻⁷)·2¹²⁷，故正指数的可表示边界为 S≈ln(bf16_max)=88.71893。
    - 本实现使用 `ex2.approx.ftz.f32`，不能依赖 subnormal 保留微小指数；以 FP32 最小正规数 2⁻¹²⁶ 为界，负指数的 FTZ 边界约为 S=126 ln2=87.33654。
    - 常数 a 下 S_C=Ca，所以临界平均衰减按 1/C 缩小。以上是范围边界：舍入、近似指数与浮点累加会影响实际首次归零/溢出的位置，不能当作逐位精确阈值。
    
    | CHUNK | 最坏 S_C=5C | exp(+5C)，实数值 | exp(-5C)，实数值 | FTZ 参考 a=87.33654/C | 正指数范围参考 a=88.71893/C |
    | --- | --- | --- | --- | --- | --- |
    | 16 | 80 | 5.5406e34 | 1.8049e-35 | 5.45853 | 5.54493 |
    | 32 | 160 | 3.0698e69 | 3.2575e-70 | 2.72927 | 2.77247 |
    | 64 | 320 | 9.4240e138 | 1.0611e-139 | 1.36463 | 1.38623 |
    
    **结论：C=16 的指数因子在这个 gate 上界下仍有范围余量；C=32/64 不再有保证。** 当 a=5，两个边界均在第 18 个 token 被越过，因此 32 和 64 都会触发，64 并不是必须等到块尾才失败。此保证仅针对指数因子，不保证后续乘法、求逆和整体精度。
    
    ### 1.2 B300 验证
    
    令 a=-g_act∈[0,5]。实际执行 `ex2.approx.ftz.f32`、bf16 转换及乘法，首次观测的常数 gate 阈值如下。
    累计衰减扫描步长为 0.05；这些是采样边界，不是无限精度阈值。
    
    '''+table(['CHUNK','exp(G) 首次归零的平均 a','exp(-G) 首次非有限的平均 a'],bound)+'''
    
    - C=32/64 首次归零的累计衰减约 87.35，逆指数首次非有限约 88.75。C=16 最坏累计衰减为 80。
    - 归零也是本项记录的数值异常，不能只检查 Inf/NaN。

    证据：[C16 范围](chunk16/RESULTS.md#range)、[C32 范围](chunk32/RESULTS.md#range)、[C64 范围](chunk64/RESULTS.md#range)。汇总已测正常/异常区间、最后正常点及首次异常点；原始测量记录折叠保留。

    ## 2. Neumann：FP16 中间部分和溢出
    
    ### 2.1 纸面推导：有限级数及展开阶数
    
    L 是 C×C 严格下三角矩阵，所以 Lᶜ=0，且
    
    `(I+L)⁻¹ = I−L+L²−…+(−L)^(C−1)`。
    
    这是有限多项式恒等式，不需要无限 Neumann 级数的范数小于 1 条件；但恒等式不保证浮点求值稳定。C 为 2 的幂时可写为
    
    `(I−L)(I+L²)(I+L⁴)…(I+L^(C/2))`。
    
    当前实现从 I−L 开始，逐步使用 L²、L⁴、L⁸ 等幂更新部分和。C16、C32、C64 的完整展开分别到 L¹⁵、L³¹、L⁶³；恒等式在实数运算中成立，但 FP16 中间部分和可能先超出数值范围。

    ### 2.2 纸面推导：最终逆很小，中间部分和仍可溢出
    
    选取可解析压力输入：所有 k 相同且单位归一化、a=0、beta=b=0.9，忽略输入量化。于是 L=bT，T 的严格下三角元素全为 1。对于距离 d=i−j≥p，选择 p−1 个中间索引可得
    
    `(Lᵖ)ᵢⱼ = bᵖ · binom(d−1,p−1)`。
    
    完整逆的非对角元素则由二项式恒等式给出
    
    `Σ(p=1…d) (−b)ᵖ binom(d−1,p−1) = −b(1−b)^(d−1)`。
    
    所以完整逆对角为 1，其余元素绝对值不超过 0.9；问题出在大中间量的抵消，而非真实逆必须很大。使用 L⁴ 完成更新后，部分和是 S₇=I−L+L²−…−L⁷。用上式进行精确有理数计算：
    
    | CHUNK | L⁴ 左下角 | S₇ 左下角 | S₇ 中绝对值超过 FP16 最大值 65,504 的元素数 |
    | --- | --- | --- | --- |
    | 16 | 238.8204 | −780.5397987 | 0 |
    | 32 | 2,663.766 | −222,079.5381015 | 15 |
    | 64 | 24,813.702 | −26,270,033.5038591 | 703 |
    
    具体地，距离 d=26 时 S₇≈−62,630.94，d=27 时约 −82,498.84；在本例 d=27…63 均超出范围。因此 C=32 有 5+4+3+2+1=15 个，C=64 有 37+36+…+1=703 个。L⁴ 自身仍在 FP16 范围内，而部分和已经超出。
    
    **结论：零衰减时指数完全安全，但 C32/64 的 FP16 Neumann 仍可先失败。** 上述 15/703 与下述阶段实测一致；实际 bf16 k、FP16 L 与逐步舍入会改变具体数值，理想解析例不是任意输入的误差预测，也不能推出全部 case 的失败率。
    
    ### 2.3 B300 验证
    
    本实验的 L 由 BF16 k 的 FP64 Gram 矩阵与 FP64 指数构造，再转 FP16 并乘 FP16 beta；没有调用生产 K1 来生成 L。原 K1 先舍入 BF16 衰减操作数再做 MMA，两者的舍入位置不同。因此本节异常计数与 inverse 相对误差只描述这组隔离求逆输入，不能直接当作生产 KDA 的失败率或输出误差。

    将有限 FP16 L 输入完整 FP16 Neumann 实现，对照同一输入的 B300 fp64 triangular solve。每种尺寸 108 个 case：3 seeds × 4 种 k（random/repeated/alternating/correlated）× a={0,0.05,1} × beta={0.1,0.9,0.999}。

    '''+table(['CHUNK','Inf/NaN case 数 / 总数','有限结果的最大 inverse 相对 L2'],inv)+'''

    - repeated k、a=0、beta=0.9 的例子中，在使用 L⁴ 更新部分和 I−L+L²−…−L⁷ 时，C32 出现 15 个 Inf，C64 出现 703 个 Inf。此时 L⁴ 自身仍有限。
    - C16 的 108 个 case 没有 Inf/NaN，但最大 inverse 相对 L2 约 23.79%。所以“没有非有限值”不等于“误差很小”。
    - 表中有限结果的误差只统计 finite=True 的 case；不能据此忽略 C32/C64 已失败的 36 个 case。

    证据：[C16 求逆](chunk16/RESULTS.md#inverse)、[C32 求逆](chunk32/RESULTS.md#inverse)、[C64 求逆](chunk64/RESULTS.md#inverse)、[C32 阶段异常](chunk32/RESULTS.md#inverse_stages)、[C64 阶段异常](chunk64/RESULTS.md#inverse_stages)。`inverse` 表按 k、a、beta 汇总完整 FP16 实现的异常条件；阶段表展示代表输入的异常前、首次异常及最终部分和。实际 L 输入保存在各 `analysis/inverse_inputs.pt`。fp64 求解仅用于核对正确性，不作为优化方案比较。

    ## 3. MMA：C32/C64 均能分块发射，形状匹配
    
    ### 3.1 纸面推导：分块即可，不要求 CHUNK 等于指令的 M 维
    
    对 m16n8k16 指令，一次处理 M=16、N=8、K=16 的 tile。完整矩阵可以由多条指令覆盖，CHUNK 不必等于 16。
    
    | CHUNK | 当 M=C 时的 M16 分块数 | 当 N=C 时的 N8 分块数 | 当 K=C 时的 K16 分块数 | padding |
    | --- | --- | --- | --- | --- |
    | 16 | 1 | 2 | 1 | 不需要 |
    | 32 | 2 | 4 | 2 | 不需要 |
    | 64 | 4 | 8 | 4 | 不需要 |
    
    本次四类 GEMM 的 (M,N,K) 为 (C,C,128)、(C,128,128)、(C,128,C)、(128,128,C)。固定维度 128 也能被 16、8、16 整除，所以三种 CHUNK 的四类形状全部匹配。
    
    **结论：C32 可以拆成两个 M16 tile，C64 可以拆成四个 M16 tile，均可发射 MMA；不存在因这些尺寸无法整齐分块而必须选 C16 的限制。**
    
    本项以指令 tile 的整除关系判断能否分块发射，不将形状验证混入数值异常数据表。

    ## 4. 文件和复现
    
    - `chunk16/`、`chunk32/`、`chunk64/`：基线及两种尺寸的关键证据。日志不保留；C16/C32/C64 的表格数据在各自 `RESULTS.md`，输入与环境在 `analysis/`，筛选后的 profiler 证据在 `reports/`。
    - 每个 `reports/` 保留 `full.ncu-rep`、`source.ncu-rep`；每份 11 个实际 launch（指数、求逆和完整诊断路径）；MMA 映射性能测试记录已移除。
    
    | CHUNK | 分析数据 | NCU full | NCU source |
    | --- | --- | --- | --- |
    | 16 | [RESULTS.md](chunk16/RESULTS.md) | [full](chunk16/reports/full.ncu-rep) | [source](chunk16/reports/source.ncu-rep) |
    | 32 | [RESULTS.md](chunk32/RESULTS.md) | [full](chunk32/reports/full.ncu-rep) | [source](chunk32/reports/source.ncu-rep) |
    | 64 | [RESULTS.md](chunk64/RESULTS.md) | [full](chunk64/reports/full.ncu-rep) | [source](chunk64/reports/source.ncu-rep) |
    
    - C16/C32/C64 各用一份 `RESULTS.md` 集中保存三类数值异常证据；`analysis/` 仅保留求逆实际输入、环境、profile case 映射及精简 NCU 指标汇总。三个 CHUNK 使用相同目录结构。
    - 环境参数保存在各 `analysis/environment.json`；[校验清单](SHA256SUMS.json) 用于核验，编译命令见 [run.sh](harness/run.sh)。
    - `harness/` 仅保留 `bench.py`（统一测试、NCU 解析、结果存储和报告入口）、`kernels.cu`（CUDA 实现）、`run.sh`（编译及 srun/NCU 调度）。
    - `.venv/bin/python profile/01_chunk_size_analysis/harness/bench.py run --chunk 32` 运行指数与 FP16 求逆异常验证；所有 GPU 测试必须在 B300 分配内执行。
    - [harness/run.sh](harness/run.sh) 可重跑实验；已有 `.ncu-rep` 不覆盖，需先归档。仅重新生成汇总可运行 `.venv/bin/python profile/01_chunk_size_analysis/harness/bench.py report`。
    
    当前实验目录为 `profile/01_chunk_size_analysis`。rep 由原始采集结果离线筛选导出，移除了全部 8 条 MMA 映射性能测试记录；保留记录的内嵌历史源码及实测指标未改写。独立 MMA 计时 CSV、resident 形状测试结果、资源对比和全部日志已删除。其余指标可由 [bench.py analyze](harness/bench.py) 从 rep 重新提取。
    
    实验均为合成输入，未测试真实模型任务质量、尾块或 varlen；任何精度筛选线都不是产品质量验收标准。
    '''
    # Function indentation must not turn Markdown prose into code blocks.
    report = ''.join(line[4:] if line.startswith('    ') else line for line in report.splitlines(keepends=True))
    (ROOT/'REPORT.md').write_text(report)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts and 'build' not in p.parts}
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Consolidated REPORT.md and checksums generated.')


def tidy():
    import shutil
    for c in (16, 32, 64):
        for name in ('full.ncu-rep', 'source.ncu-rep'):
            assert (ROOT / f'chunk{c}/reports' / name).is_file(), (c, name)
        assert (ROOT / f'chunk{c}/analysis/full_summary.json').is_file()
    provenance = ROOT / 'logs/binary-provenance.json'
    if provenance.exists():
        dest = ROOT / 'metadata/binary-provenance.json'
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == provenance.read_bytes()
            provenance.unlink()
        else:
            provenance.rename(dest)
    for directory in sorted(ROOT.rglob('logs'), reverse=True):
        shutil.rmtree(directory)
    for f in list(ROOT.rglob('*.log')):
        f.unlink()
    for c in (16, 32, 64):
        p = ROOT / f'chunk{c}'
        candidates = list((p / 'analysis').glob('full_[0-9][0-9]_*.json'))
        candidates += list((p / 'analysis').glob('source_[0-9][0-9]_*.json'))
        candidates += [p / 'analysis/source_summary.json', p / 'REPORT.md']
        for f in candidates:
            if f.is_file():
                f.unlink()
    for name in ('PLAN.md', 'README.md'):
        (ROOT / name).unlink(missing_ok=True)
    for directory in list(ROOT.rglob('__pycache__')):
        shutil.rmtree(directory)
    build = ROOT / 'harness/build'
    if build.exists():
        shutil.rmtree(build)
    print('Kept report evidence and reproduction code; removed logs and redundant artifacts.')

def run_tests(c):
    folder = ROOT / f'chunk{c}'
    (folder / 'analysis').mkdir(parents=True, exist_ok=True)
    prop = torch.cuda.get_device_properties(0)
    meta = dict(GPU=prop.name, CC=f'{prop.major}.{prop.minor}', SMs=prop.multi_processor_count,
                torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                TF32=False, device=str(prop), seed_policy='GPU generator 0,1,2; MMA torch.manual_seed(123)')
    (folder / 'analysis/environment.json').write_text(json.dumps(meta, indent=2) + '\n')
    range_test(c, folder)
    inverse_test(c, folder)
    trace_inverse(c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'profile', 'analyze', 'report', 'pack', 'tidy'])
    parser.add_argument('--chunk', type=int, choices=[16, 32, 64])
    parser.add_argument('--tag', choices=['full', 'source'], default='full')
    args = parser.parse_args()
    if args.command in ('run', 'profile', 'analyze') and args.chunk is None:
        parser.error('--chunk is required for this command')
    if args.command in ('run', 'profile'):
        folder = ROOT / f'chunk{args.chunk}'
        if args.command == 'run' and (list((folder / 'reports').glob('*.ncu-rep')) or (folder / 'RESULTS.md').exists() or (folder / 'analysis/inverse.csv').exists()):
            parser.error('Archive existing chunk results before reproducing; measurement evidence must not be overwritten')
        initialize_gpu()
    if args.command == 'run':
        run_tests(args.chunk)
    elif args.command == 'profile':
        profile(args.chunk, ROOT / f'chunk{args.chunk}')
    elif args.command == 'analyze':
        extract(args.chunk, args.tag)
    elif args.command == 'report':
        generate_report()
    elif args.command == 'pack':
        for c in ([args.chunk] if args.chunk else [16, 32, 64]):
            pack_results(c)
        generate_report()
    elif args.command == 'tidy':
        tidy()


if __name__ == '__main__':
    main()

