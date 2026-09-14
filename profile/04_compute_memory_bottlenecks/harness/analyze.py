"""Parse the retained reports with NVIDIA ncu_report; no CSV intermediates."""
from pathlib import Path
import sys,json,collections
sys.path.insert(0,'/opt/nvidia/nsight-compute/2025.3.1/extras/python')
import ncu_report
R=Path(__file__).resolve().parents[1]
cases=json.loads((R/'analysis/cases.json').read_text())
def value(m,i):
 for f in ('as_double','as_uint64'):
  try:return getattr(m,f)(i)
  except Exception:pass
 return None
def relevant(n):
 if n.startswith(('launch__','dram__','gpu__time_duration','smsp__pcsamp_','smsp__average_warps_issue_stalled_')):return True
 if n.startswith(('sm__','smsp__','l1tex__','lts__')) and ('.avg.' in n or n.endswith('hit_rate.pct')):return True
 if n.startswith('sm__cycles_active.'):return True
 if n.startswith(('l1tex__','lts__','smsp__sass_')) and n.endswith(('.sum','.ratio')):return True
 return False
result={}
for tag in ['full','source']:
 rep=ncu_report.load_report(str(R/'reports'/f'{tag}.ncu-rep'))
 acts=[rep.range_by_idx(r).action_by_idx(i) for r in range(rep.num_ranges()) for i in range(rep.range_by_idx(r).num_actions())]
 assert len(acts)==len(cases),(tag,len(acts))
 rows=[]
 for case,a in zip(cases,acts):
  assert ('prepare' if case['kernel']=='K1' else 'recurrence') in a.name()
  metrics={};pm={};pcs=[];hot=collections.defaultdict(lambda:collections.defaultdict(float))
  for name in a.metric_names():
   m=a[name]
   try:
    v=m.value()
    if isinstance(v,(int,float,str)) and relevant(name):metrics[name]=v
   except Exception:pass
   if name.startswith('pmsampling:'):
    vs=[value(m,i) for i in range(m.num_instances())];vs=[v for v in vs if v is not None]
    if vs:pm[name]=dict(instances=len(vs),bins=[sum(vs[j*len(vs)//10:(j+1)*len(vs)//10])/max(1,len(vs[j*len(vs)//10:(j+1)*len(vs)//10])) for j in range(10)])
   if tag=='source' and name.startswith('smsp__pcsamp_warps_issue_stalled_') and not name.endswith(('_not_issued','_selected')):
    try:
     if not m.has_correlation_ids():continue
     ids=m.correlation_ids()
     for i in range(m.num_instances()):
      v=value(m,i)
      if not v:continue
      si=a.source_info(ids.as_uint64(i));key=(str(si.file_name()),int(si.line())) if si else ('unknown',0)
      hot[key][name.split('stalled_')[-1]]+=v
      pcs.append(dict(file=key[0],line=key[1],reason=name.split('stalled_')[-1],samples=v,sass=str(a.sass_by_pc(ids.as_uint64(i)))))
    except Exception:pass
  if 'launch__grid_size' in metrics:assert int(metrics['launch__grid_size'])==case['expected_ctas']
  hotspots=[dict(file=f,line=l,counts=dict(v)) for (f,l),v in sorted(hot.items(),key=lambda kv:-sum(kv[1].values()))[:12]]
  rows.append(dict(**case,name=a.name(),metrics=metrics,pm_deciles=pm,hotspots=hotspots,top_pcs=sorted(pcs,key=lambda p:-p["samples"])[:12]))
 result[tag]=rows
(R/'analysis/metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# Topic4 NCU 实测证据','','full 和 source 各 2 个 action，均为对应场景 warmup 20 次后的第 21 次 forward。原始指标和源码采样保存在折叠区；PM 按采样序号等分为 10 段，不等同于 SM 利用率时间线。','','| 场景 | Kernel | CTA | NCU µs | SM % | Tensor % | L2 hit % | DRAM read GB/s | DRAM write GB/s |','|---|---|---|---|---|---|---|---|---|']
for x in result['full']:
 m=x['metrics']
 def f(n,s=1):
  v=m.get(n);return f'{v*s:.3f}' if isinstance(v,(int,float)) else 'NA'
 lines.append('| '+' | '.join([x['case'],x['kernel'],str(x['expected_ctas']),f('gpu__time_duration.sum',.001),f('sm__throughput.avg.pct_of_peak_sustained_elapsed'),f('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),f('lts__t_sector_hit_rate.pct'),f('dram__bytes_read.sum.per_second',1e-9),f('dram__bytes_write.sum.per_second',1e-9)])+' |')
lines+=['','## 同步与依赖证据','','source 采样比例以该 action 全部 stall 类别（含 selected/not_selected，排除重复的 not_issued）之和为分母，不是运行时间占比。','','| 场景 | Kernel | 样本数 | barrier % | short scoreboard % | wait % | long scoreboard % | sleeping % |','|---|---|---|---|---|---|---|---|']
for x in result['source']:
 d={n.split('stalled_')[-1]:v for n,v in x['metrics'].items() if n.startswith('smsp__pcsamp_warps_issue_stalled_') and not n.endswith('_not_issued')};total=sum(d.values())
 lines.append('| '+' | '.join([x['case'],x['kernel'],str(total)]+[f'{100*d.get(k,0)/total:.2f}' if total else 'NA' for k in ['barrier','short_scoreboard','wait','long_scoreboard','sleeping']])+' |')
lines+=['','完整 full 指标仍在 rep 中；这里仅提取瓶颈判定相关指标。PM 原始采样含采集窗口两端的零值，不把它当成 kernel 空闲时间，也不据此计算尾部耗时。','','<details><summary>相关指标、PM 采样分箱与源码/SASS 热点</summary>','','```json',json.dumps(result,ensure_ascii=False),'```','','</details>']
(R/'analysis/RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:15]))
