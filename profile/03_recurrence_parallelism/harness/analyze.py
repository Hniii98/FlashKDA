"""Extract NCU evidence, retain compact Markdown and raw JSON in one file."""
from pathlib import Path
import sys,json,hashlib,collections
sys.path.insert(0,'/opt/nvidia/nsight-compute/2025.3.1/extras/python')
import ncu_report
ROOT=Path(__file__).resolve().parents[1]
PERSISTENT='--persistent' in sys.argv
HEADS='--heads' in sys.argv
R=ROOT/('persistent' if PERSISTENT else 'multi_head' if HEADS else 'column_split')
PREFIX=''
cases=json.loads((R/'reports'/(PREFIX+'cases.json')).read_text())
def val(m,i):
 for f in ('as_double','as_uint64'):
  try:return getattr(m,f)(i)
  except Exception:pass
 return None
result={}
for tag in ['full','source']:
 rep=ncu_report.load_report(str(R/'reports'/f'{PREFIX}{tag}.ncu-rep'));acts=[rep.range_by_idx(r).action_by_idx(i) for r in range(rep.num_ranges()) for i in range(rep.range_by_idx(r).num_actions())];assert len(acts)==len(cases)
 rows=[]
 for case,a in zip(cases,acts):
  metrics={};pm={};hot=collections.defaultdict(lambda:collections.defaultdict(float))
  for name in a.metric_names():
   m=a[name]
   if name.startswith('pmsampling:'):
    values=[val(m,i) for i in range(m.num_instances())];values=[v for v in values if v is not None]
    if values:pm[name]=dict(instances=len(values),bins=[sum(values[j*len(values)//10:(j+1)*len(values)//10])/max(1,len(values[j*len(values)//10:(j+1)*len(values)//10])) for j in range(10)])
   elif any(x in name for x in ['launch__','sm__warps_active.avg.pct','sm__cycles_active.avg.pct','sm__throughput.avg.pct','dram__throughput.avg.pct','dram__bytes_read.sum','dram__bytes_write.sum','pipe_tensor','smsp__warp_issue_stalled','smsp__warps_issue_stalled','l1tex__data_bank_conflicts']):
    try:
     v=m.value()
     if isinstance(v,(int,float,str)):metrics[name]=v
    except Exception:pass
   if tag=='source' and name.startswith('smsp__pcsamp_warps_issue_stalled_') and not name.endswith(('_not_issued','_selected')):
    try:
     if not m.has_correlation_ids():continue
     ids=m.correlation_ids()
     for i in range(m.num_instances()):
      v=val(m,i)
      if not v:continue
      pc=ids.as_uint64(i);si=a.source_info(pc);key=(str(si.file_name()),int(si.line())) if si else ('unknown',0)
      hot[key][name.split('stalled_')[-1]]+=v
    except Exception:pass
  hotspots=[dict(file=f,line=l,counts=dict(v)) for (f,l),v in sorted(hot.items(),key=lambda kv:-sum(kv[1].values()))[:8]]
  rows.append(dict(**case,metrics=metrics,pm_deciles=pm,hotspots=hotspots))
 result[tag]=rows
lines=['# Topic3 NCU 证据','','NCU replay 时间不用于加速比。PM 分箱按采样序号等分，仅用于查看趋势。','','| Case | Mode | CTA 数 | 寄存器/线程 | Shared/CTA B | achieved occupancy % | SM throughput % | DRAM throughput % |','|---|---|---|---|---|---|---|']
for x in result['full']:
 m=x['metrics'];get=lambda n:m.get(n,'未提供')
 lines.append('| '+' | '.join(map(str,[x['case'],x['mode'],get('launch__grid_size'),get('launch__registers_per_thread'),get('launch__shared_mem_per_block'),get('sm__warps_active.avg.pct_of_peak_sustained_active'),get('sm__throughput.avg.pct_of_peak_sustained_elapsed'),m.get('dram__throughput.avg.pct_of_peak_sustained_elapsed',m.get('FBSP.TriageCompute.dram__throughput.avg.pct_of_peak_sustained_elapsed','未提供'))]))+' |')
lines+=['','<details><summary>指标、PM 趋势与源码热点</summary>','','```json',json.dumps(result,ensure_ascii=False),'```','','</details>']
(R/(PREFIX+'PROFILE.md')).write_text('\n'.join(lines)+'\n')
print('Parsed',len(cases),'cases in each report')
