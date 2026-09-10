import json,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'/opt/nvidia/nsight-compute/2025.3.1/extras/python')
import ncu_report
RUN=Path(__file__).resolve().parents[1]
KEYS=['gpu__time_duration.sum','launch__grid_size','launch__block_size','launch__registers_per_thread',
'launch__shared_mem_per_block','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem',
'launch__waves_per_multiprocessor','sm__warps_active.avg.pct_of_peak_sustained_active',
'sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed',
'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','l1tex__t_sector_hit_rate.pct','lts__t_sector_hit_rate.pct',
'smsp__sass_inst_executed_op_local_ld.sum','smsp__sass_inst_executed_op_local_st.sum',
'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
for path in sorted((RUN/'reports').glob('*.ncu-rep')):
 rep=ncu_report.load_report(str(path));results=[]
 for ri in range(rep.num_ranges()):
  rng=rep.range_by_idx(ri)
  for ai in range(rng.num_actions()):
   a=rng.action_by_idx(ai);allm={};pm={};hot=defaultdict(lambda:defaultdict(float))
   for name in a.metric_names():
    m=a[name]
    try:allm[name]=m.value()
    except Exception:continue
    if name.startswith('pmsampling:'):
     try:
      vals=[m.as_double(i) for i in range(m.num_instances())]
      pm[name]=vals
     except Exception:pass
    if name.startswith('smsp__pcsamp_warps_issue_stalled_') and m.has_correlation_ids():
     cor=m.correlation_ids()
     for i in range(m.num_instances()):
      try:
       v=m.as_uint64(i);pc=cor.as_uint64(i);src=a.source_info(pc)
       key=f'{src.file_name()}:{src.line()}' if src else '?'
       hot[key][name.removeprefix('smsp__pcsamp_warps_issue_stalled_')]+=v
      except Exception:pass
   keym={k:allm.get(k) for k in KEYS}
   keym.update({k:v for k,v in allm.items() if ('warps_issue_stalled_' in k and 'per_issue_active.ratio' in k)})
   prefix=f'{path.stem}_{ai}'
   (RUN/'analysis'/f'{prefix}_all.json').write_text(json.dumps(allm,indent=2,default=str)+'\n')
   (RUN/'analysis'/f'{prefix}_pm.json').write_text(json.dumps(pm,default=str)+'\n')
   hotspots=sorted([dict(source=k,total=sum(x for k,x in v.items() if not k.endswith("_not_issued") and k != "selected"),stalls=dict(v)) for k,v in hot.items()],key=lambda x:-x['total'])
   (RUN/'analysis'/f'{prefix}_hotspots.json').write_text(json.dumps(hotspots,indent=2)+'\n')
   results.append(dict(name=a.name(),metrics=keym,top_sources=hotspots[:12]))
  (RUN/'analysis'/f'{path.stem}_key.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
 print(path.name,'actions',len(results))
