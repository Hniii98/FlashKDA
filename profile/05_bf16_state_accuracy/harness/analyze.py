"""Summarize the controlled A/B/C state-storage experiment."""
import json, collections, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
data=json.loads((R/'analysis/DATA.md').read_text().split('```json\n')[1].split('\n```')[0]);rows=data['rows']
assert data['metadata']['design']=='controlled_abc_v2'
assert len(rows)==65 and len({(r['kind'],r['seed'],r['param']) for r in rows})==65
assert all(r[k]['max_abs']==0 for r in rows for k in ['alignment_output','alignment_state'])
assert all(r[k]['finite'] for r in rows for k in ['output','output_tail','state'])
groups=collections.defaultdict(list)
for r in rows:groups[r['kind'],r['param']].append(r)
labels={'random':'A 随机基线','beta':'B beta logit 扫描','gate':'C chunk gate 扫描'}
def span(values):return f'{min(values):.4f}–{max(values):.4f}'
lines=['# BF16 状态保存：A/B/C 实测结果','','65 组输入，5 个 seed；N=1、T=8192、H=96、D=128、CHUNK=16。全部 BF16 microbench 与真实 kernel 的输出、最终状态逐元素一致，所有比较结果有限。','','下表为 seed 范围。差异是 BF16 保存相对 FP32 主状态保存的差异，不是数学真值误差或模型任务准确率。状态统一格式指 FP32 最终状态仅在比较时转 BF16。','','| 组 | 参数 | 输出相对 L2 % | 末尾 1024 token 相对 L2 % | 最终状态相对 L2 %（统一格式） | 最大绝对输出差异 | 实际 gate 均值 | 实际 beta 均值 |','|---|---|---|---|---|---|---|---|']
for (kind,p),rs in groups.items():
 vals=[span([100*r[k]['relative_l2'] for r in rs]) for k in ['output','output_tail','state_same_storage']]
 lines.append('| '+' | '.join([labels[kind],str(p)]+vals+[f"{max(r['output']['max_abs'] for r in rs):.7g}",f"{statistics.mean(r['retention_mean'] for r in rs):.9g}",f"{statistics.mean(r['beta_mean'] for r in rs):.9g}"])+' |')
lines+=['','非零更新丢失比例仅保留在原始诊断数据，不作为准确性判据。没有预设任意的通过阈值。','','[原始数据](analysis/DATA.md) · [实验设计](REPORT.md)','','![输出差异](analysis/figures/output.png)','','![状态差异](analysis/figures/state.png)']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
(R/'analysis/figures').mkdir(exist_ok=True)
for metric,name in [('output','output'),('state_same_storage','state')]:
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for ax,kind in zip(axes,['random','beta','gate']):
  items=[(p,rs) for (k,p),rs in groups.items() if k==kind]
  for i,(p,rs) in enumerate(items):
   vs=[100*r[metric]['relative_l2'] for r in rs];m=statistics.median(vs)
   ax.errorbar(i,m,yerr=[[m-min(vs)],[max(vs)-m]],fmt='o',capsize=4)
  ax.set_xticks(range(len(items)),[str(p) for p,_ in items],rotation=30)
  ax.set_title({'random':'A: random baseline','beta':'B: beta logits (gate=0.9999)','gate':'C: chunk retention (random beta)'}[kind]);ax.set_ylabel('Relative L2 difference (%)');ax.grid(alpha=.2)
 fig.savefig(R/f'analysis/figures/{name}.png',dpi=180);plt.close(fig)
print('\n'.join(lines))
