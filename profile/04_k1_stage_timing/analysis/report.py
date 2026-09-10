import json
from pathlib import Path
r=Path(__file__).resolve().parents[1];a=r/'analysis'
s=json.loads((a/'stages.json').read_text());t=json.loads((a/'runtime_ablation.json').read_text())
v=s['variants'];names={'input_load':'输入加载与TMA等待','qk_normalize':'Q/K归一化','gate_cumsum_gt':'gate激活、cumsum、g_total及尾部清零','decay':'decay变换','L_Mqk':'L/Mqk矩阵乘','tril_beta_init':'三角处理、beta及INV初始化','neumann':'Neumann展开','workspace_store':'workspace写回及等待'}
out='''# K1阶段计时：原C16 vs C32 launch bounds=4

结论：C32每CTA平均被测延迟比C16多约2.902 μs，Neumann增加0.872 μs，gate/cumsum增加0.805 μs，decay增加0.469 μs。Neumann和gate/cumsum合计约占净增量的57.8%。不能把K1剩余差距只归因于Neumann或spill。

## 插桩方法与控制

在八段现有同步边界记录9个 `%globaltimer` 时间戳，单位ns。每32个沿时间维分布的CTA采一个、仅thread0记录，写入独立device数组；不增加 `__syncthreads()`，不拆kernel、不删除阶段。1536个C16 CTA或768个C32 CTA/launch，重复10次，保留原始样本 `analysis/C16T_stage_samples.pt`、`C32T_stage_samples.pt`。

记录的是每个CTA的阶段经过时间，包含执行、调度和同步等待；不是各阶段单独占用整张GPU的时间。选中CTA自身有时间戳与store开销，不能将样本平均直接乘CTA数作为kernel wall time，也不能把1.8%的整体扰动当成各阶段误差上界。全程时间戳差非负，统计包括均值、中位数、P95，见stages.json。

阶段0从有效tile判定后开始，不含kernel入口至tile定位的前导代码；最后一段包含TMA store wait和原有同步。gate段还包含尾部k清零、g_total指数及少量视图设置；Neumann段包含其后的fence和同步。没有对每轮Neumann或gate内部进一步拆分。

来源：C16计时版基于只读原baseline，C32计时版基于本工作区原helper并使用已测过的launch bounds=4；未使用B1失败候选。独立overlay仅对插桩相关的K1、binding、读取时间戳接口生成诊断源码，其余文件链接原来源。新增修改见analysis/*.patch。未修改或重建未插桩baseline，C16工作区仍clean；该轮插桩实验未修改生产源码；后续已将主源码及默认构建切换为launch bounds=4，计时数据仍属于原实验。

## 插桩扰动与同轮计时

B300，T=8192、H=96、D=128，单条等长序列，FP32 state接口，seed0。H是单卡实际头数；非TP8每卡12头。C32采用rescale=2^-96、inverse_rescale=1，lower_bound=-5；C16沿用原计算。每个实现预热30次、5轮×200次完整forward CUDA event，正反顺序交替；K1/K2各另采样20次。基于既有benchmark文件，输入及精度设置保持不变。

| 实现 | 完整forward/ms | K1/ms | K2/ms |
|---|---:|---:|---:|
'''
for k,x in t['rows'].items():out+=f"| {x['variant']} | {x['full']['mean_ms']:.6f} | {x['kernels']['K1']['mean_ms']:.6f} | {x['kernels']['K2']['mean_ms']:.6f} |\n"
out+='''
T后缀为插桩版。C16 K1从0.273331→0.278416 ms（+1.86%），C32从0.394319→0.401359 ms（+1.79%）。插桩没有造成整体数量级变化，适合判断主要阶段；不据此宣称得到了无扰动的精确阶段耗时。

编译记录：C16仍无spill；C32插桩版静态spill stores/loads为8/8 bytes，而未插桩LB4为16/16 bytes。因此插桩改变了少量编译结果，进一步说明阶段值是诊断值，最终优化仍必须用未插桩版计时。

## 每CTA阶段延迟

单位μs，均值；C16和C32每个CTA分别处理16和32个token，不能把单CTA比值直接解释成固定工作量计算量比值。

| 阶段 | C16 | C32 LB4 | C32−C16 |
|---|---:|---:|---:|
'''
for k,label in names.items():
 x=v['C16T']['phases'][k]['mean']/1000;y=v['C32T']['phases'][k]['mean']/1000;out+=f'| {label} | {x:.3f} | {y:.3f} | {y-x:+.3f} |\n'
out+=f"| 被测阶段合计 | {v['C16T']['mean_total']/1000:.3f} | {v['C32T']['mean_total']/1000:.3f} | {(v['C32T']['mean_total']-v['C16T']['mean_total'])/1000:+.3f} |\n"
out+='''
C32的绝对阶段占比中，gate/cumsum约23.4%，Neumann约14.2%，decay约9.5%；Neumann是最大的单项净增量，却不是C32耗时最大的阶段。写回增加约0.424 μs；输入加载的样本延迟反而减少0.215 μs，体现经过时间还受并发及调度影响。

结合前轮NCU资源证据，C16为49152 CTAs、约8blocks/SM，C32为24576 CTAs、约4blocks/SM，148SM上二者理论waves均约41.5。C32虽CTA数减半，驻留能力也减半，因此每CTA总延迟增长与整kernel增长并不矛盾。这个waves估算仅解释趋势，不用于将表中阶段精确分摊为kernel的毫秒成本。

## 数值验证

两种计时版分别运行现有tests/compare_naive.py全部12用例，参考未改，输出与最终state均无Inf/NaN。C32计时版12用例所有误差统计与原C32记录相同。主性能形状上，两个计时版与各自未插桩版的output、最终state均逐元素bitwise相等（stages.json/equality）。

各版误差如下，FP32/BF16 state接口统计相同合并展示；未设新门槛。

| 计时版 | H | seq_lens | output最大绝对误差 | output相对L2 | state最大绝对误差 | state相对L2 |
|---|---|---|---:|---:|---:|---:|
'''
for name in ['C16','C32']:
 for row in json.loads((a/f'naive_{name}.json').read_text())['rows']:
  if row['state_dtype']!='torch.float32':continue
  o=row['output'];f=row['final_state'];out+=f"| {name} | {row['H']} | {row['seq_lens']} | {o['max_abs']:.9g} | {o['relative_l2']:.9g} | {f['max_abs']:.9g} | {f['relative_l2']:.9g} |\n"
out+='''
原始参考哈希、二进制哈希、命令见naive_C16.json和naive_C32.json。C16 loader仅补充参考脚本所需的默认scale元数据，调用C16时不传scale关键字；保存metadata明确chunk=16。

## 下一步

优先检查gate/cumsum的串行依赖和Neumann单warp执行，两者分别是最大的绝对计算阶段和最大的单项净增量。decay随后排查。优化后应重新naive对拍，并使用未插桩版完整forward/K1/K2确认收益。当前没有更改gate算法、MMA、精度或并行组织。

复现：harness/build.py、harness/collect.sh。重复采集使用新run目录；本轮保留全部原始样本、误差和计时证据。初次overlay准备缺少csrc父目录而失败，补齐目录创建后重新构建；该失败发生在编译和GPU执行前。
'''
(r/'REPORT.md').write_text('# 实验四：拆解K1剩余阶段耗时\n\n## 本目录做什么\n\n回答采用launch bounds=4后，C32的K1仍慢于C16的时间主要花在哪些阶段。\n\n**实验方法：** 独立插桩C16和C32 LB4，在原同步边界采集八阶段时间戳，同时测插桩扰动、进行naive对拍及输出一致性检查。\n\n**结论及下一步：** 每CTA延迟净增加约2.90 μs，Neumann和gate/cumsum合计约占58%；插桩对K1整体耗时影响约1.8%。这些是包含等待的CTA诊断延迟，不是各阶段独占整张GPU的耗时。\n\n脚本用途及依赖见[README](../README.md)。以下保留本轮详细记录，历史数据不改写为新测量。\n\n'+out)
