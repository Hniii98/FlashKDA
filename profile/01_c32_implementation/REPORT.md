# 实验一：C32实现与数值验证

## 目录

把原C16的K1/K2适配到CHUNK=32，在保持算法、MMA指令和阶段精度的前提下，验证新增数据覆盖、Neumann展开和递推流程；通过rescale扫描解决指数范围问题，再与原C16比较基础性能。

## 实验内容

1. K1归一化覆盖32行、mask/初始化覆盖1024个元素，Neumann扩展到31次幂；K2适配两个time tile及CHUNK=32归约，同步修改workspace、TMA和beta缓冲。
2. 引入rescale及配套抵消，初始值1；用未修改的naive参考记录output和最终state误差及Inf/NaN。
3. 扫描rescale=2^e，选定默认2^-96、inverse_rescale=1，复查全部12个用例。
4. 同设备同轮测量完整forward和K1/K2，建立后续性能诊断的基线。

## 结论

初始rescale=1、lower_bound=-5时12个用例均出现NaN。现有用例上e=-105…-92均得到有限结果，最终采用2^-96。默认配置output相对L2约0.00343433–0.00400208，最终state约0.00456350–0.00462786，Inf/NaN均为0；未新增误差门槛。

初版LB8的18个配对性能配置均慢于C16，因此继续开展实验二。当前主源码及默认二进制已在实验三结论基础上采用LB4；
完整方法、绝对误差表和历史数据见总报告的[实现](../REPORT.md#c32-implementation)、[参数扫描](../REPORT.md#c32-sweep)、[默认对拍](../REPORT.md#c32-recheck)、[基础benchmark](../REPORT.md#c32-benchmark)章节。

## 复现

- `build.py`：构建当前默认C32。
- `run_diagnostics.sh`：在lower_bound=-1下做未缩放及非默认scale诊断。 
(ref dir 请对应修改为assignment02/team/c1_flashkda/fla_kda_ref)
- `run_sweep.sh`：复现细粒度rescale扫描。
- 仓库 `tests/compare_naive.py` 和 `benchmarks/bench_c32_vs_c16.py`：完整对拍及基础性能。

