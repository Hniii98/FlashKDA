# 实验二：定位C32为什么变慢

## 目录

回答初版C32相对C16的新增时间落在哪里，以及是否由rescale引起。

**实验方法：** 使用同一二进制改变rescale和lower_bound，测完整forward与K1/K2，再结合NCU的spill、occupancy及源码等待热点定位。

**结论及下一步：** 初版增量主要在K1，存在大量spill及同步等待；运行时scale对照不能把退化归因于rescale算术。因此实验三只改变launch bounds，检验寄存器预算的影响。

脚本用途及依赖见[README](../README.md)。以下保留本轮详细记录，历史数据不改写为新测量。

# C32 时间成本定位

本轮实测：新增 kernel 时间约 94.5% 在 K1。NCU 确认 K1 大量寄存器溢出访存与同步等待；运行时启用默认 rescale 没有造成观察到的整体变慢。尚未通过编译消融分别测出 rescale 代码、Neumann 算术、spill 各自的独立毫秒成本。

## 实验方法

- 复用 C16、C32 既有二进制，不重建 baseline，不修改生产 K1/K2。
- NVIDIA B300 SXM6 AC；同进程、同设备、同轮交替顺序。T=8192、H=96、D=128，单条等长序列，FP32 state 接口；沿用既有 benchmark 输入生成及初始 state，seed=0。这里 H=96 是本次单卡实际输入头数，不是 TP8 每卡12头的测试。
- 每条件预热30次，5轮×200次 CUDA event 完整 forward；另用 PyTorch profiler 采样20次 K1/K2。完整 forward 包含 wrapper 的 workspace 分配及内部处理；分解采样与 event 测量独立，不要求严格相加。
- 同一个 C32 二进制测试 rescale=1 与 2^-96，inverse_rescale 均为1；gate lower_bound=-5 为主条件，-1 为两种 scale 都有限的诊断对照。
- C16/C32 各采集 full+PmSampling+PmSampling_WarpStates、source+SourceCounters 两份 NCU 报告；通过 ncu_report 提取指标和源码采样。NCU replay 时间不作为 benchmark 结果。

## 测量结果

单位 ms，均值：

| lower_bound | 实现 | rescale | 完整 forward | K1 | K2 | output/state 均有限 |
|---|---|---|---:|---:|---:|---|
| -5 | C16 | 原始 | 1.734569 | 0.459480 | 1.262088 | 是 |
| -5 | C32 | 1 | 3.714262 | 2.353762 | 1.341333 | 否 |
| -5 | C32 | 2^-96 | 3.466021 | 2.096555 | 1.357471 | 是 |
| -1 | C16 | 原始 | 1.734981 | 0.460236 | 1.261454 | 是 |
| -1 | C32 | 1 | 3.714044 | 2.349777 | 1.348313 | 是 |
| -1 | C32 | 2^-96 | 3.465262 | 2.098717 | 1.352073 | 是 |

主条件 C32 默认相对 C16：forward +1.731452 ms；K1 +1.637075 ms；K2 +0.095382 ms。K1 占 K1+K2 增量的94.5%。

在 -1 有限对照下，默认 rescale 相对1，forward 减少0.248781 ms，K1减少0.251059 ms，K2增加0.003760 ms。说明当前运行时 rescale 分支的净效应没有解释 C32 的大幅退化。rescale=1 仍保留参数、分支及其编译期寄存器影响，不等价于从源码删除 rescale；分支差异为何反而加速仍需 SASS/编译消融确认，不能推断 rescale 本身是优化。

本轮只增加性能及有限性诊断，没有重新执行 naive 全量误差对拍。生产 C32 二进制未变，其误差证据沿用 `../01_c32_implementation/naive_default_full_recheck.json`：12用例均有限，output 相对L2约0.343%–0.400%，最终state约0.456%–0.463%。C32 rescale=1、lower_bound=-5 的失败仅用于性能诊断，不作为正确候选。

## 成本位置与证据

### K1：溢出访存、单 warp 阶段与同步等待

| NCU 指标 | C16 K1 | C32 K1 |
|---|---:|---:|
| 寄存器/thread | 32 | 32 |
| shared memory/block，bytes | 22272 | 42752 |
| shared memory 限制的 blocks/SM | 9 | 5 |
| 实际 occupancy | 97.06% | 62.08% |
| local load 指令计数 | 0 | 31776768 |
| local store 指令计数 | 0 | 25165824 |
| local load sectors | 0 | 150700032 |
| local store sectors | 0 | 132022272 |
| L1/TEX 命中率 | 99.55% | 22.19% |
| barrier stall / issue-active | 8.64 | 25.55 |
| long-scoreboard stall / issue-active | 2.23 | 9.61 |
| Tensor pipe active / elapsed | 5.89% | 3.74% |

NCU 明确将56942592次 local memory 请求判为 spilling，overhead=100%。local sector 总量折合约9.05 GB的 L1 请求流量，不能解释成9.05 GB HBM流量。DRAM active约24.21%，不支持 HBM 带宽饱和解释；NCU 提示 local load/store 每32-byte sector平均仅利用约0.9 bytes。

`csrc/smxx/utils.cuh:194` 的 Neumann helper 仅一个 warp 执行，C32 中 B=2，power/inv/next/product 的矩阵分块数组增长，并补齐更高阶展开。`fwd_kernel1.cuh:119` 仍保留原来的 `__launch_bounds__(NumThreads, 8)`，编译器仍将每线程寄存器压到32；C32 的 shared memory 实际只允许5块驻留。两者共同构成值得优先处理的寄存器压力证据，但未实验测定修改 launch bounds 的收益。

源码等待热点：`fwd_kernel1.cuh:529`（Neumann后的同步）、`:507`（L/Mqk后的同步）、`:600`（store后的同步）；另有 BF16 转换处的 long-scoreboard 热点。同步行代表其他 warp 在等待，不代表 barrier 指令自身消耗了全部时间，不能直接将采样百分比换成某阶段毫秒。既有单 warp 组织在每块工作增大后放大等待。

PM sampling 按有效区间分8段，K1 的 barrier/long-scoreboard 等待贯穿中间各段，未观察到只集中于末尾的长尾模式。本轮为等长输入，未测试 varlen 尾部不均衡。原始序列及分桶见 `analysis/*_pm.json`、`analysis/timeline_bins.json`；分桶用于观察时间趋势，不将不同 replay 的采样累计量当可比周期数。

### K2：增量较小，现有并行度与依赖等待

K2 无 local spill。寄存器74→86，shared memory99456→162816 bytes；shared memory 驻留上限2→1。两者实际 occupancy 都约9.37%：当前只启动96个block，GPU有148个SM，不能把驻留上限减半直接解释成当前并行度减半。

Tensor active约20.09%→20.19%；short-scoreboard stall/issue-active 0.60→1.30，wait 1.14→1.80。主要源码热点在 MMA、state 更新相关位置。说明更大的 tile 及执行依赖值得后续检查，但本轮没有将K2新增0.095 ms进一步做独立阶段计时。DRAM active约10.66%，也不是 HBM 带宽满载。

复现入口：`harness/collect.sh`、`harness/run.py`。
