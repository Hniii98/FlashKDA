# 指定负载：N=1、T=8192、H=96、D=128

**结论：这个负载的主要耗时在 K2，表现为状态链并行度不足，以及递推内部的依赖和片上供数等待；不支持“整体 HBM 带宽已饱和”或“Tensor 算力已饱和”的判断。K1 则是指令发射、片上数据访问与同步的混合限制。**

用户指定 T=8192、H=96、D=128；未指定 batch，本次取 N=1。CHUNK 固定为 16。本目录只保留这一种正式负载的测试、报告和原始 NCU 数据。T=16/31 仅用于小规模正确性检查，不作为其他性能负载。

## 实验设计

 NVIDIA B300 SXM6 AC（148 SM、CC 10.3）运行。使用 CUDA 13.0、Nsight Compute 2025.3.1，将当前仓库原始 `csrc/smxx/fwd_launch.cu` 与 host bridge 编译为独立动态库，保留 `-lineinfo`，目标 `sm_103a`。没有修改 GPU kernel。完整编译命令、源文件 SHA256 和二进制 SHA256 见 [source.json](analysis/source.json)。

调用 `launch_fwd<128,true,true,false,false>`：D=128、提供初始状态、输出最终状态、BF16 状态、等长序列。Q/K/V/g 和状态为 BF16；随机 seed=42，Q/K/V 为标准正态乘 0.125，g/beta 为标准正态，初始状态为标准正态乘 0.02，A_log/dt_bias 为 0，lower_bound=-5。这是指定 shape 的合成输入测量，不是模型真实张量回放。

输入分配、beta 转置和参考检查在计时前完成。先运行 20 次完整 K1→K2；独立计时为 CUDA events 测 9 组×20 次 forward，取每次 forward 耗时的中位数。NCU 使用 application replay、cache-control none、clock-control none，full+PM 和 source 两次采集均只覆盖第 21 次 forward；每份报告准确包含 K1、K2 两个 action，并核对名称及 CTA 数。该协议测预热后的原始 kernel，不包含输入投影 `in_proj_qkvgfab` 或 Python 公开接口的分配与布局转换。

## 实测结果

| Kernel | CTA 数 | NCU 耗时 µs | Compute/SM % | Tensor % | L1/TEX % | L2 % | DRAM % | DRAM 读写 GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | 49152 | 456.896 | 73.22 | 5.87 | 71.04 | 54.90 | 35.77 | 2743.83 |
| K2 | 96 | 1291.552 | 20.89 | 19.62 | 38.23 | 19.92 | 11.32 | 868.35 |

所有吞吐百分比统一采用 `pct_of_peak_sustained_elapsed`。Compute/SM 是 `sm__throughput`，本次最大分项为指令发射 `sm__issue_active`，不能解释成 Tensor FLOPS 使用率。L1/TEX 最大分项为 `l1tex__data_pipe_lsu_wavefronts`；Tensor 单列使用 `sm__pipe_tensor_cycles_active`。L2 使用 `lts__throughput`。DRAM 使用有效的 `dram__cycles_active`，与本次 DRAM 读、写字节吞吐百分比之和一致；GB/s 为读写之和，采用十进制。

独立完整 forward 中位数 **1733.864 µs（1.734 ms）**，见 [RESULTS.md](RESULTS.md)。NCU 分 kernel 时间合计 1748.448 µs，其中 K2 占 **73.87%**；NCU 重放与独立计时口径不同，不要求二者完全相等。

## 判断依据

### 1. 并行度与 occupancy

K1 每个 chunk/head 一个 CTA，grid=8192/16×96=49152，41.51 waves/SM，achieved occupancy 96.87%，32 registers/thread、22272 B shared/CTA。其低 Tensor 利用率不能归因于简单的 CTA 数不足。

K2 每个 batch/head 一个 CTA，只有 96 条状态链，每条链顺序处理 512 个 chunk。96 个 CTA 最多覆盖 96/148=64.86% 的 SM；增加 T 只延长每条链，不增加 grid。NCU 给出 small-grid 提示，waves/SM=0.324。每 CTA 192 threads、73 registers/thread、99456 B shared，shared 限制最多 2 CTA/SM，理论 occupancy 18.75%；当前只有一轮且 CTA 数不足，实测 active occupancy 9.37%。每 scheduler 仅 0.346 个 eligible warps/cycle，active issue 32.63%。这些证据说明可并行的工作量与延迟隐藏不足，是 K2 的关键限制。

### 2. 等长负载与活动周期分布

所有 head 都有 512 个 chunk，没有变长输入造成的工作量差异。K1 的 SM active cycles min/avg/max 为 488201/492647/495154；K2 为 0/901415/1404859。K2 的零值与存在空闲 SM 一致，不能直接把全卡 min/max 差异称为长尾负载不均衡。

### 3. 源码等待与 Tensor 利用率

| Kernel | barrier % | short scoreboard % | wait % | long scoreboard % | sleeping % |
|---|---:|---:|---:|---:|---:|
| K1 | 40.65 | 13.49 | 9.11 | 10.70 | 0.00 |
| K2 | 3.47 | 17.74 | 26.51 | 8.35 | 10.81 |

分母为 source 报告全部 stall 类别样本之和，包含 selected/not_selected，排除重复的 not_issued 指标。这是采样比例，不是运行时间占比或可直接消除的时间。

K1 的 barrier 热点映射到 [fwd_kernel1.cuh:629](../../csrc/smxx/fwd_kernel1.cuh#L629) 的 EXIT（2456 样本）和 [395 行](../../csrc/smxx/fwd_kernel1.cuh#L395) 的 LDCU（1907 样本）。编译后的同步等待可以归到后继指令，不能据此认为 EXIT 本身消耗了这些同步时间。结合 SM issue 73.22%、L1/TEX 71.04% 与 Tensor 5.87%，支持混合限制判断。

K2 的 short scoreboard 包含 [583 行](../../csrc/smxx/fwd_kernel2.cuh#L583) 和 [737 行](../../csrc/smxx/fwd_kernel2.cuh#L737) 的 BF16 HMMA（分别 454、401 样本）；[881 行](../../csrc/smxx/fwd_kernel2.cuh#L881) 的 `WARPSYNC.ALL` 有 3314 个 sleeping 样本。wait 26.51% 与 short scoreboard 17.74% 支持计算依赖和片上供数等待的判断，不能把它们全部解释为 HBM 等待。Tensor elapsed 19.62% 也没有显示全卡 Tensor 饱和。

### 4. PM 采样

full 报告保留 14 类 PM 序列。K2 的 wait/short-scoreboard 在有效采样区间形成持续平台，与循环内持续等待一致。当前序列是 warp 状态而非 SM 利用率时间线，采集窗口两端还有零值；不据此计算尾部耗时或空闲 SM 比例。原始序列分箱见 [分析结果](analysis/RESULTS.md)。

### 5. 访存层级与访问模式

K1 的 L1/TEX 71.04% 明显高于 DRAM 35.77%，shared load/store bank conflicts 分别为 5046979/1678770；NCU 将其估计为 shared load/store wavefront 的 18.78%/10.16%。K2 的 L1/TEX 38.23%、L2 19.92%、DRAM 11.32%，并未显示全卡某级带宽饱和；其中 L1/TEX active 为 59.74%，不要与主表 elapsed 混用。

K1/K2 L2 hit 分别为 42.08%/51.43%。普通 LSU global 指标不覆盖所有 TMA 流量：K2 的普通 LSU global requests 为 0，但实测 HBM 读写仍有 868.35 GB/s。因此不能用普通 global 请求数为 0 推断“没有访存”。K1 普通 global load 的 sectors/request=1，NCU 同时提示每线程每 sector 使用 4 B；这个提示本身不足以证明全 kernel 合并访问差。两 kernel 的 local load/store sectors 均为 0，当前未观察到 local-memory spill 流量。

### 6. NCU 规则与后续优先级

[规则输出](analysis/rules.txt) 对 K2 给出 small-grid 估计 35.14%、occupancy 相关估计 61.77%，对 K1 barrier 给出 local speedup 41.85%，shared load/store 冲突估计 13.48%/7.292%。这些是规则模型提示，互相重叠，不能相加或承诺为实测收益。

后续应优先在同一指定负载上验证 K2 沿独立 value 列拆分是否能增加 CTA 并缩短递推时间；它必须权衡重复供数和同步成本。其次分析 K2 供数与依赖隐藏，再考虑 K1 shared 布局及同步。当前只完成原始实现测量，没有把 profile03 的其他 shape 或 2-CTA microbench 收益转移到这里。

## 正确性与复现

调用仓库 `tests/torch_ref.py`，N/H/T=1/2/16 和 2/2/31 的输出、最终状态 relative L2 均为 0。正式 N=1/H=96/T=8192 的输出与状态均为有限值；正式大 shape 未做完整参考对拍，这里不声称它有零误差。

入口为 [harness/run.sh](harness/run.sh)。脚本依次编译、运行独立计时、采集 full/source、调用 NVIDIA `ncu_report` 解析；已有 rep 时拒绝覆盖。再次测量应复制为新的 run 目录并调整路径。PATH 需要虚拟环境、CUDA 与 NCU 的 bin，首次检查参考实现还会调用 Ninja 编译扩展。

- [完整计时及环境](RESULTS.md)
- [指标 JSON](analysis/metrics.json)、[指标及源码采样 Markdown](analysis/RESULTS.md)
- [full NCU](reports/full.ncu-rep)、[source NCU](reports/source.ncu-rep)
- [源文件与编译溯源](analysis/source.json)、[场景及 CTA 校验](analysis/cases.json)

full 采集提示 6 个 CTC 收发指标不可访问；本报告结论使用的 SM、Tensor、L1/L2、DRAM 和源码指标均已取得，不依赖 CTC 指标。

## 目录整理与采集溯源

本次测量已统一迁入 `profile/04_compute_memory_bottlenecks`，替换原四负载实验；独立补测目录已移除。测试脚本均使用当前路径。原始 rep、日志、独立计时和 `analysis/source.json` 中的采集路径保留测量时原值；这是历史溯源，不是运行入口。迁移未重新编译或测量，原始二进制及 rep 的 SHA256 保持不变。

纸面算术强度及瘦 GEMM 对照见 [METHOD.md](analysis/METHOD.md)；当前 CHUNK=16、每链 L=512，K2 逻辑矩阵算术强度约 **76.837 FLOP/B**，仅用于提出假设，不代替硬件计数器。
