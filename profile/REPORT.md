# FlashKDA 实验总报告

本报告合并原Baseline、topic1–6七份主报告及topic4方法说明。保留数值、失败结果和适用范围；各实验日期与计时协议不同，不能跨轮拼接加速比。原始profile、JSON、日志及自动生成的详细数据留在本地，不随仓库提交。本次仅整理目录与文档，没有运行GPU实验或修改生产kernel算法。脚本入口见[README](README.md)。

## 目录

- [原始KDA基线](#00_baseline)
- [CHUNK大小与数值范围](#01_chunk_size_analysis)
- [tcgen05计算重组](#02_tcgen05_evaluation)
- [递推并行方案](#03_recurrence_parallelism)
- [计算与访存瓶颈](#04_compute_memory_bottlenecks)
- [BF16状态存储精度](#05_bf16_state_accuracy)
- [SM100a专版设计决策](#06_sm100a_design_decision)
- [计算与访存采集方法](#measurement-method)

<a id="00_baseline"></a>

## 原始KDA基线

原目录：`profile/00_baseline`；本节代码块及路径文字仍按该目录上下文理解。

### 00_baseline：原始KDA基线

#### 本目录实验目标

建立原FlashKDA的正确性与完整forward性能基线，为后续实验提供对照。

**实验方法：** 比较FlashKDA与FLA路径在H=96/64、等长和变长输入下的完整forward，并用assignment02的naive参考检查output与最终state。

**结果与边界：** 性能与误差口径见下文；不能把不同计时协议的微基准直接与完整forward相除。

脚本、运行顺序和依赖见[profile README](README.md)，全部主题的报告合并见[总报告](REPORT.md)。以下保留原实验的测量和限制说明；本次整理没有重新执行GPU实验。

### KDA forward benchmark (Blackwell / B300)

- Command: `srun -G 1 python benchmarks/generate_benchmark_md.py -o profile/00_baseline/REPORT.md --device-label "Blackwell / B300"`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

##### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 1.7367 | 3.9900 | 2.30× | 1.9990 | 1.15× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.5020 | 4.1155 | 2.74× | 2.0592 | 1.37× |
| Varlen, `seq_lens`=`1024 x 8` | 1.2168 | 4.0127 | 3.30× | 1.9642 | 1.61× |

##### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 1.5857 | 2.7114 | 1.71× | 1.3551 | 0.85× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.1332 | 2.8696 | 2.53× | 1.4753 | 1.30× |
| Varlen, `seq_lens`=`1024 x 8` | 0.8246 | 2.6475 | 3.21× | 1.3030 | 1.58× |

#### 运行与验证

- 官方 `bench_fwd.py` 和 `generate_benchmark_md.py` 未修改；形状与 `BENCHMARK_GB200.md` 一致。表格采用 FP32 initial/final state，输入 BF16。没有设置额外 CPU 数量或 `FLA_FLASH_KDA` 环境变量。
- B300 SXM6 AC，SM103；驱动 580.126.09，CUDA toolkit 13.0.88，PyTorch 2.14.0+cu130，FLA 0.5.2，Nsight Compute 2025.3.1。
- 直接执行仓库 `benchmarks/ncu.sh`。仅将其中安装命令补为 `pip install -e . --no-build-isolation`，以使用当前环境中的 torch；ncu 命令及采集参数保持原样。
- ncu 作业 19960 成功结束；扩展重装后再以官方命令完成最终 benchmark（作业 19962），确保表格和 SASS 对应同一份扩展。
- ncu 按官方脚本覆盖 H=96 的 fixed、两组 varlen，以及 BF16 state / no state / FP32 state；共 **36 + 72 = 108** 条 prepare/recurrence 记录。H=64 已完成 benchmark，但官方 ncu.sh 不采集 H=64。

#### SM80 MMA 证据

**实际矩阵计算使用 SM80 风格的 warp-level `mma.sync` / HMMA，编译目标仍是 B300 的 `sm_103a`。**

- `csrc/smxx/fwd_kernel2.cuh:469` 使用 `SM80_16x8x16_F32BF16BF16F32_TN`；`cutlass/include/cute/arch/mma_sm80.hpp:239` 对应 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`。
- 已通过 `ncu_report` API 验证全部 108 条记录：动态 HMMA 计数均大于 0，没有动态 HGMMA/UTCMMA 计数。
- 唯一保留的 flash_kda_sm103a.sass（本地证据：`flash_kda_sm103a.sass`） 从实际扩展导出，只含 FP32 state 路径在 fixed/varlen 下的 4 个 kernel 实例，其中有 **272 条 `HMMA.16816.F32.BF16`、24 条 `HMMA.16816.F16`**。这是静态指令数量，不是运行次数。

下表取各形状 FP32 state 的第一次调用，指标为 `sass__inst_executed_per_opcode` 中的 **HMMA 动态 warp 指令数**：

| H=96, T=8192, D=128 | prepare | recurrence |
|---|---:|---:|
| Fixed | 2,162,688 | 20,447,232 |
| Varlen [1300,547,2048,963,271,3063] | 2,175,360 | 20,567,040 |
| Varlen [1024] × 8 | 2,162,688 | 20,447,232 |

例如 fixed recurrence 的 `HMMA.16816.F32.BF16 R12, R20, R36, RZ` 与 `fwd_kernel2.cuh:552` 关联；该调用的 BF16 HMMA 运算计数为 **83,751,862,272**。因此结论不仅来自源码声明，也有实际执行计数和 SASS 支持。

本轮 fixed recurrence 的 NCU 平均 SM 时钟约 **1.09 GHz**（`sm__cycles_elapsed.avg.per_second`）。官方脚本使用 `--clock-control none`，未锁频；不能把不同运行状态下的耗时、加速比差异直接当作随机误差。NCU replay 的耗时未用于上面的 benchmark 表。

#### 复现

```bash
source .venv/bin/activate
srun -G 1 bash benchmarks/ncu.sh
srun -G 1 python benchmarks/generate_benchmark_md.py \
  -o profile/00_baseline/REPORT.md --device-label "Blackwell / B300"
cuobjdump --dump-sass flash_kda_C.cpython-312-x86_64-linux-gnu.so
```

最后一条命令导出完整扩展 SASS；本目录保留的是其中与 FP32 state 表格对应的 4 个实例。再次运行生成器会重写报告，需要重新补充本节验证结果。

按要求已删除临时 `.ncu-rep` 和日志，只保留本报告及 SASS；没有额外 harness、脚本副本或分析目录。若需重新打开 NCU 指标，执行上述官方采集命令。

本次扩展 SHA256：`039e25c8d9187fdbb73efd8c00612f80fc051f9a8b9b98275d95b0557cff7f2b`。

#### 与 `fla_kda_ref/naive.py` 对拍（2026-09-10）

直接通过文件路径加载材料快照中未修改的 `naive_recurrent_kda`，使用其逐 token 的 FP32 PyTorch 递推作为参考；没有经 FLA backend dispatch，也没有用 FlashKDA 自身作参考。本节未运行 `chunk.py`。

- 运行：Slurm 作业 24782，GPU **NVIDIA B300 SXM6 AC**，PyTorch `2.14.0+cu130`；扩展 SHA256 与上面的 baseline 一致。
- 参考文件：`/home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref/naive.py`；SHA256：`60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016`。
- 覆盖上表全部 6 组形状，每组分别运行 FP32 / BF16 initial、final state，共 12 次对拍。每组固定 `torch.manual_seed(0)`，输入生成方式遵循 `benchmarks/bench_fwd.py`：BF16 Q/K/V/g/beta，Q/K 先归一化再转 BF16，A_log、dt_bias 为 FP32 uniform；initial state 为 `arange(N*H*D*D).reshape(N,H,D,D).bfloat16()`，FP32 路径再转为 FP32。
- 参考预处理：Q/K 使用 `x / sqrt(sum(x²)+1e-6)`；gate 为 `-5*sigmoid(exp(A_log)*(g+dt_bias))`；beta 为 `sigmoid(beta)`，scale 为 `1/sqrt(128)`，全部在 FP32 计算，关闭 TF32。参考 V 使用 FP32，使输出不额外舍入为 BF16；因此统计包含 FlashKDA 的输出量化误差及内部近似误差。
- state 布局由 FlashKDA 的 `[N,H,V,K]` 显式转成参考的 `[N,H,K,V]`，返回后转回；varlen 按 cu_seqlens 分段独立调用参考，使用各自的 initial state。
- relative L2 = `||actual-reference||₂ / ||reference||₂`；mean/max abs 为绝对差的均值/最大值，指标归约使用 FP64。完整的 12 组指标及有限值检查见 fla_ref_errors.json（本地证据：`fla_ref_errors.json`），脚本见 [compare_fla_ref.py](00_baseline/compare_fla_ref.py)。

| H | Case | State dtype | Output rel L2 | Output mean abs | Output max abs | Final state rel L2 | Final state mean abs | Final state max abs |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 96 | Fixed | float32 | 0.385815% | 0.00658665 | 314.684 | 0.467087% | 0.000115812 | 0.00904006 |
| 96 | Fixed | bfloat16 | 0.385815% | 0.00658665 | 314.684 | 0.467087% | 0.000115812 | 0.00904006 |
| 96 | Varlen 不等长（同上） | float32 | 0.347523% | 0.221123 | 1796.06 | 0.458659% | 0.000108429 | 0.00949746 |
| 96 | Varlen 不等长（同上） | bfloat16 | 0.347523% | 0.221123 | 1796.06 | 0.458659% | 0.000108429 | 0.00949746 |
| 96 | Varlen 1024 × 8 | float32 | 0.400175% | 0.431685 | 3514.62 | 0.456153% | 0.000108385 | 0.00916034 |
| 96 | Varlen 1024 × 8 | bfloat16 | 0.400175% | 0.431685 | 3514.62 | 0.456153% | 0.000108385 | 0.00916034 |
| 64 | Fixed | float32 | 0.391238% | 0.00410538 | 163.449 | 0.455767% | 0.000110144 | 0.00847822 |
| 64 | Fixed | bfloat16 | 0.391238% | 0.00410538 | 163.449 | 0.455767% | 0.000110144 | 0.00847822 |
| 64 | Varlen 不等长（同上） | float32 | 0.355750% | 0.158197 | 1268.53 | 0.459242% | 0.0001117 | 0.00973064 |
| 64 | Varlen 不等长（同上） | bfloat16 | 0.355750% | 0.158197 | 1268.53 | 0.459242% | 0.0001117 | 0.00973064 |
| 64 | Varlen 1024 × 8 | float32 | 0.343960% | 0.272445 | 1425.25 | 0.460594% | 0.000109872 | 0.00847822 |
| 64 | Varlen 1024 × 8 | bfloat16 | 0.343960% | 0.272445 | 1425.25 | 0.460594% | 0.000109872 | 0.00847822 |

全部输出和最终状态均无 NaN/Inf。输出 relative L2 为 **0.344%–0.400%**，最终状态为 **0.456%–0.467%**。FP32/BF16 state 路径在这组输入上的误差统计一致；initial state 本来就取 BF16 可表示值，kernel 内部的递推状态仍舍入到 BF16，FP32 外部 state buffer 不等价于 FP32 递推。

这里的 initial state 最大达到约 1.26e7，因此 output 的 max abs 可达 3514.625；应结合 relative L2 阅读，不能将该绝对误差直接理解为单位尺度输入下的误差。本节仅报告固定种子、benchmark 分布的实测数值，没有设置通过阈值，也不替代弱衰减、长程记忆或多种子压力验证。

复现（从仓库根目录执行，`--ref-dir` 可替换为材料目录）：

```bash
srun -G 1 --mem=32G .venv/bin/python profile/00_baseline/compare_fla_ref.py \
  --ref-dir /home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref
```

该命令更新 JSON；本节 Markdown 为该次 JSON 的结果摘要。原 benchmark 生成器仍会重写整份 REPORT.md，重跑时需保留本节。

<a id="01_chunk_size_analysis"></a>

## CHUNK大小与数值范围

原目录：`profile/01_chunk_size_analysis`；本节代码块及路径文字仍按该目录上下文理解。

### 01_chunk_size_analysis：CHUNK大小与数值范围

#### 本目录实验目标

检查CHUNK扩大到32/64时，指数范围、FP16 Neumann及MMA形状分别受到什么限制。

**实验方法：** 使用独立CUDA microbench比较C16/C32/C64的指数异常、求逆异常及矩阵形状映射。

**结果与边界：** C32/C64出现指数与Neumann中间值异常，但仍能按SM80 MMA形状分块；本组不是生产C32 forward实现。

脚本、运行顺序和依赖见[profile README](README.md)，全部主题的报告合并见[总报告](REPORT.md)。以下保留原实验的测量和限制说明；本次整理没有重新执行GPU实验。

### CHUNK=16/32/64：B300 实验结论汇总

日期：2026-09-05。NVIDIA B300 SXM6 AC，CC 10.3，148 SM；CUDA 13.0，Nsight Compute 2025.3.1。

**结论：**

- **C32 已出现指数归零、溢出，C64 在更小衰减下就会出现。** 每个 token 使用相同衰减 a∈[0,5] 时，B300 扫描中 C32 首次负指数归零的 a≈2.72969，逆指数首次非有限的 a≈2.77344；C64 分别为 1.36484 和 1.38672。负指数归零会丢失原本非零的衰减贡献，逆指数溢出会使后续计算产生 Inf/NaN。C16 在本次 a≤5 扫描中没有出现这两种异常。
- **C32/C64 的 FP16 Neumann 求逆都已算出 Inf/NaN。** 把整个 32×32 或 64×64 矩阵直接做 Neumann 展开，并用 FP16 保存、计算中间矩阵时，各有 36/108 个测试 case 得到非有限结果。例如所有 k 相同、a=0、beta=0.9 时，指数没有问题，但计算部分和 I−L+L²−…−L⁷ 时，中间数值超过 FP16 范围，C32 出现 15 个 Inf，C64 出现 703 个 Inf，无法正常完成求逆。对应理想输入的正确逆矩阵元素绝对值不超过 1，异常来自展开过程。
- **C32/C64 都匹配 m16n8k16 MMA 形状，可以分块发射。** C32 沿 M 维拆成 2 个 M16 tile，C64 拆成 4 个；相关 N、K 维也能分别被 8、16 整除，无需 padding。因此，MMA 形状匹配不能作为 CHUNK 必须选 16 的理由。

本报告只验证三件事：指数是否归零或溢出、FP16 Neumann 求逆是否产生数值异常，以及 MMA 形状能否分块发射。

**实验范围：使用独立 CUDA microbench，在 B300 上复现所讨论的指数与求逆计算；并非修改后的生产 FlashKDA kernel。C16 是相同测试的尺寸对照。**

#### 1. bf16 动态范围：C32 已出现指数归零和溢出，C64 的衰减阈值更低

##### 1.1 纸面推导：累计衰减边界

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

##### 1.2 B300 验证

令 a=-g_act∈[0,5]。实际执行 `ex2.approx.ftz.f32`、bf16 转换及乘法，首次观测的常数 gate 阈值如下。
累计衰减扫描步长为 0.05；这些是采样边界，不是无限精度阈值。

| CHUNK | exp(G) 首次归零的平均 a | exp(-G) 首次非有限的平均 a |
| --- | --- | --- |
| 16 | a≤5 未出现 | a≤5 未出现 |
| 32 | 2.7297 | 2.7734 |
| 64 | 1.3648 | 1.3867 |

- C=32/64 首次归零的累计衰减约 87.35，逆指数首次非有限约 88.75。C=16 最坏累计衰减为 80。
- 归零也是本项记录的数值异常，不能只检查 Inf/NaN。

证据：C16 范围（本地证据：`chunk16/RESULTS.md`）、C32 范围（本地证据：`chunk32/RESULTS.md`）、C64 范围（本地证据：`chunk64/RESULTS.md`）。汇总已测正常/异常区间、最后正常点及首次异常点；原始测量记录折叠保留。

#### 2. Neumann：FP16 中间部分和溢出

##### 2.1 纸面推导：有限级数及展开阶数

L 是 C×C 严格下三角矩阵，所以 Lᶜ=0，且

`(I+L)⁻¹ = I−L+L²−…+(−L)^(C−1)`。

这是有限多项式恒等式，不需要无限 Neumann 级数的范数小于 1 条件；但恒等式不保证浮点求值稳定。C 为 2 的幂时可写为

`(I−L)(I+L²)(I+L⁴)…(I+L^(C/2))`。

当前实现从 I−L 开始，逐步使用 L²、L⁴、L⁸ 等幂更新部分和。C16、C32、C64 的完整展开分别到 L¹⁵、L³¹、L⁶³；恒等式在实数运算中成立，但 FP16 中间部分和可能先超出数值范围。

##### 2.2 纸面推导：最终逆很小，中间部分和仍可溢出

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

##### 2.3 B300 验证

本实验的 L 由 BF16 k 的 FP64 Gram 矩阵与 FP64 指数构造，再转 FP16 并乘 FP16 beta；没有调用生产 K1 来生成 L。原 K1 先舍入 BF16 衰减操作数再做 MMA，两者的舍入位置不同。因此本节异常计数与 inverse 相对误差只描述这组隔离求逆输入，不能直接当作生产 KDA 的失败率或输出误差。

将有限 FP16 L 输入完整 FP16 Neumann 实现，对照同一输入的 B300 fp64 triangular solve。每种尺寸 108 个 case：3 seeds × 4 种 k（random/repeated/alternating/correlated）× a={0,0.05,1} × beta={0.1,0.9,0.999}。

| CHUNK | Inf/NaN case 数 / 总数 | 有限结果的最大 inverse 相对 L2 |
| --- | --- | --- |
| 16 | 0/108 | 0.23788 |
| 32 | 36/108 | 0.00016508 |
| 64 | 36/108 | 0.0019403 |

- repeated k、a=0、beta=0.9 的例子中，在使用 L⁴ 更新部分和 I−L+L²−…−L⁷ 时，C32 出现 15 个 Inf，C64 出现 703 个 Inf。此时 L⁴ 自身仍有限。
- C16 的 108 个 case 没有 Inf/NaN，但最大 inverse 相对 L2 约 23.79%。所以“没有非有限值”不等于“误差很小”。
- 表中有限结果的误差只统计 finite=True 的 case；不能据此忽略 C32/C64 已失败的 36 个 case。

证据：C16 求逆（本地证据：`chunk16/RESULTS.md`）、C32 求逆（本地证据：`chunk32/RESULTS.md`）、C64 求逆（本地证据：`chunk64/RESULTS.md`）、C32 阶段异常（本地证据：`chunk32/RESULTS.md`）、C64 阶段异常（本地证据：`chunk64/RESULTS.md`）。`inverse` 表按 k、a、beta 汇总完整 FP16 实现的异常条件；阶段表展示代表输入的异常前、首次异常及最终部分和。实际 L 输入保存在各 `analysis/inverse_inputs.pt`。fp64 求解仅用于核对正确性，不作为优化方案比较。

#### 3. MMA：C32/C64 均能分块发射，形状匹配

##### 3.1 纸面推导：分块即可，不要求 CHUNK 等于指令的 M 维

对 m16n8k16 指令，一次处理 M=16、N=8、K=16 的 tile。完整矩阵可以由多条指令覆盖，CHUNK 不必等于 16。

| CHUNK | 当 M=C 时的 M16 分块数 | 当 N=C 时的 N8 分块数 | 当 K=C 时的 K16 分块数 | padding |
| --- | --- | --- | --- | --- |
| 16 | 1 | 2 | 1 | 不需要 |
| 32 | 2 | 4 | 2 | 不需要 |
| 64 | 4 | 8 | 4 | 不需要 |

本次四类 GEMM 的 (M,N,K) 为 (C,C,128)、(C,128,128)、(C,128,C)、(128,128,C)。固定维度 128 也能被 16、8、16 整除，所以三种 CHUNK 的四类形状全部匹配。

**结论：C32 可以拆成两个 M16 tile，C64 可以拆成四个 M16 tile，均可发射 MMA；不存在因这些尺寸无法整齐分块而必须选 C16 的限制。**

本项以指令 tile 的整除关系判断能否分块发射，不将形状验证混入数值异常数据表。

#### 4. 文件和复现

- `chunk16/`、`chunk32/`、`chunk64/`：基线及两种尺寸的关键证据。日志不保留；C16/C32/C64 的表格数据在各自 `RESULTS.md`，输入与环境在 `analysis/`，筛选后的 profiler 证据在 `reports/`。
- 每个 `reports/` 保留 `full.ncu-rep`、`source.ncu-rep`；每份 11 个实际 launch（指数、求逆和完整诊断路径）；MMA 映射性能测试记录已移除。

| CHUNK | 分析数据 | NCU full | NCU source |
| --- | --- | --- | --- |
| 16 | RESULTS.md（本地证据：`chunk16/RESULTS.md`） | full（本地证据：`chunk16/reports/full.ncu-rep`） | source（本地证据：`chunk16/reports/source.ncu-rep`） |
| 32 | RESULTS.md（本地证据：`chunk32/RESULTS.md`） | full（本地证据：`chunk32/reports/full.ncu-rep`） | source（本地证据：`chunk32/reports/source.ncu-rep`） |
| 64 | RESULTS.md（本地证据：`chunk64/RESULTS.md`） | full（本地证据：`chunk64/reports/full.ncu-rep`） | source（本地证据：`chunk64/reports/source.ncu-rep`） |

- C16/C32/C64 各用一份 `RESULTS.md` 集中保存三类数值异常证据；`analysis/` 仅保留求逆实际输入、环境、profile case 映射及精简 NCU 指标汇总。三个 CHUNK 使用相同目录结构。
- 环境参数保存在各 `analysis/environment.json`；校验清单（本地证据：`SHA256SUMS.json`） 用于核验，编译命令见 [run.sh](01_chunk_size_analysis/harness/run.sh)。
- `harness/` 仅保留 `bench.py`（统一测试、NCU 解析、结果存储和报告入口）、`kernels.cu`（CUDA 实现）、`run.sh`（编译及 srun/NCU 调度）。
- `.venv/bin/python profile/01_chunk_size_analysis/harness/bench.py run --chunk 32` 运行指数与 FP16 求逆异常验证；所有 GPU 测试必须在 B300 分配内执行。
- [harness/run.sh](01_chunk_size_analysis/harness/run.sh) 可重跑实验；已有 `.ncu-rep` 不覆盖，需先归档。仅重新生成汇总可运行 `.venv/bin/python profile/01_chunk_size_analysis/harness/bench.py report`。

当前实验目录为 `profile/01_chunk_size_analysis`。rep 由原始采集结果离线筛选导出，移除了全部 8 条 MMA 映射性能测试记录；保留记录的内嵌历史源码及实测指标未改写。独立 MMA 计时 CSV、resident 形状测试结果、资源对比和全部日志已删除。其余指标可由 [bench.py analyze](01_chunk_size_analysis/harness/bench.py) 从 rep 重新提取。

实验均为合成输入，未测试真实模型任务质量、尾块或 varlen；任何精度筛选线都不是产品质量验收标准。

<a id="02_tcgen05_evaluation"></a>

## tcgen05计算重组

原目录：`profile/02_tcgen05_evaluation`；本节代码块及路径文字仍按该目录上下文理解。

### 02_tcgen05_evaluation：tcgen05计算重组

#### 本目录实验目标

检验CHUNK=16能否通过共享操作数合并或转置匹配tcgen05，以及是否有实际收益。

**实验方法：** 对原阶段GEMM作代数等价重组，运行独立正确性、计时和NCU执行验证。

**结果与边界：** 局部GEMM有收益，69项检查通过，但没有证明完整KDA换指令后的端到端收益。

脚本、运行顺序和依赖见[profile README](README.md)，全部主题的报告合并见[总报告](REPORT.md)。以下保留原实验的测量和限制说明；本次整理没有重新执行GPU实验。

### Topic2：CHUNK=16 的 tcgen05 计算重组与 B300 实测

**结论：padding 不是唯一办法。源码确认共享操作数合并和完整转置两类重组，可以减少或消除无效计算。B300 的 69 组独立 GEMM 正确性检查通过；共享 state 的 M32 合并运算在 batch=1 下，ws 从 mma.sync 的 15.0920 µs 降至 14.2695 µs，旧/新=1.058×，耗时下降 5.45%。batch=16384 时，所有测试的新路径仍慢于同任务 mma.sync。不能再用“CHUNK16 只能 padding，所以 tcgen05 不值得”解释结果。**

此处“保持数学结果”指 GEMM 代数等价；不是逐位相等保证，也不是完整 KDA kernel 已验证。生产代码未修改。

#### 1. 逐项源码审查

以 CHUNK=16、D=128 为准。P 表示 Neumann 当前的矩阵幂，R 表示已按原代码算出并转换成 BF16 的 residual。

| 阶段及源码 | 原运算 | 可用重组 | 形状与限制 | 本次验证 |
| --- | --- | --- | --- | --- |
| [K1 构造 L/Mqk](../csrc/smxx/fwd_kernel1.cuh#L481) | K_decayed@K_inv 与 Q_decayed@K_inv | 堆叠两组左操作数，共用 K_inv | (32,16,128)；普通仍需 M64，ws 仍需 N64；比两次分别 padding 更省部分无效计算 | K1 shared-B pair |
| [Neumann 初始 L²](../csrc/smxx/utils.cuh#L262) | L@L | 方阵转置仍是 M16；当前序列中无已就绪的同 B 伴随运算 | (16,16,16)，保留单运算时仍需 padding | Neumann square |
| [Neumann L⁴/L⁸ 及逆更新](../csrc/smxx/utils.cuh#L267) | INV@P 和 P@P | 堆叠 INV 与 P，共用旧 P；完成后分别执行原 FP16 加法与下一阶段 | (32,16,16)；普通 M64，ws N64；最后 INV@L⁸ 无下一次平方可配对 | Neumann shared-power pair，仅测试配对 GEMM |
| [K2 Phase 1](../csrc/smxx/fwd_kernel2.cuh#L533) | K_decayed@S 与 Q_decayed@S | 堆叠 K/Q，共用 S | (32,128,128)，ws 无 padding；普通仍需 M64 | K2 shared-state pair |
| [K2 Phase 3](../csrc/smxx/fwd_kernel2.cuh#L589) | INV@R | 完整转置 Rᵀ@INVᵀ，结果转回 | (128,16,16)，普通无 padding；不得跳过 residual 的 BF16 舍入 | INV@residual / Mqk@U |
| [K2 Phase 4](../csrc/smxx/fwd_kernel2.cuh#L625) | Mqk@U | 完整转置 Uᵀ@Mqkᵀ，结果转回 | (128,16,16)，普通无 padding；U 依赖前一阶段，不能与 INV@R 当作共享 B 的独立乘法合并 | 与上一项同 GEMM 测试，实际中间数据未移植 |
| K2 Phase 1 的单项备选 | K/Q@S | Sᵀ@(K/Q)ᵀ，结果转回 | (128,16,128)，普通无 padding | single K/Q@state |
| [K2 Phase 6](../csrc/smxx/fwd_kernel2.cuh#L659) | K_restoredᵀ@U | 原形状已匹配；保留即可 | (128,128,16)，无 padding | state update |

共享 state 合并满足 `[K;Q]S=[KS;QS]`。转置必须同时交换并转置两个操作数：`AB=(BᵀAᵀ)ᵀ`，绝不是只转置 A 或 B。本次 CUDA 转置路径直接按原输出方向合并写回，不额外启动输出转置 kernel。

K1 两组乘法都使用 BF16 输入、FP32 累加，但分别写为 FP16 的 L 与 BF16 的 Mqk；合并时必须分别恢复这些转换。Neumann 的每次乘法输出和 INV 加法仍需保留 FP16 舍入位置。K2 中 BF16 residual、U 和输出加法的转换顺序也不能删除。因此本次仅验证重组乘法本身，不能将其说成完整算法正确性验证。

#### 2. 其他组合的审查结果

- **Mqk@U 与 K_restoredᵀ@U 也共享 B。** 堆叠后有效 M=144，超过单条指令 M128。用 ws M128+M32 共执行 160 行，与分别执行 M128、M32 相同；普通 M128+M64 共 192 行，也与分别 padding 相同。它可能通过同 CTA 内复用 TMEM、同步和 U 改善调度，但没有单靠形状合并减少 FLOPs 的收益。本次未实现融合两阶段的 kernel，不对这种调度的收益下结论。
- **Neumann 依赖必须保留。** 配对的是使用同一旧 P 的 INV@P 与 P@P；不能把依赖新 INV 或新 P 的后续乘法提前。初始 L²和最后 INV@L⁸仍是单项。
- **不同时间 chunk 的 S 不共享且存在递推依赖。** 不同序列/head 也通常具有不同操作数，不能套用共享 B 堆叠。可以计算所有 A_iB_j 再取对角块，但会引入无用的交叉乘积；不等同于免费 batch，也未证明比 padding 更好。
- **gate、beta、mask、残差和 state 衰减**不是待替换的 GEMM，保持原有计算及精度转换。本轮不将这些操作融合进 GEMM 或更改 Neumann 算法。

#### 3. 公平比较的边界

所有路径使用相同有效矩阵、相同 batch、相同 FP16/FP32 累加类型和 FP32 输出接口。共享 B 的配对任务让 **mma.sync 也使用同一 CTA、相同堆叠输入和 B 复用**，避免用新路径一次 launch 对比旧路径两次 launch。各路径仍需各自的正确布局和同步。

输入物理打包和 padding 均在计时前准备，转置路径的输入重排也在计时前准备。计时包含 global→shared、MMA、TMEM 管理、同步和原方向输出写回。故结果回答的是“操作数已经按路径布局准备好之后”的独立 GEMM 性能；完整 KDA 接入时的输入重排、寄存器到 shared 的转换尚未计入。这不是生产 kernel 的融合时序模拟，也不证明已找到最优 tcgen05 实现。

每项 3 个随机 seed，每个 seed 3 个矩阵，对量化后输入的 FP64 GEMM 检查有限性及相对 L2；FP16 阈值 0.003，BF16/FP32 阈值 2e-5。7 项共 69 个检查记录。计时每条路径 12 轮，轮换顺序，CUDA Graph 与 events；单任务每图 32 次、大批每图 4 次。NCU replay 时间不作为性能数据。

#### 4. 实测与复现

全部时间、正确性误差、12 轮原始样本及机器信息集中在 RESULTS.md（本地证据：`RESULTS.md`）。

- 共享 state 配对，batch=1：mma.sync 15.0920 µs，ws 14.2695 µs；采样区间分别 15.0710–15.1080、14.2520–14.3230 µs，本轮不重叠。
- 同一配对，batch=16384：mma.sync 1089.6200 µs，ws 2760.9360 µs，旧/新 0.395×。无 padding 仍不保证快。
- M16×N128、K16，batch=16384：普通 padding 1611.3520 µs，完整转置 1236.0720 µs，减少约 23.29%；但 mma.sync 仅 83.9280 µs。
- M16×N128、K128，batch=16384：普通 padding 3237.1320 µs，完整转置 3194.8240 µs；差距较小，不能当作稳定收益。ws padding 为 2725.6560 µs，仍比该转置实现快。

统一测试入口 [bench.py](02_tcgen05_evaluation/harness/bench.py)、CUDA 实现 [kernels.cu](02_tcgen05_evaluation/harness/kernels.cu)、编译和 B300 调度 [run.sh](02_tcgen05_evaluation/harness/run.sh)。运行 `bash profile/02_tcgen05_evaluation/harness/run.sh` 可重现。本次 basic.ncu-rep（本地证据：`reports/basic.ncu-rep`） 使用 basic 集合，采集 batch=16384 的 23 个路径，作为实际 CUDA kernel 执行证据；不是完整 stall/source 性能归因报告。

**采用建议：优先把共享 state 的 M32 合并作为后续集成候选，而非把所有 M16 运算统一 padding。当前独立测试尚不支持全面改用 tcgen05；生产 KDA 中的 TMEM 复用和输入布局接入成本仍需实际集成才能判定。**

<a id="03_recurrence_parallelism"></a>

## 递推并行方案

#### 实验设计与实现路径

**为什么本题聚焦 K2？** K1 生成不依赖历史状态的块内系数，各 chunk 可以直接并行；它不负责跨 chunk 的递推状态更新。K2 同样包含矩阵乘，但下一 chunk 必须等待上一 chunk 更新后的状态，因此时间维并行受限，本题重点是从 head、value 分片和任务调度中寻找额外并行度。这不是因为“K1 使用矩阵乘，所以不需要考虑并行度”，也不表示 K1 的性能已经最优。前三组实验不改 K1 的并行策略，单独测 K2 判断方案效果，再用 K1+K2 检查整体收益；新增 DSM 组覆盖 K1/K2 的实际 GEMM 形状，但只验证局部协作计算。

**实验实现逻辑：修改什么代码、怎样编译、测哪一层。** 前三组不是只调整 launch 参数；它们从当前生产 K1/K2 源码生成隔离副本，在保留主要数学计算与精度路径的基础上改动任务分工。DSM 双 CTA 目前是另写的局部 GEMM 实现，尚未接入生产 K2。二者的实验粒度不同。

| 方案 | 源码具体改动 | 编译出的对照版本 | GPU 实测范围 |
| --- | --- | --- | --- |
| value 分片 | grid 增加分片维；计算线程数改为 `128/TOPIC_SPLIT`；用 `blockIdx.z` 映射 value 范围；改输出与最终状态的分片写回 | `original.so`、`split1.so`、`split2.so`、`split4.so` | 完整 K2，以及 K1+K2 |
| 多 head 共 CTA | 用 `local_tid` 和 `head_slot` 划分线程；映射相邻 head；为各 head 隔离 shared storage、流水线与 barrier；调整 block/grid | `original.so`、`heads2.so` | 完整 K2，以及 K1+K2 |
| persistent | 在原 K2 完整递推外加任务循环；静态步长或原子队列领取下一条链；切换时等待写回、处理旧 barrier、重置状态与流水线 | `original.so`、`static.so`、`dynamic.so`；运行时配置 148/296 worker | 完整 K2，以及 K1+K2；动态队列初始化计时 |
| DSM 双 CTA | 独立 GEMM 中沿 N 分片；创建双 block cluster；rank0 加载公共 A，rank1 经 DSM 读取；加入跨 CTA 同步 | 一个库内的 `single_cta`、`independent_split2`、`cooperative_cluster2` 三个 mode | 一次完整 GEMM kernel，包含加载、DSM、同步、MMA 与写回，不含 K2 递推 |

**前三组如何构建和调用。** `harness/build.py` 读取生产 `fwd.h`、`utils.cuh`、`fwd_kernel1.cuh`、`fwd_kernel2.cuh` 和 `fwd_launch.cu`，写入对应的 `harness/build_<group>/`，只修改副本。随后用 nvcc 的 `-O3 -std=c++17 --use_fast_math -lineinfo -gencode arch=compute_103a,code=sm_103a -Xcompiler=-fPIC -shared` 等选项编译副本的 `fwd_launch.cu`，通过 `TOPIC_SPLIT/TOPIC_MANUAL`、`TOPIC_HEADS` 或 `TOPIC_SCHED` 生成不同变体；宏是选择已写好的改动路径，不是仅改一个参数便自动得到新算法。生产 `csrc` 不改动。

副本导出统一 `extern "C" run(phase, ...)`，`bench.py` 用 ctypes 加载各库并传 GPU 张量指针。`phase=1` 只运行 K1，`phase=2` 只运行 K2，`phase=0` 运行 K1+K2。只测 K2 时先运行 K1 准备同一份 workspace，各变体从同一初始状态执行全部 chunk；先比较 output/final state，再预热和用 CUDA Graph/events 计时。完整计时则把 K1 一并放入被测调用。`split1` 额外隔离了分片写回方式本身的成本，主要基线仍是 original。

**为什么协作 2-CTA 目前另写 microbench？** 这是本轮实现范围的选择，并非生产 K2 不能采用相同的副本构建方法。前三组主要让每个 CTA 独立维护自己的状态和流水线；共享输入的协作 K2 需要进一步规定由谁加载公共系数、何时允许远端读取、何时可以覆盖缓冲或退出，以及两条流水线如何配合。原先 CTA 内的 barrier 不能直接替代这些跨 CTA 同步。

本轮先固定实际 KDA GEMM 形状，比较单 CTA、独立分片和 DSM 共享输入，回答“这一具体共享方案是否减少局部耗时”。因此没有把生产 K2 包装成协作版，也不能用局部结果代替完整递推结论。若后续统一比较完整 K2，应在生产 split2 副本上增加 cluster、公共系数供数与每个流水阶段的跨 CTA 生命周期管理，再用同一套初始状态、workspace、对拍与计时接口验证；该集成尚未完成。不是所有双 CTA 方案都必然需要这些交换，独立 split2 正是无需跨 CTA 交换的对照。

详细报告：[03_recurrence_parallelism/REPORT.md](03_recurrence_parallelism/REPORT.md)。本节未带前缀的实验文件路径均相对于 `profile/03_recurrence_parallelism/`。

**结论：并行度收益取决于任务数量和分工。** value 分片在少链长序列有收益，直接双 head 合并未测到中位数收益，动态 persistent 仅在特定分配倾斜下有小幅收益。新增的 DSM 协作 2-CTA 在所测 30 个形状/任务数组合中，中位数均慢于本组单 CTA 和独立 split2；该结果仅针对独立 GEMM，不能推广为完整 K2 或所有双 CTA 协作方案的结论。

#### 实验分组与证据范围

| 实验组 | 实际执行内容 | 对照 | 结果与边界 |
| --- | --- | --- | --- |
| `column_split` | 生产 K2 的 value 分片，保持完整时间递推 | original / split1 / split2 / split4 | 少链长序列 split4 有收益；多链和短任务可退化 |
| `multi_head` | 两个 head 放入一个 CTA，各自保留状态和流水线 | original / heads2 | 直接复制资源未测到中位数收益，不代表更精细的复用设计无效 |
| `persistent` | 固定 worker 静态或动态领取完整状态链 | original / static148/296 / dynamic148/296 | 动态队列修复静态倾斜，但原 grid 已有调度，不能只对比最差静态版 |
| `cooperative_2cta` | 固定 CHUNK16 实际 GEMM 形状，cluster 内经 DSM 共享 A | single_cta / independent_split2 / cooperative_cluster2 | 75 项对拍通过；30 个条件均未测到协作中位数收益；不含完整状态递推 |

前三组采用 2026-09-10 的 B300（CC10.3、148 SM）实测，修正了封装中遗漏的 `log2(e)`，统一使用自然指数约定 `lower_bound=-5`。新增协作组采用 2026-09-11 的同型号 GPU 实测。本次整理只更新报告，没有重新运行 GPU 实验。

前三组保留 CHUNK=16、D=128、BF16 状态、真实 K1/K2 和 mma.sync，分别报告 K2 及完整 K1+K2。第四组仅按这些阶段的实际矩阵形状测一次 GEMM，使用相同 MMA 类型做软件 cluster 协作，不是 `tcgen05.cta_group::2`。

四组均使用 CUDA Graph 每图 3 次调用、9 轮轮换采样和 CUDA events 中位数；加速比使用各组同轮基线，不能跨组相除，NCU replay 时间不参与加速比。前三组的 `PROFILE.md` 是 NCU 硬件指标解析，第四组本次没有采集 NCU，不能从计时推断同步或带宽开销的比例。

##### 4.1 独立链、实现一致性与门控检查

K1 可跨 chunk 并行；K2 原版每个 sequence/head 对应一个 CTA，独立链数 P=N×H_local。96 heads、TP8 时单序列每卡只有 12 条链，最多覆盖 148 SM 的 8.11%；value 四分后可增加到 48 个 CTA，纸面上限 32.43%。这只是任务映射推算，本组没有恰好 H=12 的计时。

三组原有的 48、16、80 个 output/final-state 比较共 144 项全部与各自原版逐位一致，覆盖非零初始状态、尾 chunk、变长序列、T4096，以及 persistent 的 600 条任务跨任务复用。它们验证实现一致性，不能独立证明两者都符合数学参考。

新增门控回归检查直接读取 K1 workspace 中的 GT：令 g=A_log=dt_bias=0、chunk 长度为 16，预期 GT=exp(-5×16/2)=4.248354255×10⁻¹⁸。该检查不以测试封装的原版作为参考，旧的直接传 -5 会得到约 9.094947018×10⁻¹³，因而无法通过。

三组共 11 项门控检查全部通过，最大相对误差为 5.48e-06（阈值 1e-4）。完整性能记录共 206 条。

##### 4.2 value 列分片

对 value 列集合 J，R_J、U_J、O_J、S'_J 只依赖对应的 S_J 和 V_J，不需要其他 value 列的归约。将原来的 4 个计算 warp 分给 2/4 个 CTA，每个 CTA 保留完整时间递推、相同 MMA 顺序和 BF16 舍入。输入和完整状态仍由 TMA 加载，应用 shared storage 保持 98,432 B/CTA；因此会重复搬运数据，尚未实现紧凑分片缓存。

写出改为 store warp 协作写回，split1 使用相同写出方式作控制组。必须以 original 为主要基线，不能把相对较慢 split1 的改善全部算作对原实现的收益。

以下为 K2 时间（µs），最后一列为完整 K1+K2 加速比。

| 场景 | 链数 | 原 K2 | split1 | split2 | split4 | split4 K2 加速比 | split4 K1+K2 加速比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one_long | 1 | 362.720 | 1708.480 | 468.576 | 304.725 | 1.190× | 1.195× |
| few_long | 8 | 365.781 | 1711.904 | 468.373 | 305.333 | 1.198× | 1.189× |
| 64_chains | 64 | 99.445 | 446.933 | 131.243 | 100.597 | 0.989× | 0.990× |
| 148_chains | 148 | 101.536 | 451.328 | 168.043 | 197.803 | 0.513× | 0.625× |
| 256_chains | 256 | 138.784 | 498.261 | 325.739 | 382.475 | 0.363× | 0.488× |
| many_short | 256 | 13.547 | 60.320 | 57.291 | 67.488 | 0.201× | 0.290× |
| equal_64 | 64 | 51.861 | 234.827 | 73.899 | 60.053 | 0.864× | 0.903× |
| skew_64 | 64 | 355.189 | 1664.875 | 456.267 | 330.336 | 1.075× | 1.084× |

单链和 8 链的 split4 K2 加速比分别为 **1.190×/1.198×**；148/256 链分别为 **0.513×/0.363×**。对应完整 forward，单链为 375.573→314.272 µs，8 链为 387.435→325.835 µs。

少链长序列有扩大工作分布的机会；多链、短任务的重复加载和资源成本则构成反例。equal_64 和 skew_64 同为 8 序列×8 heads、总长度 4096，仅长度分布不同；后者的长链决定尾部，不能把该结果解释成已实现动态调度。

##### 4.3 两个 head 共 CTA

同序列的相邻两个 head 放入一个 CTA，每个 head 保留独立 state、输入/输出流水线、mbarrier 和计算 named barrier。线程数由 192 增至 384，应用 shared storage 由 98,432 增至 196,864 B，CTA 数从 P 降为 P/2。未共享不同 head 的 GEMM 操作数，也未压缩加载/写出角色，只支持偶数 H；奇数 H 明确返回错误。

四 head 直接复制需要 393,728 B，超过本机 232,448 B/block 的 opt-in 上限，因此没有测试该配置。

| 场景 | 原 K2 µs | 双 head µs | K2 加速比 | K1+K2 加速比 |
| --- | --- | --- | --- | --- |
| two_long | 620.011 | 897.259 | 0.691× | 0.704× |
| few_long | 622.336 | 900.597 | 0.691× | 0.708× |
| 64_chains | 167.979 | 238.859 | 0.703× | 0.758× |
| 148_chains | 171.125 | 240.085 | 0.713× | 0.797× |
| 256_chains | 238.229 | 241.483 | 0.987× | 0.989× |
| many_short | 22.379 | 22.837 | 0.980× | 0.985× |
| equal_64 | 87.253 | 126.869 | 0.688× | 0.753× |
| skew_64 | 611.648 | 875.339 | 0.699× | 0.714× |

few_long 的耗时变化为 44.71%。各场景是否存在明确收益应结合下面的原始采样范围判断；相交区间不能据中位数的小幅差异声称稳健加速或退化。

| 场景 | 原 K2 采样区间 µs | 双 head 采样区间 µs |
| --- | --- | --- |
| few_long | 621.707–624.640 | 899.776–901.621 |
| 256_chains | 237.419–240.224 | 240.992–241.760 |
| many_short | 21.728–25.557 | 22.379–25.387 |

直接合并 head 减少了可分配到不同 SM 的 CTA，是“合并 head 一定提高并行度”的反例；这不排除进一步复用缓冲、共享角色或交错流水线的设计。

##### 4.4 跨链 persistent worker

worker 数取 min(P,148/296)。任务顺序为 seq=job%N、head=job/N；静态版按 job+=gridDim.x 前进，动态版完成当前完整链后用 atomicAdd 领取下一条链。每次切换等待 TMA 写出、处理旧 barrier，再初始化下一条链的 state 和流水线。动态队列初始化计入 K2 时间。本实验是软件队列，不是硬件 CLC，也没有拆开时间上依赖的 chunk。

以下为 K2 时间（µs）。

| 场景 | 链数 | 原 K2 | static148 | dynamic148 | static296 | dynamic296 |
| --- | --- | --- | --- | --- | --- | --- |
| many_equal | 592 | 48.789 | 68.672 | 73.973 | 49.024 | 52.352 |
| long_first | 592 | 338.165 | 1214.048 | 339.253 | 614.208 | 328.107 |
| long_last | 592 | 348.608 | 1216.651 | 357.323 | 627.061 | 339.445 |

long_first 的静态 148-worker 为 1214.048 µs，动态为 339.253 µs，但原 grid 已为 338.165 µs。四条长链编号为 0、148、296、444，恰好全部落在 static148 的 worker0；动态队列纠正这种分配冲突，不等于相对原 grid 获得相同倍率的收益。long_last 反转长序列位置作对照。

| 场景 | dynamic296 K2 加速比 | dynamic296 完整加速比 | 原完整采样区间 µs | dynamic296 完整采样区间 µs |
| --- | --- | --- | --- | --- |
| many_equal | 0.932× | 0.942× | 81.024–84.053 | 85.408–87.509 |
| long_first | 1.031× | 1.020× | 369.856–373.845 | 360.960–367.776 |
| long_last | 1.027× | 1.013× | 380.608–385.685 | 374.752–381.408 |

本轮 long_first 的 dynamic296 完整执行加速为 1.020×，9 轮样本区间与原版分离；long_last 为 1.013×，区间存在重叠。这是特定输入的局部证据，不能扩大成普遍收益。

任务不足时 worker 最多执行一条链，persistent 不能创造独立工作；任务充分时，固定 worker 数又可能限制并发。当前静态步长不适合作为通用替代；动态队列应与原 grid 比较完整执行时间，不能只选择静态 persistent 作基线。

##### 4.5 NCU 证据与结论边界

| 实验/场景 | 路径 | CTA | 寄存器/线程 | shared B/CTA | occupancy %（active） | SM throughput %（elapsed） |
| --- | --- | --- | --- | --- | --- | --- |
| column_split/few_long | original | 8 | 72 | 99456 | 9.36 | 1.62 |
| column_split/few_long | split4 | 32 | 96 | 99456 | 4.68 | 4.01 |
| column_split/256_chains | original | 256 | 72 | 99456 | 16.38 | 36.97 |
| column_split/256_chains | split4 | 1024 | 96 | 99456 | 8.45 | 26.39 |
| multi_head/few_long | original | 8 | 73 | 99456 | 9.36 | 1.74 |
| multi_head/few_long | heads2 | 4 | 74 | 197888 | 18.72 | 1.22 |
| persistent/long_first | original | 592 | 73 | 99456 | 15.57 | 2.74 |
| persistent/long_first | static148 | 148 | 77 | 99584 | 9.42 | 0.79 |
| persistent/long_first | dynamic148 | 148 | 77 | 99584 | 9.33 | 2.85 |

NCU 的 shared 分配量与应用结构体大小口径不同；active occupancy 也不等于全卡覆盖率。平均 SM throughput 不能还原每时刻活跃 SM 数。full/source 分别保留指标、PM 采样和源码热点；PM 按采样序号分箱，不作为精确的 SM 利用率时间线。不可用或返回零的派生 DRAM 指标不用于判断带宽瓶颈。

| 候选方案 | 机会 | 必须面对的反例与边界 |
| --- | --- | --- |
| value 列分片 | 增加少链的可调度 CTA | 多链/短任务重复搬运，当前缓存未缩小；应按形状选择 |
| 多 head 共 CTA | 进一步复用缓冲或摊销角色 | 当前直接复制资源会减少 CTA、扩大 shared；不能由更高 occupancy 推导更快 |
| Persistent | 跨链复用、处理任务倾斜 | 原 grid 已有硬件调度；单长链不可拆，固定 worker 可能限制并发 |
| 协作 2-CTA | 协作计算和加载 | 2026-09-11 新增 DSM cluster2 独立 GEMM 对照；5 个实际形状、6 档 batch 下均未测到中位数收益，详见 03 报告 4.6；非完整 K2 或 tcgen05.cta_group::2 |

以上前三组是完整递推实现的证据；协作 2-CTA 的独立 GEMM 实现和数据见下节。不能将不同计算范围的绝对时间直接比较。

##### 4.6 CHUNK16 实际形状的协作 2-CTA microbench

**结论：本次实现的 DSM 共享 A 协作版，在 5 种形状 × 6 档任务数的 30 个条件中，中位数均慢于本组单 CTA 和独立 split2。** 这是 2026-09-11 在 B300（CC10.3、148 SM）新增的独立 GEMM 实验，不是完整 K2，也不是 `tcgen05.cta_group::2`。前三组生产递推测试及其历史数据保持原样。

###### 4.6.1 实际 KDA 形状与接口

固定 CHUNK=16、D=128，只改变独立任务数 `batch∈{1,8,64,148,256,16384}`，覆盖少任务、接近 SM 数、多任务和大批量。batch 不是序列长度，不将相邻 chunk 当作可独立执行的状态任务。

| shape ID | KDA 中的计算 | M,N,K | 输入 / 累加 |
| --- | --- | --- | --- |
| 0 | Neumann 中的一次乘法 | 16,16,16 | FP16 / FP16 |
| 1 | K1 的 KD×KIᵀ 或 QD×KIᵀ | 16,16,128 | BF16 / FP32 |
| 2 | K2 的 INV×R 或 Mqk×U | 16,128,16 | BF16 / FP32 |
| 3 | K2 的 KD×Sᵀ 或 QD×Sᵀ | 16,128,128 | BF16 / FP32 |
| 4 | K2 的 KRᵀ×U（状态更新的转置方向） | 128,128,16 | BF16 / FP32 |

接口为 `run_cooperative(shape, mode, batch, A, B, output, stream)`；shape 选择上表的预编译实例，不改变 CHUNK。A/B 是相同的 8×8 interleave 打包输入，B 按转置视图打包，输出统一为 FP32。尚未融合 beta、gate、mask、残差、状态衰减和中间格式转换，不执行完整 Neumann 级数。

###### 4.6.2 三条对照路径

- `single_cta`：一个 CTA 完成一个有效 GEMM，最多 4 个计算 warp。
- `independent_split2`：两个独立 CTA 沿输出 N 维各算一半，最多各 2 个 warp，分别从 global 加载同一 A 及自己所需的 B 分片。K2 中 N 对应 value 通道；这是本微基准的独立分片控制组，不是历史 production split2 kernel。
- `cooperative_cluster2`：相同 N 分片，但两个 CTA 组成实际的 2-block cluster。rank0 加载 A，rank1 通过 `cluster.map_shared_rank` 读取 rank0 的 A 到自己的 shared buffer；两者各自加载 B 分片。第一次 `cluster.sync()` 保证 A 已就绪、两个 block 都存活，第二次保证远程读取结束后源 block 才能退出。随后各自 MMA 并写回不重叠的输出分片。

三条路径都使用相同 `mma.sync.m16n8k16`、相同每元素 K 归约顺序，不通过换 Tensor Core 指令引入额外变量。对于很小的形状，warp 数按输出 tile 数缩小；single 与两个 split CTA 的总计算 warp 数一致。N=16 直接分成两个 N8，没有 padding。

该协作设计明确尝试用 DSM 传输替代重复的 A global load：逻辑 global 输入量从 independent 的 `2|A|+|B|` 降到 `|A|+|B|`，但增加 `|A|` 的 DSM 传输、两个 cluster barrier 和 cluster 调度要求。各 CTA 仍保留自己的 A shared buffer，不声称节省每 CTA shared memory；缓存也会影响实际 HBM 流量，不能把逻辑加载减少等同于 HBM 字节同比减少。代码用 CUDA cluster API 实现的软件数据协作，不保证两个 CTA 必须落在不同 SM，也不应写成已验证 2-SM tcgen05。

同步依据：[CUDA 13.0 Programming Guide — Distributed Shared Memory](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#distributed-shared-memory)。要求访问远端 shared 时源 block 仍存活，并在退出前完成远端访问。

###### 4.6.3 输入、对拍和计时

输入采用 `randn×0.125` 后量化到对应类型；这是匹配 KDA 形状的独立 GEMM 输入，不是从真实模型抽出的 K1 workspace。正确性覆盖 3 个随机 seed、全零 A、带交替符号的结构化矩阵，batch=3。5 形状×5 输入×3 路径共 75 项检查，对照同一量化输入的 FP64 GEMM；FP16 相对 L2 阈值 0.003，BF16/FP32 阈值 2e-5。所有输出有限，且分片与协作结果均与单 CTA 逐位一致。计时的所有 batch 也检查输出有限和逐位一致。

每路径预热 5 次；CUDA Graph 内重复 3 次，图再预热回放 3 次；CUDA events 计时 9 轮，轮换路径顺序，报告除以图内调用次数后的中位数。输入打包与 buffer 分配不计时；global→shared、DSM 传输、cluster 同步、MMA 和写回均计时。结果是完成整个 batch 的一次 kernel 耗时，未除以 batch。输入和输出 buffer 重复使用，属于热缓存条件；没有模拟 K2 的状态更新和逐 chunk 供数。原始 90 条性能记录及 9 轮样本保存在 `cooperative_2cta/RESULTS.md`。

Compute Sanitizer 的 memcheck、racecheck、synccheck 分别检查相同正确性用例，三项均通过，memcheck/synccheck 为 0 errors，racecheck 为 0 errors、0 warnings，输出保留于 `cooperative_2cta/validation/`。本组不采集 NCU，不从计时结果推断带宽瓶颈或同步开销占比；提供 `--profile` 入口供后续采集。

###### 4.6.4 代表性结果

单位 μs；single/new、split2/new 均针对协作版，>1 才代表协作更快。

| 计算 | Batch | single | split2 | cluster2 | single/cluster2 | split2/cluster2 |
| --- | --- | --- | --- | --- | --- | --- |
| neumann_product | 1 | 3.8400 | 4.3947 | 7.1253 | 0.539× | 0.617× |
| neumann_product | 8 | 4.4373 | 4.5653 | 7.3173 | 0.606× | 0.624× |
| neumann_product | 16384 | 19.3707 | 26.9867 | 147.4133 | 0.131× | 0.183× |
| k1_l_mqk | 1 | 7.5307 | 10.0800 | 23.7973 | 0.316× | 0.424× |
| k1_l_mqk | 8 | 8.0960 | 10.5707 | 25.6960 | 0.315× | 0.411× |
| k1_l_mqk | 16384 | 76.9600 | 129.1840 | 830.3146 | 0.093× | 0.156× |
| k2_inv_r_mqk_u | 1 | 5.8880 | 5.1733 | 7.0827 | 0.831× | 0.730× |
| k2_inv_r_mqk_u | 8 | 5.8133 | 5.3867 | 7.7867 | 0.747× | 0.692× |
| k2_inv_r_mqk_u | 16384 | 85.3333 | 84.3307 | 219.1680 | 0.389× | 0.385× |
| k2_state_projection | 1 | 14.1013 | 15.5520 | 22.9013 | 0.616× | 0.679× |
| k2_state_projection | 8 | 14.7840 | 16.1280 | 24.2773 | 0.609× | 0.664× |
| k2_state_projection | 16384 | 592.8320 | 679.4560 | 990.4747 | 0.599× | 0.686× |
| k2_state_update | 1 | 11.3920 | 12.0533 | 19.4667 | 0.585× | 0.619× |
| k2_state_update | 8 | 11.5840 | 12.6187 | 20.8320 | 0.556× | 0.606× |
| k2_state_update | 16384 | 684.2027 | 718.9227 | 1174.4960 | 0.583× | 0.612× |

当前协作实现未测到正收益，独立 split2 在部分修正量乘法上反而更快；因此“增加两个 CTA”与“需要跨 CTA 协作”不是同一件事。这是一个可复现的反例，而非对协作 2-CTA 全部设计的否定。不能用本组单 CTA 的绝对耗时对比 4.2–4.4 的完整递推时间，也不能把单次 GEMM 加速比直接当作完整 KDA 加速比。

###### 4.6.5 复现与文件

- `harness/cooperative_2cta.cu`：三条 kernel 路径与统一 C 接口。
- `harness/cooperative_2cta.py`：shape 表、构建、正确性、计时及结果保存；`--batches` 接收独立任务数列表，`--check-only` 仅做对拍。
- `cooperative_2cta/RESULTS.md`：完整结果、原始样本和 GPU 环境。
- `cooperative_2cta/SOURCE.json`：源码、bench、二进制哈希和编译命令。
- `cooperative_2cta/validation/`：memcheck、racecheck、synccheck 日志。

```bash
bash profile/03_recurrence_parallelism/harness/run.sh cooperative_2cta
```

该组先构建，依次执行三种 sanitizer，再执行独立计时；复测前自动将已有该组结果和验证记录复制到 `cooperative_2cta/archive/`，不会覆盖前三组数据。直接调用 Python 计时入口会重写本组 `RESULTS.md`，需要保留历史时应使用上述 shell 入口。`all` 已包含此新增组；该组保留编译库供检查和后续采集，不执行前三组的 full/source NCU 流程。

##### 复现与数据索引

| 组 | 结果文件 | 额外证据 | 执行入口参数 |
| --- | --- | --- | --- |
| value 分片 | [column_split/RESULTS.md](03_recurrence_parallelism/column_split/RESULTS.md) | `PROFILE.md`、`reports/full.ncu-rep`、`reports/source.ncu-rep`、`SOURCE.json` | `column_split` |
| 双 head | [multi_head/RESULTS.md](03_recurrence_parallelism/multi_head/RESULTS.md) | 同上，位于对应组目录 | `multi_head` |
| persistent | [persistent/RESULTS.md](03_recurrence_parallelism/persistent/RESULTS.md) | 同上，位于对应组目录 | `persistent` |
| 协作 2-CTA | [cooperative_2cta/RESULTS.md](03_recurrence_parallelism/cooperative_2cta/RESULTS.md) | `SOURCE.json`、`validation/memcheck.log`、`racecheck.log`、`synccheck.log`；无本次 NCU 数据 | `cooperative_2cta` |

前三组共 144 项 output/final-state 对照、11 项门控检查、206 条性能记录；第四组另有 75 项独立 GEMM 对拍和 90 条性能记录。这两类检查对应的计算范围不同，不合并成“完整 KDA 正确性通过次数”。

从 FlashKDA 根目录运行：

```bash
# 只运行新增协作组
bash profile/03_recurrence_parallelism/harness/run.sh cooperative_2cta
# 运行四组
bash profile/03_recurrence_parallelism/harness/run.sh all
```

前三组从生产源码生成隔离副本，由 `build.py`、`bench.py`、`analyze.py` 构建、实测并解析 NCU；新组使用 `cooperative_2cta.cu/.py`，先执行三种 sanitizer，再计时。生产 `csrc` 不修改。新组复测前自动归档已有本组结果；前三组脚本仍会更新对应组结果，保留历史时需先归档。全部结果文件包含原始样本和环境记录。

<a id="04_compute_memory_bottlenecks"></a>

## 计算与访存瓶颈

### 指定负载：N=1、T=8192、H=96、D=128

**结论：这个负载的主要耗时在 K2，表现为状态链并行度不足，以及递推内部的依赖和片上供数等待；不支持“整体 HBM 带宽已饱和”或“Tensor 算力已饱和”的判断。K1 则是指令发射、片上数据访问与同步的混合限制。**

用户指定 T=8192、H=96、D=128；未指定 batch，本次取 N=1。CHUNK 固定为 16。本目录只保留这一种正式负载的测试、报告和原始 NCU 数据。T=16/31 仅用于小规模正确性检查，不作为其他性能负载。

#### 实验设计

2026-09-11 在 NVIDIA B300 SXM6 AC（148 SM、CC 10.3）运行。使用 CUDA 13.0、Nsight Compute 2025.3.1，将当前仓库原始 `csrc/smxx/fwd_launch.cu` 与 host bridge 编译为独立动态库，保留 `-lineinfo`，目标 `sm_103a`。没有修改 GPU kernel。完整编译命令、源文件 SHA256 和二进制 SHA256 见 [source.json](04_compute_memory_bottlenecks/analysis/source.json)。

调用 `launch_fwd<128,true,true,false,false>`：D=128、提供初始状态、输出最终状态、BF16 状态、等长序列。Q/K/V/g 和状态为 BF16；随机 seed=42，Q/K/V 为标准正态乘 0.125，g/beta 为标准正态，初始状态为标准正态乘 0.02，A_log/dt_bias 为 0，lower_bound=-5。这是指定 shape 的合成输入测量，不是模型真实张量回放。

输入分配、beta 转置和参考检查在计时前完成。先运行 20 次完整 K1→K2；独立计时为 CUDA events 测 9 组×20 次 forward，取每次 forward 耗时的中位数。NCU 使用 application replay、cache-control none、clock-control none，full+PM 和 source 两次采集均只覆盖第 21 次 forward；每份报告准确包含 K1、K2 两个 action，并核对名称及 CTA 数。该协议测预热后的原始 kernel，不包含输入投影 `in_proj_qkvgfab` 或 Python 公开接口的分配与布局转换。

#### 实测结果

| Kernel | CTA 数 | NCU 耗时 µs | Compute/SM % | Tensor % | L1/TEX % | L2 % | DRAM % | DRAM 读写 GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | 49152 | 456.896 | 73.22 | 5.87 | 71.04 | 54.90 | 35.77 | 2743.83 |
| K2 | 96 | 1291.552 | 20.89 | 19.62 | 38.23 | 19.92 | 11.32 | 868.35 |

所有吞吐百分比统一采用 `pct_of_peak_sustained_elapsed`。Compute/SM 是 `sm__throughput`，本次最大分项为指令发射 `sm__issue_active`，不能解释成 Tensor FLOPS 使用率。L1/TEX 最大分项为 `l1tex__data_pipe_lsu_wavefronts`；Tensor 单列使用 `sm__pipe_tensor_cycles_active`。L2 使用 `lts__throughput`。DRAM 使用有效的 `dram__cycles_active`，与本次 DRAM 读、写字节吞吐百分比之和一致；GB/s 为读写之和，采用十进制。

独立完整 forward 中位数 **1733.864 µs（1.734 ms）**，见 [RESULTS.md](04_compute_memory_bottlenecks/RESULTS.md)。NCU 分 kernel 时间合计 1748.448 µs，其中 K2 占 **73.87%**；NCU 重放与独立计时口径不同，不要求二者完全相等。

#### 判断依据

##### 1. 并行度与 occupancy

K1 每个 chunk/head 一个 CTA，grid=8192/16×96=49152，41.51 waves/SM，achieved occupancy 96.87%，32 registers/thread、22272 B shared/CTA。其低 Tensor 利用率不能归因于简单的 CTA 数不足。

K2 每个 batch/head 一个 CTA，只有 96 条状态链，每条链顺序处理 512 个 chunk。96 个 CTA 最多覆盖 96/148=64.86% 的 SM；增加 T 只延长每条链，不增加 grid。NCU 给出 small-grid 提示，waves/SM=0.324。每 CTA 192 threads、73 registers/thread、99456 B shared，shared 限制最多 2 CTA/SM，理论 occupancy 18.75%；当前只有一轮且 CTA 数不足，实测 active occupancy 9.37%。每 scheduler 仅 0.346 个 eligible warps/cycle，active issue 32.63%。这些证据说明可并行的工作量与延迟隐藏不足，是 K2 的关键限制。

##### 2. 等长负载与活动周期分布

所有 head 都有 512 个 chunk，没有变长输入造成的工作量差异。K1 的 SM active cycles min/avg/max 为 488201/492647/495154；K2 为 0/901415/1404859。K2 的零值与存在空闲 SM 一致，不能直接把全卡 min/max 差异称为长尾负载不均衡。

##### 3. 源码等待与 Tensor 利用率

| Kernel | barrier % | short scoreboard % | wait % | long scoreboard % | sleeping % |
|---|---:|---:|---:|---:|---:|
| K1 | 40.65 | 13.49 | 9.11 | 10.70 | 0.00 |
| K2 | 3.47 | 17.74 | 26.51 | 8.35 | 10.81 |

分母为 source 报告全部 stall 类别样本之和，包含 selected/not_selected，排除重复的 not_issued 指标。这是采样比例，不是运行时间占比或可直接消除的时间。

K1 的 barrier 热点映射到 [fwd_kernel1.cuh:629](../csrc/smxx/fwd_kernel1.cuh#L629) 的 EXIT（2456 样本）和 [395 行](../csrc/smxx/fwd_kernel1.cuh#L395) 的 LDCU（1907 样本）。编译后的同步等待可以归到后继指令，不能据此认为 EXIT 本身消耗了这些同步时间。结合 SM issue 73.22%、L1/TEX 71.04% 与 Tensor 5.87%，支持混合限制判断。

K2 的 short scoreboard 包含 [583 行](../csrc/smxx/fwd_kernel2.cuh#L583) 和 [737 行](../csrc/smxx/fwd_kernel2.cuh#L737) 的 BF16 HMMA（分别 454、401 样本）；[881 行](../csrc/smxx/fwd_kernel2.cuh#L881) 的 `WARPSYNC.ALL` 有 3314 个 sleeping 样本。wait 26.51% 与 short scoreboard 17.74% 支持计算依赖和片上供数等待的判断，不能把它们全部解释为 HBM 等待。Tensor elapsed 19.62% 也没有显示全卡 Tensor 饱和。

##### 4. PM 采样

full 报告保留 14 类 PM 序列。K2 的 wait/short-scoreboard 在有效采样区间形成持续平台，与循环内持续等待一致。当前序列是 warp 状态而非 SM 利用率时间线，采集窗口两端还有零值；不据此计算尾部耗时或空闲 SM 比例。原始序列分箱见 [分析结果](04_compute_memory_bottlenecks/analysis/RESULTS.md)。

##### 5. 访存层级与访问模式

K1 的 L1/TEX 71.04% 明显高于 DRAM 35.77%，shared load/store bank conflicts 分别为 5046979/1678770；NCU 将其估计为 shared load/store wavefront 的 18.78%/10.16%。K2 的 L1/TEX 38.23%、L2 19.92%、DRAM 11.32%，并未显示全卡某级带宽饱和；其中 L1/TEX active 为 59.74%，不要与主表 elapsed 混用。

K1/K2 L2 hit 分别为 42.08%/51.43%。普通 LSU global 指标不覆盖所有 TMA 流量：K2 的普通 LSU global requests 为 0，但实测 HBM 读写仍有 868.35 GB/s。因此不能用普通 global 请求数为 0 推断“没有访存”。K1 普通 global load 的 sectors/request=1，NCU 同时提示每线程每 sector 使用 4 B；这个提示本身不足以证明全 kernel 合并访问差。两 kernel 的 local load/store sectors 均为 0，当前未观察到 local-memory spill 流量。

##### 6. NCU 规则与后续优先级

[规则输出](04_compute_memory_bottlenecks/analysis/rules.txt) 对 K2 给出 small-grid 估计 35.14%、occupancy 相关估计 61.77%，对 K1 barrier 给出 local speedup 41.85%，shared load/store 冲突估计 13.48%/7.292%。这些是规则模型提示，互相重叠，不能相加或承诺为实测收益。

后续应优先在同一指定负载上验证 K2 沿独立 value 列拆分是否能增加 CTA 并缩短递推时间；它必须权衡重复供数和同步成本。其次分析 K2 供数与依赖隐藏，再考虑 K1 shared 布局及同步。当前只完成原始实现测量，没有把 profile03 的其他 shape 或 2-CTA microbench 收益转移到这里。

#### 正确性与复现

调用仓库 `tests/torch_ref.py`，N/H/T=1/2/16 和 2/2/31 的输出、最终状态 relative L2 均为 0。正式 N=1/H=96/T=8192 的输出与状态均为有限值；正式大 shape 未做完整参考对拍，这里不声称它有零误差。

入口为 [harness/run.sh](04_compute_memory_bottlenecks/harness/run.sh)。脚本依次编译、运行独立计时、采集 full/source、调用 NVIDIA `ncu_report` 解析；已有 rep 时拒绝覆盖。再次测量应复制为新的 run 目录并调整路径。PATH 需要虚拟环境、CUDA 与 NCU 的 bin，首次检查参考实现还会调用 Ninja 编译扩展。

- [完整计时及环境](04_compute_memory_bottlenecks/RESULTS.md)
- [指标 JSON](04_compute_memory_bottlenecks/analysis/metrics.json)、[指标及源码采样 Markdown](04_compute_memory_bottlenecks/analysis/RESULTS.md)
- [full NCU](04_compute_memory_bottlenecks/reports/full.ncu-rep)、[source NCU](04_compute_memory_bottlenecks/reports/source.ncu-rep)
- [源文件与编译溯源](04_compute_memory_bottlenecks/analysis/source.json)、[场景及 CTA 校验](04_compute_memory_bottlenecks/analysis/cases.json)

full 采集提示 6 个 CTC 收发指标不可访问；本报告结论使用的 SM、Tensor、L1/L2、DRAM 和源码指标均已取得，不依赖 CTC 指标。

#### 目录整理与采集溯源

本次测量已统一迁入 `profile/04_compute_memory_bottlenecks`，替换原四负载实验；独立补测目录已移除。测试脚本均使用当前路径。原始 rep、日志、独立计时和 `analysis/source.json` 中的采集路径保留测量时原值；这是历史溯源，不是运行入口。迁移未重新编译或测量，原始二进制及 rep 的 SHA256 保持不变。

纸面算术强度及瘦 GEMM 对照见 [METHOD.md](04_compute_memory_bottlenecks/analysis/METHOD.md)；当前 CHUNK=16、每链 L=512，K2 逻辑矩阵算术强度约 **76.837 FLOP/B**，仅用于提出假设，不代替硬件计数器。

<a id="05_bf16_state_accuracy"></a>

## BF16 状态存储精度

### 05：指定负载的 BF16 状态精度验证

#### 实验目标

官方 Numerical Precision 说明，K2 的状态更新使用 FP32 FMA，但在两次更新之间将状态保存为 BF16，其内部推理 benchmark 未测到精度损失。本实验检查：**每个 chunk 更新后保存 BF16，相比保留 FP32，会给状态和输出带来多少差异，以及哪些输入在指定长度下会产生明显差异。** 这里的状态是 K2 跨 chunk 传递的历史状态矩阵，不是前面测试的指数或逆矩阵。

#### 实验设计

**1. 用 PyTorch 实现 K2 的计算逻辑作为 microbench，并与实际 K2 kernel 对齐。**

先运行原始 K1→K2，取出真实 K1 workspace 中的 `kd、qd、kr、gt、inv、mqk`，同时保存原始 K2 的输出和最终状态。microbench 接收这些中间结果，以及同一份 V、beta 和初始状态，用 PyTorch GPU 运算逐 chunk 执行 K2 的计算：

$$
R=(V-K_dS)\odot\beta,\quad
U=INV\,R,\quad
Y=Q_dS+M_{qk}U,\quad
S_{new}=G_T\odot S+K_r^TU.
$$

其中 beta 已经过 sigmoid；公式表示计算顺序，实际代码也保留原 K2 对中间结果的 BF16 舍入。矩阵乘法通过 `torch.bmm` 执行，状态更新复用仓库的 FP32 FMA 辅助函数。它复现数值计算，不复现原 kernel 的线程调度与 shared memory 流水线。

先让 microbench 每个 chunk 更新后保存 BF16，将其输出和状态与原始 K2 比较，要求最大绝对差异为 0。完成 8192 个 token 后，将全部输出和最终状态与同一输入、同一初始状态的原始 K2 比较。只有对齐后，才分析状态精度差异。

**2. 在同一 microbench 中分别保存 BF16、FP32 状态，比较结果和更新过程。**

两种方式共用输入、初始状态、K1 中间结果和计算流程：

| 项目 | 每次保存 BF16 | 每次保留 FP32 |
| --- | --- | --- |
| 初始状态 | 同一份 BF16 初始状态 | 同左 |
| GEMM 操作数 | BF16 | BF16，计算时临时转换状态副本 |
| 状态更新 | FP32 FMA | FP32 FMA |
| 更新后保存 | 舍入 BF16，供下一 chunk 使用 | 保留 FP32，下一次更新仍使用它 |

FP32 方式做矩阵乘法时只生成临时 BF16 副本，不覆盖保存的 FP32 状态。这样只改变状态保存精度，不同时提高 gate、求逆和 GEMM 的精度。

原 K2 的状态更新会直接执行 `BF16(旧状态 * gate + 增量)`，再写回 shared memory。使用 PyTorch microbench，方便同时保留更新前、FP32 更新后、转 BF16 后的值，并插入以下比较：

| 比较内容 | 要回答的问题 |
| --- | --- |
| 两种保存方式的全部前缀输出 | 状态保存差异是否影响实际输出，影响多大？ |
| 两种保存方式在同一长度处的状态 | 历史状态是否已产生差异？ |
| 将 FP32 最终状态只在读出时转一次 BF16，再与 BF16 路径比较 | 差异是否仅来自最终保存格式，还是之前的递推已经产生差异？ |
| 每个 chunk 的 FP32 更新结果与旧状态 | 每次状态变化实际有多大？ |
| FP32 更新改变了值，但转 BF16 后又等于旧值的元素数 | 有多少非零变化被保存操作完全舍掉？ |

固定序列长度为 8192，比较不同输入场景下的这些指标。microbench 的作用是方便切换保存方式和记录中间数值；输入场景仍由输入生成代码和真实 K1 构造。直接修改原 K2 并增加中间值导出也能完成这些比较。本实验不使用 microbench 的时间代表原 K2 性能。

#### 输入设计：三个场景及其目的

已有实验在 B300 SXM6 AC（148 SM）上执行，固定 N=1、H=96、D=128、CHUNK=16。q/k 为 BF16 正态输入，由真实 K1 归一化；v 为正态数×0.125；`A_log=dt_bias=0`、`lower_bound=-5`。这是精度实验的输入范围，与第 4 节性能负载分开。

状态更新可以理解为“旧状态乘一个系数，再加本次更新”。下面的保留系数指每个 chunk 更新时旧状态乘以多少，例如 0.9999 表示旧状态这一项保留 99.99%，不表示最终新状态一定等于旧状态的 99.99%。

| 场景 | 如何生成输入 | 为什么需要这个场景 |
| --- | --- | --- |
| 1. 随机输入 | g/beta 正态；初始状态为零或正态数×0.02；每种初始状态使用 10 个 seed | 检查普通随机输入，以及从零开始和带非零状态开始时的差异。但随机 gate 可能使旧状态迅速变小，仅测随机输入可能看不到累积问题 |
| 2. 每次保留较多旧状态 | 保留系数目标 0.9、0.99、0.999、0.9999；beta 正态；初始状态正态数×0.02；每档 3 个 seed | 让旧状态对后续计算持续产生影响，检查保存误差是否随更新次数增加 |
| 3. 保留较多旧状态，每次变化又很小 | 保留系数目标 0.9999；beta 在 sigmoid 前的输入为 -5、-7、-9；初始状态正态数×1；每档 3 个 seed | BF16 在 1 附近的间隔为 0.0078125，`1→1.001` 可能保存后仍为 1；这组专门检查细小变化是否反复被舍掉 |

场景三同时调整 beta 和初始状态大小，是为了构造“旧状态较大、每次变化较小”的输入，不能把它与场景二的差异全部归因于 beta。是否真的形成小变化，要根据实际更新幅度核对。

对于保留系数目标 $r$，输入生成代码使用

$$
p=-\ln(r)/(5\times16),\qquad g=\ln(p/(1-p))
$$

构造 gate 输入，再交给真实 K1。实际保留系数从 K1 workspace 读取，不假定目标值能精确实现。

所有输入都固定为 **T=8192、H=96、D=128**，每条链更新 512 个 chunk，仅在完成 8192 个 token 后比较完整输出和最终状态。共 20 组随机输入、12 组场景二输入、9 组场景三输入，合计 **41 组输入、41 次完整负载检查**。

#### 实测结果与分析

本次只测 **N=1、T=8192、H=96、D=128、CHUNK=16**，每条状态链更新 512 次。41 组输入均在完整 8192 个 token 计算结束后比较；没有测试其他长度。41 组 microbench 的 BF16 输出和最终状态与原始 K2 的最大绝对差异全部为 **0**，所有结果均为有限值。

**指标含义。** 输出相对 L2 是两种保存方式的全部 8192 个 token 输出差值范数除以 FP32 保存方式的输出范数；状态相对 L2 则比较计算结束后的状态。“统一 BF16 格式”表示将 FP32 最终状态仅在读出时转一次 BF16，再比较，以排除单纯最终格式不同的影响。非零更新丢失比例表示 FP32 更新改变了值，但保存 BF16 后又等于旧值的次数比例；它不是输出误差百分比。所有误差统计使用 FP64。

**场景一：随机输入。** 零初始状态和非零初始状态各 10 个 seed，输出相对差异均为 **0**。直接比较不同保存格式的最终状态，相对差异为 **0.1656%～0.1662%**；统一最终 BF16 格式后差异为 **0**。这组随机输入没有观察到输出差异，但不能代表其他 gate 和更新幅度也没有差异。

**场景二：每次保留较多旧状态，beta 随机。** 以下为 3 个 seed 中的最大差异：

| 实际 chunk 保留系数 | 输出相对 L2 差异 | 状态相对 L2 差异（统一 BF16 格式） |
| --- | --- | --- |
| 0.899421692 | 0.5113% | 0.4419% |
| 0.990206361 | 0.7079% | 0.6597% |
| 0.999015808 | 0.7533% | 0.7094% |
| 0.999899983 | 0.7585% | 0.7154% |

旧状态保留得较多时，两种保存方式已产生可测差异，但本组输出差异均低于 0.76%。目标保留系数 0.9999 组实际平均更新/旧状态范数比约 **35.50%～35.53%**，非零更新丢失比例约 **0.54%**；旧状态保留得多，但每次仍有较明显变化。本次只有 T=8192 一个长度，不据此判断差异随长度如何变化。

**场景三：保留较多旧状态，每次变化又很小。** 保留系数目标固定 0.9999。下表的更新幅度和丢失比例为 seed 范围，输出、状态差异为 seed 最大值：

| beta 在 sigmoid 前的输入 | 平均更新/旧状态范数比 | 输出相对 L2 差异 | 状态相对 L2 差异（统一 BF16 格式） | 非零更新丢失比例 |
| --- | --- | --- | --- | --- |
| -5 | 0.2554%～0.2555% | 14.5528% | 30.8353% | 51.9511%～51.9630% |
| -7 | 0.0388%～0.0388% | 6.3993% | 11.5559% | 92.2576%～92.2772% |
| -9 | 0.0123%～0.0123% | 3.4456% | 6.0902% | 98.9662%～98.9781% |

**在题目指定的 T=8192 下，每次变化很小时，BF16 状态保存已经产生明显精度差异。** beta=-5 的最大输出相对差异为 **14.5528%**，统一最终格式后的状态差异为 **30.8353%**；不需要借助更长序列才能观察到这个现象。状态统一最终格式后仍有差异，说明差异不只是最后一次格式转换产生的。

beta=-7 时，约 **92.26%～92.28%** 的非零元素变化被保存操作完全舍掉，输出最大相对差异为 **6.3993%**，最大绝对差异为 **0.0488281**；FP32 最终状态范数为 **1122.942～1125.290**，BF16 为 **1252.259～1254.890**。这说明微小变化未被完整保留，512 次更新后状态和输出均已偏离 FP32 保存方式。

更新丢失比例不能直接当成输出误差：beta=-9 的丢失比例更高，但其每次变化也更小，所以输出差异低于 beta=-5。需要同时看实际更新幅度、输出和状态差异。

#### 文件与复现

数据和代码统一位于 `profile/05_bf16_state_accuracy/`：`harness/bench.py` 保留原 PyTorch 递推与两种状态保存方式，仅将 H 设为 96，所有输入和检查长度固定为 8192；`RESULTS.md` 汇总各 seed 范围，`analysis/DATA.md` 保存全部 41 组原始数据，`analysis/run_8192.log` 保存运行记录，`analysis/figures/` 为固定负载下各输入的结果图。

本次复用了 04 已编译的原始 K1/K2 二进制；已核对当前 csrc 哈希、bridge 内容及二进制校验值一致。编译溯源和复用说明保存在 `analysis/source.json`。`harness/run.sh` 仍可从源码构建并运行；已有原始数据时会停止，避免意外覆盖。

```bash
bash profile/05_bf16_state_accuracy/harness/run.sh
```

#### 结论

**在 N=1、T=8192、H=96、D=128 的指定负载下，BF16 状态保存的精度表现取决于输入。** 随机输入没有观察到输出差异；旧状态保留较多时，输出最大相对差异约为 **0.5113%～0.7585%**；旧状态保留较多且每次变化很小时，三档输入的输出最大相对差异分别为 **14.5528%、6.3993%、3.4456%**。因此，更新使用 FP32 FMA 不能保证 BF16 状态保存对所有输入都无损。

这些结果比较的是两种状态保存方式，并非相对于数学真值或模型任务准确率。实验只测指定长度，不对更长序列或误差随长度变化作结论；官方内部推理 benchmark 的准确率表述仍需相应模型输入与评测指标才能验证。


<a id="06_sm100a_design_decision"></a>

## SM100a专版设计决策

### 06_sm100a_design_decision：SM100a专版设计决策

#### 本目录实验目标

综合数值、指令、并行度、硬件瓶颈和状态精度证据，判断v2是否应另做SM100a专版。

**实验方法：** 汇总前五组与基线结果，比较收益、兼容性和维护成本；本目录没有新GPU测量或独立实验脚本。

**结果与边界：** 当前证据不支持新增专版；B300 sm_103a上的局部收益不能直接当成SM100设备或完整KDA的收益。

脚本、运行顺序和依赖见[profile README](README.md)，全部主题的报告合并见[总报告](REPORT.md)。以下保留原实验的测量和限制说明；本次整理没有重新执行GPU实验。

### Topic6：作为作者，v2 要不要出 SM100a 专版？

**我们的决定：v2 不做 SM100a 专版，继续沿用现有共享实现。现有实验没有证明专用路径能带来足以支撑额外开发和维护成本的完整 KDA 性能收益；引入专用指令又会缩小该路径的设备适用范围，增加构建、分发和正确性验证成本。综合收益与通用性，本轮没有必要另做专版，也不将可选专用后端列入 v2 开发计划。**

理由不是“CHUNK16 用不了 tcgen05”。02_tcgen05_evaluation 已证明可以合并共享操作数或完整转置来匹配形状，且有局部收益。但局部可行性不足以支持专版立项：这些办法尚未在真实 K1→K2 的数据布局、递推、同步和精度约束下证明稳定的端到端收益。topic3、topic4、05_bf16_state_accuracy 分别给出了并行度、片上数据访问和状态精度方面必须处理的限制。

本篇综合已有实验，不新增 GPU 测量。所有既有 GPU 实测来自 **B300、CC 10.3、148 SM**；SM100a 专用后端在 SM100 设备上的性能尚未验证。

#### 1. 先把“专版”和“可移植”说清楚

当前 [setup.py](../setup.py#L19) 已列出 `90a、100a、103a、120a` 编译目标，编译的仍是同一份 `csrc/smxx/fwd_launch.cu`。所以本题不是“要不要添加一个 sm100a 编译选项”，而是“要不要另写 tcgen05/TMEM 等架构专用计算路径，并承担维护和验证成本”。目标列表也不是这些设备都已实测通过的证明。

B300 的实验使用 `sm_103a`，不能直接改名为 `sm_100a` 的性能结果。带 `a` 的架构专用代码不能依赖一般 PTX 的向前兼容保证；SM100 与 SM103 应分别构建并在对应设备验证。CUDA 还提供 family-specific 的 `100f` 目标，可覆盖相应 10.x 家族特性；如果所用指令都属于该特性集合，可评估减少二进制分裂，但仍需核对工具链和具体指令支持，不能直接更换后缀就宣布兼容。[NVIDIA 兼容性指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)、[family-specific 架构说明](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)。

另一方面，原实现虽使用 SM80 `mma.sync`，还依赖 [SM90 TMA](../csrc/smxx/fwd_launch.cu#L88) 等功能。因此它的优势是共享更多源码、减少新指令依赖，**不是现有 kernel 就能直接运行在 A100 等所有 SM80 GPU 上**。源码可复用、可编译、二进制兼容和实机验证是四件不同的事。

#### 2. 支持做专用后端：峰值之外，也有真实的数据流机会

##### 2.1 CHUNK16 可以匹配新指令，并非只能 padding

[topic2](02_tcgen05_evaluation/REPORT.md) 对原计算逐项检查后，找到了以下保持 GEMM 代数结果的重组：

| 运算 | 可用方案 | 为什么值得继续 |
|---|---|---|
| K2 的 K@S 与 Q@S | `[K;Q]@S`，得到 M32×N128×K128 的共享 state 配对 | `.ws` 路径可免 padding，并复用同一 S |
| INV@R、Mqk@U | 使用 `AB=(BᵀAᵀ)ᵀ`，变成 M128×N16×K16 | 普通 tcgen05 路径可免 padding |
| 状态更新 K_restoredᵀ@U | 原形状 M128×N128×K16 | 原形状已经匹配，不必增加 CHUNK |
| K1 的 L/Mqk、Neumann 部分配对 | 堆叠共享右操作数的独立乘法 | 可减少分别 padding 的无效工作，但仍有形状和精度限制 |

02_tcgen05_evaluation 的 **B300、BF16 输入/FP32 累加、共享 state 配对 M32×N128×K128、batch=1** 实测，`mma.sync` 为 **15.0920 µs**，`.ws` 为 **14.2695 µs**，耗时下降 **5.45%**。这是支持进一步集成的正面证据；69 个独立 GEMM 正确性检查通过，说明并非只有形状上的设想。详细结果（本地证据：`../02_tcgen05_evaluation/RESULTS.md`）。

这里 `.ws` 指 weight-stationary 指令变体，不能把名称直接解释为整套 warp-specialization 调度。具体语义和支持范围以 [PTX tcgen05 指令规范](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma) 为准。

##### 2.2 TMEM、异步执行可能改善当前瓶颈

新后端的价值可以来自减少数据搬运和调度指令，而不只是更高的 FLOPS。Blackwell Ultra 提供独立 TMEM，并支持异步第五代 MMA 和双 CTA 协作能力。[NVIDIA 架构介绍](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)。

04_compute_memory_bottlenecks 的指定负载 N=1/H=96/T=8192/D=128 中，K2 只有 96 CTA，L1/TEX 为 **38.23% elapsed**、Tensor 为 **19.62%**；wait/short-scoreboard 采样分别为 **26.51%/17.74%**。因此，改善递推内部供数和依赖隐藏有研究依据，但也必须解决状态链并行度不足；仅增加指令峰值或迁入 TMEM 不会自动消除这些限制。完整融合方案尚未验证。

05_bf16_state_accuracy 还提出了精度方面的需求：持续保留 FP32 状态可能比反复写回 BF16 更可靠。TMEM 能否帮助容纳 FP32 中间状态，是值得研究的方向；但任意逐元素 gate/FMA 更新不等于一次 MMA，TMEM 的读写、同步和操作数转换仍有成本，不能宣称放进去就免费解决精度与性能。

##### 2.3 专用后端可以与主线共存

不必让所有设备和形状都运行同一个新 kernel。可以共享 API、数学参考和精度测试，根据设备能力与已验证的形状选择后端。未支持设备、短任务或测得退化的形状继续使用现有实现。

这是支持专版的一项架构论据：双后端可以保留整个产品的跨设备服务能力。不过，它不能消除专用路径自身的适用范围限制，还会扩大分发、构建和测试矩阵。在当前完整收益证据不足的情况下，我们不认为维护双后端值得。

#### 3. 支持保留现有主线：已测结果没有支撑全面替换

##### 3.1 局部 GEMM 的收益不等于完整 KDA 的收益

同一个共享 state 配对任务在 **batch=16384** 时，`mma.sync` 为 **1089.6200 µs**，`.ws` 为 **2760.9360 µs**，旧/新仅 **0.395×**。02_tcgen05_evaluation 所有大 batch 新路径均慢于同任务旧路径。

另外，02_tcgen05_evaluation 在计时前已经完成输入布局准备；生产 KDA 中原本驻留寄存器的数据如何转为新指令所需布局，以及整个递推中如何复用 TMEM，都没有完整接入。独立 kernel 的 global→shared、同步和输出接口也不同于生产融合时序。因此既不能拿 5.45% 局部收益承诺整层加速，也不能拿当前大 batch 退化证明所有可能的 tcgen05 融合都失败。

##### 3.2 更高的计算峰值不能直接解决 CTA 数量不足

[topic3](03_recurrence_parallelism/REPORT.md) 保留真实递推逻辑，得到：

- value 列分片在 1/8 条长链时，K2 加速 **1.190×/1.198×**；148/256 条链时仅 **0.513×/0.363×**。
- 两 head 合进同一 CTA，8 个场景没有明确收益，少链耗时增加约 **44.7%**。
- 动态 persistent 在特定倾斜场景以 **339.253 µs** 胜过静态分配的 **1214.048 µs**，但原 K2 已是 **338.165 µs**。

这些结果支持按形状选择并行策略，也说明“更多工作放进 CTA”“改成 persistent”不是通用收益来源。现有 K2 已在 CTA 内连续处理多个 chunk，不存在每个 chunk 单独 launch 的开销可供再次消除。

列分片的正面结果使用原 MMA 就已实现，不是 SM100a 独有的收益。CLC 和协作 2-CTA 尚未测试；不能把普通 persistent 的结果冒充硬件 CLC 测试，也不能把独立列分片当成双 CTA MMA 协作。

##### 3.3 实测并没有显示 Tensor 峰值是唯一限制

[topic4](04_compute_memory_bottlenecks/REPORT.md) 的原 kernel、20 次 warmup 后 step21 测量：

| B300 场景 | 关键数据 | 对专版决策的含义 |
|---|---|---|
| N1/H96/T8192/D128 的 K1 | SM 73.22%、Tensor 5.87%、L1/TEX 71.04%；barrier 采样 40.65% | 指令发射、片上访问和同步都要处理 |
| N1/H96/T8192/D128 的 K2 | 96 CTA / 148 SM；Tensor 19.62%、L1/TEX 38.23%、DRAM 11.32% | 状态链并行度不足，并伴随递推依赖和片上供数等待；不是全卡 Tensor 或 HBM 饱和 |

表中吞吐均为 elapsed 口径。采样比例不是可直接删除的时间比例，不能把 40.65% barrier 样本换算成确定的加速上限。该证据仅对应指定负载，不推断未测的其他形状。

##### 3.4 为匹配大 tile 而直接增大 CHUNK，会引入数值风险

[topic1](01_chunk_size_analysis/REPORT.md) 的独立 B300 microbench 中，C32 的常数 token 衰减约 **2.7297** 已出现指数归零，约 **2.7734** 出现逆指数非有限；C64 阈值约减半。完整 FP16 Neumann 在 C32/C64 都有 **36/108** 个压力 case 出现 Inf/NaN。

所以不能直接把 C16 改成 C64，以“喂满新 Tensor Core”为由忽略指数重缩放和求逆稳定性。另一方面，C32/C64 都能用 M16 MMA 整齐分块，形状并不是数值失败的原因。保持 C16 并重组 GEMM 才是当前更合理的试验起点；大 CHUNK 属于额外的数值算法设计。

01_chunk_size_analysis 的 C16 求逆虽然没有非有限结果，压力输入的最大相对误差仍约 **23.79%**。因此保留旧实现也不等于已经证明其所有精度场景都安全。

##### 3.5 维护成本包括数值语义，而不只是编译

[topic5](05_bf16_state_accuracy/REPORT.md) 的 BF16 参考在 **N=1、T=8192、H=96、D=128 的 41 组输入**上与原 kernel 完全一致；旧状态保留较多且每次变化很小的三档输入，相对 FP32 状态保存方式的最大输出差异分别为 **14.5528%、6.3993%、3.4456%**，且没有 Inf/NaN。

因此专版不能只测试单个 GEMM allclose 或“结果有限”。它必须保留或明确改变 residual、U、Neumann 和状态的舍入位置，并重新验证长递推。原版与新版逐位一致只是实现一致性证据；两者也可能继承相同的精度问题。

持续 FP32 状态、重缩放或更稳定求逆若被引入，需要独立的精度契约和性能对照。现有 BF16 路径可以作为性能回退，但不能对所有压力输入自动充当“精度安全回退”。

#### 4. 峰值论据应当怎样使用

官方 HGX B300 表给出八卡 FP16/BF16 **36 PFLOPS（稀疏）**，注明 dense 为其一半，因此换算单卡 dense 为 **36÷2÷8=2.25 PFLOPS**。不能将八卡、稀疏、FP4 峰值拿来预测当前单卡 dense BF16 KDA 的性能；这一数值也不是同一 B300 上 tcgen05 相比 mma.sync 的实测倍率。[HGX 官方规格与脚注](https://www.nvidia.com/en-sg/data-center/hgx/)。

更不能把 Blackwell Ultra 相比前代的 NVFP4 增幅直接用于 BF16 状态递推；它们涉及不同数据类型和数值约束。本机 148 SM 的实测也应优先于满规格芯片的 SM 数假设。[Blackwell Ultra 架构及 SKU 说明](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)。

可以做的纸面推算是明确假设下的 Amdahl 分析。04_compute_memory_bottlenecks 指定负载 N=1/H=96/T=8192/D=128的 NCU 时间为 K1=456.896 µs、K2=1291.552 µs，K2 占两者和的 **73.87%**。假设只改变 K2、K1 不变、没有额外接入成本：

`总加速比 = (456.896 + 1291.552) / (456.896 + 1291.552 / s)`。

即使完整 K2 假设加速 2 倍，两 kernel 时间和也只加速 **1.586×**；假设 K2 耗时趋零，上限为 **3.827×**。这只是以已测 NCU 时间构造的条件模型，不是无 profiler 的端到端预测，更不代表 02_tcgen05_evaluation 已实现了 s=2。不能把 Tensor 利用率直接当成可加速时间占比，也不能将不同 topic 的加速比相乘。

#### 5. 我们会怎样安排 v2

| 决策项 | 当前选择 | 依据 |
|---|---|---|
| 正式默认路径 | 保留现有实现及统一 API | 尚无完整 tcgen05 KDA 的稳定收益数据 |
| CHUNK | 先保持 16 | 01_chunk_size_analysis 的范围/求逆风险，02_tcgen05_evaluation 又证明无需增大 CHUNK 才能匹配 |
| Blackwell 专用后端 | v2 不开发、不发布 | 局部正面结果不足以支撑完整专版投入，通用性与维护成本更有确定性 |
| K1/Neumann | 暂不全面迁移 tcgen05 | 小矩阵、同步与数值稳定性限制，Tensor 利用率很低 |
| 少链场景 | 优先评估已有列分片，按实测范围选择 | 03_recurrence_parallelism 已有完整递推正面证据，但多链会退化 |
| 多 head/persistent/2-CTA | 不作默认方案；2-CTA/CLC 保留为未验证候选 | 不能从硬件功能存在推导收益 |
| 状态精度 | 为 BF16 明确适用范围，评估独立的更高精度路径 | 05_bf16_state_accuracy 已证明无条件“无损”不成立 |
| 二进制与分发 | 沿用共享源码按目标架构构建，不增加专用计算后端 | 维持现有适配范围；编译目标仍须分别验证 |

以下是未来重新考虑这一决定所需的新证据，不是 v2 的专版开发计划，也不是已经完成的结果：

1. **完整路径收益**：在实际 SM100 设备及 B300 上，包含输入布局接入、K1、K2、TMEM 管理、同步和输出的完整执行；与同轮、同精度、同语义的基线比较，warmup 后独立计时，NCU 只用于归因。
2. **形状适用范围**：少链/多链、短序列/长序列、尾块/varlen 分别报告；只对有稳定收益的区域选择新后端。收益必须超过测量波动；商业发布所需的最小收益应在测试前按目标负载确定，不用孤立最佳点代替。
3. **精度覆盖**：重跑 01_chunk_size_analysis 的边界和 05_bf16_state_accuracy 的状态压力测试，验证完整递推与状态接续；如更改精度策略，应同时报告数值误差与性能，不能用降低质量换来的时间宣称等价加速。
4. **维护可承受**：支持的设备/工具链明确，未支持设备能选择已有路径；共享参考和输入生成，避免两套后端长期出现不可解释的语义差异。

**最终选择：不要，v2 不做 SM100a 专版。我们优先在现有实现上处理已被实测证明的并行度、数据访问和精度问题。专版目前只有局部机会，没有足够的完整性能收益证据，却确定会增加设备适配和维护负担，因此不值得在本轮投入。这个决策不意味着现有代码完全不用优化，也不意味着所有未来的专用设计都不可能更快。**

#### 6. 证据入口

- [topic1：CHUNK 数值范围、Neumann 与 MMA 形状](01_chunk_size_analysis/REPORT.md)：独立 microbench，不是 C32/C64 生产 kernel。
- [topic2：tcgen05 重组](02_tcgen05_evaluation/REPORT.md)、原始测量汇总（本地证据：`../02_tcgen05_evaluation/RESULTS.md`）：独立 GEMM，不是完整 KDA 专版。
- [topic3：真实递推并行方案](03_recurrence_parallelism/REPORT.md)：列分片、多 head、软件 persistent；未测试协作 2-CTA/CLC。
- [topic4：原始 K1/K2 的 NCU 瓶颈](04_compute_memory_bottlenecks/REPORT.md)：原源码、20 次 warmup、step21。
- [topic5：状态保存精度](05_bf16_state_accuracy/REPORT.md)、误差范围（本地证据：`../05_bf16_state_accuracy/RESULTS.md`）：原 kernel 对齐的受控状态精度比较，不是模型任务评测。

架构信息核对使用 KernelWiki 的 [hw-tcgen05-mma 页面](../../.codex/skills/KernelWiki/wiki/hardware/tcgen05-mma.md)作为检索入口；最终指令、兼容性和规格论据以本文链接的 NVIDIA 原始文档及本仓库源码为准，不采用 wiki 中的简化形状表推翻已有实测。本文不复制旧 rep、日志或 CSV，各结论直接链接原 topic 的证据。

<a id="measurement-method"></a>

## 计算与访存采集方法

### 实验口径与纸面模型

#### 调用与 warmup

直接编译仓库 [fwd_launch.cu](../csrc/smxx/fwd_launch.cu)，调用 `launch_fwd<128,true,true,false,false>`。这个原始 launcher 发射 prepare（K1）与 recurrence（K2），K1 的真实 workspace 直接供 K2 使用。[bridge.cu](04_compute_memory_bottlenecks/harness/bridge.cu) 只有主机侧 ABI 转接，没有新增或改写 GPU kernel。`csrc` 各文件的 SHA256、编译命令及本次二进制 SHA256 保存在 [source.json](04_compute_memory_bottlenecks/analysis/source.json)。

固定 D=128、CHUNK=16、BF16 输入/输出/初末状态、普通等长 batch。q/k/v 为 seed=42 的正态随机数×0.125，g/beta 为正态随机数，A_log/dt_bias=0，lower_bound=-5，非零初始状态为正态随机数×0.02。数据代表本次合成负载，不声称来自真实模型 trace。输入分配、beta 转置、workspace 分配在计量区外完成。

| 场景 | N | H | T | K1 CTA | K2 CTA | 每个 K2 CTA 的 chunk 数 |
|---|---:|---:|---:|---:|---:|---:|
| task_n1_h96_t8192 | 1 | 96 | 8192 | 49152 | 96 | 512 |

采用 assignment02/cuda/common.h:65 的 20 次 warmup，额外在计量前显式同步。每轮 application replay 都重新创建输入并执行 20 次完整 forward，然后以 `cudaProfilerStart/Stop` 和 NVTX `场景/step21` 标记第 21 次 forward。只过滤原始 K1/K2；full、source 分开采集，均应包含 2 个 action，解析时检查名称、数量及 CTA 数。

NCU 使用 application replay、`--cache-control none --clock-control none`：让每轮经过同样 warmup，不强制刷新 K1→K2 的 workspace 缓存。NCU 仍会插入采集/同步操作，所以时间与缓存指标属于该测量协议，不能完全等同无 profiler 的端到端行为。频率不锁定，不能把小幅差异解释为优化收益。full 加 PmSampling、PmSampling_WarpStates；source 加 SourceCounters；均带源码关联。

独立 CUDA event 计时采用 20 次 warmup 后 9 组×20 次完整 forward，报告每组每次耗时的中位数。它包含 launcher 与 Python 发射可能造成的 GPU 间隙，不用于代替 NCU 的单 kernel 时间。正确性调用仓库 tests/torch_ref.py，覆盖 T=16、T=31（含尾块）、非零初末状态；正式指定形状另检查输出和状态均为有限值。

#### 纸面算术强度：只作假设，不代替计数器

D=128、C=16，每个 head/chunk 的 workspace 是 3×C×D×2 + D×4 + 2×C²×2 = **13,824 B**。

K1 两个 C×D 乘 D×C，加 Neumann 的六次 C×C 乘 C×C，共 `4 C² D + 12 C³ = 180,224` 个矩阵 FLOP（FMA=2）。q/k/g 逻辑读取 12,288 B，beta 32 B，写 workspace 13,824 B；忽略跨 chunk 共享的 A_log/dt_bias，得到 `180224/26144 = 6.894 FLOP/B`。这不是完整 K1 的 FLOP 计数：归一化、sigmoid、指数、归约、类型转换和同步不能用 Tensor 峰值衡量。

K2 有三个 C×D×D 级 GEMM（K@S、Q@S、状态更新）和两个 C×C×D GEMM（INV@U、Mqk@U），矩阵 FLOP 为 `6 C D² + 4 C² D = 1,703,936`。每 chunk 逻辑读 workspace 13,824 B、v 4,096 B、beta 32 B，写 output 4,096 B，共 22,048 B。初末状态每条链另读写 65,536 B，因此 L 个 chunk 的逻辑强度为 `1703936/(22048+65536/L)`：L=512 时 **76.837 FLOP/B**。循环中的状态在 CTA 内复用，不能按每 chunk 都读写 HBM 算。

这些是逻辑张量字节，不包括 cache 命中、事务放大或重复指令访问。尤其 warmup 后的小负载可命中 L2，实际 HBM 字节可能远小于上述模型。应以报告的 DRAM 字节/带宽和 L2 指标检验。

#### 与 assignment 4.5 的关系

参考 [assignment02 writeup 4.5](/home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/handout/src/writeup.md:714)。其 `in_proj_qkvgfab` 是 KDA 输入投影，N=6288、K=7168，M 是投影的 token 行数，不是本 kernel 的 CHUNK。

BF16 GEMM 理想强度为 `AI=2MNK/(2MK+2NK+2MN)=1/(1/M+1/N+1/K)`。沿用该作业的 2250 TFLOPS、8000 GB/s，平衡点 281.25 FLOP/B，对应 M≈307；M=16 的 AI≈15.9，M=1024 的 AI≈784.2。它说明形状会改变算术强度，但不能把投影结论直接套给递推 kernel。该表的 GB/s 是逻辑最低字节量除以时间，也不能当成本实验 NCU 测到的 DRAM GB/s。

这里没有重新测试输入投影，也没有引入 02_tcgen05_evaluation/3 的 kernel 变体。

