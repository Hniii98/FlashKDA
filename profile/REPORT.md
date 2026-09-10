# C32 完整实验报告

当前主源码为CHUNK=32、rescale=2^-96、inverse_rescale=1、K1 launch bounds=4，脚本入口见[README](README.md)。

## 目录

- [C32 实现与初版诊断](#c32-implementation)
- [rescale 参数扫描](#c32-sweep)
- [默认参数完整对拍](#c32-recheck)
- [原 C16 / C32 benchmark](#c32-benchmark)
- [C32 时间成本定位](#cost-attribution)
- [launch bounds 对照](#launch-bounds)
- [K1 八阶段计时](#stage-timing)

<a id="c32-implementation"></a>

## C32 实现与初版诊断

原文位置：`c32/RESULTS.md`。本节内未改写的相对路径仍以该原文所在目录为基准。

### C32 第一版：原算法与精度，rescale 默认 1

后续参数 sweep 已完成，见 [SWEEP.md](#c32-sweep)。当前默认值已改为 `rescale=2^-96`、`inverse_rescale=1`；以下保留第一版默认参数及诊断记录。

实现基于本工作区原 C16 commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`，未复制旧候选实现。设备为 NVIDIA B300 SXM6 AC，编译目标 sm_103a，PyTorch 2.13.0+cu130。本版已编译并完成 naive 对拍；默认配置有数值失败，不能宣称正确性通过。未进行性能测量。

#### 实现

- K1：CHUNK=32；归一化覆盖 32 行；mask/初始化覆盖 1024 个元素。沿用 cooperative_gemm 和 SM80 MMA。FP16 Neumann 用 16×16 tile 完成整个 32×32 矩阵的乘法，依次计算 L²、L⁴、L⁸、L¹⁶，部分和最终覆盖到 31 次幂；没有改成分块求逆或其他求解算法。
- K2：每个计算 warp 保留两个 value-column tile，扩展到两个 time tile。D 维投影仍归约 128，INV@R、Mqk@U、Krᵀ@U 沿 CHUNK 归约 32；每个 C32 块完成一次 state 更新。沿用 BF16 操作数、FP32 MMA 累加及原输出/state 舍入位置类型。
- launcher、workspace 尺寸和 TMA 随 C32 扩展。beta 缓冲扩展到 CHUNK+8 项，容纳原有最多 7 项对齐偏移；同步更新 TMA 字节数。原线程数、pipeline stage 数与 launch bounds 保留。
- Python 和 C++ 接口追加 `rescale=1.0`、`inverse_rescale=1.0`，原调用无需传入新参数。

#### 缩放参数

以下 G 为逐通道的块内累计 log-gate，Ge 为块末累计值。

`rescale=s>0`：

- Kd=K exp(G−log(s))，Qd=Q exp(G−log(s)) scale。
- Ki=K exp(−G+log(s))；L/Mqk 中 s 配套抵消。
- Kr=Ki exp(Ge−log(s))；在生成指数前应用偏移，避免先将 exp(Ge) 清零再除以 s。
- K2 的 Kd@S 和 Qd@S 在转 BF16 前乘 s，恢复原数学贡献。
- 用于旧 state 衰减的 g_total 仍为 exp(Ge)，不改变遗忘率。

`inverse_rescale=r>0`：令对角矩阵 D[i,i]=r^i，在原 FP16 Neumann 前应用 Ls=D⁻¹LD，求逆后应用 INV=D INVs D⁻¹，再转为原 BF16 workspace。它仍是同一个 Neumann 算法。缩放系数和中间量也有低精度范围限制，并不保证任意正数都数值可用。

两参数为 1 时显式跳过缩放，使用原指数表达式、低精度操作和转换；新增 C32 tile/归约自身会改变相对 C16 的舍入。非 1 参数只做了执行检查，尚未 sweep 或确定可用范围。

#### 对拍来源与口径

唯一对拍文件：[tests/compare_naive.py](../tests/compare_naive.py)，基于原 baseline 的 `profile/Baseline/compare_fla_ref.py` 适配。保留其 seed=0、H={96,64}、总 T=8192、三种序列划分和 FP32/BF16 state 接口，共 12 组。非有限结果保留，不再直接 assert 中断。

参考为题目未修改的 `fla_kda_ref/naive.py::naive_recurrent_kda`。harness 在 FP32 中做数学上的 Q/K 归一化、gate 与 beta 激活，并适配初始/最终 state 的转置；naive 内部为 FP32 recurrent state。参考路径、SHA256、扩展路径和 SHA256 均写在结果 JSON 中。

输入 initial_state 沿用原对拍文件的 arange 后 BF16 量化；幅值很大，所以 output 最大绝对误差不能脱离相对误差解释。FP32/BF16 state 接口输入表示相同数值，且 kernel 内部 state 均为 BF16，因此两种接口结果相同。

误差为 max(abs(candidate−reference)) 和 ||candidate−reference||₂ / ||reference||₂，以 FP64 统计。出现非有限值时误差记为 null（未定义），同时报告 NaN/Inf 个数，不只筛选有限元素。未设定通过阈值。

#### 默认配置结果

`lower_bound=-5, rescale=1, inverse_rescale=1`：12/12 组 output 和最终 state 均含 NaN，参考均有限。完整记录：naive_rescale1.json（本地文件：`c32/naive_rescale1.json`）。

官方形状 H=96、T=8192、单序列（两种 state 接口相同）：

| 量 | NaN 数 | Inf 数 | 元素数 |
| --- | ---: | ---: | ---: |
| output | 100663296 | 0 | 100663296 |
| 最终 state | 1572864 | 0 | 1572864 |
| K1 k_restored | 13546657 | 0 | 100663296 |
| K1 Mqk | 1617267 | 0 | 25165824 |
| K1 INV | 25165824 | 0 | 25165824 |

同一输入中，K1 的 Kd/Qd 均有限，g_total 有 2549960/3145728 个零。零值计数本身不是误差判据。实际 workspace 已证明异常出现在 K1、早于 K2；此诊断没有导出临时 Ki/L，因此不能仅凭 INV 的 NaN 区分上游污染与 Neumann 自身溢出。

#### 有限值条件下的流程诊断

仍使用同一个对拍文件和相同 12 组输入生成方式，仅将双方 `lower_bound` 改为 -1；两缩放参数仍为 1。用于隔离强衰减指数问题，不能替代上面的 -5 失败结果。

12/12 组 output/state 均有限，output 相对 L2 范围 0.00353655～0.00406399，最终 state 相对 L2 范围 0.00458670～0.00476006。完整记录：naive_gate1.json（本地文件：`c32/naive_gate1.json`）。

| H=96、单序列 T=8192 | 最大绝对误差 | 相对 L2 |
| --- | ---: | ---: |
| output | 619.9375 | 0.00406398814 |
| 最终 state | 0.00663065910 | 0.00473843263 |

同样的有限值输入使用 `rescale=0.5, inverse_rescale=1.01`，两种 state 接口均有限：output 相对 L2=0.00406403351，最终 state 相对 L2=0.00473869569；最大绝对误差分别为 619.9375、0.00663065910。该检查仅证明参数分支执行并取得上述结果，不是参数 sweep 或可用值验收。naive_scale_smoke.json（本地文件：`c32/naive_scale_smoke.json`）

#### 构建与复现

内存检查：在同一对拍文件的 H=96、变长序列 `[1300,547,2048,963,271,3063]` 用例上，使用 lower_bound=-1、两缩放参数为 1，FP32/BF16 state 接口各运行一次。Compute Sanitizer memcheck 过滤 `_flash_kda_` kernels，报告 `ERROR SUMMARY: 0 errors`；此结论限于该用例及 memcheck 范围。memcheck.log（本地文件：`c32/memcheck.log`）、naive_memcheck.json（本地文件：`c32/naive_memcheck.json`）

仅构建候选，复用已有 candidate-venv、Python 头文件和原仓库 CUTLASS 头文件，不安装或覆盖原 baseline。

```bash
PATH=/home/lcpu/60990375/topic7-envs/candidate-venv/bin:$PATH /home/lcpu/60990375/topic7-envs/candidate-venv/bin/python profile/c32/build.py
srun --partition=gpu --ntasks=1 --gpus=1 --cpus-per-task=8 --time=00:20:00 /home/lcpu/60990375/topic7-envs/candidate-venv/bin/python tests/compare_naive.py --ref-dir /home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref --diagnostics --rescale 1 --inverse-rescale 1 --output profile/c32/naive_rescale1.json
```

诊断命令见 [run_diagnostics.sh](c32/run_diagnostics.sh)。构建日志为 build.log（本地文件：`c32/build.log`）。编译器报告 K1 有寄存器 spill；本版按约束保留原 launch bounds，未据此调整设置或声称性能收益。

<a id="c32-sweep"></a>

## rescale 参数扫描

原文位置：`c32/SWEEP.md`。本节内未改写的相对路径仍以该原文所在目录为基准。

### C32 rescale sweep（2026-09-10）

后续已按用户要求将默认值改为 `rescale=2^-96`、`inverse_rescale=1`，定义在 `csrc/smxx/fwd_config.h`；以下为此前 sweep 记录。

保留已编译的 C32 kernel，未修改算法、dtype、gate 或默认参数。扩展同一 `tests/compare_naive.py`，使多个参数共享相同输入和 naive 参考。设备：NVIDIA B300 SXM6 AC；PyTorch 2.13.0+cu130。

#### 结论

在现有 12 组对拍用例上，固定 `lower_bound=-5`、`inverse_rescale=1`，`rescale=2^e` 的整数采样点 e=-105…-92（共 14 点）全部得到有限 output 和最终 state。推荐将区间内的 `2^-96 = 1.262177448353619e-29` 作为后续研究配置；没有将 API 默认值从 1 改掉。

这是消除本组输入 NaN/Inf 的实测结果，不是精确连续参数边界，也不代表其他输入都可用。未设定正确性通过阈值。未进行性能测试。

#### 范围与证据

粗扫官方单序列形状：e={0,-16,-32,-48,-64,-80,-96,-112,-120}，两种 state 接口，共 18 条结果。细扫 e=-80…-112、步长 -1，沿用 H={96,64}、三种序列划分、FP32/BF16 state 接口，共 396 条结果。全部保留，含失败项。

粗扫原始结果（本地文件：`c32/sweep_coarse.json`）、细扫原始结果（本地文件：`c32/sweep_fine.json`）、按参数汇总（本地文件：`c32/sweep_summary.json`）。JSON 内保存 binary/reference/harness SHA256 与调用参数；每行的 rescale 是实际运行值，顶层 rescale 是 CLI 的单值默认参数，列表 sweep 以 rescale_grid 和各行值为准。

| H | 序列划分（总 T=8192） | output/state 均有限的整数 e 采样点 |
| --- | --- | --- |
| 96 | [8192] | -108…-90 |
| 96 | [1300, 547, 2048, 963, 271, 3063] | -105…-92 |
| 96 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | -105…-90 |
| 64 | [8192] | -108…-90 |
| 64 | [1300, 547, 2048, 963, 271, 3063] | -106…-89 |
| 64 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | -105…-90 |

两种 state 接口结果相同。全用例交集为 -105…-92，不能根据单个形状的更宽区间宣称所有用例均有限。

#### rescale=2^-96 的误差

参考为题目未修改的 naive_recurrent_kda，预处理与内部参考计算为 FP32。误差在 FP64 中统计。下表两种 state 接口相同。

| H | 序列划分 | output 最大绝对误差 | output 相对 L2 | state 最大绝对误差 | state 相对 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 96 | [8192] | 314.683594 | 0.00386518907 | 0.00791704655 | 0.00458132069 |
| 96 | [1300, 547, 2048, 963, 271, 3063] | 1796.0625 | 0.00347572915 | 0.0094974637 | 0.0045906454 |
| 96 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 3514.625 | 0.00400208345 | 0.00905925035 | 0.00456350217 |
| 64 | [8192] | 163.449219 | 0.00390619811 | 0.00813734531 | 0.00459004285 |
| 64 | [1300, 547, 2048, 963, 271, 3063] | 1268.53125 | 0.00355141306 | 0.00696307421 | 0.00459145836 |
| 64 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 1425.25 | 0.00343433461 | 0.00871014595 | 0.00462786221 |

上述 12 组 output 相对 L2 为 0.00343433～0.00400208，state 相对 L2 为 0.00456350～0.00462786。所有 output/state 的 NaN/Inf 数均为 0。initial_state 沿用原对拍文件的 arange 后 BF16 量化，幅值很大，因此同时报告相对误差和绝对误差。

14 个全用例有限采样点的误差基本一致；这次没有证据支持仅靠继续调 s 显著降低剩余约 0.34%～0.46% 的相对 L2 误差。

#### 两侧失败原因

- s 偏大（如 2^-91）：H=96 的非整块变长用例仍有 K1 k_restored 2 个 NaN、Mqk 2 个 NaN，最终 output 有 223232 个 NaN。指数范围问题尚未完全消除。
- s 偏小（如 2^-106）：部分用例 K1 workspace 所有保存量均有限，但 output 出现 NaN，异常在 K2 计算中出现。根据代码，Qd/Kd 按 1/s 放大，在投影与恢复尺度之前有 FP32 累加溢出的风险；该具体中间投影没有导出，因此此机制为源码推断，workspace 与 output 的有限性差异是实测。
- g_total 的零值仍可能存在；rescale 不改变旧 state 的真实遗忘因子。有限结果不能证明全部下溢误差都消失。
- 此次无需调 inverse_rescale 就可消除当前用例的非有限值。topic1 中弱衰减、高相关 K 的 Neumann 压力输入不在本次原有对拍用例中，本结果不能宣称解决该独立问题。

#### 复现

```bash
srun --partition=gpu --ntasks=1 --gpus=1 --cpus-per-task=8 --time=00:20:00 bash profile/c32/run_sweep.sh
```

本轮 kernel 二进制 SHA256：`99ea3737d623e1ec192f7b444305d9d37ecf2cd35472fd212faeb7b53af1a25f`。与 C32 第一版相同，没有重编译 kernel。

<a id="c32-recheck"></a>

## 默认参数完整对拍

原文位置：`c32/RECHECK.md`。本节内未改写的相对路径仍以该原文所在目录为基准。

### 当前默认参数：完整 naive 对拍复查

设备：NVIDIA B300 SXM6 AC；参考：`naive_recurrent_kda`，FP32 数学预处理及 FP32 递推。

参数：rescale=1.262177448353619e-29 (2^-96)，inverse_rescale=1.0，lower_bound=-5.0。此次调用未显式传入缩放参数。

12 组结果；下表每行包括 FP32/BF16 两种 state 接口，两者误差相同。

| H | 序列长度 | output 最大绝对误差 | output 相对 L2 | state 最大绝对误差 | state 相对 L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 96 | [8192] | 314.683594 | 0.00386518907 | 0.00791704655 | 0.00458132069 |
| 96 | [1300, 547, 2048, 963, 271, 3063] | 1796.0625 | 0.00347572915 | 0.0094974637 | 0.0045906454 |
| 96 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 3514.625 | 0.00400208345 | 0.00905925035 | 0.00456350217 |
| 64 | [8192] | 163.449219 | 0.00390619811 | 0.00813734531 | 0.00459004285 |
| 64 | [1300, 547, 2048, 963, 271, 3063] | 1268.53125 | 0.00355141306 | 0.00696307421 | 0.00459145836 |
| 64 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 1425.25 | 0.00343433461 | 0.00871014595 | 0.00462786221 |

全部 output/state 有限：True；NaN 总数 0，Inf 总数 0。

相对 L2 = ||candidate-reference||₂ / ||reference||₂，使用 FP64 统计。初始 state 沿用对拍文件的 arange 后 BF16 量化，幅值较大；因此同时列出绝对误差和相对误差。未设定正确性通过阈值。

完整 JSON（本地文件：`c32/naive_default_full_recheck.json`） 包含全部 12 组数值、参考与二进制 SHA256 及运行命令。

<a id="c32-benchmark"></a>

## 原 C16 / C32 benchmark

原文位置：`c32/benchmark/REPORT.md`。本节内未改写的相对路径仍以该原文所在目录为基准。

### C32 与原 C16：同轮 benchmark

日期：2026-09-10；GPU：NVIDIA B300 SXM6 AC；PyTorch：2.13.0+cu130；Slurm job：25029。

同一进程加载原 C16 和当前 C32 二进制，使用同一组输入，在同一张 GPU 上交替运行。C16 来源 commit 为 `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`，源工作区干净，复用现成二进制，未重建。C32 使用默认 `rescale=2^-96`、`inverse_rescale=1`，lower_bound=-5。

结论：当前 C32 第一版在全部 18 个配对配置中均慢于 C16。尚未做性能优化；本轮未修改 kernel。

#### 完整 forward（CUDA event）

沿用官方计时方式：预热 30 次、每轮 200 次、5 轮，共每实现每配置 1000 个样本。各轮交替 C16/C32 顺序；统计下表平均值。包含 public wrapper 对应的 workspace 分配、beta 转置和完整 forward，输入/输出张量预先分配。未使用 CUDA Graph，未锁 GPU 时钟。

主表为 FP32 initial/final state，与历史官方 benchmark 汇总的 state 模式一致；每行 T=8192、D=128。

| H | 序列形状 | C16 ms | C32 ms | C32/C16 耗时比 |
| ---: | --- | ---: | ---: | ---: |
| 96 | 单序列 8192 | 1.7051 | 3.4139 | 2.002× |
| 96 | 不等长 varlen | 1.4920 | 3.4408 | 2.306× |
| 96 | 1024 × 8 | 1.2138 | 3.2081 | 2.643× |
| 64 | 单序列 8192 | 1.5528 | 2.7333 | 1.760× |
| 64 | 不等长 varlen | 1.1239 | 2.4476 | 2.178× |
| 64 | 1024 × 8 | 0.8217 | 2.1577 | 2.626× |

#### K1/K2 分项（独立 profiler 采样）

每项 20 次完整 forward，从 PyTorch profiler CUDA 时间线提取实际 K1/K2 kernel duration；未重编译成孤立 kernel。以下为平均值，单位 ms。profiler 不在上述完整 forward 计时区域内；两种计时独立，不能要求分项之和严格等于 event 耗时。varlen 的 prefix kernel 另存于 JSON。

| H | 序列形状 | C16 K1 | C32 K1 | C16 K2 | C32 K2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 96 | 单序列 8192 | 0.4554 | 2.0747 | 1.2359 | 1.3252 |
| 96 | 不等长 varlen | 0.4806 | 2.1190 | 0.9941 | 1.3018 |
| 96 | 1024 × 8 | 0.4795 | 2.0961 | 0.7151 | 1.0917 |
| 64 | 单序列 8192 | 0.3075 | 1.3966 | 1.2327 | 1.3232 |
| 64 | 不等长 varlen | 0.3238 | 1.4224 | 0.7824 | 1.0077 |
| 64 | 1024 × 8 | 0.3221 | 1.4130 | 0.4811 | 0.7303 |

K1 是主要耗时增量，K2 也没有取得收益。源码与构建日志显示，C32 的整块 Neumann 增加计算量，K1 在保留原 launch bounds 的情况下出现寄存器 spill；本轮未进一步采集硬件计数器，不能量化两者各自的耗时占比。

#### 全部 state 模式

| H | 序列形状 | state 模式 | C16 ms | C32 ms | C32/C16 耗时比 |
| ---: | --- | --- | ---: | ---: | ---: |
| 96 | 单序列 8192 | bf16 | 1.7458 | 3.4752 | 1.991× |
| 96 | 单序列 8192 | none | 1.7530 | 3.4520 | 1.969× |
| 96 | 单序列 8192 | fp32 | 1.7051 | 3.4139 | 2.002× |
| 96 | 不等长 varlen | bf16 | 1.4719 | 3.4641 | 2.354× |
| 96 | 不等长 varlen | none | 1.4712 | 3.4255 | 2.328× |
| 96 | 不等长 varlen | fp32 | 1.4920 | 3.4408 | 2.306× |
| 96 | 1024 × 8 | bf16 | 1.1850 | 3.2002 | 2.701× |
| 96 | 1024 × 8 | none | 1.1822 | 3.1566 | 2.670× |
| 96 | 1024 × 8 | fp32 | 1.2138 | 3.2081 | 2.643× |
| 64 | 单序列 8192 | bf16 | 1.5991 | 2.7944 | 1.748× |
| 64 | 单序列 8192 | none | 1.6044 | 2.7716 | 1.728× |
| 64 | 单序列 8192 | fp32 | 1.5528 | 2.7333 | 1.760× |
| 64 | 不等长 varlen | bf16 | 1.1108 | 2.4700 | 2.224× |
| 64 | 不等长 varlen | none | 1.1129 | 2.4331 | 2.186× |
| 64 | 不等长 varlen | fp32 | 1.1239 | 2.4476 | 2.178× |
| 64 | 1024 × 8 | bf16 | 0.8019 | 2.1547 | 2.687× |
| 64 | 1024 × 8 | none | 0.7993 | 2.1263 | 2.660× |
| 64 | 1024 × 8 | fp32 | 0.8217 | 2.1577 | 2.626× |

本轮所有配置的两种实现 output（以及有 state 时的最终 state）均有限。精度依据上一轮 [naive 对拍复查](#c32-recheck)：C32 output 相对 L2 约 0.3434%～0.4002%，最终 state 约 0.4564%～0.4628%，用户已接受其与 C16 接近的误差。本轮仅测性能及有限性，不重复 naive 对拍。

#### 原始证据与复现

results.json（本地文件：`c32/benchmark/results.json`） 保存完整轮次统计、mean/median/min/max、K1/K2/prefix 时间及环境；同目录各 trace JSON 保存 CUDA 时间线。

二进制：
- C16：`/home/lcpu/60990375/topic7-envs/baseline/flash_kda_C.cpython-312-x86_64-linux-gnu.so`，SHA256 `d5f324a698d7811730c3b55d3badc8bd92819634dddd239524a58d01e95ce4b1`。
- C32：`/home/lcpu/60990375/kda-chunk32-baseline/profile/c32/build/flash_kda_C.so`，SHA256 `fb809f65e9a12d1cf123a633f88569ad837a643f01359180f5a151ecab4e3d98`。

```bash
srun --partition=gpu --ntasks=1 --gpus=1 --cpus-per-task=8 --time=00:20:00 /home/lcpu/60990375/topic7-envs/candidate-venv/bin/python benchmarks/bench_c32_vs_c16.py
```

此结果与本轮 C16 比较，不用历史不同环境的 benchmark 数值计算加速比。

<a id="cost-attribution"></a>

## C32 时间成本定位

原文位置：`c32-cost-attribution-20260910/REPORT.md`。本节内未改写的相对路径仍以该原文所在目录为基准。

### C32 时间成本定位

本轮实测：新增 kernel 时间约 94.5% 在 K1。NCU 确认 K1 大量寄存器溢出访存与同步等待；运行时启用默认 rescale 没有造成观察到的整体变慢。尚未通过编译消融分别测出 rescale 代码、Neumann 算术、spill 各自的独立毫秒成本。

#### 实验方法

- 复用 C16、C32 既有二进制，不重建 baseline，不修改生产 K1/K2。二进制 SHA256 见 `analysis/runtime_ablation.json`。
- NVIDIA B300 SXM6 AC；同进程、同设备、同轮交替顺序。T=8192、H=96、D=128，单条等长序列，FP32 state 接口；沿用既有 benchmark 输入生成及初始 state，seed=0。这里 H=96 是本次单卡实际输入头数，不是 TP8 每卡12头的测试。
- 每条件预热30次，5轮×200次 CUDA event 完整 forward；另用 PyTorch profiler 采样20次 K1/K2。完整 forward 包含 wrapper 的 workspace 分配及内部处理；分解采样与 event 测量独立，不要求严格相加。
- 同一个 C32 二进制测试 rescale=1 与 2^-96，inverse_rescale 均为1；gate lower_bound=-5 为主条件，-1 为两种 scale 都有限的诊断对照。
- C16/C32 各采集 full+PmSampling+PmSampling_WarpStates、source+SourceCounters 两份 NCU 报告；通过 ncu_report 提取指标和源码采样。NCU replay 时间不作为 benchmark 结果。

#### 测量结果

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

本轮只增加性能及有限性诊断，没有重新执行 naive 全量误差对拍。生产 C32 二进制未变，其误差证据沿用 `../c32/naive_default_full_recheck.json`：12用例均有限，output 相对L2约0.343%–0.400%，最终state约0.456%–0.463%。C32 rescale=1、lower_bound=-5 的失败仅用于性能诊断，不作为正确候选。

#### 成本位置与证据

##### K1：溢出访存、单 warp 阶段与同步等待

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

##### K2：增量较小，现有并行度与依赖等待

K2 无 local spill。寄存器74→86，shared memory99456→162816 bytes；shared memory 驻留上限2→1。两者实际 occupancy 都约9.37%：当前只启动96个block，GPU有148个SM，不能把驻留上限减半直接解释成当前并行度减半。

Tensor active约20.09%→20.19%；short-scoreboard stall/issue-active 0.60→1.30，wait 1.14→1.80。主要源码热点在 MMA、state 更新相关位置。说明更大的 tile 及执行依赖值得后续检查，但本轮没有将K2新增0.095 ms进一步做独立阶段计时。DRAM active约10.66%，也不是 HBM 带宽满载。

#### 下一步优先级

1. 优先减少 K1 的临时矩阵同时存活范围、spill，以及单 warp 阶段引发的等待；保持既定算法、MMA和精度。每次改动仍用现有 naive 对拍文件和本 harness 验证。
2. 若要实验放宽 K1 launch bounds，应另立诊断版本：这改变了当前保留的设置，本轮没有做。对照应同时比较spill、occupancy和完整forward，不能只追求更少寄存器溢出。
3. 如需精确分离 rescale 的编译成本，另比较数学等价的固定默认scale专门化版本与动态参数版本。当前 scale=1 实验不能回答这部分。
4. K2 的增量远小于K1，本轮先不以更改MMA或并行度重构作为结论。

复现入口：`harness/collect.sh`、`harness/run.py`。NCU原报告在 `reports/`，完整指标、源码热点、运行时消融及 NCU规则输出在 `analysis/`。重复采集应新建 run 目录，保留本轮证据。

<a id="launch-bounds"></a>

## launch bounds 对照

原文位置：`c32-launch-bounds-20260910/REPORT.md`。本节内未改写的相对路径仍以该原文所在目录为基准。

### K1 launch bounds：8 → 5 / 4 / 2

结论：本轮代表形状上4最快，相对原C32完整forward加速1.779倍（耗时减少43.78%），K1加速3.306倍。2虽然零spill，却因寄存器占用增加、驻留block减少而比4慢。推荐下一步采用4进行更广形状的性能验证。该轮测试时主版本为LB8；现已按用户要求将主源码及默认构建切换为LB4，下面保留原同轮对照数据。

#### 控制变量和复现

只将本工作区 `csrc/smxx/fwd_kernel1.cuh:119` 的 `__launch_bounds__(NumThreads, 8)` 第二参数替换为5、4、2；没有应用上一轮B1临时变量改动。用户明确授权本次launch bounds对照。算法、Neumann展开、MMA、阶段精度、state、gate、rescale=2^-96、inverse_rescale=1以及K2完全不变。

`harness/build_all.py` 顺序更改该一处、用原环境和原编译参数独立构建，finally恢复原文件；`analysis/build_manifest.json`记录原始/恢复源码哈希一致与各二进制哈希。现有C16/C32源码或二进制没有被复制成候选，原二进制不重建、不覆盖。复现候选用该脚本，验证入口 `harness/collect.sh`；再次采集应新建run目录。

#### 同轮性能

B300 SXM6 AC，PyTorch2.13.0+cu130。T=8192、H=96、D=128，单条等长序列，FP32 state接口，seed0，沿用既有benchmark输入。H=96是本次单卡实际输入，未测试TP8每卡12头。

每条件预热30次，5轮×200次CUDA event，顺序正反交替；另外20次PyTorch profiler获得K1/K2分解。完整forward包含原wrapper路径，分解与event独立采样，不要求严格相加。没有使用NCU replay时间作benchmark。

| 版本 | forward/ms | K1/ms | K2/ms | 相对原C32 forward加速 |
|---|---:|---:|---:|---:|
| C16 | 0.999137 | 0.273104 | 0.716128 | 2.075× |
| LB8 | 2.073561 | 1.301401 | 0.762703 | 1.000× |
| LB5 | 1.328292 | 0.555259 | 0.762553 | 1.561× |
| LB4 | 1.165728 | 0.393610 | 0.762787 | 1.779× |
| LB2 | 1.385485 | 0.627969 | 0.763120 | 1.497× |

本轮原C32绝对时间与前轮不同；这里只比较本轮同进程、同设备、同输入的结果，不跨轮计算收益。LB4依然比本轮原C16完整forward慢约16.7%。没有据此声称全部形状都最优。

#### Spill与并行度证据

所有版本每block动态shared memory均为41728 bytes（计driver保留后42752）。K1线程数仍为256。NCU报告的shared memory配置容量随资源需求变化，LB8/5为233472、LB4为200704、LB2为135168 bytes；这是实际采集到的设备配置差异，本实验未手动修改carveout。

| K1指标 | LB8 | LB5 | LB4 | LB2 |
|---|---:|---:|---:|---:|
| 寄存器/thread | 32 | 48 | 64 | 93 |
| 定长静态spill stores/loads bytes | 3920/4164 | 750/750 | 16/16 | 0/0 |
| varlen静态spill stores/loads bytes | 3928/4180 | 774/778 | 16/16 | 0/0 |
| 寄存器限制blocks/SM | 8 | 5 | 4 | 2 |
| 实际occupancy | 61.94% | 61.86% | 48.93% | 24.54% |
| 动态local load+store指令 | 56942592 | 13049856 | 172032 | 0 |
| local load sectors | 150700032 | 31555584 | 393216 | 0 |
| local store sectors | 132022272 | 31555584 | 393216 | 0 |
| barrier stall / issue-active | 27.20 | 13.58 | 6.90 | 5.46 |
| long-scoreboard stall / issue-active | 11.92 | 3.80 | 1.69 | 1.40 |
| Tensor pipe active / elapsed | 3.44% | 7.99% | 11.31% | 7.27% |

LB4动态spill指令减少99.70%，保留4block/SM；LB2进一步去除剩余spill却将可驻留block减到2，与实测K1变慢一致。不能只以零spill选择参数。这里的local sector是缓存请求，不等于HBM流量。

每版采full+PmSampling+PmSampling_WarpStates和source+SourceCounters，原报告见reports，ncu_report提取见analysis。源码同步热点用于解释等待，不能将采样计数直接换成阶段毫秒。等长输入下grid不变，K1改动没有增加工作量不均衡；未针对varlen性能长尾作结论。PM原始序列保留，未以曲线推断新的阶段瓶颈。K1 DRAM active依次约38.89%、36.53%、42.05%、27.00%，并非HBM满载证据。K2未改动，耗时基本一致。

#### naive数值对拍

三个候选各使用既有 `tests/compare_naive.py` 全部12个用例，共36条结果；只通过loader选择独立二进制，不修改参考、用例或门槛。三版output和最终state的所有误差指标均与原C32既有记录相同，全部Inf/NaN为0。这是误差统计逐项相同，没有声称做了候选之间逐元素bitwise比较。

下表三种launch bounds均相同，FP32/BF16 state接口统计也相同，合并展示：

| H | seq_lens | output最大绝对误差 | output相对L2 | state最大绝对误差 | state相对L2 |
|---|---|---:|---:|---:|---:|
| 96 | [8192] | 314.683594 | 0.00386518907 | 0.00791704655 | 0.00458132069 |
| 96 | [1300, 547, 2048, 963, 271, 3063] | 1796.0625 | 0.00347572915 | 0.0094974637 | 0.0045906454 |
| 96 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 3514.625 | 0.00400208345 | 0.00905925035 | 0.00456350217 |
| 64 | [8192] | 163.449219 | 0.00390619811 | 0.00813734531 | 0.00459004285 |
| 64 | [1300, 547, 2048, 963, 271, 3063] | 1268.53125 | 0.00355141306 | 0.00696307421 | 0.00459145836 |
| 64 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 1425.25 | 0.00343433461 | 0.00871014595 | 0.00462786221 |

各版完整数值与参考哈希见 `analysis/naive_5.json`、`naive_4.json`、`naive_2.json`。原C32比较文件为 `../c32/naive_default_full_recheck.json`。

安全验证：5、4、2分别使用既有H=96六条变长序列用例、两种state接口运行compute-sanitizer memcheck，三版均报告0 errors，日志与数值输出保存在 `analysis/memcheck_*.log/json`。没有新增测试套件，未运行racecheck。

<a id="stage-timing"></a>

## K1 八阶段计时

原文位置：`k1-stage-timing-20260910/REPORT.md`。本节内未改写的相对路径仍以该原文所在目录为基准。

### K1阶段计时：原C16 vs C32 launch bounds=4

结论：C32每CTA平均被测延迟比C16多约2.902 μs，Neumann增加0.872 μs，gate/cumsum增加0.805 μs，decay增加0.469 μs。Neumann和gate/cumsum合计约占净增量的57.8%。不能把K1剩余差距只归因于Neumann或spill。

#### 插桩方法与控制

在八段现有同步边界记录9个 `%globaltimer` 时间戳，单位ns。每32个沿时间维分布的CTA采一个、仅thread0记录，写入独立device数组；不增加 `__syncthreads()`，不拆kernel、不删除阶段。1536个C16 CTA或768个C32 CTA/launch，重复10次，保留原始样本 `analysis/C16T_stage_samples.pt`、`C32T_stage_samples.pt`。

记录的是每个CTA的阶段经过时间，包含执行、调度和同步等待；不是各阶段单独占用整张GPU的时间。选中CTA自身有时间戳与store开销，不能将样本平均直接乘CTA数作为kernel wall time，也不能把1.8%的整体扰动当成各阶段误差上界。全程时间戳差非负，统计包括均值、中位数、P95，见stages.json。

阶段0从有效tile判定后开始，不含kernel入口至tile定位的前导代码；最后一段包含TMA store wait和原有同步。gate段还包含尾部k清零、g_total指数及少量视图设置；Neumann段包含其后的fence和同步。没有对每轮Neumann或gate内部进一步拆分。

来源：C16计时版基于只读原baseline，C32计时版基于本工作区原helper并使用已测过的launch bounds=4；未使用B1失败候选。独立overlay仅对插桩相关的K1、binding、读取时间戳接口生成诊断源码，其余文件链接原来源。新增修改见analysis/*.patch。未修改或重建未插桩baseline，C16工作区仍clean；该轮插桩实验未修改生产源码；后续已将主源码及默认构建切换为launch bounds=4，计时数据仍属于原实验。

#### 插桩扰动与同轮计时

B300，T=8192、H=96、D=128，单条等长序列，FP32 state接口，seed0。H是单卡实际头数；非TP8每卡12头。C32采用rescale=2^-96、inverse_rescale=1，lower_bound=-5；C16沿用原计算。每个实现预热30次、5轮×200次完整forward CUDA event，正反顺序交替；K1/K2各另采样20次。基于既有benchmark文件，输入及精度设置保持不变。

| 实现 | 完整forward/ms | K1/ms | K2/ms |
|---|---:|---:|---:|
| C16 | 1.000378 | 0.273331 | 0.716782 |
| C32 | 1.166200 | 0.394319 | 0.762733 |
| C16T | 1.005720 | 0.278416 | 0.717077 |
| C32T | 1.172991 | 0.401359 | 0.762896 |

T后缀为插桩版。C16 K1从0.273331→0.278416 ms（+1.86%），C32从0.394319→0.401359 ms（+1.79%）。插桩没有造成整体数量级变化，适合判断主要阶段；不据此宣称得到了无扰动的精确阶段耗时。

编译记录：C16仍无spill；C32插桩版静态spill stores/loads为8/8 bytes，而未插桩LB4为16/16 bytes。因此插桩改变了少量编译结果，进一步说明阶段值是诊断值，最终优化仍必须用未插桩版计时。

#### 每CTA阶段延迟

单位μs，均值；C16和C32每个CTA分别处理16和32个token，不能把单CTA比值直接解释成固定工作量计算量比值。

| 阶段 | C16 | C32 LB4 | C32−C16 |
|---|---:|---:|---:|
| 输入加载与TMA等待 | 1.697 | 1.481 | -0.215 |
| Q/K归一化 | 0.621 | 0.752 | +0.131 |
| gate激活、cumsum、g_total及尾部清零 | 1.327 | 2.133 | +0.805 |
| decay变换 | 0.399 | 0.868 | +0.469 |
| L/Mqk矩阵乘 | 0.448 | 0.624 | +0.176 |
| 三角处理、beta及INV初始化 | 0.240 | 0.479 | +0.239 |
| Neumann展开 | 0.424 | 1.296 | +0.872 |
| workspace写回及等待 | 1.070 | 1.493 | +0.424 |
| 被测阶段合计 | 6.225 | 9.127 | +2.902 |

C32的绝对阶段占比中，gate/cumsum约23.4%，Neumann约14.2%，decay约9.5%；Neumann是最大的单项净增量，却不是C32耗时最大的阶段。写回增加约0.424 μs；输入加载的样本延迟反而减少0.215 μs，体现经过时间还受并发及调度影响。

结合前轮NCU资源证据，C16为49152 CTAs、约8blocks/SM，C32为24576 CTAs、约4blocks/SM，148SM上二者理论waves均约41.5。C32虽CTA数减半，驻留能力也减半，因此每CTA总延迟增长与整kernel增长并不矛盾。这个waves估算仅解释趋势，不用于将表中阶段精确分摊为kernel的毫秒成本。

#### 数值验证

两种计时版分别运行现有tests/compare_naive.py全部12用例，参考未改，输出与最终state均无Inf/NaN。C32计时版12用例所有误差统计与原C32记录相同。主性能形状上，两个计时版与各自未插桩版的output、最终state均逐元素bitwise相等（stages.json/equality）。

各版误差如下，FP32/BF16 state接口统计相同合并展示；未设新门槛。

| 计时版 | H | seq_lens | output最大绝对误差 | output相对L2 | state最大绝对误差 | state相对L2 |
|---|---|---|---:|---:|---:|---:|
| C16 | 96 | [8192] | 314.683594 | 0.00385814965 | 0.00904005766 | 0.00467086969 |
| C16 | 96 | [1300, 547, 2048, 963, 271, 3063] | 1796.0625 | 0.00347523178 | 0.0094974637 | 0.00458659226 |
| C16 | 96 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 3514.625 | 0.00400174923 | 0.00916033983 | 0.00456152792 |
| C16 | 64 | [8192] | 163.449219 | 0.00391237906 | 0.00847822428 | 0.00455767378 |
| C16 | 64 | [1300, 547, 2048, 963, 271, 3063] | 1268.53125 | 0.0035574994 | 0.00973063707 | 0.00459241982 |
| C16 | 64 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 1425.25 | 0.00343960019 | 0.00847822428 | 0.00460593683 |
| C32 | 96 | [8192] | 314.683594 | 0.00386518907 | 0.00791704655 | 0.00458132069 |
| C32 | 96 | [1300, 547, 2048, 963, 271, 3063] | 1796.0625 | 0.00347572915 | 0.0094974637 | 0.0045906454 |
| C32 | 96 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 3514.625 | 0.00400208345 | 0.00905925035 | 0.00456350217 |
| C32 | 64 | [8192] | 163.449219 | 0.00390619811 | 0.00813734531 | 0.00459004285 |
| C32 | 64 | [1300, 547, 2048, 963, 271, 3063] | 1268.53125 | 0.00355141306 | 0.00696307421 | 0.00459145836 |
| C32 | 64 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] | 1425.25 | 0.00343433461 | 0.00871014595 | 0.00462786221 |

原始参考哈希、二进制哈希、命令见naive_C16.json和naive_C32.json。C16 loader仅补充参考脚本所需的默认scale元数据，调用C16时不传scale关键字；保存metadata明确chunk=16。