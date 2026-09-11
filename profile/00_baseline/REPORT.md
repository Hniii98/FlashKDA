# 00_baseline：原始KDA基线

## 本目录实验目标

建立原FlashKDA的正确性与完整forward性能基线，为后续实验提供对照。

**实验方法：** 比较FlashKDA与FLA路径在H=96/64、等长和变长输入下的完整forward，并用assignment02的naive参考检查output与最终state。

**结果与边界：** 性能与误差口径见下文；不能把不同计时协议的微基准直接与完整forward相除。

脚本、运行顺序和依赖见[profile README](../README.md)，全部主题的报告合并见[总报告](../REPORT.md)。以下保留原实验的测量和限制说明；本次整理没有重新执行GPU实验。

# KDA forward benchmark (Blackwell / B300)

- Generated: 2026-09-05

- Command: `srun -G 1 python benchmarks/generate_benchmark_md.py -o profile/00_baseline/REPORT.md --device-label "Blackwell / B300"`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 1.7367 | 3.9900 | 2.30× | 1.9990 | 1.15× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.5020 | 4.1155 | 2.74× | 2.0592 | 1.37× |
| Varlen, `seq_lens`=`1024 x 8` | 1.2168 | 4.0127 | 3.30× | 1.9642 | 1.61× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 1.5857 | 2.7114 | 1.71× | 1.3551 | 0.85× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.1332 | 2.8696 | 2.53× | 1.4753 | 1.30× |
| Varlen, `seq_lens`=`1024 x 8` | 0.8246 | 2.6475 | 3.21× | 1.3030 | 1.58× |

## 运行与验证

- 官方 `bench_fwd.py` 和 `generate_benchmark_md.py` 未修改；形状与 `BENCHMARK_GB200.md` 一致。表格采用 FP32 initial/final state，输入 BF16。没有设置额外 CPU 数量或 `FLA_FLASH_KDA` 环境变量。
- B300 SXM6 AC，SM103；驱动 580.126.09，CUDA toolkit 13.0.88，PyTorch 2.14.0+cu130，FLA 0.5.2，Nsight Compute 2025.3.1。
- 直接执行仓库 `benchmarks/ncu.sh`。仅将其中安装命令补为 `pip install -e . --no-build-isolation`，以使用当前环境中的 torch；ncu 命令及采集参数保持原样。
- ncu 作业 19960 成功结束；扩展重装后再以官方命令完成最终 benchmark（作业 19962），确保表格和 SASS 对应同一份扩展。
- ncu 按官方脚本覆盖 H=96 的 fixed、两组 varlen，以及 BF16 state / no state / FP32 state；共 **36 + 72 = 108** 条 prepare/recurrence 记录。H=64 已完成 benchmark，但官方 ncu.sh 不采集 H=64。

## SM80 MMA 证据

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

## 复现

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

## 与 `fla_kda_ref/naive.py` 对拍

直接通过文件路径加载材料快照中未修改的 `naive_recurrent_kda`，使用其逐 token 的 FP32 PyTorch 递推作为参考；没有经 FLA backend dispatch，也没有用 FlashKDA 自身作参考。本节未运行 `chunk.py`。

- 运行：Slurm 作业 24782，GPU **NVIDIA B300 SXM6 AC**，PyTorch `2.14.0+cu130`；扩展 SHA256 与上面的 baseline 一致。
- 参考文件：`/home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/fla_kda_ref/naive.py`；SHA256：`60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016`。
- 覆盖上表全部 6 组形状，每组分别运行 FP32 / BF16 initial、final state，共 12 次对拍。每组固定 `torch.manual_seed(0)`，输入生成方式遵循 `benchmarks/bench_fwd.py`：BF16 Q/K/V/g/beta，Q/K 先归一化再转 BF16，A_log、dt_bias 为 FP32 uniform；initial state 为 `arange(N*H*D*D).reshape(N,H,D,D).bfloat16()`，FP32 路径再转为 FP32。
- 参考预处理：Q/K 使用 `x / sqrt(sum(x²)+1e-6)`；gate 为 `-5*sigmoid(exp(A_log)*(g+dt_bias))`；beta 为 `sigmoid(beta)`，scale 为 `1/sqrt(128)`，全部在 FP32 计算，关闭 TF32。参考 V 使用 FP32，使输出不额外舍入为 BF16；因此统计包含 FlashKDA 的输出量化误差及内部近似误差。
- state 布局由 FlashKDA 的 `[N,H,V,K]` 显式转成参考的 `[N,H,K,V]`，返回后转回；varlen 按 cu_seqlens 分段独立调用参考，使用各自的 initial state。
- relative L2 = `||actual-reference||₂ / ||reference||₂`；mean/max abs 为绝对差的均值/最大值，指标归约使用 FP64。完整的 12 组指标及有限值检查见 fla_ref_errors.json（本地证据：`fla_ref_errors.json`），脚本见 [compare_fla_ref.py](compare_fla_ref.py)。

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
