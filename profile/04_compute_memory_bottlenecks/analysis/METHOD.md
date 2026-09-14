# 实验口径与纸面模型

## 调用与 warmup

直接编译仓库 [fwd_launch.cu](../../../csrc/smxx/fwd_launch.cu)，调用 `launch_fwd<128,true,true,false,false>`。这个原始 launcher 发射 prepare（K1）与 recurrence（K2），K1 的真实 workspace 直接供 K2 使用。[bridge.cu](../harness/bridge.cu) 只有主机侧 ABI 转接，没有新增或改写 GPU kernel。`csrc` 各文件的 SHA256、编译命令及本次二进制 SHA256 保存在 [source.json](source.json)。

固定 D=128、CHUNK=16、BF16 输入/输出/初末状态、普通等长 batch。q/k/v 为 seed=42 的正态随机数×0.125，g/beta 为正态随机数，A_log/dt_bias=0，lower_bound=-5，非零初始状态为正态随机数×0.02。数据代表本次合成负载，不声称来自真实模型 trace。输入分配、beta 转置、workspace 分配在计量区外完成。

| 场景 | N | H | T | K1 CTA | K2 CTA | 每个 K2 CTA 的 chunk 数 |
|---|---:|---:|---:|---:|---:|---:|
| task_n1_h96_t8192 | 1 | 96 | 8192 | 49152 | 96 | 512 |

采用 assignment02/cuda/common.h:65 的 20 次 warmup，额外在计量前显式同步。每轮 application replay 都重新创建输入并执行 20 次完整 forward，然后以 `cudaProfilerStart/Stop` 和 NVTX `场景/step21` 标记第 21 次 forward。只过滤原始 K1/K2；full、source 分开采集，均应包含 2 个 action，解析时检查名称、数量及 CTA 数。

NCU 使用 application replay、`--cache-control none --clock-control none`：让每轮经过同样 warmup，不强制刷新 K1→K2 的 workspace 缓存。NCU 仍会插入采集/同步操作，所以时间与缓存指标属于该测量协议，不能完全等同无 profiler 的端到端行为。频率不锁定，不能把小幅差异解释为优化收益。full 加 PmSampling、PmSampling_WarpStates；source 加 SourceCounters；均带源码关联。

独立 CUDA event 计时采用 20 次 warmup 后 9 组×20 次完整 forward，报告每组每次耗时的中位数。它包含 launcher 与 Python 发射可能造成的 GPU 间隙，不用于代替 NCU 的单 kernel 时间。正确性调用仓库 tests/torch_ref.py，覆盖 T=16、T=31（含尾块）、非零初末状态；正式指定形状另检查输出和状态均为有限值。

## 纸面算术强度：只作假设，不代替计数器

D=128、C=16，每个 head/chunk 的 workspace 是 3×C×D×2 + D×4 + 2×C²×2 = **13,824 B**。

K1 两个 C×D 乘 D×C，加 Neumann 的六次 C×C 乘 C×C，共 `4 C² D + 12 C³ = 180,224` 个矩阵 FLOP（FMA=2）。q/k/g 逻辑读取 12,288 B，beta 32 B，写 workspace 13,824 B；忽略跨 chunk 共享的 A_log/dt_bias，得到 `180224/26144 = 6.894 FLOP/B`。这不是完整 K1 的 FLOP 计数：归一化、sigmoid、指数、归约、类型转换和同步不能用 Tensor 峰值衡量。

K2 有三个 C×D×D 级 GEMM（K@S、Q@S、状态更新）和两个 C×C×D GEMM（INV@U、Mqk@U），矩阵 FLOP 为 `6 C D² + 4 C² D = 1,703,936`。每 chunk 逻辑读 workspace 13,824 B、v 4,096 B、beta 32 B，写 output 4,096 B，共 22,048 B。初末状态每条链另读写 65,536 B，因此 L 个 chunk 的逻辑强度为 `1703936/(22048+65536/L)`：L=512 时 **76.837 FLOP/B**。循环中的状态在 CTA 内复用，不能按每 chunk 都读写 HBM 算。

这些是逻辑张量字节，不包括 cache 命中、事务放大或重复指令访问。尤其 warmup 后的小负载可命中 L2，实际 HBM 字节可能远小于上述模型。应以报告的 DRAM 字节/带宽和 L2 指标检验。

## 与 assignment 4.5 的关系

参考 [assignment02 writeup 4.5](/home/lcpu/60990375/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/handout/src/writeup.md:714)。其 `in_proj_qkvgfab` 是 KDA 输入投影，N=6288、K=7168，M 是投影的 token 行数，不是本 kernel 的 CHUNK。

BF16 GEMM 理想强度为 `AI=2MNK/(2MK+2NK+2MN)=1/(1/M+1/N+1/K)`。沿用该作业的 2250 TFLOPS、8000 GB/s，平衡点 281.25 FLOP/B，对应 M≈307；M=16 的 AI≈15.9，M=1024 的 AI≈784.2。它说明形状会改变算术强度，但不能把投影结论直接套给递推 kernel。该表的 GB/s 是逻辑最低字节量除以时间，也不能当成本实验 NCU 测到的 DRAM GB/s。

这里没有重新测试输入投影，也没有引入 02_tcgen05_evaluation/3 的 kernel 变体。
