# 02_tcgen05_evaluation：tcgen05计算重组

## 本目录实验目标

检验CHUNK=16能否通过共享操作数合并或转置匹配tcgen05，以及是否有实际收益。

**实验方法：** 对原阶段GEMM作代数等价重组，运行独立正确性、计时和NCU执行验证。

**结果与边界：** 局部GEMM有收益，69项检查通过，但没有证明完整KDA换指令后的端到端收益。

脚本、运行顺序和依赖见[profile README](../README.md)，全部主题的报告合并见[总报告](../REPORT.md)。以下保留原实验的测量和限制说明；本次整理没有重新执行GPU实验。

# Topic2：CHUNK=16 的 tcgen05 计算重组与 B300 实测

**结论：padding 不是唯一办法。源码确认共享操作数合并和完整转置两类重组，可以减少或消除无效计算。B300 的 69 组独立 GEMM 正确性检查通过；共享 state 的 M32 合并运算在 batch=1 下，ws 从 mma.sync 的 15.0920 µs 降至 14.2695 µs，旧/新=1.058×，耗时下降 5.45%。batch=16384 时，所有测试的新路径仍慢于同任务 mma.sync。不能再用“CHUNK16 只能 padding，所以 tcgen05 不值得”解释结果。**

此处“保持数学结果”指 GEMM 代数等价；不是逐位相等保证，也不是完整 KDA kernel 已验证。生产代码未修改。

## 1. 逐项源码审查

以 CHUNK=16、D=128 为准。P 表示 Neumann 当前的矩阵幂，R 表示已按原代码算出并转换成 BF16 的 residual。

| 阶段及源码 | 原运算 | 可用重组 | 形状与限制 | 本次验证 |
| --- | --- | --- | --- | --- |
| [K1 构造 L/Mqk](../../csrc/smxx/fwd_kernel1.cuh#L481) | K_decayed@K_inv 与 Q_decayed@K_inv | 堆叠两组左操作数，共用 K_inv | (32,16,128)；普通仍需 M64，ws 仍需 N64；比两次分别 padding 更省部分无效计算 | K1 shared-B pair |
| [Neumann 初始 L²](../../csrc/smxx/utils.cuh#L262) | L@L | 方阵转置仍是 M16；当前序列中无已就绪的同 B 伴随运算 | (16,16,16)，保留单运算时仍需 padding | Neumann square |
| [Neumann L⁴/L⁸ 及逆更新](../../csrc/smxx/utils.cuh#L267) | INV@P 和 P@P | 堆叠 INV 与 P，共用旧 P；完成后分别执行原 FP16 加法与下一阶段 | (32,16,16)；普通 M64，ws N64；最后 INV@L⁸ 无下一次平方可配对 | Neumann shared-power pair，仅测试配对 GEMM |
| [K2 Phase 1](../../csrc/smxx/fwd_kernel2.cuh#L533) | K_decayed@S 与 Q_decayed@S | 堆叠 K/Q，共用 S | (32,128,128)，ws 无 padding；普通仍需 M64 | K2 shared-state pair |
| [K2 Phase 3](../../csrc/smxx/fwd_kernel2.cuh#L589) | INV@R | 完整转置 Rᵀ@INVᵀ，结果转回 | (128,16,16)，普通无 padding；不得跳过 residual 的 BF16 舍入 | INV@residual / Mqk@U |
| [K2 Phase 4](../../csrc/smxx/fwd_kernel2.cuh#L625) | Mqk@U | 完整转置 Uᵀ@Mqkᵀ，结果转回 | (128,16,16)，普通无 padding；U 依赖前一阶段，不能与 INV@R 当作共享 B 的独立乘法合并 | 与上一项同 GEMM 测试，实际中间数据未移植 |
| K2 Phase 1 的单项备选 | K/Q@S | Sᵀ@(K/Q)ᵀ，结果转回 | (128,16,128)，普通无 padding | single K/Q@state |
| [K2 Phase 6](../../csrc/smxx/fwd_kernel2.cuh#L659) | K_restoredᵀ@U | 原形状已匹配；保留即可 | (128,128,16)，无 padding | state update |

共享 state 合并满足 `[K;Q]S=[KS;QS]`。转置必须同时交换并转置两个操作数：`AB=(BᵀAᵀ)ᵀ`，绝不是只转置 A 或 B。本次 CUDA 转置路径直接按原输出方向合并写回，不额外启动输出转置 kernel。

K1 两组乘法都使用 BF16 输入、FP32 累加，但分别写为 FP16 的 L 与 BF16 的 Mqk；合并时必须分别恢复这些转换。Neumann 的每次乘法输出和 INV 加法仍需保留 FP16 舍入位置。K2 中 BF16 residual、U 和输出加法的转换顺序也不能删除。因此本次仅验证重组乘法本身，不能将其说成完整算法正确性验证。

## 2. 其他组合的审查结果

- **Mqk@U 与 K_restoredᵀ@U 也共享 B。** 堆叠后有效 M=144，超过单条指令 M128。用 ws M128+M32 共执行 160 行，与分别执行 M128、M32 相同；普通 M128+M64 共 192 行，也与分别 padding 相同。它可能通过同 CTA 内复用 TMEM、同步和 U 改善调度，但没有单靠形状合并减少 FLOPs 的收益。本次未实现融合两阶段的 kernel，不对这种调度的收益下结论。
- **Neumann 依赖必须保留。** 配对的是使用同一旧 P 的 INV@P 与 P@P；不能把依赖新 INV 或新 P 的后续乘法提前。初始 L²和最后 INV@L⁸仍是单项。
- **不同时间 chunk 的 S 不共享且存在递推依赖。** 不同序列/head 也通常具有不同操作数，不能套用共享 B 堆叠。可以计算所有 A_iB_j 再取对角块，但会引入无用的交叉乘积；不等同于免费 batch，也未证明比 padding 更好。
- **gate、beta、mask、残差和 state 衰减**不是待替换的 GEMM，保持原有计算及精度转换。本轮不将这些操作融合进 GEMM 或更改 Neumann 算法。

## 3. 公平比较的边界

所有路径使用相同有效矩阵、相同 batch、相同 FP16/FP32 累加类型和 FP32 输出接口。共享 B 的配对任务让 **mma.sync 也使用同一 CTA、相同堆叠输入和 B 复用**，避免用新路径一次 launch 对比旧路径两次 launch。各路径仍需各自的正确布局和同步。

输入物理打包和 padding 均在计时前准备，转置路径的输入重排也在计时前准备。计时包含 global→shared、MMA、TMEM 管理、同步和原方向输出写回。故结果回答的是“操作数已经按路径布局准备好之后”的独立 GEMM 性能；完整 KDA 接入时的输入重排、寄存器到 shared 的转换尚未计入。这不是生产 kernel 的融合时序模拟，也不证明已找到最优 tcgen05 实现。

每项 3 个随机 seed，每个 seed 3 个矩阵，对量化后输入的 FP64 GEMM 检查有限性及相对 L2；FP16 阈值 0.003，BF16/FP32 阈值 2e-5。7 项共 69 个检查记录。计时每条路径 12 轮，轮换顺序，CUDA Graph 与 events；单任务每图 32 次、大批每图 4 次。NCU replay 时间不作为性能数据。

## 4. 实测与复现

全部时间、正确性误差、12 轮原始样本及机器信息集中在 RESULTS.md（本地证据：`RESULTS.md`）。

- 共享 state 配对，batch=1：mma.sync 15.0920 µs，ws 14.2695 µs；采样区间分别 15.0710–15.1080、14.2520–14.3230 µs，本轮不重叠。
- 同一配对，batch=16384：mma.sync 1089.6200 µs，ws 2760.9360 µs，旧/新 0.395×。无 padding 仍不保证快。
- M16×N128、K16，batch=16384：普通 padding 1611.3520 µs，完整转置 1236.0720 µs，减少约 23.29%；但 mma.sync 仅 83.9280 µs。
- M16×N128、K128，batch=16384：普通 padding 3237.1320 µs，完整转置 3194.8240 µs；差距较小，不能当作稳定收益。ws padding 为 2725.6560 µs，仍比该转置实现快。

统一测试入口 [bench.py](harness/bench.py)、CUDA 实现 [kernels.cu](harness/kernels.cu)、编译和 B300 调度 [run.sh](harness/run.sh)。运行 `bash profile/02_tcgen05_evaluation/harness/run.sh` 可重现。本次 basic.ncu-rep（本地证据：`reports/basic.ncu-rep`） 使用 basic 集合，采集 batch=16384 的 23 个路径，作为实际 CUDA kernel 执行证据；不是完整 stall/source 性能归因报告。

**采用建议：优先把共享 state 的 M32 合并作为后续集成候选，而非把所有 M16 运算统一 padding。当前独立测试尚不支持全面改用 tcgen05；生产 KDA 中的 TMEM 复用和输入布局接入成本仍需实际集成才能判定。**
