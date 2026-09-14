# 03_recurrence_parallelism：递推并行方案

### 实验设计与实现路径

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

## 本目录实验目标

在chunk间存在state依赖时，寻找仍可增加GPU并行度的方式。

**实验方法：** 比较value列分片、双head合并、静态/动态persistent，保留原CHUNK/MMA/精度；检查门控约定、输出一致性、完整及K2耗时和NCU。

**结果与边界：** 列分片在少链长序列有收益；双head合并未测到中位数收益；persistent收益依赖任务分布，不能通用替换。新增 CHUNK16 真实形状的协作 2-CTA 独立 GEMM 测试，见 4.6；其计时不与前三组完整 K2 混用。

脚本、运行顺序和依赖见[profile README](../README.md)，全部主题的报告合并见[总报告](../REPORT.md)。以下保留原实验的测量和限制说明；本次整理没有重新执行GPU实验。

# Topic3：递推依赖下的并行方案

**结论：列分片在少链长序列上仍有收益，直接双 head 合并未测到中位数收益；动态 persistent 可以修复静态分配倾斜，个别场景有小幅收益，但不支持通用替换原 grid。**

本轮已修正三组封装中遗漏的 `log2(e)` 转换，统一采用自然指数约定的 `lower_bound=-5`，即传给 launcher 的 `gate_scale=-5×log2(e)`。以下数据为 2026-09-10 在 B300（CC10.3、148 SM）重新编译、检查、计时和采集 NCU 的结果，替代旧参数下的测量。

三组保留 CHUNK=16、D=128、BF16 状态、真实 K1/K2 和 mma.sync。每项计时采用 CUDA Graph（每图 3 次调用）、9 轮轮换采样和 CUDA events 中位数；K2 单独计时排除 K1，K1+K2 包含预处理和完整递推。加速比=同轮原版耗时/变体耗时，NCU replay 时间不参与加速比。不同组、不同轮次的绝对时间不可直接相除，也不能把旧测量与本轮的时间差归因为门控参数修正。

### 4.1 独立链、实现一致性与门控检查

K1 可跨 chunk 并行；K2 原版每个 sequence/head 对应一个 CTA，独立链数 P=N×H_local。96 heads、TP8 时单序列每卡只有 12 条链，最多覆盖 148 SM 的 8.11%；value 四分后可增加到 48 个 CTA，纸面上限 32.43%。这只是任务映射推算，本组没有恰好 H=12 的计时。

三组原有的 48、16、80 个 output/final-state 比较共 144 项全部与各自原版逐位一致，覆盖非零初始状态、尾 chunk、变长序列、T4096，以及 persistent 的 600 条任务跨任务复用。它们验证实现一致性，不能独立证明两者都符合数学参考。

新增门控回归检查直接读取 K1 workspace 中的 GT：令 g=A_log=dt_bias=0、chunk 长度为 16，预期 GT=exp(-5×16/2)=4.248354255×10⁻¹⁸。该检查不以测试封装的原版作为参考，旧的直接传 -5 会得到约 9.094947018×10⁻¹³，因而无法通过。

三组共 11 项门控检查全部通过，最大相对误差为 5.48e-06（阈值 1e-4）。完整性能记录共 206 条。

### 4.2 value 列分片

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

### 4.3 两个 head 共 CTA

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

### 4.4 跨链 persistent worker

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

### 4.5 NCU 证据与结论边界

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
| 协作 2-CTA | 两个 CTA 协作完成同一任务的加载和计算 | 短任务中同步与交换成本可能超过收益；value 分片本可独立时，协作可能增加不必要的通信；独立链充足时，每链占用两个 CTA 可能降低吞吐。本轮未实现、未测，以上为待验证反例 |

**协作 2-CTA 的候选反例（一般设计推理；具体 DSM 实现的新增实测见 4.6）。** 协作不能解除 chunk 间状态依赖：两个 CTA 即使共同算完当前 chunk，下一 chunk 仍必须等待更新后的状态。它只能尝试缩短每个 chunk 的处理时间，不能把相邻 chunk 直接变成独立任务。

- **短序列或只有一两个 chunk：** 可分摊的计算量较少；若方案需要额外 barrier、跨 CTA 数据交换或结果汇合，这些成本可能超过协作节省的时间。反驳“分给两个 CTA 就一定更快”。
- **按 value 分片已经可以独立计算：** KDA 不同 value 分片可各自维护状态并完成整条递推。若协作方案仍做相同的分片，却额外引入通信或共享阶段，就可能慢于无需相互等待的独立 split2。协作必须有实际的数据复用或工作分摊收益，不能仅以 CTA 数增加为依据。
- **独立序列/head 已足够多：** 原 grid 已有足够任务覆盖 GPU，给每条链分配两个 CTA 会增加资源占用；若每链耗时没有相应下降，整体吞吐可能反而降低。若采用 cluster/共驻留约束，还需考虑成组调度和尾部利用率。
- **两个 CTA 工作不均衡：** 若分工造成一方长期等待，或每个 chunk 都需等待较慢的一方，协作收益受慢方限制。不能将两个 CTA 的理想算力简单相加。

这些一般性反例是候选设计需要面对的条件；4.6 仅验证一个具体 DSM 协作实现，不能据此认为全部协作设计都已验证。应同时对照原版单 CTA 和独立 split2，检查少链长序列的延迟、短序列开销及多链吞吐，而不是只选择最有利的输入。

以上是当前三种实现和合成输入的结论，不是其设计空间的性能上限。

### 4.6 新增：CHUNK16 实际形状的协作 2-CTA microbench

**结论：本次实现的 DSM 共享 A 协作版，在 5 种形状 × 6 档任务数的 30 个条件中，中位数均慢于本组单 CTA 和独立 split2。** 这是 2026-09-11 在 B300（CC10.3、148 SM）新增的独立 GEMM 实验，不是完整 K2，也不是 `tcgen05.cta_group::2`。前三组生产递推测试及其历史数据保持原样。

#### 4.6.1 实际 KDA 形状与接口

固定 CHUNK=16、D=128，只改变独立任务数 `batch∈{1,8,64,148,256,16384}`，覆盖少任务、接近 SM 数、多任务和大批量。batch 不是序列长度，不将相邻 chunk 当作可独立执行的状态任务。

| shape ID | KDA 中的计算 | M,N,K | 输入 / 累加 |
| --- | --- | --- | --- |
| 0 | Neumann 中的一次乘法 | 16,16,16 | FP16 / FP16 |
| 1 | K1 的 KD×KIᵀ 或 QD×KIᵀ | 16,16,128 | BF16 / FP32 |
| 2 | K2 的 INV×R 或 Mqk×U | 16,128,16 | BF16 / FP32 |
| 3 | K2 的 KD×Sᵀ 或 QD×Sᵀ | 16,128,128 | BF16 / FP32 |
| 4 | K2 的 KRᵀ×U（状态更新的转置方向） | 128,128,16 | BF16 / FP32 |

接口为 `run_cooperative(shape, mode, batch, A, B, output, stream)`；shape 选择上表的预编译实例，不改变 CHUNK。A/B 是相同的 8×8 interleave 打包输入，B 按转置视图打包，输出统一为 FP32。尚未融合 beta、gate、mask、残差、状态衰减和中间格式转换，不执行完整 Neumann 级数。

#### 4.6.2 三条对照路径

- `single_cta`：一个 CTA 完成一个有效 GEMM，最多 4 个计算 warp。
- `independent_split2`：两个独立 CTA 沿输出 N 维各算一半，最多各 2 个 warp，分别从 global 加载同一 A 及自己所需的 B 分片。K2 中 N 对应 value 通道；这是本微基准的独立分片控制组，不是历史 production split2 kernel。
- `cooperative_cluster2`：相同 N 分片，但两个 CTA 组成实际的 2-block cluster。rank0 加载 A，rank1 通过 `cluster.map_shared_rank` 读取 rank0 的 A 到自己的 shared buffer；两者各自加载 B 分片。第一次 `cluster.sync()` 保证 A 已就绪、两个 block 都存活，第二次保证远程读取结束后源 block 才能退出。随后各自 MMA 并写回不重叠的输出分片。

三条路径都使用相同 `mma.sync.m16n8k16`、相同每元素 K 归约顺序，不通过换 Tensor Core 指令引入额外变量。对于很小的形状，warp 数按输出 tile 数缩小；single 与两个 split CTA 的总计算 warp 数一致。N=16 直接分成两个 N8，没有 padding。

该协作设计明确尝试用 DSM 传输替代重复的 A global load：逻辑 global 输入量从 independent 的 `2|A|+|B|` 降到 `|A|+|B|`，但增加 `|A|` 的 DSM 传输、两个 cluster barrier 和 cluster 调度要求。各 CTA 仍保留自己的 A shared buffer，不声称节省每 CTA shared memory；缓存也会影响实际 HBM 流量，不能把逻辑加载减少等同于 HBM 字节同比减少。代码用 CUDA cluster API 实现的软件数据协作，不保证两个 CTA 必须落在不同 SM，也不应写成已验证 2-SM tcgen05。

同步依据：[CUDA 13.0 Programming Guide — Distributed Shared Memory](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#distributed-shared-memory)。要求访问远端 shared 时源 block 仍存活，并在退出前完成远端访问。

#### 4.6.3 输入、对拍和计时

输入采用 `randn×0.125` 后量化到对应类型；这是匹配 KDA 形状的独立 GEMM 输入，不是从真实模型抽出的 K1 workspace。正确性覆盖 3 个随机 seed、全零 A、带交替符号的结构化矩阵，batch=3。5 形状×5 输入×3 路径共 75 项检查，对照同一量化输入的 FP64 GEMM；FP16 相对 L2 阈值 0.003，BF16/FP32 阈值 2e-5。所有输出有限，且分片与协作结果均与单 CTA 逐位一致。计时的所有 batch 也检查输出有限和逐位一致。

每路径预热 5 次；CUDA Graph 内重复 3 次，图再预热回放 3 次；CUDA events 计时 9 轮，轮换路径顺序，报告除以图内调用次数后的中位数。输入打包与 buffer 分配不计时；global→shared、DSM 传输、cluster 同步、MMA 和写回均计时。结果是完成整个 batch 的一次 kernel 耗时，未除以 batch。输入和输出 buffer 重复使用，属于热缓存条件；没有模拟 K2 的状态更新和逐 chunk 供数。原始 90 条性能记录及 9 轮样本保存在 `cooperative_2cta/RESULTS.md`。

Compute Sanitizer 的 memcheck、racecheck、synccheck 分别检查相同正确性用例，三项均通过，memcheck/synccheck 为 0 errors，racecheck 为 0 errors、0 warnings，输出保留于 `cooperative_2cta/validation/`。本组不采集 NCU，不从计时结果推断带宽瓶颈或同步开销占比；提供 `--profile` 入口供后续采集。

#### 4.6.4 代表性结果

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

#### 4.6.5 复现与文件

- `harness/cooperative_2cta.cu`：三条 kernel 路径与统一 C 接口。
- `harness/cooperative_2cta.py`：shape 表、构建、正确性、计时及结果保存；`--batches` 接收独立任务数列表，`--check-only` 仅做对拍。
- `cooperative_2cta/RESULTS.md`：完整结果、原始样本和 GPU 环境。


```bash
bash profile/03_recurrence_parallelism/harness/run.sh cooperative_2cta
```

该组先构建，依次执行三种 sanitizer，再执行独立计时；复测前自动将已有该组结果和验证记录复制到 `cooperative_2cta/archive/`，不会覆盖前三组数据。直接调用 Python 计时入口会重写本组 `RESULTS.md`，需要保留历史时应使用上述 shell 入口。`all` 已包含此新增组；该组保留编译库供检查和后续采集，不执行前三组的 full/source NCU 流程。

### 复现与数据

运行 `bash profile/03_recurrence_parallelism/harness/run.sh all` 重跑全部，或指定 `column_split`、`multi_head`、`persistent`、`cooperative_2cta`。脚本会更新所选组的结果；重跑前如需保留历史应自行归档。前三组构建从生产源码生成隔离副本，新增协作组使用独立 GEMM harness；生产 csrc 不作修改。各组 RESULTS.md 保存门控检查、实现对拍、全部计时样本和环境；PROFILE.md 保存 NCU 解析；SOURCE.json 保存构建源码指纹。
