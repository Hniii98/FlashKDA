# FlashKDA profile：实验与脚本索引

完整实验记录见[总报告](REPORT.md)。这里保留各主题主报告、方法说明、复现脚本及手写CUDA源文件；原始NCU、日志、JSON、图像、编译产物和自动生成的详细结果文件仍在本地，不随仓库提交。

## 实验目录

| 目录与报告 | 实验问题 | 原目录 |
|---|---|---|
| [00_baseline](00_baseline/REPORT.md) | 原始KDA的正确性与完整forward性能怎样？ | Baseline |
| [01_chunk_size_analysis](01_chunk_size_analysis/REPORT.md) | CHUNK=16/32/64的数值范围、求逆及MMA形状限制是什么？ | topic1 |
| [02_tcgen05_evaluation](02_tcgen05_evaluation/REPORT.md) | CHUNK=16能否通过计算重组使用tcgen05，并取得收益？ | topic2 |
| [03_recurrence_parallelism](03_recurrence_parallelism/REPORT.md) | 递推依赖下，列分片、多head、persistent及协作2-CTA是否有用？ | topic3 |
| [04_compute_memory_bottlenecks](04_compute_memory_bottlenecks/REPORT.md) | K1/K2的计算、访存、同步和并行度瓶颈在哪里？ | topic4 |
| [05_bf16_state_accuracy](05_bf16_state_accuracy/REPORT.md) | 每个chunk将state保存为BF16带来多大精度代价？ | topic5 |
| [06_sm100a_design_decision](06_sm100a_design_decision/REPORT.md) | 综合证据，是否应新增SM100a专版？ | topic6 |

第06目录是综合设计结论，没有新GPU实验。以上实验不是另一个C32实现仓库的LB4实验；不要混用两套报告或二进制。历史原始数据、NCU内嵌源码及编译记录中的旧目录名保留，按本表映射查找；可执行脚本使用新目录名。

## 环境与数据来源

- 既有实验使用B300（CC10.3、148SM），CUDA13.0与Nsight Compute2025.3.1；各报告注明更具体的环境及计时口径。不能把这些结果称作SM100设备的实测。
- 需要配置仓库的 `.venv`、PyTorch/CUDA、nvcc，以及依赖的CUTLASS头文件。部分脚本写死 `/usr/local/cuda` 和 `/opt/nvidia/nsight-compute/2025.3.1` 路径，换环境须调整。基线benchmark还需要原报告所用的FLA环境和已构建FlashKDA扩展。
- 数学参考从课程assignment02的 `fla_kda_ref/` 获取，本次材料位置为 `assignment02/team/c1_flashkda/fla_kda_ref/`。使用未经修改的 `naive.py`；对拍脚本用 `--ref-dir` 指向该目录。
- 03/04/05的构建依赖本仓库 `csrc`；04/05是桥接真实kernel，不是只依赖profile内的独立实现。复现旧数据前应确认源码与原报告记录一致，不能把不同源码或门控参数的结果混合比较。
- 总报告与各目录说明是记录，不会随脚本自动同步更新。部分脚本会重写对应目录REPORT或RESULTS；重新生成后需检查实验目标和总报告是否仍一致。

## 00：基线

| 文件 | 用途 | 运行及输出 |
|---|---|---|
| [compare_fla_ref.py](00_baseline/compare_fla_ref.py) | 原FlashKDA与naive参考的output/final-state对拍 | `--ref-dir`必填，`--output`可指定结果JSON；需GPU及已构建扩展 |

完整基线报告由profile外的 [generate_benchmark_md.py](../benchmarks/generate_benchmark_md.py) 生成；[bench_fwd.py](../benchmarks/bench_fwd.py) 是原benchmark入口。示例：

```bash
srun -G 1 .venv/bin/python benchmarks/generate_benchmark_md.py -o profile/00_baseline/REPORT.md --device-label "Blackwell / B300"
srun -G 1 .venv/bin/python profile/00_baseline/compare_fla_ref.py --ref-dir "$REF_DIR" --output profile/00_baseline/fla_ref_errors.json
```

## 01：CHUNK大小

| 文件 | 用途 | 参数／输出 |
|---|---|---|
| [kernels.cu](01_chunk_size_analysis/harness/kernels.cu) | 指数范围、Neumann及相关矩阵计算的独立CUDA实验实现 | 由run.sh编译，不是生产C32 forward |
| [bench.py](01_chunk_size_analysis/harness/bench.py) | 数值实验、NCU区域标记、指标解析和报告生成 | 命令 `run/profile/analyze/report/pack/tidy`；前三者需要 `--chunk 16/32/64`，解析支持 `--tag full/source` |
| [run.sh](01_chunk_size_analysis/harness/run.sh) | 编译并顺序完成三种CHUNK的实验、NCU和汇总 | `bash profile/01_chunk_size_analysis/harness/run.sh`；遇到既有结果会要求先归档 |

`pack`整理生成结果；`tidy`清理部分中间文件。它们不是只读查看命令，执行前检查脚本。需要重跑时用新输出位置或先归档旧数据。

## 02：tcgen05

| 文件 | 用途 | 参数／输出 |
|---|---|---|
| [kernels.cu](02_tcgen05_evaluation/harness/kernels.cu) | mma.sync与tcgen05配对／转置GEMM微基准 | 保留作手写复现源码；由run.sh编译 |
| [bench.py](02_tcgen05_evaluation/harness/bench.py) | GEMM正确性与延迟测量；`--profile`框定NCU区域 | 生成RESULTS及环境、计时记录 |
| [run.sh](02_tcgen05_evaluation/harness/run.sh) | 构建、Slurm运行、basic NCU采集与哈希清单 | `bash profile/02_tcgen05_evaluation/harness/run.sh`；完成后清理临时库 |

## 03：递推并行度

| 文件 | 用途 | 参数／输出 |
|---|---|---|
| [build.py](03_recurrence_parallelism/harness/build.py) | 从当前生产源码生成隔离的列分片、多head或persistent变体 | 默认column_split，`--heads`或`--persistent`选择另外两组；生成独立库和源码指纹 |
| [bench.py](03_recurrence_parallelism/harness/bench.py) | 门控检查、实现一致性、K2及完整K1+K2计时 | 同样支持组选择；`--profile`框定NCU，`--check`只执行检查路径 |
| [analyze.py](03_recurrence_parallelism/harness/analyze.py) | 解析该组NCU的资源、stall和PM数据 | 支持 `--heads`、`--persistent`；默认列分片，生成PROFILE.md |
| [run.sh](03_recurrence_parallelism/harness/run.sh) | 各组构建与测量；前三组采集 full/source NCU，协作组执行 sanitizer 和独立计时 | `all`（默认）或 `column_split/multi_head/persistent/cooperative_2cta` |
| [cooperative_2cta.py](03_recurrence_parallelism/harness/cooperative_2cta.py) / [CUDA](03_recurrence_parallelism/harness/cooperative_2cta.cu) | CHUNK16 实际 GEMM 形状的单 CTA、独立 split2、DSM cluster2 对照 | `--build`、`--check-only`、`--batches 1 8 64 148 256 16384`；结果保存于 `cooperative_2cta/`，不是完整 K2 |

```bash
bash profile/03_recurrence_parallelism/harness/run.sh all
# 或仅复现一组
bash profile/03_recurrence_parallelism/harness/run.sh column_split
```

## 04：计算与访存

| 文件 | 用途 | 参数／输出 |
|---|---|---|
| [bridge.cu](04_compute_memory_bottlenecks/harness/bridge.cu) | 将原生产forward暴露给Python ctypes测试 | 与生产fwd_launch.cu共同编译 |
| [build.py](04_compute_memory_bottlenecks/harness/build.py) | 构建原kernel桥接库并记录源码哈希 | 输出harness/build和analysis/source.json |
| [bench.py](04_compute_memory_bottlenecks/harness/bench.py) | N=1/H=96/T=8192/D=128 的检查及完整forward计时 | `--profile`框定K1/K2用于NCU |
| [analyze.py](04_compute_memory_bottlenecks/harness/analyze.py) | full/source报告与源码热点、PM分析 | 在采集完成后运行，生成analysis/RESULTS.md |
| [run.sh](04_compute_memory_bottlenecks/harness/run.sh) | 完整构建、检查、NCU采集、分析流程 | `bash profile/04_compute_memory_bottlenecks/harness/run.sh`；已有NCU报告时停止，保留测量二进制与日志 |

采集口径见 [METHOD.md](04_compute_memory_bottlenecks/analysis/METHOD.md)。

## 05：BF16持久状态精度

| 文件 | 用途 | 参数／输出 |
|---|---|---|
| [bridge.cu](05_bf16_state_accuracy/harness/bridge.cu) | 桥接原KDA及所需workspace供状态精度对照 | 与原fwd_launch.cu共同编译 |
| [build.py](05_bf16_state_accuracy/harness/build.py) | 构建桥接库并记录来源 | 输出harness/build/original.so和analysis/source.json |
| [bench.py](05_bf16_state_accuracy/harness/bench.py) | 用真实K1中间量比较原kernel、BF16状态与FP32持久状态路径 | `--check`执行快速检查；默认运行完整前缀/seed对照，生成analysis/DATA.md |
| [analyze.py](05_bf16_state_accuracy/harness/analyze.py) | 从DATA.md提取误差、更新丢失统计和图像 | 在bench后执行，生成RESULTS.md及figures |
| [run.sh](05_bf16_state_accuracy/harness/run.sh) | 构建、快速检查、完整实验及分析 | `bash profile/05_bf16_state_accuracy/harness/run.sh`；已有DATA.md时停止，完成后清理临时构建和快速检查文件 |

## 06：设计结论

无独立脚本或CUDA源码。阅读 [REPORT.md](06_sm100a_design_decision/REPORT.md)，结合00–05的证据检查推论，不应将它作为新增benchmark结果。

## 使用顺序与文件保留

先准备基线和依赖，再按需要运行01–05；03、04、05用各自同轮原kernel作对照，最后阅读06。上述run.sh已经包含srun，不要把它们当成完全不涉及作业调度的本地脚本。多数流程会生成、覆盖或清理中间文件，需先核对输出路径；本次目录整理没有执行这些脚本，也没有删除已有数据。

被忽略的结果在干净检出中不存在，须先运行测量/采集再运行分析。报告中的“本地证据”路径仅供实验机器检索，不是缺失的源码依赖。原始记录和新目录采用同一相对层级，历史路径按上表迁移；已有构建缓存需要由脚本重新生成。
