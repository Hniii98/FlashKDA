# Profile 脚本索引


## 实验顺序与目录

**实现C32 → 定位变慢原因 → 调整launch bounds → 分析剩余阶段耗时。**

| 目录与实验报告 | 回答的问题 |
|---|---|
| [01_c32_implementation](01_c32_implementation/REPORT.md) | C32如何实现，rescale如何选择，正确性及基础性能怎样？ |
| [02_slowdown_diagnosis](02_slowdown_diagnosis/REPORT.md) | 为什么慢，rescale、spill和同步等待各有什么证据？ |
| [03_launch_bounds_tuning](03_launch_bounds_tuning/REPORT.md) | 8/5/4/2中哪个更快，为什么采用4？ |
| [04_k1_stage_timing](04_k1_stage_timing/REPORT.md) | 采用4后，K1剩余耗时落在哪些阶段？ |

目录迁移对应关系：`c32` → `01_c32_implementation`；`c32-cost-attribution-20260910` → `02_slowdown_diagnosis`；`c32-launch-bounds-20260910` → `03_launch_bounds_tuning`；`k1-stage-timing-20260910` → `04_k1_stage_timing`。历史JSON、日志、NCU和二进制中的旧路径保留，脚本与文档使用新路径；搬迁后的构建缓存会由构建脚本重新生成。

## 1. 01_c32_implementation：实现、数值诊断和参数扫描

| 脚本 | 作用 | 输入依赖与输出 |
|---|---|---|
| [build.py](01_c32_implementation/build.py) | 使用现有PyTorch/CUDA环境编译当前主源码 | 需要CUTLASS和Python头文件；输出 `01_c32_implementation/build/flash_kda_C.so`，不构建原C16 |
| [run_diagnostics.sh](01_c32_implementation/run_diagnostics.sh) | 在lower_bound=-1下检查未缩放路径，并执行一组非默认scale诊断 | 调用仓库的naive对拍文件；输出 `naive_gate1.json/log` 和 `naive_scale_smoke.json/log`；不包含初版-5失败用例的重跑 |
| [run_sweep.sh](01_c32_implementation/run_sweep.sh) | 扫描rescale=2^e，e=-80…-112，记录实际K1 workspace诊断 | 依赖主C32二进制和未修改参考；输出 `sweep_fine.json/log` |

默认参数对拍与完整benchmark的公共入口在profile外：

- [tests/compare_naive.py](../tests/compare_naive.py)：output及最终state的最大绝对误差、相对L2、Inf/NaN；默认12个用例，也支持scale sweep和workspace诊断。
- [benchmarks/bench_c32_vs_c16.py](../benchmarks/bench_c32_vs_c16.py)：同轮C16/C32完整forward及K1/K2分解；输出结果JSON与profiler trace。默认C32路径是 `01_c32_implementation/build/`。

## 2. 02_slowdown_diagnosis：时间成本定位

| 脚本 | 作用 | 输入依赖与输出 |
|---|---|---|
| [harness/run.py](02_slowdown_diagnosis/harness/run.py) | `--mode timing` 比较C16/C32及scale=1、2^-96、lower_bound=-5/-1；`--mode profile`仅框定一次forward供NCU采集 | 默认T=8192、H=96、D=128、单序列、FP32 state；需要现成C16/C32二进制；C32现在默认是LB4，复现历史LB8须用--binary指定LB8构建。timing输出 `analysis/runtime_ablation.json` 和trace；支持 `--binary` 指定额外候选 |
| [harness/collect.sh](02_slowdown_diagnosis/harness/collect.sh) | 顺序执行运行时对照和两版NCU采集 | 需要GPU及ncu；生成 `reports/full_*.ncu-rep`、`source_*.ncu-rep` 和采集日志 |
| [analysis/parse_reports.py](02_slowdown_diagnosis/analysis/parse_reports.py) | 用ncu_report解析当前目录的NCU报告 | 在collect之后执行；导出关键指标、完整指标、源码热点和PM采样JSON |

## 3. 03_launch_bounds_tuning：寄存器预算实验

| 脚本 | 作用 | 输入依赖与输出 |
|---|---|---|
| [harness/build_all.py](03_launch_bounds_tuning/harness/build_all.py) | 读取当前K1 launch bounds，依次独立编译8、5、4、2，结束时恢复进入脚本前的源码 | 支持当前LB4源码；复用 `01_c32_implementation/build.py`，输出 `build_8/5/4/2/`、构建日志和哈希清单；不得与其他源码编辑或构建并行 |
| [harness/compare.py](03_launch_bounds_tuning/harness/compare.py) | 预加载指定候选，随后运行公共naive对拍文件 | 首个位置参数是候选 `.so` 路径，其余参数传给compare_naive；不修改参考或用例 |
| [harness/run.py](03_launch_bounds_tuning/harness/run.py) | 对C16、LB8/5/4/2同轮计时，或框定指定版本供NCU采集 | 需要独立LB8/5/4/2全部构建完成；默认代表形状同上；输出 `analysis/runtime_ablation.json` 和trace |
| [harness/collect.sh](03_launch_bounds_tuning/harness/collect.sh) | 顺序完成三版对拍、同轮计时、四版NCU以及三版memcheck | 需要GPU、ncu、compute-sanitizer；输出位于 `analysis/` 和 `reports/` |
| [analysis/parse_reports.py](03_launch_bounds_tuning/analysis/parse_reports.py) | 解析该实验的NCU报告 | 在collect之后执行；生成关键指标、源码热点与PM数据 |

## 4. 04_k1_stage_timing：K1八阶段计时

| 脚本 | 作用 | 输入依赖与输出 |
|---|---|---|
| [harness/build.py](04_k1_stage_timing/harness/build.py) | 从原C16和当前C32生成独立插桩构建；C32使用LB4，在现有同步边界记录时间戳 | 需要原C16源码、当前C32和现有构建环境；生成 `harness/C16/`、`harness/C32/`、`build_C16/`、`build_C32/`、补丁和构建清单；不改原源码 |
| [harness/compare.py](04_k1_stage_timing/harness/compare.py) | 预加载指定计时版并运行公共naive对拍 | 首个参数是 `.so`；为C16补充脚本所需默认参数元数据，并将输出metadata标为chunk=16；不向C16传入scale关键字 |
| [harness/run.py](04_k1_stage_timing/harness/run.py) | 测插桩前后完整forward/K1/K2、采集八阶段延迟，并检查主形状output/state逐元素一致性 | 需要原C16、独立未插桩LB4及两种计时版；输出 `runtime_ablation.json`、`stages.json`、原始 `.pt` 样本与trace |
| [harness/collect.sh](04_k1_stage_timing/harness/collect.sh) | 顺序执行两种计时版的naive对拍、阶段计时及C32 memcheck | 需要GPU；输出对拍、计时和阶段数据。现已补入C32插桩版的变长用例memcheck，输出 `analysis/memcheck.json/log`；需要compute-sanitizer |
| [analysis/report.py](04_k1_stage_timing/analysis/report.py) | 读取本轮阶段、完整计时和两版naive JSON，生成分目录阶段报告 | 在collect之后执行；覆盖该目录的 `REPORT.md`，不更新 `profile/REPORT.md`，也不自动追加单独memcheck结论 |

## 参考实现获取与配置

参考实现使用课程材料 **assignment02 下的 `fla_kda_ref/`**。从课程仓库或作业材料中取得该目录，保持内容不变；本次实验的相对位置为 `assignment02/team/c1_flashkda/fla_kda_ref/`。对拍调用其中的 `naive.py`，不要替换成其他版本的FLA参考。

直接运行对拍文件时，把 `--ref-dir` 指向含有 `naive.py` 的目录，例如：

```bash
export REF_DIR=/path/to/assignment02/team/c1_flashkda/fla_kda_ref
python tests/compare_naive.py --ref-dir "$REF_DIR" --output profile/01_c32_implementation/naive_default_full_recheck.json
```

现有shell脚本仍使用实验机器上的参考路径；换环境时还需同步修改脚本的 `ref`、`ref_dir` 或 `--ref-dir`，仅设置REF_DIR不会自动覆盖这些路径。

## 02历史LB8实验的复现说明

02报告对应launch bounds=8。运行前，仅将 `csrc/smxx/fwd_kernel1.cuh` 中的 `__launch_bounds__(NumThreads, 4)` 改为 `__launch_bounds__(NumThreads, 8)`，并执行 `python profile/01_c32_implementation/build.py` 重建。然后运行02的collect和解析脚本。仅改源码而不重建，仍会运行旧二进制。

采集结束后，将参数恢复为4并重建，以恢复当前默认版本；rescale及其他设置保持不变。对应说明也已写入02的collect脚本注释，脚本本身不会自动修改源码。原实验的绝对耗时取决于设备运行状态，重跑用于核验趋势和结论。

## 建议运行顺序

1. 配好既有CUDA/PyTorch环境、CUTLASS、原C16源码/二进制与未修改参考。
2. 执行 `01_c32_implementation/build.py`，再运行公共对拍和benchmark；按需运行诊断或sweep。
3. 要复现02的历史成本定位，先把K1 launch bounds从4改为8并重建默认C32，再执行对应collect和parse_reports；完成后改回4并再次重建。
4. 要复现launch bounds，先build_all，再collect，最后parse_reports。
5. 要复现阶段计时，先完成LB4构建，再执行阶段build、collect，最后按需生成分目录报告。



```bash
# Python需使用现有CUDA/PyTorch环境；REF_DIR由使用者指定。
python profile/01_c32_implementation/build.py
python tests/compare_naive.py --ref-dir "$REF_DIR" --output profile/01_c32_implementation/naive_default_full_recheck.json
python benchmarks/bench_c32_vs_c16.py

python profile/03_launch_bounds_tuning/harness/build_all.py
bash profile/03_launch_bounds_tuning/harness/collect.sh
python profile/03_launch_bounds_tuning/analysis/parse_reports.py

python profile/04_k1_stage_timing/harness/build.py
bash profile/04_k1_stage_timing/harness/collect.sh
python profile/04_k1_stage_timing/analysis/report.py
```

