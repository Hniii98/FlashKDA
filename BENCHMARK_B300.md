# KDA forward benchmark (Blackwell / B300)

- Generated: 2026-09-13

- Command: `python benchmarks/generate_benchmark_md.py -o BENCHMARK_B300.md --device-label 'Blackwell / B300' --include-fused`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- Native columns use FP32 initial/final state, with the original BF16-rounded `arange` initial state.
- `flash_kda_fused` uses `use_fused=True`; all columns use the original eager CUDA Event timer.
- Every fused workload uses the single VTile Direct kernel body, with no multi-family routing.
- Device: NVIDIA B300 SXM6 AC, 148 SMs; CUDA 13.0, PyTorch 2.14.0+cu130, FLA 0.5.2.
- GDN is a different operator with a scalar gate; its latency is included as in the original report.

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` K1/K2 (ms) | `flash_kda_fused` (ms) | Speedup vs K1/K2 | `fla_chunk_kda` (ms) | Fused speedup vs KDA | `fla_chunk_gdn` (ms) | Fused speedup vs GDN |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Fixed | 1.7049 | 0.6328 | 2.69× | 3.9403 | 6.23× | 1.9741 | 3.12× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.4932 | 0.5169 | 2.89× | 4.0746 | 7.88× | 2.0414 | 3.95× |
| Varlen, `seq_lens`=`1024 x 8` | 1.2114 | 0.5579 | 2.17× | 3.9733 | 7.12× | 1.9454 | 3.49× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` K1/K2 (ms) | `flash_kda_fused` (ms) | Speedup vs K1/K2 | `fla_chunk_kda` (ms) | Fused speedup vs KDA | `fla_chunk_gdn` (ms) | Fused speedup vs GDN |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Fixed | 1.5526 | 0.6326 | 2.45× | 2.6764 | 4.23× | 1.3398 | 2.12× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.1237 | 0.3595 | 3.13× | 2.8377 | 7.89× | 1.4567 | 4.05× |
| Varlen, `seq_lens`=`1024 x 8` | 0.8196 | 0.3774 | 2.17× | 2.6160 | 6.93× | 1.2870 | 3.41× |

### Validation

- `FLA_FLASH_KDA=0 bash tests/test.sh`: original K1/K2 exact-reference tests passed;
  all 51 fused tests passed, including torch/K1/K2/FLA comparisons with fused
  tolerance `atol=rtol=1e-2`, state/tail handling, graph replay and single-kernel
  launch checks for the benchmark shapes.
- `compute-sanitizer --tool memcheck --error-exitcode 1 python tests/test_fwd_fused.py`:
  27 native eager/graph cases passed; 0 memory errors.
- `FLA_FLASH_KDA=0` was also set during benchmarking so FLA used its independent implementation.
