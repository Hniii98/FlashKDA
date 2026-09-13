# KDA forward benchmark (Blackwell / B300)

- Generated: 2026-09-13

- Command: `python benchmarks/generate_benchmark_md.py -o /home/lcpu/60990375/cake_kda_align/.cache/fused-validation/dedup/after-final.md --device-label 'Blackwell / B300' --include-fused`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- Native columns use FP32 initial/final state, with the original BF16-rounded `arange` initial state.
- `flash_kda_fused` uses `use_fused=True`; all columns use the original eager CUDA Event timer.
- GDN is a different operator with a scalar gate; its latency is included as in the original report.

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` K1/K2 (ms) | `flash_kda_fused` (ms) | Speedup vs K1/K2 | `fla_chunk_kda` (ms) | Fused speedup vs KDA | `fla_chunk_gdn` (ms) | Fused speedup vs GDN |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Fixed | 1.7080 | 0.6357 | 2.69× | 3.9503 | 6.21× | 1.9797 | 3.11× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.4941 | 0.5557 | 2.69× | 4.0819 | 7.35× | 2.0461 | 3.68× |
| Varlen, `seq_lens`=`1024 x 8` | 1.2134 | 0.5494 | 2.21× | 3.9785 | 7.24× | 1.9480 | 3.55× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` K1/K2 (ms) | `flash_kda_fused` (ms) | Speedup vs K1/K2 | `fla_chunk_kda` (ms) | Fused speedup vs KDA | `fla_chunk_gdn` (ms) | Fused speedup vs GDN |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Fixed | 1.5539 | 0.5908 | 2.63× | 2.6797 | 4.54× | 1.3401 | 2.27× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.1248 | 0.3857 | 2.92× | 2.8398 | 7.36× | 1.4556 | 3.77× |
| Varlen, `seq_lens`=`1024 x 8` | 0.8198 | 0.3764 | 2.18× | 2.6182 | 6.96× | 1.2862 | 3.42× |
