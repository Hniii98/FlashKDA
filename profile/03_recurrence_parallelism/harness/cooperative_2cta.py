"""CHUNK16 KDA shape microbench: single CTA / independent split2 / DSM cluster2.
No sequence-length axis: batch is the number of independent GEMM tasks.
"""
import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cooperative_2cta'
SOURCE = Path(__file__).with_suffix('.cu')
LIBRARY = ROOT / 'harness/build_cooperative_2cta/kernels.so'
CASES = [
    ('neumann_product', 16, 16, 16, 'float16'),
    ('k1_l_mqk', 16, 16, 128, 'bfloat16'),
    ('k2_inv_r_mqk_u', 16, 128, 16, 'bfloat16'),
    ('k2_state_projection', 16, 128, 128, 'bfloat16'),
    ('k2_state_update', 128, 128, 16, 'bfloat16'),
]
MODES = ['single_cta', 'independent_split2', 'cooperative_cluster2']


def build():
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    flags = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-lineinfo',
             '-gencode', 'arch=compute_103a,code=sm_103a', '-Xcompiler=-fPIC', '-shared',
             str(SOURCE), '-o', str(LIBRARY)]
    subprocess.run(flags, check=True)
    OUT.mkdir(exist_ok=True)
    (OUT / 'SOURCE.json').write_text(json.dumps({
        'compile_command': flags,
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'bench_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'binary_sha256': hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 64, 148, 256, 16384])
    args = parser.parse_args()
    if args.build:
        build()
        return
    import torch
    assert torch.cuda.is_available() and 'B300' in torch.cuda.get_device_name()
    assert all(b > 0 for b in args.batches)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    lib = ctypes.CDLL(str(LIBRARY))
    lib.run_cooperative.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 4
    lib.run_cooperative.restype = ctypes.c_int

    def pack(x):
        batch, rows, cols = x.shape
        return x.reshape(batch, rows//8, 8, cols//8, 8).permute(0, 1, 3, 2, 4).contiguous()

    def inputs(shape, batch, seed, pattern='random'):
        _, m, n, k, dtype = CASES[shape]
        gen = torch.Generator(device='cuda').manual_seed(seed)
        dt = getattr(torch, dtype)
        a = (torch.randn(batch, m, k, generator=gen, device='cuda') * .125).to(dt)
        b = (torch.randn(batch, k, n, generator=gen, device='cuda') * .125).to(dt)
        if pattern == 'zero':
            a.zero_()
        elif pattern == 'structured':
            a.fill_(.125)
            b.fill_(.0625)
            b[:, :, 1::2].neg_()
        return a, b, pack(a), pack(b.transpose(1, 2))

    def run(shape, mode, batch, a, b, out):
        result = lib.run_cooperative(shape, mode, batch, a.data_ptr(), b.data_ptr(),
                                     out.data_ptr(), torch.cuda.current_stream().cuda_stream)
        assert result == 0, (shape, mode, result)

    checks = []
    if not args.profile:
        for shape, (name, m, n, k, dtype) in enumerate(CASES):
            for seed, pattern in [(0, 'random'), (1, 'random'), (2, 'random'), (3, 'zero'), (4, 'structured')]:
                a, b, pa, pb = inputs(shape, 3, seed, pattern)
                ref = a.double() @ b.double()
                baseline = None
                for mode, label in enumerate(MODES):
                    out = torch.full((3, m, n), float('nan'), device='cuda')
                    run(shape, mode, 3, pa, pb, out)
                    torch.cuda.synchronize()
                    error = float((out.double()-ref).norm() / ref.norm().clamp_min(1e-30))
                    max_abs = float((out.double()-ref).abs().max())
                    threshold = .003 if dtype == 'float16' else 2e-5
                    assert bool(out.isfinite().all()) and error <= threshold, (name, label, seed, error)
                    if mode == 0:
                        baseline = out.clone()
                    equal = torch.equal(out, baseline)
                    assert equal, (name, label, 'CTA decomposition changed result')
                    checks.append(dict(case=name, mode=label, seed=seed, pattern=pattern,
                                       batch=3, finite=True, relative_l2=error, max_abs=max_abs,
                                       bitwise_equal_single=equal, threshold=threshold))
        print(f'Correctness: {len(checks)} checks passed, including zero and structured inputs.', flush=True)
        if args.check_only:
            return

    timing = []
    for shape, (name, m, n, k, dtype) in enumerate(CASES):
        for batch in args.batches:
            a, b, pa, pb = inputs(shape, batch, 42)
            outputs = [torch.empty((batch, m, n), device='cuda') for _ in MODES]
            if args.profile:
                for mode, label in enumerate(MODES):
                    run(shape, mode, batch, pa, pb, outputs[mode])
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push(f'{name}/batch{batch}/{label}')
                    torch.cuda.cudart().cudaProfilerStart()
                    run(shape, mode, batch, pa, pb, outputs[mode])
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    torch.cuda.nvtx.range_pop()
                continue
            graphs = []
            inner = 3
            for mode in range(3):
                for _ in range(5):
                    run(shape, mode, batch, pa, pb, outputs[mode])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(inner):
                        run(shape, mode, batch, pa, pb, outputs[mode])
                for _ in range(3):
                    graph.replay()
                graphs.append(graph)
            torch.cuda.synchronize()
            for out in outputs:
                assert torch.equal(out, outputs[0]), (name, batch, 'timed batch mismatch')
                assert bool(out.isfinite().all())
            samples = [[] for _ in MODES]
            for repeat in range(9):
                for pos in range(3):
                    mode = (repeat + pos) % 3
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[mode].replay()
                    end.record()
                    end.synchronize()
                    samples[mode].append(start.elapsed_time(end) * 1000 / inner)
            base = statistics.median(samples[0])
            independent = statistics.median(samples[1])
            for mode, label in enumerate(MODES):
                vals = samples[mode]
                row = dict(case=name, shape=[m, n, k], dtype=dtype, batch=batch, mode=label,
                           median_us=statistics.median(vals), min_us=min(vals), max_us=max(vals),
                           speedup_vs_single=base/statistics.median(vals),
                           speedup_vs_split2=independent/statistics.median(vals), samples_us=vals,
                           ctas=batch*(1 if mode == 0 else 2), timed_batch_bitwise_equal=True)
                timing.append(row)
                print(name, batch, label, f'{row["median_us"]:.4f} us', flush=True)
            del graphs, graph, outputs, out, a, b, pa, pb
            torch.cuda.empty_cache()
    if args.profile:
        return
    OUT.mkdir(exist_ok=True)
    environment = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                       sms=torch.cuda.get_device_properties(0).multi_processor_count,
                       capability=torch.cuda.get_device_capability(),
                       nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True))
    data = dict(environment=environment, checks=checks, timing=timing,
                protocol=dict(warmup=5, graph_warmup=3, graph_inner=3, rounds=9, seed=42,
                              packing_timed=False, chunk=16, feature_dim=128),
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    lines = ['# Cooperative 2-CTA: CHUNK16 KDA GEMM results', '',
             '独立 GEMM microbench；不是完整 K2 或 K1+K2。所有路径使用相同 MMA 类型和打包输入；打包不计时。',
             '协作版采用 2-block cluster + DSM 共享 A；不是 tcgen05.cta_group::2。', '',
             '| Case | Batch | Path | Median μs | Min–max μs | single/new | split2/new |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in timing:
        lines.append(f'| {row["case"]} | {row["batch"]} | {row["mode"]} | {row["median_us"]:.4f} | '
                     f'{row["min_us"]:.4f}–{row["max_us"]:.4f} | {row["speedup_vs_single"]:.3f} | {row["speedup_vs_split2"]:.3f} |')
    lines += ['', f'{len(checks)} correctness checks passed; all split/cluster outputs bitwise equal to single CTA.',
              '计时批量也全部检查有限性及与 single CTA 逐位一致。', '',
              '<details><summary>Raw measurements and environment</summary>', '', '```json',
              json.dumps(data, ensure_ascii=False), '```', '', '</details>', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
