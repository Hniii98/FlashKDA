"""Same-device C16/C32 forward timing plus separate CUDA kernel timeline samples."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path('/home/lcpu/60990375/topic7-envs/baseline/flash_kda_C.cpython-312-x86_64-linux-gnu.so')
CANDIDATE = ROOT / 'profile/01_c32_implementation/build/flash_kda_C.so'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name + '.flash_kda_C', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values):
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                min_ms=min(values), max_ms=max(values), count=len(values))


def telemetry():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu',
                             '--format=csv'], text=True, capture_output=True)
    return result.stdout.strip() or result.stderr.strip()


def timed(fn, iterations):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    torch.cuda.synchronize()
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def kernel_times(fn, count, path):
    # Separate from event timing: these durations are profiler samples, not additive
    # components of the independently measured end-to-end CUDA-event mean.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(count):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())['traceEvents']
    groups = {name: [] for name in ['K1', 'K2', 'prefix']}
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        name = event.get('name', '')
        for key, marker in [('K1', '_flash_kda_fwd_prepare'), ('K2', '_flash_kda_fwd_recurrence'),
                            ('prefix', '_flash_kda_build_tile_prefix')]:
            if marker in name:
                groups[key].append(event['dur'] / 1000.0)  # trace microseconds -> ms
    for key in ['K1', 'K2']:
        if len(groups[key]) != count:
            raise RuntimeError(f'{path}: expected {count} {key} launches, got {len(groups[key])}')
    return {key: stats(values) for key, values in groups.items() if values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--profile-iters', type=int, default=20)
    p.add_argument('--heads', type=int, nargs='+', default=[96, 64])
    p.add_argument('--output-dir', type=Path, default=ROOT/'profile/01_c32_implementation/benchmark')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    mods = {'C16': load('c16', BASELINE), 'C32': load('c32', CANDIDATE)}
    rows = []
    result = dict(command=os.sys.argv, gpu=torch.cuda.get_device_name(),
                  device_properties=str(torch.cuda.get_device_properties(0)),
                  torch=torch.__version__, job_id=os.getenv('SLURM_JOB_ID'),
                  cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                  start_telemetry=telemetry(), warmup=args.warmup, iters=args.iters, repeats=args.repeats,
                  profile_iters=args.profile_iters, seed=0, lower_bound=-5,
                  C32_rescale=mods['C32'].DEFAULT_RESCALE,
                  C32_inverse_rescale=mods['C32'].DEFAULT_INVERSE_RESCALE,
                  artifacts={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                             for name, path in [('C16', BASELINE), ('C32', CANDIDATE)]}, rows=rows)
    cases = [('fixed', [8192]), ('varlen', [1300, 547, 2048, 963, 271, 3063]), ('eight', [1024]*8)]
    for H in args.heads:
        for case, lengths in cases:
            torch.manual_seed(0)
            D, T, N = 128, sum(lengths), len(lengths)
            shape = (1, T, H, D)
            q, k = [F.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16() for _ in range(2)]
            v, g = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
            beta = torch.randn(1, T, H, device='cuda', dtype=torch.bfloat16)
            al, bias = torch.rand(H, device='cuda'), torch.rand(H, D, device='cuda')
            h0 = torch.arange(N*H*D*D, device='cuda', dtype=torch.float32).reshape(N, H, D, D).bfloat16()
            cu = torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()), device='cuda', dtype=torch.int64) if N>1 else None
            for mode in ['bf16', 'none', 'fp32']:
                initial = None if mode=='none' else h0.to(torch.float32 if mode=='fp32' else torch.bfloat16)
                outputs = {key: torch.empty_like(q) for key in mods}
                finals = {key: None if initial is None else torch.empty_like(initial) for key in mods}
                def make_fn(key):
                    module = mods[key]
                    def call():
                        # Match the public wrapper: workspace allocation and beta transpose
                        # are inside forward, all input/output tensor allocations are outside.
                        workspace = torch.empty(module.get_workspace_size(T,H,N),device='cuda',dtype=torch.uint8)
                        module.fwd(q,k,v,g,beta,D**-0.5,outputs[key],workspace,al,bias,-5,
                                   initial_state=initial, final_state=finals[key], cu_seqlens=cu)
                    return call
                fns = {key: make_fn(key) for key in mods}
                for _ in range(args.warmup):
                    for fn in fns.values():
                        fn()
                torch.cuda.synchronize()
                samples = {key: [] for key in mods}
                rounds = []
                for repeat in range(args.repeats):
                    order = ['C16','C32'] if repeat%2==0 else ['C32','C16']
                    current = {'order': order}
                    for key in order:
                        values = timed(fns[key], args.iters)
                        samples[key].extend(values)
                        current[key] = stats(values)
                    rounds.append(current)
                row = dict(H=H,T=T,D=D,case=case,seq_lens=lengths,state=mode,
                           forward={key:stats(values) for key,values in samples.items()}, rounds=rounds)
                row['speedup_C16_over_C32'] = row['forward']['C16']['mean_ms']/row['forward']['C32']['mean_ms']
                row['finite'] = {key: bool(outputs[key].isfinite().all()) and
                                 (finals[key] is None or bool(finals[key].isfinite().all())) for key in mods}
                for key in mods:
                    trace = args.output_dir/f'{H}_{case}_{mode}_{key}.json'
                    row.setdefault('kernel_profile',{})[key] = kernel_times(fns[key], args.profile_iters, trace)
                rows.append(row)
                print(json.dumps(row),flush=True)
                result['end_telemetry'] = telemetry()
                (args.output_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
                del fns, outputs, finals
    print('Completed',len(rows),'paired benchmark cases',flush=True)


if __name__ == '__main__':
    main()
