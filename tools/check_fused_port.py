"""Compare all 18 benchmark contracts with a saved upstream implementation.

Run --write-reference with the frozen implementation, then --reference with
this implementation. Checks output and final state bit-for-bit, before timing.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import flash_kda


@torch.inference_mode()
def main(args):
    layouts = ([8192], [1300, 547, 2048, 963, 271, 3063], [1024] * 8)
    directory = args.write_reference or args.reference
    directory.mkdir(parents=True, exist_ok=True)
    for heads in (64, 96):
        for index, lengths in enumerate(layouts):
            for state_type in ("bf16", "none", "fp32"):
                torch.manual_seed(42)
                shape = (1, sum(lengths), heads, 128)
                q, k = [F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
                v, g = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
                beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
                a_log = torch.rand(heads, device="cuda")
                dt_bias = torch.rand(heads, 128, device="cuda")
                initial = final = None
                if state_type != "none":
                    dtype = torch.bfloat16 if state_type == "bf16" else torch.float32
                    initial = torch.randn((len(lengths), heads, 128, 128), device="cuda").bfloat16().to(dtype)
                    final = torch.empty_like(initial)
                cu = None if len(lengths) == 1 else torch.tensor([0] + list(lengths), dtype=torch.int64, device="cuda").cumsum(0)
                out = torch.empty_like(q)
                flash_kda.fwd(q, k, v, g, beta, 128**-0.5, out, a_log, dt_bias, -5.0,
                              initial_state=initial, final_state=final, cu_seqlens=cu, use_fused=True)
                torch.cuda.synchronize()
                if args.check_launches:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        flash_kda.fwd(q, k, v, g, beta, 128**-0.5, out, a_log, dt_bias, -5.0,
                                      initial_state=initial, final_state=final, cu_seqlens=cu, use_fused=True)
                        torch.cuda.synchronize()
                    kernels = [event.name for event in profile.events()
                               if event.device_type == torch.autograd.DeviceType.CUDA]
                    assert len(kernels) == 1 and "_flash_kda_fwd_fused_" in kernels[0], kernels
                    print("single launch:", kernels[0], flush=True)
                result = dict(out=out.cpu(), state=final.cpu() if final is not None else None)
                path = directory / f"h{heads}_layout{index}_{state_type}.pt"
                if args.write_reference:
                    torch.save(result, path)
                else:
                    reference = torch.load(path, weights_only=True)
                    for name, value in result.items():
                        if value is not None:
                            torch.testing.assert_close(value, reference[name], atol=0, rtol=0)
                print(path.name, "saved" if args.write_reference else "bitwise match", flush=True)
    print("18 benchmark contracts passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-reference", type=Path)
    group.add_argument("--reference", type=Path)
    parser.add_argument("--check-launches", action="store_true")
    main(parser.parse_args())
