"""Exercise native prefill launches without loading Triton/FLA.

compute-sanitizer --tool memcheck --error-exitcode 1 python tools/check_fused_memory.py
Numerical torch/FLA comparisons live in tests/test_fwd_fused.py.
"""
import torch

import flash_kda
from flash_kda.fused import prepare_plan

CASES = [
    (3, [1, 1], False),
    (3, [16, 16], False),
    (3, [17, 17], False),
    (3, [513, 513], False),
    (3, [0, 1, 15, 0, 16, 17, 129, 0], True),
    (64, [512], False),
    (96, [4096], False),
    (64, [512] * 8, True),
    (96, [512] * 8, True),
    (64, [1, 15, 257, 16, 32, 513], True),
    (64, [256, 256], False),
    (12, [128, 128], False),
    (12, [256, 256], False),
    (12, [1024, 1024], False),
    (1, [2048], False),
    (32, [512] * 10, False),
]


@torch.inference_mode()
def check_case(heads, lengths, packed, dtype, has_in=True, has_out=True):
    batch, length = (1, sum(lengths)) if packed else (len(lengths), lengths[0])
    shape = (batch, length, heads, 128)
    q, k, v, g = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(heads, device="cuda")
    bias = torch.randn(heads, 128, device="cuda")
    state_shape = (len(lengths), heads, 128, 128)
    initial = torch.randn(state_shape, device="cuda", dtype=dtype) if has_in else None
    final = torch.empty(state_shape, device="cuda", dtype=dtype) if has_out else None
    cu = torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0) if packed else None
    out = torch.empty_like(q)

    def run():
        flash_kda.fwd(q, k, v, g, beta, 128**-0.5, out, a_log, bias, -5.0,
                      initial_state=initial, final_state=final, cu_seqlens=cu, use_fused=True)

    run()
    torch.cuda.synchronize()
    assert out.isfinite().all()
    expected = out.clone()
    expected_state = final.clone() if has_out else None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        if has_out:
            assert final.isfinite().all() and torch.equal(final, expected_state)
    plan, _ = prepare_plan(q, cu, initial, final, 128**-0.5, -5.0)
    print(heads, lengths, dtype, has_in, has_out, plan[2], "OK", flush=True)


if __name__ == "__main__":
    torch.manual_seed(123)
    for dtype in (torch.bfloat16, torch.float32):
        for heads, lengths, packed in CASES:
            check_case(heads, lengths, packed, dtype)
        for has_in, has_out in ((False, False), (False, True), (True, False)):
            check_case(3, [0, 1, 16, 17, 129, 0], True, dtype, has_in, has_out)
        # Generic direct handles unsupported specialized state contracts.
        check_case(32, [512] * 10, False, dtype, False, False)
    print("40 native eager/graph cases passed", flush=True)
