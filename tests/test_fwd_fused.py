"""One VTile Direct implementation: numerical, boundary, state and graph checks.

Native memory checks: compute-sanitizer --tool memcheck --error-exitcode 1
    python tests/test_fwd_fused.py
"""
import pytest
import torch

import flash_kda
from torch_ref import torch_ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="VTile Direct requires SM100/SM103",
)


def inputs(heads, lengths, packed):
    torch.manual_seed(42)
    batch, length = (1, sum(lengths)) if packed else (len(lengths), lengths[0])
    shape = (batch, length, heads, 128)
    q, k, v, g = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0) if packed else None
    kwargs = dict(A_log=torch.rand(heads, device="cuda"),
                  dt_bias=torch.randn(heads, 128, device="cuda"), lower_bound=-5.0, cu_seqlens=cu)
    return (q, k, v, g, beta, 128**-0.5), kwargs


@pytest.mark.parametrize("lengths,packed", [([1, 1], False), ([16, 16], False),
                                          ([33, 33], False), ([0, 1, 15, 0, 16, 17, 65, 0], True)])
@pytest.mark.parametrize("has_in,has_out,dtype", [
    (False, False, torch.bfloat16), (True, False, torch.bfloat16),
    (False, True, torch.bfloat16), (True, True, torch.bfloat16),
    (True, False, torch.float32), (False, True, torch.float32), (True, True, torch.float32),
])
@torch.inference_mode()
def test_torch_reference(lengths, packed, has_in, has_out, dtype):
    args, kwargs = inputs(3, lengths, packed)
    initial = torch.randn(len(lengths), 3, 128, 128, device="cuda", dtype=dtype) if has_in else None
    initial_copy = initial.clone() if has_in else None
    outputs, states = [], []
    for implementation in ("fused", "split", "torch"):
        out = torch.full_like(args[0], float("nan"))
        final = torch.full((len(lengths), 3, 128, 128), float("nan"), device="cuda", dtype=dtype) if has_out else None
        state_kwargs = dict(initial_state=initial, final_state=final)
        if implementation == "torch":
            torch_ref(*args, out, **kwargs, **state_kwargs)
        else:
            flash_kda.fwd(*args, out, **kwargs, **state_kwargs, use_fused=implementation == "fused")
        outputs.append(out)
        states.append(final)
    torch.cuda.synchronize()
    assert torch.equal(outputs[1], outputs[2])
    torch.testing.assert_close(outputs[0], outputs[2], atol=1e-2, rtol=1e-2)
    if has_out:
        assert torch.equal(states[1], states[2])
        torch.testing.assert_close(states[0], states[2], atol=1e-2, rtol=1e-2)
    if has_in:
        assert torch.equal(initial, initial_copy)


@pytest.mark.parametrize("heads", [64, 96])
@pytest.mark.parametrize("lengths", [[8192], [1300, 547, 2048, 963, 271, 3063], [1024] * 8])
@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_benchmark_contracts(heads, lengths, dtype):
    from fla.ops.kda import chunk_kda
    args, kwargs = inputs(heads, lengths, len(lengths) > 1)
    initial = torch.randn(len(lengths), heads, 128, 128, device="cuda").bfloat16().to(dtype) if dtype else None
    initial_copy = initial.clone() if initial is not None else None
    final = torch.empty_like(initial) if initial is not None else None
    out = torch.empty_like(args[0])

    def run():
        flash_kda.fwd(*args, out, **kwargs, initial_state=initial, final_state=final, use_fused=True)

    run()
    expected, expected_state = chunk_kda(
        q=args[0], k=args[1], v=args[2], g=args[3], beta=args[4], scale=args[5],
        initial_state=initial.float() if initial is not None else None,
        output_final_state=final is not None, use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        transpose_state_layout=True, **kwargs)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
    if final is not None:
        torch.testing.assert_close(final.float(), expected_state.float(), atol=1e-2, rtol=1e-2)
    split_out = torch.empty_like(out)
    split_state = torch.empty_like(final) if final is not None else None
    flash_kda.fwd(*args, split_out, **kwargs, initial_state=initial, final_state=split_state)
    torch.testing.assert_close(out, split_out, atol=1e-2, rtol=1e-2)
    if final is not None:
        torch.testing.assert_close(final, split_state, atol=1e-2, rtol=1e-2)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert len(kernels) == 1 and "_flash_kda_fwd_fused_vtile_direct" in kernels[0], kernels

    expected, expected_state = out.clone(), final.clone() if final is not None else None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    if final is not None:
        assert torch.equal(final, expected_state)
        assert torch.equal(initial, initial_copy)


@pytest.mark.parametrize("heads,scale,bound", [(1, 0.1, -1.0), (12, 0.25, 0.0), (64, 0.1, -3.0)])
@torch.inference_mode()
def test_runtime_scalars_and_inplace(heads, scale, bound):
    args, kwargs = inputs(heads, [17, 33], True)
    args = (*args[:-1], scale)
    kwargs["lower_bound"] = bound
    initial = torch.randn(2, heads, 128, 128, device="cuda").bfloat16().float()
    expected, expected_state = torch.empty_like(args[0]), torch.empty_like(initial)
    flash_kda.fwd(*args, expected, **kwargs, initial_state=initial, final_state=expected_state)
    state, out = initial.clone(), torch.empty_like(args[0])
    flash_kda.fwd(*args, out, **kwargs, initial_state=state, final_state=state, use_fused=True)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(state, expected_state, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("has_in", [False, True])
@torch.inference_mode()
def test_all_empty(has_in):
    args, kwargs = inputs(3, [0, 0], True)
    initial = torch.randn(2, 3, 128, 128, device="cuda") if has_in else None
    final = torch.full((2, 3, 128, 128), float("nan"), device="cuda")
    flash_kda.fwd(*args, torch.empty_like(args[0]), **kwargs,
                  initial_state=initial, final_state=final, use_fused=True)
    assert torch.equal(final, initial if has_in else torch.zeros_like(final))


@torch.inference_mode()
def check_native_memory():
    cases = [(3, [0, 1, 15, 16, 17, 65, 0]), (1, [1]), (12, [33])]
    cases += [(h, lengths) for h in (64, 96)
              for lengths in ([8192], [1300, 547, 2048, 963, 271, 3063], [1024] * 8)]
    for heads, lengths in cases:
        args, kwargs = inputs(heads, lengths, len(lengths) > 1)
        for dtype in (None, torch.bfloat16, torch.float32):
            initial = torch.randn(len(lengths), heads, 128, 128, device="cuda").to(dtype) if dtype else None
            final = torch.empty_like(initial) if initial is not None else None
            out = torch.empty_like(args[0])
            def run():
                flash_kda.fwd(*args, out, **kwargs, initial_state=initial, final_state=final, use_fused=True)
            run()
            expected, expected_state = out.clone(), final.clone() if final is not None else None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.cuda.synchronize()
            assert out.isfinite().all() and torch.equal(out, expected)
            if final is not None:
                assert final.isfinite().all() and torch.equal(final, expected_state)
            print(heads, lengths, dtype, "eager/graph OK", flush=True)
    print("27 native eager/graph cases passed", flush=True)


if __name__ == "__main__":
    check_native_memory()
