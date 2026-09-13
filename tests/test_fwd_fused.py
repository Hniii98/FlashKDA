"""Fused forward vs K1/K2 and the independent torch reference.

Run: pytest tests/test_fwd_fused.py -x -v
Race checks: compute-sanitizer --tool racecheck python -m pytest tests/test_fwd_fused.py -x
"""

import math

import pytest
import torch

import flash_kda
from torch_ref import torch_ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Native fused prefill requires SM100/SM103",
)


STATE_VARIANTS = [
    (False, False, torch.bfloat16),
    (True, False, torch.bfloat16),
    (False, True, torch.bfloat16),
    (True, True, torch.bfloat16),
    (True, False, torch.float32),
    (False, True, torch.float32),
    (True, True, torch.float32),
]


def make_inputs(lengths, varlen):
    torch.manual_seed(42)
    B, T = (1, sum(lengths)) if varlen else (len(lengths), lengths[0])
    H, D = 3, 128  # Exercise beta offsets across heads and sequence boundaries.
    shape = (B, T, H, D)
    q, k, v, g = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device="cuda")
    cu_seqlens = None
    if varlen:
        cu_seqlens = torch.tensor([0] + lengths, dtype=torch.int64, device="cuda").cumsum(0)
    kwargs = dict(
        A_log=torch.rand(H, device="cuda"),
        dt_bias=torch.randn((H, D), device="cuda"),
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
    )
    return (q, k, v, g, beta, 1 / math.sqrt(D)), kwargs


@pytest.mark.parametrize("has_in,has_out,state_dtype", STATE_VARIANTS)
@pytest.mark.parametrize("lengths,varlen", [
    ([1, 1], False),
    ([16, 16], False),
    ([17, 17], False),
    ([513, 513], False),  # Multiple wraps of both pipeline rings.
    ([0, 1, 15, 0, 16, 17, 129, 0], True),
])
@torch.inference_mode()
def test_fwd_fused(lengths, varlen, has_in, has_out, state_dtype):
    args, kwargs = make_inputs(lengths, varlen)
    q = args[0]
    state_shape = (len(lengths), q.shape[2], 128, 128)
    initial_state = torch.randn(state_shape, dtype=state_dtype, device="cuda") if has_in else None
    initial_copy = initial_state.clone() if has_in else None
    outputs, states = [], []

    for implementation in ("fused", "split", "torch"):
        out = torch.full_like(q, float("nan"))
        final_state = torch.full(state_shape, float("nan"), dtype=state_dtype, device="cuda") if has_out else None
        state_kwargs = dict(initial_state=initial_state, final_state=final_state)
        if implementation == "torch":
            torch_ref(*args, out, **kwargs, **state_kwargs)
        else:
            flash_kda.fwd(*args, out, **kwargs, **state_kwargs, use_fused=implementation == "fused")
        outputs.append(out)
        states.append(final_state)

    torch.cuda.synchronize()
    assert torch.equal(outputs[1], outputs[2])
    for out in outputs[1:]:
        torch.testing.assert_close(outputs[0], out, atol=1e-2, rtol=1e-2)
    if has_out:
        assert torch.equal(states[1], states[2])
        for state in states[1:]:
            torch.testing.assert_close(states[0], state, atol=1e-2, rtol=1e-2)
    if has_in:
        assert torch.equal(initial_state, initial_copy)


@torch.inference_mode()
def test_fwd_fused_graph_and_inplace_state():
    args, kwargs = make_inputs([17, 65, 1], varlen=True)
    q = args[0]
    initial_state = torch.randn((3, q.shape[2], 128, 128), device="cuda")
    state = initial_state.clone()
    out = torch.empty_like(q)
    expected_out = torch.empty_like(q)
    expected_state = torch.empty_like(state)
    torch_ref(*args, expected_out, **kwargs, initial_state=initial_state, final_state=expected_state)

    # Warm up on a side stream, then replay with initial_state == final_state.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flash_kda.fwd(*args, out, **kwargs, initial_state=state, final_state=state, use_fused=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_kda.fwd(*args, out, **kwargs, initial_state=state, final_state=state, use_fused=True)
    for _ in range(3):
        state.copy_(initial_state)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(state, expected_state, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("heads,lengths,packed", [
    (64, [512], False),                         # independent M64
    (96, [4096], False),                        # direct V-tile
    (64, [512] * 8, True),                      # four-task V-tile
    (96, [512] * 8, True),                      # six-task V-tile
    (64, [1, 15, 257, 16, 32, 513], True),      # scalar LPT
    (64, [256, 256], False),                    # tensor-decay regression: prediction-first
    (12, [128, 128], False),                   # generic N16
    (12, [256, 256], False),                   # generic N16, multiple chunks
    (12, [1024, 1024], False),                  # BT16 state regression: direct N16
    (1, [2048], False),                        # generic direct fallback
    (32, [512] * 10, False),                     # generic direct fallback
])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_prefill_schedules(heads, lengths, packed, state_dtype):
    from test_fwd import run_fla_gold_reference
    from flash_kda.fused import prepare_plan

    torch.manual_seed(123)
    batch, length = (1, sum(lengths)) if packed else (len(lengths), lengths[0])
    shape = (batch, length, heads, 128)
    q, k, v, g = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    A_log = torch.rand(heads, device="cuda")
    dt_bias = torch.randn((heads, 128), device="cuda")
    initial = torch.randn((len(lengths), heads, 128, 128), device="cuda", dtype=state_dtype)
    final = torch.empty_like(initial)
    cu = torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0) if packed else None
    out = torch.empty_like(q)
    scale = 128**-0.5
    flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, -5.0,
                  initial_state=initial, final_state=final, cu_seqlens=cu, use_fused=True)
    plan, _ = prepare_plan(q, cu, initial, final, scale, -5.0)
    print(f"route={plan[2]} variant={plan[0]['name']}")
    reference = torch.empty_like(out)
    reference_state = torch.empty_like(initial)
    torch_ref(q, k, v, g, beta, scale, reference, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
              initial_state=initial, final_state=reference_state, cu_seqlens=cu)
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final, reference_state, atol=1e-2, rtol=1e-2)
    gold_out, gold_state, chunk_out, chunk_state = run_fla_gold_reference(
        q, k, v, g, beta, initial.float(), A_log, dt_bias, scale, -5.0, cu)
    for expected in (gold_out, chunk_out):
        torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)
    for expected in (gold_state, chunk_state):
        torch.testing.assert_close(final.float(), expected.float(), atol=1e-2, rtol=1e-2)

    captured_out, captured_state = out.clone(), final.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, -5.0,
                      initial_state=initial, final_state=final, cu_seqlens=cu, use_fused=True)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, captured_out)
        assert torch.equal(final, captured_state)
