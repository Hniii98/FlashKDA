"""Shape dispatch for the native fused kernels in fwd_kernel_fused.cuh."""
from __future__ import annotations

import functools
import json
import struct
from pathlib import Path

import torch

from . import _fused_policy as policy

_REGISTRY = json.loads(Path(__file__).with_name("fused_registry.json").read_text())["variants"]
_DIRECT = "direct_m128"
_N16 = "direct_m128_n16"
_M64 = "independent_dvsplit_m64"
_VTILE = "source599_vtile_m128"
_SCALAR = "scalar_chunk_lpt_m128"

# Graph nodes retain raw metadata addresses. Pin warmed metadata used in capture
# independently of the bounded eager caches, for the lifetime of the process.
_CAPTURED_METADATA = {}


@functools.lru_cache(maxsize=64)
def _offsets(cu_seqlens, version, batch, length, device):
    # Keep the tensor alive in the key: allocator address reuse must not return
    # stale lengths. PyTorch in-place edits invalidate the key via _version.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm fused forward with these sequence lengths before CUDA graph capture")
    if cu_seqlens is None:
        values = tuple(i * length for i in range(batch + 1))
        cu_seqlens = torch.tensor(values, dtype=torch.int64, device=device)
    else:
        values = tuple(cu_seqlens.cpu().tolist())
        if values[0] != 0 or values[-1] != batch * length or any(a > b for a, b in zip(values, values[1:])):
            raise ValueError("cu_seqlens must be nondecreasing, start at zero, and end at B*T")
    return values, cu_seqlens


@functools.lru_cache(maxsize=64)
def _plan(offsets, heads, fixed, capability, sms, fp32, has_in, has_out, scale, lower_bound, device):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm this fused shape/state configuration before CUDA graph capture")
    lengths = tuple(b - a for a, b in zip(offsets, offsets[1:]))
    arch = f"sm_{capability[0]}{capability[1]}a"
    tasks = len(lengths) * heads
    maximum = max(lengths)
    uniform = len(set(lengths)) == 1
    # Short tiles and H12 use upstream N16; generic N32 uses the validated
    # prediction-first path. Tensor-state-decay is intentionally not compiled.
    route = _N16 if heads == 12 or maximum <= 16 else _DIRECT
    variant_id = (18 if fp32 else 14) if route == _N16 else (6 if fp32 else 7)
    specialized = has_in and has_out and scale == 128**-0.5 and lower_bound == -5.0 and 0 not in lengths
    shape = dict(compute_capability=capability, num_heads=heads,
                 num_sequences=len(lengths), max_sequence_length=maximum)
    if specialized and heads == 64 and policy._should_use_independent_dvsplit(
            **shape, sm_count=sms, fixed_layout=fixed):
        route, variant_id = _M64, 32 if fp32 else 30
    elif specialized and policy._should_use_source_vtile_direct(
            **shape, sm_count=sms, fixed_layout=fixed, uniform_sequences=uniform):
        route, variant_id = _VTILE, 25 if fp32 else 29
    elif specialized and policy._should_use_source_vtile_persistent(
            **shape, fixed_layout=fixed, uniform_sequences=uniform):
        route = _VTILE
        variant_id = (27 if fp32 else 26) if heads == 64 else (24 if fp32 else 28)
    elif specialized and policy._should_use_scalar_chunk_lpt(
            **shape, sm_count=sms, uniform_sequences=uniform):
        route = _SCALAR
        variant_id = (17 if fp32 else 10) if heads == 64 else (8 if fp32 else 12)
    variant = next(v for v in _REGISTRY if v["arch"] == arch and v["id"] == variant_id)
    order = range(len(lengths)) if route == _SCALAR else sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    buffers = {"seq_order": torch.tensor(list(order), dtype=torch.int32, device=device)}
    if 0 in lengths:
        buffers["empty_indices"] = torch.tensor([i for i, n in enumerate(lengths) if n == 0], device=device)
    config = dict(variant=variant_id, blocks=tasks)
    if route == _M64:
        config["blocks"] = 2 * tasks
    elif route == _VTILE:
        workers = tasks if fixed else 128
        config.update(blocks=workers, uniform_seq_len=maximum,
                      persistent_tasks=tasks // workers, persistent_stride=workers)
    elif route == _SCALAR:
        schedule, counts, stride = policy._build_generated_scalar_schedule(
            lengths, num_heads=heads, worker_count=sms, device=device)
        buffers.update(tile_schedule=schedule, tile_schedule_counts=counts)
        config.update(blocks=sms, schedule_stride=stride)
    return variant, route, route, config, buffers


def prepare_plan(q, cu_seqlens, initial_state, final_state, scale, lower_bound):
    batch, length, heads, dim = q.shape
    if dim != 128:
        raise ValueError("fused prefill requires head dimension 128")
    capability = torch.cuda.get_device_capability(q.device)
    if capability not in ((10, 0), (10, 3)):
        raise ValueError("FlashInfer fused prefill requires SM100/SM103 (B200/B300)")
    if cu_seqlens is not None and (batch != 1 or cu_seqlens.dtype != torch.int64 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2):
        raise ValueError("packed fused prefill requires B=1 and int64 cu_seqlens[N+1]")
    # Inference tensors have no version counter: their length metadata must
    # remain immutable, as for a prepared upstream prefill workspace.
    try:
        version = cu_seqlens._version if cu_seqlens is not None else 0
    except RuntimeError:
        version = None
    offsets, cu = _offsets(cu_seqlens, version, batch, length, q.device)
    fp32 = any(s is not None and s.dtype == torch.float32 for s in (initial_state, final_state))
    # Kernel scalars use float32. Canonicalize equivalent spellings of the
    # default scale (1/sqrt(128) versus 128**-0.5) for the frozen selector key.
    if struct.pack("f", scale) == struct.pack("f", 128**-0.5):
        scale = 128**-0.5
    plan = _plan(offsets, heads, cu_seqlens is None, capability,
                 torch.cuda.get_device_properties(q.device).multi_processor_count,
                 fp32, initial_state is not None, final_state is not None,
                 float(scale), float(lower_bound), q.device)
    if torch.cuda.is_current_stream_capturing():
        _CAPTURED_METADATA[(id(plan), id(cu))] = (plan, cu)
    return plan, cu


def fwd_fused(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None):
    from flash_kda_C import _fwd_fused

    if q.numel() == 0:
        if final_state is not None:
            final_state.zero_() if initial_state is None else final_state.copy_(initial_state)
        return
    (variant, _, route, config, metadata), cu = prepare_plan(
        q, cu_seqlens, initial_state, final_state, scale, lower_bound)
    buffers = dict(metadata, cu_seqlens=cu)
    if not torch.cuda.is_current_stream_capturing():
        stream = torch.cuda.current_stream(q.device)
        for tensor in buffers.values():
            tensor.record_stream(stream)
    heads = q.shape[2]
    tokens = q.shape[0] * q.shape[1]
    def allocate(name, shape, dtype=torch.bfloat16):
        buffers[name] = torch.empty(shape, dtype=dtype, device=q.device)
    chunk = variant["launch"]["chunk_tokens"]
    if heads % 8 == 0 and tokens >= chunk:
        buffers["beta_carrier"] = beta.view(-1, heads)
    else:
        padded_heads = (heads + 7) // 8 * 8
        allocate("beta_carrier", (max(tokens, chunk), padded_heads))
        buffers["beta_carrier"].zero_()
        buffers["beta_carrier"][:tokens, :heads].copy_(beta.view(tokens, heads))
    if "empty_indices" in buffers and final_state is not None:
        indices = buffers.pop("empty_indices")
        if initial_state is None:
            final_state.index_fill_(0, indices, 0)
        else:
            final_state.index_copy_(0, indices, initial_state.index_select(0, indices))
    _fwd_fused(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
               initial_state, final_state, buffers, config)
