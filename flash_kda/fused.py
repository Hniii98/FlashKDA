"""Sequence metadata for the single VTile Direct fused implementation."""
import functools

import torch

# Captured launches keep raw pointers. Pin their metadata beyond LRU eviction.
_CAPTURED_METADATA = {}


@functools.lru_cache(maxsize=64)
def _sequence_metadata(cu_seqlens, version, batch, length, device):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm fused forward with these sequence lengths before CUDA Graph capture")
    if cu_seqlens is None:
        lengths = [length] * batch
    else:
        if (batch != 1 or cu_seqlens.device != device or cu_seqlens.dtype != torch.int64
                or cu_seqlens.ndim != 1 or not cu_seqlens.is_contiguous() or cu_seqlens.numel() < 2):
            raise ValueError("Packed fused input requires B=1 and contiguous CUDA int64 cu_seqlens[N+1]")
        offsets = cu_seqlens.cpu().tolist()
        if offsets[0] != 0 or offsets[-1] != batch * length or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("cu_seqlens must be nondecreasing and span the packed input")
        lengths = [b - a for a, b in zip(offsets, offsets[1:])]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    return torch.tensor(order, device=device, dtype=torch.int32), all(n % 32 == 0 for n in lengths)


def prepare_metadata(q, cu_seqlens):
    try:
        version = cu_seqlens._version if cu_seqlens is not None else 0
    except RuntimeError:
        # Inference tensors have no version counter: their offsets must remain
        # unchanged, or callers must provide a new tensor for a new segmentation.
        version = -1
    metadata = _sequence_metadata(cu_seqlens, version, q.shape[0], q.shape[1], q.device)
    if torch.cuda.is_current_stream_capturing():
        _CAPTURED_METADATA[(id(metadata), id(cu_seqlens))] = (metadata, cu_seqlens)
    else:
        stream = torch.cuda.current_stream(q.device)
        metadata[0].record_stream(stream)
    return metadata
