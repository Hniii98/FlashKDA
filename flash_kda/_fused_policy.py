# Copyright (c) 2026 NVIDIA CORPORATION. Apache-2.0.
# Shape guards and scalar LPT schedule from FlashInfer c9f0f0d.
# Only the H64/H96 benchmark portfolio is retained.
import heapq

import torch

_FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_FLASH_KDA_INDEPENDENT_DVSPLIT_CTAS = 2
_FLASH_KDA_INDEPENDENT_DVSPLIT_MIN_SEQUENCE_LENGTH = 512
_FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS = 128
_FLASH_KDA_M128_CHUNK = 32


def _should_use_independent_dvsplit(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    max_sequence_length: int,
) -> bool:
    """Select M64 when its doubled fixed-layout grid remains one resident wave."""

    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and fixed_layout
        and num_sequences == 1
        and max_sequence_length >= _FLASH_KDA_INDEPENDENT_DVSPLIT_MIN_SEQUENCE_LENGTH
        and _FLASH_KDA_INDEPENDENT_DVSPLIT_CTAS * num_heads <= sm_count
    )


def _should_use_source_vtile_direct(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
) -> bool:
    """Select the source one-wave M128 schedule for long dense H96 work."""

    return (
        compute_capability == (10, 3)
        and fixed_layout
        and uniform_sequences
        and num_heads == 96
        and num_sequences * num_heads <= sm_count
        and max_sequence_length >= 4096
    )


def _should_use_source_vtile_persistent(
    *,
    compute_capability: tuple[int, int],
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
) -> bool:
    """Select the source persistent M128 schedule by work-per-CTA bucket."""

    total_tasks = num_sequences * num_heads
    return (
        compute_capability == (10, 3)
        and not fixed_layout
        and uniform_sequences
        and num_heads in (64, 96)
        and total_tasks % _FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS == 0
        and total_tasks // _FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS in (4, 6)
        and max_sequence_length >= 512
    )


def _should_use_scalar_chunk_lpt(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
) -> bool:
    """Select the complete-chain scalar LPT schedule on mixed dense work."""

    total_tasks = num_sequences * num_heads
    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and not uniform_sequences
        and num_heads in (64, 96)
        and max_sequence_length > 0
        and 2 * sm_count <= total_tasks < 1024
        and (max_sequence_length + _FLASH_KDA_M128_CHUNK - 1) // _FLASH_KDA_M128_CHUNK
        < 256
    )


def _build_generated_scalar_schedule(
    sequence_lengths: tuple[int, ...],
    *,
    num_heads: int,
    worker_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the exact one-wave scalar-chunk LPT schedule."""

    if worker_count <= 0 or num_heads <= 0:
        raise ValueError("scalar schedule requires positive workers and heads")
    bins: list[tuple[int, int, list[tuple[int, int]]]] = [
        (0, worker, []) for worker in range(worker_count)
    ]
    heapq.heapify(bins)
    ordered_sequences = sorted(
        range(len(sequence_lengths)),
        key=lambda sequence: (sequence_lengths[sequence] + 31) // 32,
        reverse=True,
    )
    for sequence in ordered_sequences:
        chunks = (sequence_lengths[sequence] + 31) // 32
        for head in range(num_heads):
            load, worker, tasks = heapq.heappop(bins)
            tasks.append((sequence * num_heads + head, chunks))
            heapq.heappush(bins, (load + chunks, worker, tasks))

    # Rebalance two bins exactly before encoding the scalar-chunk schedule.
    while True:
        light_index = min(
            range(worker_count), key=lambda index: (bins[index][0], bins[index][1])
        )
        heavy_index = max(
            range(worker_count), key=lambda index: (bins[index][0], -bins[index][1])
        )
        light_load, light_worker, light_tasks = bins[light_index]
        heavy_load, heavy_worker, heavy_tasks = bins[heavy_index]
        pair_tasks = light_tasks + heavy_tasks
        pair_load = light_load + heavy_load
        reachable = {0: 0}
        for task_index, (_task, chunks) in enumerate(pair_tasks):
            for load, mask in list(reachable.items()):
                reachable.setdefault(load + chunks, mask | (1 << task_index))
        split_load = min(
            reachable,
            key=lambda load: (
                max(load, pair_load - load),
                abs(pair_load - 2 * load),
            ),
        )
        if max(split_load, pair_load - split_load) >= heavy_load:
            break
        split_mask = reachable[split_load]
        bins[light_index] = (
            split_load,
            light_worker,
            [
                task
                for index, task in enumerate(pair_tasks)
                if split_mask & (1 << index)
            ],
        )
        bins[heavy_index] = (
            pair_load - split_load,
            heavy_worker,
            [
                task
                for index, task in enumerate(pair_tasks)
                if not split_mask & (1 << index)
            ],
        )

    while worker_count >= 3:
        ordered_bins = sorted(
            range(worker_count), key=lambda index: (bins[index][0], bins[index][1])
        )
        light_index = ordered_bins[0]
        heavy_index = ordered_bins[-1]
        average_load = sum(load for load, _worker, _tasks in bins) / worker_count
        middle_index = min(
            ordered_bins[1:-1],
            key=lambda index: (
                abs(bins[index][0] - average_load),
                -max(chunks for _task, chunks in bins[index][2]),
                bins[index][1],
            ),
        )
        selected_indices = (light_index, middle_index, heavy_index)
        selected_tasks = [task for index in selected_indices for task in bins[index][2]]
        selected_load = sum(chunks for _task, chunks in selected_tasks)
        heavy_load = bins[heavy_index][0]
        if selected_load > 1024:
            break
        reachable_pairs = {(0, 0): 0}
        processed_load = 0
        for task_index, (_task, chunks) in enumerate(selected_tasks):
            next_pairs = dict(reachable_pairs)
            for (first_load, second_load), assignment in reachable_pairs.items():
                third_load = processed_load - first_load - second_load
                if first_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load + chunks, second_load),
                        assignment | (1 << (2 * task_index)),
                    )
                if second_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load, second_load + chunks),
                        assignment | (2 << (2 * task_index)),
                    )
                if third_load + chunks >= heavy_load:
                    next_pairs.pop((first_load, second_load), None)
            reachable_pairs = next_pairs
            processed_load += chunks
        if not reachable_pairs:
            break
        (first_load, second_load), assignment = min(
            reachable_pairs.items(),
            key=lambda item: max(item[0][0], item[0][1], selected_load - sum(item[0])),
        )
        split_loads = (
            first_load,
            second_load,
            selected_load - first_load - second_load,
        )
        if max(split_loads) >= heavy_load:
            break
        split_tasks: list[list[tuple[int, int]]] = [[], [], []]
        for task_index, task in enumerate(selected_tasks):
            encoded_group = (assignment >> (2 * task_index)) & 3
            group = 0 if encoded_group == 1 else 1 if encoded_group == 2 else 2
            split_tasks[group].append(task)
        for index, load, tasks in zip(
            selected_indices, split_loads, split_tasks, strict=True
        ):
            _old_load, worker, _old_tasks = bins[index]
            bins[index] = (load, worker, tasks)

    bins.sort(key=lambda item: item[1])
    counts = [load for load, _worker, _tasks in bins]
    stride = max(counts)
    schedule = [0] * (worker_count * stride)
    for _load, worker, tasks in bins:
        slot = 0
        for encoded_task, chunks in tasks:
            for local_chunk in range(chunks):
                schedule[worker * stride + slot] = (
                    encoded_task | (local_chunk << 10) | (chunks << 18)
                )
                slot += 1
    return (
        torch.tensor(schedule, dtype=torch.int32, device=device),
        torch.tensor(counts, dtype=torch.int32, device=device),
        stride,
    )
