// Copyright (c) 2026 NVIDIA CORPORATION. Licensed under Apache-2.0.
// Adapted from FlashInfer c9f0f0d90a22b1734733297ec09f0f44d21927cf.
// See ../../NOTICE.flashinfer for provenance and local modifications.
#pragma once

#include "fwd.h"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>

namespace flash_kda::fused {
using BF16  = __nv_bfloat16;
using BF162 = __nv_bfloat162;

// ==================== SM100 primitives ====================
// Keep the upstream TMEM/barrier protocol; the K1/K2 pipelines have different
// ownership and completion counts. Scalar approximations come from utils.cuh.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100)
__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
                 " P1, [%0], %1;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}\n" ::"r"(mbar_addr),
                 "r"(phase)
                 : "memory");
}

__device__ __forceinline__ void mma_ts_step(int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
                                            uint32_t i_desc, int enable_d) {
    asm volatile("{\n\t"
                 ".reg .pred leader, p;\n\t"
                 ".reg .b32 dhi;\n\t"
                 ".reg .b64 db;\n\t"
                 "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                 "setp.ne.b32 p, %5, 0;\n\t"
                 "mov.b32 dhi, %3;\n\t"
                 "mov.b64 db, {%2, dhi};\n\t"
                 "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
                 "}\n" ::"r"(taddr_out),
                 "r"(taddr_a), "r"(b_lo), "r"(b_dhi), "r"(i_desc), "r"(enable_d));
}

__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile("{\n\t"
                 ".reg .pred leader;\n\t"
                 "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                 "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                 ".shared::cluster.b64 [%0];\n\t"
                 "}\n" ::"r"(mbar_addr));
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
                 "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32"
                 " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
                 "  %9, %10, %11, %12, %13, %14, %15, %16,"
                 "  %17, %18, %19, %20, %21, %22, %23, %24,"
                 "  %25, %26, %27, %28, %29, %30, %31, %32};" ::"r"(tmem_addr),
                 "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]), "f"(src[4]), "f"(src[5]), "f"(src[6]),
                 "f"(src[7]), "f"(src[8]), "f"(src[9]), "f"(src[10]), "f"(src[11]), "f"(src[12]),
                 "f"(src[13]), "f"(src[14]), "f"(src[15]), "f"(src[16]), "f"(src[17]), "f"(src[18]),
                 "f"(src[19]), "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]), "f"(src[24]),
                 "f"(src[25]), "f"(src[26]), "f"(src[27]), "f"(src[28]), "f"(src[29]), "f"(src[30]),
                 "f"(src[31]));
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile("{\n\t"
                 ".reg .pred leader;\n\t"
                 "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                 "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                 ".shared::cluster.b64 [%0];\n\t"
                 "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                 ".shared::cluster.b64 [%1];\n\t"
                 "}\n" ::"r"(mbar_addr0),
                 "r"(mbar_addr1)
                 : "memory");
}

__device__ __forceinline__ void tma_3d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int mbar_addr) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global"
                 ".mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
                 "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
                 : "memory");
}

__device__ __forceinline__ void tma_4d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z, int w,
                                                 int mbar_addr) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global"
                 ".mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
                 "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr)
                 : "memory");
}

__device__ __forceinline__ void tma_store_4d(const void* tmap, int x, int y, int z, int w,
                                             unsigned smem_addr) {
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
                 " [%0, {%1, %2, %3, %4}], [%5];" ::"l"(tmap),
                 "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr)
                 : "memory");
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(result) : "r"(val));
    return result;
}
#endif

template <class Layouts> struct alignas(1024) SharedStorageFused {
    unsigned char data[Layouts::kSmemTotal];
};

// ==================== Fused layouts ====================
// One CTA owns all 128 value rows for one sequence/head and advances in N32 chunks.
// Shared-memory regions overlap intentionally: prepare and recurrence exchange
// five slots, whose lifetimes are coordinated by the barriers below.
// Upstream sources: flashkda_bf16_fused_m128_vtile_d589dad375, flashkda_bf16_fused_m128_vtile_4b7d3b6523
template <int NumHeads, bool FullChunks> struct FusedLayouts {
    static constexpr int kStateOffset         = 64;
    static constexpr int kUAccOffset          = 224;
    static constexpr int kU2InpOffset         = 224;
    static constexpr int kU2AccOffset         = 256;
    static constexpr int kOutOffset           = 192;
    static constexpr int kQDecayedOff         = 1024;
    static constexpr int kGRawOff             = 1024;
    static constexpr int kGRawAllOff          = 1024;
    static constexpr int kKDecayedOff         = 9216;
    static constexpr int kQRawPrefetchOff     = 17408;
    static constexpr int kFinalTransOff       = 17408;
    static constexpr int kKRestoredTransOff   = 17408;
    static constexpr int kMqkTransOff         = 25600;
    static constexpr int kInverseOff          = 29696;
    static constexpr int kStateDiag0Off       = 214016;
    static constexpr int kStateDiag1Off       = 214528;
    static constexpr int kStateDiag2Off       = 215040;
    static constexpr int kStateDiag3Off       = 215552;
    static constexpr int kStateDiag4Off       = 216064;
    static constexpr int kStateDiag5Off       = 216576;
    static constexpr int kStateDiag6Off       = 217088;
    static constexpr int kStateDiag7Off       = 217600;
    static constexpr int kVOff                = 218112;
    static constexpr int kKInverseOff         = 17408;
    static constexpr int kInverseWorkOff      = 32384;
    static constexpr int kOutOff              = 205824;
    static constexpr int kRestoreFactorAllOff = 39936;
    static constexpr int kGTotalAllOff        = 31744;
    static constexpr int kBetaAllOff          = 32256;
    static constexpr int kGateAllOff          = 25600;
    static constexpr int kSmemTotal           = 226304;
    static constexpr int kFullChunks          = FullChunks;
    static constexpr float kScaleValue        = 0.08838834764831845;
    static constexpr float kLowerBoundValue   = -5.0;
};

// ==================== Fused prepare + recurrence ====================
// Warps 0..3: state/residual; 4..7: output; 9: tcgen05 MMA; 10: V load;
// warps 12..31: five independent four-warp prepare groups. Warps 8 and 11 are idle.
// NumHeads=0 accepts runtime heads/scale/gate; 64/96 retain the measured default
// scalar constants. Every instantiation uses this same VTile Direct algorithm.
template <int NumHeads, bool FullChunks, bool StateFP32>
__global__ void __launch_bounds__(1024)
    _flash_kda_fwd_fused_vtile_direct(const __grid_constant__ FusedTensorMaps tensor_maps,
                                      const __grid_constant__ FusedParams params) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100)
    using Layouts                            = FusedLayouts<NumHeads, FullChunks>;
    auto* __restrict__ q_ptr                 = static_cast<BF16*>(params.q);
    auto const* tma_load_q                   = &tensor_maps.q;
    auto* __restrict__ k_ptr                 = static_cast<BF16*>(params.k);
    auto const* tma_load_k                   = &tensor_maps.k;
    auto* __restrict__ v_ptr                 = static_cast<BF16*>(params.v);
    auto const* tma_load_v                   = &tensor_maps.v;
    auto* __restrict__ g_ptr                 = static_cast<BF16*>(params.g);
    auto const* tma_load_g                   = &tensor_maps.g;
    auto* __restrict__ beta_ptr              = static_cast<BF16*>(params.beta);
    auto* __restrict__ A_log_ptr             = static_cast<float*>(params.A_log);
    auto* __restrict__ dt_bias_ptr           = static_cast<float*>(params.dt_bias);
    auto* __restrict__ initial_state_ptr     = static_cast<BF16*>(params.initial_state);
    auto* __restrict__ out_ptr               = static_cast<BF16*>(params.out);
    auto const* tma_store_out                = &tensor_maps.out;
    auto* __restrict__ final_state_ptr       = static_cast<BF16*>(params.final_state);
    long long state_slot_stride              = params.state_slot_stride;
    auto* __restrict__ initial_state_f32_ptr = static_cast<float*>(params.initial_state_f32);
    auto* __restrict__ final_state_f32_ptr   = static_cast<float*>(params.final_state_f32);
    const int num_heads                      = NumHeads ? NumHeads : params.num_heads;
    int use_initial_state                    = params.use_initial_state;
    int store_final_state                    = params.store_final_state;
    float scale                              = params.scale;
    float lower_bound                        = params.lower_bound;
    const int thread_idx                     = threadIdx.x;
    const int warp_idx                       = make_warp_uniform(thread_idx / 32);
    const int lane_idx                       = thread_idx % 32;
    const int sequence_idx = static_cast<const int*>(params.seq_order)[blockIdx.x / num_heads];
    const auto* offsets    = params.cu_seqlens;
    const long long sequence_begin =
        offsets ? offsets[sequence_idx] : (long long)sequence_idx * params.seq_len;
    const long long sequence_end = offsets ? offsets[sequence_idx + 1] : sequence_begin + params.seq_len;
    if (sequence_begin == sequence_end) {
        if (store_final_state) {
            const long long state_base =
                (long long)sequence_idx * state_slot_stride + (long long)(blockIdx.x % num_heads) * 128 * 128;
            for (int i = thread_idx; i < 128 * 128; i += blockDim.x) {
                if constexpr (StateFP32) {
                    final_state_f32_ptr[state_base + i] =
                        use_initial_state ? initial_state_f32_ptr[state_base + i] : 0.0f;
                } else {
                    final_state_ptr[state_base + i] =
                        use_initial_state ? initial_state_ptr[state_base + i] : BF16(0.0f);
                }
            }
        }
        return;
    }

    extern __shared__ __align__(1024) unsigned char storage[];
    auto& shared_storage = *reinterpret_cast<SharedStorageFused<Layouts>*>(storage);
    char* shared_mem     = reinterpret_cast<char*>(shared_storage.data);
    int smem_addr;
    smem_addr                         = (int)(unsigned long long)__cvta_generic_to_shared(shared_mem);
    const int mbar_base               = smem_addr;
    const int qk_full_addr            = (mbar_base + 0);
    const int gate_raw_full_addr      = (mbar_base + 40);
    const int qk_raw_full_addr        = (mbar_base + 80);
    const int v_full_addr             = (mbar_base + 120);
    const int v_free_addr             = (mbar_base + 128);
    const int smem_free_addr          = (mbar_base + 136);
    const int raw_inputs_free_addr    = (mbar_base + 176);
    const int state_inp_ready_addr    = (mbar_base + 216);
    const int old_out_ready_addr      = (mbar_base + 256);
    const int u_inp_ready_addr        = (mbar_base + 296);
    const int u2_acc_ready_addr       = (mbar_base + 336);
    const int state_diag_ready_addr   = (mbar_base + 376);
    const int u2_inp_ready_addr       = (mbar_base + 416);
    const int state_ready_addr        = (mbar_base + 456);
    const int final_ready_addr        = (mbar_base + 496);
    const int out_empty_addr          = (mbar_base + 536);
    const int tmem_dealloc_ready_addr = (mbar_base + 544);
    const int prep_diag_ready_addr    = (mbar_base + 552);
    const int prep_inv16_ready_addr   = (mbar_base + 592);
    const int state_copy_ready_addr   = (mbar_base + 632);

    __syncthreads();

    // --- shared-memory views (overlaid across pipeline phases)
    const int s_q_decayed_addr         = smem_addr + Layouts::kQDecayedOff;
    const int smem_g_raw_addr          = smem_addr + Layouts::kGRawOff;
    BF16* smem_g_raw_all               = reinterpret_cast<BF16*>(shared_mem + Layouts::kGRawAllOff);
    const int s_k_decayed_addr         = smem_addr + Layouts::kKDecayedOff;
    const int smem_q_raw_prefetch_addr = smem_addr + Layouts::kQRawPrefetchOff;
    const int smem_final_trans_addr    = smem_addr + Layouts::kFinalTransOff;
    const int s_k_restored_trans_addr  = smem_addr + Layouts::kKRestoredTransOff;
    const int s_mqk_trans_addr         = smem_addr + Layouts::kMqkTransOff;
    const int s_inverse_addr           = smem_addr + Layouts::kInverseOff;
    const int smem_state_diag0_addr    = smem_addr + Layouts::kStateDiag0Off;
    const int smem_state_diag1_addr    = smem_addr + Layouts::kStateDiag1Off;
    const int smem_state_diag2_addr    = smem_addr + Layouts::kStateDiag2Off;
    const int smem_state_diag3_addr    = smem_addr + Layouts::kStateDiag3Off;
    const int smem_state_diag4_addr    = smem_addr + Layouts::kStateDiag4Off;
    const int smem_state_diag5_addr    = smem_addr + Layouts::kStateDiag5Off;
    const int smem_state_diag6_addr    = smem_addr + Layouts::kStateDiag6Off;
    const int smem_state_diag7_addr    = smem_addr + Layouts::kStateDiag7Off;
    const int smem_v_addr              = smem_addr + Layouts::kVOff;
    const int s_k_inverse_addr         = smem_addr + Layouts::kKInverseOff;
    const int smem_inv_work_addr       = smem_addr + Layouts::kInverseWorkOff;
    const int smem_out_addr            = smem_addr + Layouts::kOutOff;
    float* smem_restore_factor_all     = reinterpret_cast<float*>(shared_mem + Layouts::kRestoreFactorAllOff);
    const int smem_restore_factor_all_addr = smem_addr + Layouts::kRestoreFactorAllOff;
    float* s_g_total_all                   = reinterpret_cast<float*>(shared_mem + Layouts::kGTotalAllOff);
    float* smem_beta_all                   = reinterpret_cast<float*>(shared_mem + Layouts::kBetaAllOff);
    float* smem_gate_all                   = reinterpret_cast<float*>(shared_mem + Layouts::kGateAllOff);
    const int smem_gate_all_addr           = smem_addr + Layouts::kGateAllOff;

    // Mbarrier init (20 groups, 83 barriers)
    // Mbarriers at smem_raw[0..664)

    if (warp_idx == 0) {
        uint32_t leader = cute::elect_one_sync();
        if (leader) {
            // --- pipeline 'chunk_pipe' ---
            // qk_full: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 0, 1);
            mbarrier_init(smem_addr + 8, 1);
            mbarrier_init(smem_addr + 16, 1);
            mbarrier_init(smem_addr + 24, 1);
            mbarrier_init(smem_addr + 32, 1);
            // gate_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 40, 1);
            mbarrier_init(smem_addr + 48, 1);
            mbarrier_init(smem_addr + 56, 1);
            mbarrier_init(smem_addr + 64, 1);
            mbarrier_init(smem_addr + 72, 1);
            // qk_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 80, 1);
            mbarrier_init(smem_addr + 88, 1);
            mbarrier_init(smem_addr + 96, 1);
            mbarrier_init(smem_addr + 104, 1);
            mbarrier_init(smem_addr + 112, 1);
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem_addr + 120, 1);
            // v_free: 1 barriers, init_count=4
            mbarrier_init(smem_addr + 128, 4);
            // smem_free: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 136, 1);
            mbarrier_init(smem_addr + 144, 1);
            mbarrier_init(smem_addr + 152, 1);
            mbarrier_init(smem_addr + 160, 1);
            mbarrier_init(smem_addr + 168, 1);
            // raw_inputs_free: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 176, 1);
            mbarrier_init(smem_addr + 184, 1);
            mbarrier_init(smem_addr + 192, 1);
            mbarrier_init(smem_addr + 200, 1);
            mbarrier_init(smem_addr + 208, 1);
            // state_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem_addr + 216, 4);
            mbarrier_init(smem_addr + 224, 4);
            mbarrier_init(smem_addr + 232, 4);
            mbarrier_init(smem_addr + 240, 4);
            mbarrier_init(smem_addr + 248, 4);
            // old_out_ready: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 256, 1);
            mbarrier_init(smem_addr + 264, 1);
            mbarrier_init(smem_addr + 272, 1);
            mbarrier_init(smem_addr + 280, 1);
            mbarrier_init(smem_addr + 288, 1);
            // u_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem_addr + 296, 4);
            mbarrier_init(smem_addr + 304, 4);
            mbarrier_init(smem_addr + 312, 4);
            mbarrier_init(smem_addr + 320, 4);
            mbarrier_init(smem_addr + 328, 4);
            // u2_acc_ready: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 336, 1);
            mbarrier_init(smem_addr + 344, 1);
            mbarrier_init(smem_addr + 352, 1);
            mbarrier_init(smem_addr + 360, 1);
            mbarrier_init(smem_addr + 368, 1);
            // state_diag_ready: 5 barriers, init_count=4
            mbarrier_init(smem_addr + 376, 4);
            mbarrier_init(smem_addr + 384, 4);
            mbarrier_init(smem_addr + 392, 4);
            mbarrier_init(smem_addr + 400, 4);
            mbarrier_init(smem_addr + 408, 4);
            // u2_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem_addr + 416, 4);
            mbarrier_init(smem_addr + 424, 4);
            mbarrier_init(smem_addr + 432, 4);
            mbarrier_init(smem_addr + 440, 4);
            mbarrier_init(smem_addr + 448, 4);
            // state_ready: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 456, 1);
            mbarrier_init(smem_addr + 464, 1);
            mbarrier_init(smem_addr + 472, 1);
            mbarrier_init(smem_addr + 480, 1);
            mbarrier_init(smem_addr + 488, 1);
            // final_ready: 5 barriers, init_count=1
            mbarrier_init(smem_addr + 496, 1);
            mbarrier_init(smem_addr + 504, 1);
            mbarrier_init(smem_addr + 512, 1);
            mbarrier_init(smem_addr + 520, 1);
            mbarrier_init(smem_addr + 528, 1);
            // out_empty: 1 barriers, init_count=1
            mbarrier_init(smem_addr + 536, 1);
            // tmem_dealloc_ready: 1 barriers, init_count=2
            mbarrier_init(smem_addr + 544, 2);
            // prep_diag_ready: 5 barriers, init_count=2
            mbarrier_init(smem_addr + 552, 2);
            mbarrier_init(smem_addr + 560, 2);
            mbarrier_init(smem_addr + 568, 2);
            mbarrier_init(smem_addr + 576, 2);
            mbarrier_init(smem_addr + 584, 2);
            // prep_inv16_ready: 5 barriers, init_count=2
            mbarrier_init(smem_addr + 592, 2);
            mbarrier_init(smem_addr + 600, 2);
            mbarrier_init(smem_addr + 608, 2);
            mbarrier_init(smem_addr + 616, 2);
            mbarrier_init(smem_addr + 624, 2);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // Publish explicit kernel-setup mbarrier initialization.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

    __syncwarp();

    // TMEM alloc (512 columns, 288 used)
    volatile int* tmem_addr_storage = (volatile int*)(shared_mem + 664);
    if (warp_idx == 0) {
        int tmem_hold = smem_addr + 664;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(tmem_hold),
                     "r"(512)
                     : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int tmem_addr = tmem_addr_storage[0];

    // --- tensor-memory views
    const int tmem_state     = tmem_addr + Layouts::kStateOffset;
    const int tmem_state_inp = tmem_addr;
    const int tmem_u_acc     = tmem_addr + Layouts::kUAccOffset;
    const int tmem_u2_inp    = tmem_addr + Layouts::kU2InpOffset;
    const int tmem_u2_acc    = tmem_addr + Layouts::kU2AccOffset;
    const int tmem_out       = tmem_addr + Layouts::kOutOffset;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp_idx >= 8 && warp_idx <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // --- state recurrence warps
    if (warp_idx <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // compute_main
            int task_idx            = blockIdx.x;
            int seq_idx             = sequence_idx;
            int head_idx            = task_idx % num_heads;
            long long bos           = sequence_begin;
            long long eos           = sequence_end;
            int seq_len             = (int)(eos - bos);
            int num_chunks          = (seq_len + 32 - 1) / 32;
            int total_chunks        = num_chunks;
            int warp_in_wg          = warp_idx % 4;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            int lane_quad           = lane_idx & 3;
            int state_row           = warp_in_wg * 32 + lane_idx;
            int warp_id_in_role     = (warp_idx - 0);
            int compute_local_warp  = warp_id_in_role;
            int state_slot          = seq_idx;
            long long state_base    = (long long)state_slot * state_slot_stride +
                                      (long long)head_idx * 128 * 128 + (long long)state_row * 128;
#pragma unroll
            for (int state_col_block = 0; state_col_block < 4; state_col_block++) {
                float state_init[32] = {};
                if (use_initial_state) {
                    if constexpr (StateFP32) {
                        {
#pragma unroll
                            for (int state_col_group = 0; state_col_group < 4; state_col_group++) {
                                {
                                    unsigned ldv8_0_0;
                                    unsigned ldv8_0_1;
                                    unsigned ldv8_0_2;
                                    unsigned ldv8_0_3;
                                    unsigned ldv8_0_4;
                                    unsigned ldv8_0_5;
                                    unsigned ldv8_0_6;
                                    unsigned ldv8_0_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(ldv8_0_0), "=r"(ldv8_0_1), "=r"(ldv8_0_2), "=r"(ldv8_0_3),
                                          "=r"(ldv8_0_4), "=r"(ldv8_0_5), "=r"(ldv8_0_6), "=r"(ldv8_0_7)
                                        : "l"((const void*)(initial_state_f32_ptr +
                                                            (state_base + (long long)(state_col_block * 32) +
                                                             (long long)(state_col_group * 8))))
                                        : "memory");
                                    state_init[state_col_group * 8 + 0] = __uint_as_float(ldv8_0_0);
                                    state_init[state_col_group * 8 + 1] = __uint_as_float(ldv8_0_1);
                                    state_init[state_col_group * 8 + 2] = __uint_as_float(ldv8_0_2);
                                    state_init[state_col_group * 8 + 3] = __uint_as_float(ldv8_0_3);
                                    state_init[state_col_group * 8 + 4] = __uint_as_float(ldv8_0_4);
                                    state_init[state_col_group * 8 + 5] = __uint_as_float(ldv8_0_5);
                                    state_init[state_col_group * 8 + 6] = __uint_as_float(ldv8_0_6);
                                    state_init[state_col_group * 8 + 7] = __uint_as_float(ldv8_0_7);
                                }
                            }
                        }
                    } else {
                        {
                            {
                                const uint4* vptr_0 = reinterpret_cast<const uint4*>(
                                    initial_state_ptr + state_base + (long long)(state_col_block * 32));
                                uint4 vld_0[4];
#pragma unroll
                                for (int blk = 0; blk < 4; blk++) {
                                    vld_0[blk]         = vptr_0[blk];
                                    uint32_t* vpairs_0 = reinterpret_cast<uint32_t*>(&vld_0[blk]);
#pragma unroll
                                    for (int pair = 0; pair < 4; pair++) {
                                        asm volatile("{\n\t"
                                                     "shl.b32 %0, %2, 16;\n\t"
                                                     "and.b32 %1, %2, 0xffff0000;\n\t"
                                                     "}\n"
                                                     : "=f"((&state_init[0 + blk * 8 + pair * 2])[0]),
                                                       "=f"((&state_init[0 + blk * 8 + pair * 2])[1])
                                                     : "r"(vpairs_0[pair]));
                                    }
                                }
                            }
                        }
                    }
                }
                tmem_st_x32_f32(tmem_addr + 64 + (unsigned int)tmem_row_base +
                                    (unsigned int)(state_col_block * 32),
                                state_init);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int zero_diag[4];
            zero_diag[0]             = 0;
            zero_diag[1]             = 0;
            zero_diag[2]             = 0;
            zero_diag[3]             = 0;
            int diag_lane_row        = lane_idx % 16;
            int diag_lane_col        = lane_idx / 16 * 8;
            int diag_block_0         = compute_local_warp;
            int diag_block_1         = compute_local_warp + 4;
            int diag_base_0          = smem_state_diag0_addr + (unsigned int)(diag_block_0 * 16 * 16 * 2);
            int diag_base_1          = smem_state_diag0_addr + (unsigned int)(diag_block_1 * 16 * 16 * 2);
            uint32_t stmatrix_addr_1 = static_cast<uint32_t>(
                (unsigned long long)(diag_base_0 +
                                     (diag_lane_col / 16 * 512 + diag_lane_row * 32 + diag_lane_col % 16 * 2 ^
                                      (diag_lane_col / 16 * 512 + diag_lane_row * 32 +
                                               diag_lane_col % 16 * 2 >>
                                           7 &
                                       1) << 4)));
            asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(stmatrix_addr_1),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[3]))
                : "memory");
            uint32_t stmatrix_addr_2 = static_cast<uint32_t>(
                (unsigned long long)(diag_base_1 +
                                     (diag_lane_col / 16 * 512 + diag_lane_row * 32 + diag_lane_col % 16 * 2 ^
                                      (diag_lane_col / 16 * 512 + diag_lane_row * 32 +
                                               diag_lane_col % 16 * 2 >>
                                           7 &
                                       1) << 4)));
            asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(stmatrix_addr_2),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[3]))
                : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int compute_stage       = 0;
            int chunk_idx                    = 0;
            unsigned int phase_qk_full       = 0;
            unsigned int phase_v_full_0      = 0;
            unsigned int phase_old_out_ready = 0;
            unsigned int phase_u2_acc_ready  = 0;
            unsigned int phase_state_ready   = 0;
#pragma unroll 1
            for (int global_chunk_idx = 0; global_chunk_idx < total_chunks; global_chunk_idx++) {
                {
                    chunk_idx = global_chunk_idx;
                }
#pragma unroll
                for (int state_col_half = 0; state_col_half < 2; state_col_half++) {
                    int state_addr_0 =
                        tmem_addr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_half * 64);
                    int state_addr_1 = state_addr_0 + 1048576;
                    float tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[0])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[1])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[2])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[3])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[4])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[5])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[6])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[7])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[8])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[9])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[10])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[11])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[12])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[13])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[14])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[15])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[16])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[17])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[18])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[19])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[20])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[21])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[22])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[23])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[24])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[25])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[26])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[27])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[28])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[29])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[30])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[31]))
                        : "r"(state_addr_0));
                    float tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[4])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[5])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[6])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[7])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[8])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[9])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[10])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[11])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[12])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[13])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[14])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[15])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[16])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[17])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[18])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[19])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[20])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[21])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[22])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[23])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[24])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[25])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[26])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[27])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[28])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[29])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[30])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[31]))
                        : "r"(state_addr_1));
                    uint32_t tmem_load_0_bf16[16];
#pragma unroll
                    for (int lp = 0; lp < 16; lp++) {
                        BF162 bf2 = __float22bfloat162_rn(
                            make_float2(tmem_load_0[lp * 2 + 0], tmem_load_0[lp * 2 + 1 + 0]));
                        tmem_load_0_bf16[lp] = *(uint32_t*)&bf2;
                    }
                    uint32_t tmem_load_1_bf16[16];
#pragma unroll
                    for (int lp = 0; lp < 16; lp++) {
                        BF162 bf2 = __float22bfloat162_rn(
                            make_float2(tmem_load_1[lp * 2 + 0], tmem_load_1[lp * 2 + 1 + 0]));
                        tmem_load_1_bf16[lp] = *(uint32_t*)&bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                            "r"(tmem_addr + (unsigned int)tmem_row_base +
                                (unsigned int)(state_col_half * 32)),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[0])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[1])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[2])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[3])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[4])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[5])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[6])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[7])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[8])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[9])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[10])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[11])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[12])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[13])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[14])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_0_bf16[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                            "r"(tmem_addr + (unsigned int)tmem_row_base + 1048576 +
                                (unsigned int)(state_col_half * 32)),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[0])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[1])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[2])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[3])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[4])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[5])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[6])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[7])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[8])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[9])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[10])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[11])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[12])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[13])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[14])),
                        "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_1_bf16[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (cute::elect_one_sync()) {
                        if (state_col_half != 0) {
                            mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                        }
                    }
                }
                mbarrier_wait(qk_full_addr + (compute_stage) * 8, phase_qk_full);
                float diag_scale_0 = 0.0f;
                float diag_scale_1 = 0.0f;
                if (lane_idx < 16) {
                    int diag_stage_f32 = compute_stage * 10240;
                    diag_scale_0       = s_g_total_all[diag_stage_f32 + compute_local_warp * 16 + lane_idx];
                    diag_scale_1 = s_g_total_all[diag_stage_f32 + (compute_local_warp + 4) * 16 + lane_idx];
                    {
                        BF16 bval_3     = __float2bfloat16_rn(diag_scale_0);
                        uint16_t bits_3 = *(uint16_t*)&bval_3;
                        uint32_t addr_3 = static_cast<uint32_t>(
                            (diag_base_0 +
                             (lane_idx / 16 * 512 + lane_idx * 32 + lane_idx % 16 * 2 ^
                              (lane_idx / 16 * 512 + lane_idx * 32 + lane_idx % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" ::"r"(addr_3), "h"(bits_3) : "memory");
                    }
                    {
                        BF16 bval_4     = __float2bfloat16_rn(diag_scale_1);
                        uint16_t bits_4 = *(uint16_t*)&bval_4;
                        uint32_t addr_4 = static_cast<uint32_t>(
                            (diag_base_1 +
                             (lane_idx / 16 * 512 + lane_idx * 32 + lane_idx % 16 * 2 ^
                              (lane_idx / 16 * 512 + lane_idx * 32 + lane_idx % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" ::"r"(addr_4), "h"(bits_4) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (cute::elect_one_sync()) {
                    mbarrier_arrive(state_diag_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(v_full_addr, phase_v_full_0);
                phase_v_full_0 ^= 1;
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, phase_old_out_ready);
                int v_stage_addr = smem_v_addr + (unsigned int)(warp_in_wg / 2 * 32 * 64 * 2);
                float tmem_load_2[16];
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32"
                             " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                             : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[8])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[9])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[10])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[11])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[12])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[13])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[14])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[15]))
                             : "r"(tmem_addr + 224 + (unsigned int)tmem_row_base));
                float tmem_load_3[16];
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32"
                             " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                             : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[0])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[1])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[2])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[3])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[4])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[5])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[6])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[7])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[8])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[9])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[10])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[11])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[12])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[13])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[14])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_3[15]))
                             : "r"(tmem_addr + 224 + (unsigned int)tmem_row_base + 1048576));
                float residual_values_0[16];
                float residual_values_1[16];
                unsigned int v_ld_bits_0[2];
                unsigned int v_ld_bits_1[2];
#pragma unroll
                for (int token_group = 0; token_group < 4; token_group++) {
                    int token_pair              = token_group * 8 + lane_quad * 2;
                    const int residual_reg_base = token_group * 4;
                    float beta_0      = smem_beta_all[compute_stage * 10240 + (unsigned int)token_pair];
                    float beta_1      = smem_beta_all[compute_stage * 10240 + (unsigned int)token_pair + 1];
                    int v_ld_matrix   = lane_idx / 8 & 1;
                    int v_ld_token    = token_group * 8 + (lane_idx & 7);
                    int v_ld_row_0    = warp_in_wg % 2 * 32 + v_ld_matrix * 8;
                    int v_ld_row_1    = v_ld_row_0 + 16;
                    int v_ld_row_addr = v_stage_addr + v_ld_token * 64 * 2;
                    int v_ld_addr_0   = (v_ld_row_addr + (v_ld_row_0 * 2 ^ (v_ld_row_addr >> 7 & 7) << 4));
                    int v_ld_addr_1   = (v_ld_row_addr + (v_ld_row_1 * 2 ^ (v_ld_row_addr >> 7 & 7) << 4));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(v_ld_bits_0[0]), "=r"(v_ld_bits_0[1])
                                 : "r"(v_ld_addr_0)
                                 : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(v_ld_bits_1[0]), "=r"(v_ld_bits_1[1])
                                 : "r"(v_ld_addr_1)
                                 : "memory");
                    float v_ld_bits_0_f32[4];
#pragma unroll
                    for (int pair = 0; pair < 2; pair++) {
                        asm volatile("{\n\t"
                                     "shl.b32 %0, %2, 16;\n\t"
                                     "and.b32 %1, %2, 0xffff0000;\n\t"
                                     "}\n"
                                     : "=f"((&v_ld_bits_0_f32[pair * 2])[0]),
                                       "=f"((&v_ld_bits_0_f32[pair * 2])[1])
                                     : "r"(v_ld_bits_0[pair]));
                    }
                    float v_ld_bits_1_f32[4];
#pragma unroll
                    for (int pair = 0; pair < 2; pair++) {
                        asm volatile("{\n\t"
                                     "shl.b32 %0, %2, 16;\n\t"
                                     "and.b32 %1, %2, 0xffff0000;\n\t"
                                     "}\n"
                                     : "=f"((&v_ld_bits_1_f32[pair * 2])[0]),
                                       "=f"((&v_ld_bits_1_f32[pair * 2])[1])
                                     : "r"(v_ld_bits_1[pair]));
                    }
                    residual_values_0[residual_reg_base] =
                        (v_ld_bits_0_f32[0] - tmem_load_2[residual_reg_base]) * beta_0;
                    residual_values_0[residual_reg_base + 1] =
                        (v_ld_bits_0_f32[1] - tmem_load_2[residual_reg_base + 1]) * beta_1;
                    residual_values_0[residual_reg_base + 2] =
                        (v_ld_bits_0_f32[2] - tmem_load_2[residual_reg_base + 2]) * beta_0;
                    residual_values_0[residual_reg_base + 3] =
                        (v_ld_bits_0_f32[3] - tmem_load_2[residual_reg_base + 3]) * beta_1;
                    residual_values_1[residual_reg_base] =
                        (v_ld_bits_1_f32[0] - tmem_load_3[residual_reg_base]) * beta_0;
                    residual_values_1[residual_reg_base + 1] =
                        (v_ld_bits_1_f32[1] - tmem_load_3[residual_reg_base + 1]) * beta_1;
                    residual_values_1[residual_reg_base + 2] =
                        (v_ld_bits_1_f32[2] - tmem_load_3[residual_reg_base + 2]) * beta_0;
                    residual_values_1[residual_reg_base + 3] =
                        (v_ld_bits_1_f32[3] - tmem_load_3[residual_reg_base + 3]) * beta_1;
                }
                uint32_t residual_values_0_bf16[8];
#pragma unroll
                for (int lp = 0; lp < 8; lp++) {
                    BF162 bf2 = __float22bfloat162_rn(
                        make_float2(residual_values_0[lp * 2 + 0], residual_values_0[lp * 2 + 1 + 0]));
                    residual_values_0_bf16[lp] = *(uint32_t*)&bf2;
                }
                uint32_t residual_values_1_bf16[8];
#pragma unroll
                for (int lp = 0; lp < 8; lp++) {
                    BF162 bf2 = __float22bfloat162_rn(
                        make_float2(residual_values_1[lp * 2 + 0], residual_values_1[lp * 2 + 1 + 0]));
                    residual_values_1_bf16[lp] = *(uint32_t*)&bf2;
                }
                asm volatile("tcgen05.st.sync.aligned.16x128b.x4.b32"
                             " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(tmem_addr + 224 +
                                                                              (unsigned int)tmem_row_base),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[0])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[1])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[2])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[3])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[4])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[5])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[6])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[7])));
                asm volatile("tcgen05.st.sync.aligned.16x128b.x4.b32"
                             " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(
                                 tmem_addr + 224 + (unsigned int)tmem_row_base + 1048576),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[0])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[1])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[2])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[3])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[4])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[5])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[6])),
                             "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[7])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (cute::elect_one_sync()) {
                    mbarrier_arrive(v_free_addr);
                    mbarrier_arrive(u_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(u2_acc_ready_addr + (compute_stage) * 8, phase_u2_acc_ready);
                float tmem_load_4[16];
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32"
                             " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                             : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[0])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[1])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[2])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[3])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[4])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[5])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[6])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[7])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[8])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[9])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[10])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[11])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[12])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[13])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[14])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_4[15]))
                             : "r"(tmem_addr + 256 + (unsigned int)tmem_row_base));
                float tmem_load_5[16];
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32"
                             " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                             : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[0])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[1])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[2])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[3])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[4])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[5])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[6])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[7])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[8])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[9])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[10])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[11])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[12])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[13])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[14])),
                               "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_5[15]))
                             : "r"(tmem_addr + 256 + (unsigned int)tmem_row_base + 1048576));
                uint32_t tmem_load_4_bf16[8];
#pragma unroll
                for (int lp = 0; lp < 8; lp++) {
                    BF162 bf2 = __float22bfloat162_rn(
                        make_float2(tmem_load_4[lp * 2 + 0], tmem_load_4[lp * 2 + 1 + 0]));
                    tmem_load_4_bf16[lp] = *(uint32_t*)&bf2;
                }
                uint32_t tmem_load_5_bf16[8];
#pragma unroll
                for (int lp = 0; lp < 8; lp++) {
                    BF162 bf2 = __float22bfloat162_rn(
                        make_float2(tmem_load_5[lp * 2 + 0], tmem_load_5[lp * 2 + 1 + 0]));
                    tmem_load_5_bf16[lp] = *(uint32_t*)&bf2;
                }
                asm volatile("tcgen05.st.sync.aligned.16x128b.x4.b32"
                             " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(tmem_addr + 224 +
                                                                              (unsigned int)tmem_row_base),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[0])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[1])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[2])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[3])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[4])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[5])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[6])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_4_bf16[7])));
                asm volatile("tcgen05.st.sync.aligned.16x128b.x4.b32"
                             " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(
                                 tmem_addr + 224 + (unsigned int)tmem_row_base + 1048576),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[0])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[1])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[2])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[3])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[4])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[5])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[6])),
                             "r"(*reinterpret_cast<const uint32_t*>(&tmem_load_5_bf16[7])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (cute::elect_one_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(state_ready_addr + (compute_stage) * 8, phase_state_ready);

                compute_stage += 1;
                if (compute_stage == 5) {
                    compute_stage = 0;
                    phase_qk_full ^= 1;
                    phase_old_out_ready ^= 1;
                    phase_u2_acc_ready ^= 1;
                    phase_state_ready ^= 1;
                }
            }
            if (store_final_state) {
#pragma unroll
                for (int state_col_block_2 = 0; state_col_block_2 < 4; state_col_block_2++) {
                    float tmem_load_7[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(tmem_load_7[0]), "=f"(tmem_load_7[1]), "=f"(tmem_load_7[2]),
                          "=f"(tmem_load_7[3]), "=f"(tmem_load_7[4]), "=f"(tmem_load_7[5]),
                          "=f"(tmem_load_7[6]), "=f"(tmem_load_7[7]), "=f"(tmem_load_7[8]),
                          "=f"(tmem_load_7[9]), "=f"(tmem_load_7[10]), "=f"(tmem_load_7[11]),
                          "=f"(tmem_load_7[12]), "=f"(tmem_load_7[13]), "=f"(tmem_load_7[14]),
                          "=f"(tmem_load_7[15]), "=f"(tmem_load_7[16]), "=f"(tmem_load_7[17]),
                          "=f"(tmem_load_7[18]), "=f"(tmem_load_7[19]), "=f"(tmem_load_7[20]),
                          "=f"(tmem_load_7[21]), "=f"(tmem_load_7[22]), "=f"(tmem_load_7[23]),
                          "=f"(tmem_load_7[24]), "=f"(tmem_load_7[25]), "=f"(tmem_load_7[26]),
                          "=f"(tmem_load_7[27]), "=f"(tmem_load_7[28]), "=f"(tmem_load_7[29]),
                          "=f"(tmem_load_7[30]), "=f"(tmem_load_7[31])
                        : "r"(tmem_addr + 64 + (unsigned int)tmem_row_base +
                              (unsigned int)(state_col_block_2 * 32)));
                    if constexpr (StateFP32) {
                        {
#pragma unroll
                            for (int state_col_group_2 = 0; state_col_group_2 < 4; state_col_group_2++) {
                                {
                                    unsigned stv8_6_0 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 0]);
                                    unsigned stv8_6_1 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 1]);
                                    unsigned stv8_6_2 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 2]);
                                    unsigned stv8_6_3 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 3]);
                                    unsigned stv8_6_4 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 4]);
                                    unsigned stv8_6_5 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 5]);
                                    unsigned stv8_6_6 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 6]);
                                    unsigned stv8_6_7 =
                                        __float_as_uint(tmem_load_7[state_col_group_2 * 8 + 7]);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(
                                            (void*)(final_state_f32_ptr +
                                                    (state_base + (long long)(state_col_block_2 * 32) +
                                                     (long long)(state_col_group_2 * 8)) +
                                                    (0))),
                                        "r"(stv8_6_0), "r"(stv8_6_1), "r"(stv8_6_2), "r"(stv8_6_3),
                                        "r"(stv8_6_4), "r"(stv8_6_5), "r"(stv8_6_6), "r"(stv8_6_7)
                                        : "memory");
                                }
                            }
                        }
                    } else {
                        {
                            {
                                {
                                    BF162 pk[4];
                                    pk[0] = __floats2bfloat162_rn(tmem_load_7[0 + 0], tmem_load_7[0 + 1]);
                                    pk[1] = __floats2bfloat162_rn(tmem_load_7[0 + 2], tmem_load_7[0 + 3]);
                                    pk[2] = __floats2bfloat162_rn(tmem_load_7[0 + 4], tmem_load_7[0 + 5]);
                                    pk[3] = __floats2bfloat162_rn(tmem_load_7[0 + 6], tmem_load_7[0 + 7]);
                                    *reinterpret_cast<uint4*>(
                                        &((BF16*)(final_state_ptr +
                                                  (state_base + (long long)(state_col_block_2 * 32))))[0]) =
                                        *reinterpret_cast<uint4*>(&pk[0]);
                                }
                                {
                                    BF162 pk[4];
                                    pk[0] = __floats2bfloat162_rn(tmem_load_7[0 + 8], tmem_load_7[0 + 9]);
                                    pk[1] = __floats2bfloat162_rn(tmem_load_7[0 + 10], tmem_load_7[0 + 11]);
                                    pk[2] = __floats2bfloat162_rn(tmem_load_7[0 + 12], tmem_load_7[0 + 13]);
                                    pk[3] = __floats2bfloat162_rn(tmem_load_7[0 + 14], tmem_load_7[0 + 15]);
                                    *reinterpret_cast<uint4*>(
                                        &((BF16*)(final_state_ptr +
                                                  (state_base + (long long)(state_col_block_2 * 32))))[8]) =
                                        *reinterpret_cast<uint4*>(&pk[0]);
                                }
                                {
                                    BF162 pk[4];
                                    pk[0] = __floats2bfloat162_rn(tmem_load_7[0 + 16], tmem_load_7[0 + 17]);
                                    pk[1] = __floats2bfloat162_rn(tmem_load_7[0 + 18], tmem_load_7[0 + 19]);
                                    pk[2] = __floats2bfloat162_rn(tmem_load_7[0 + 20], tmem_load_7[0 + 21]);
                                    pk[3] = __floats2bfloat162_rn(tmem_load_7[0 + 22], tmem_load_7[0 + 23]);
                                    *reinterpret_cast<uint4*>(
                                        &((BF16*)(final_state_ptr +
                                                  (state_base + (long long)(state_col_block_2 * 32))))[16]) =
                                        *reinterpret_cast<uint4*>(&pk[0]);
                                }
                                {
                                    BF162 pk[4];
                                    pk[0] = __floats2bfloat162_rn(tmem_load_7[0 + 24], tmem_load_7[0 + 25]);
                                    pk[1] = __floats2bfloat162_rn(tmem_load_7[0 + 26], tmem_load_7[0 + 27]);
                                    pk[2] = __floats2bfloat162_rn(tmem_load_7[0 + 28], tmem_load_7[0 + 29]);
                                    pk[3] = __floats2bfloat162_rn(tmem_load_7[0 + 30], tmem_load_7[0 + 31]);
                                    *reinterpret_cast<uint4*>(
                                        &((BF16*)(final_state_ptr +
                                                  (state_base + (long long)(state_col_block_2 * 32))))[24]) =
                                        *reinterpret_cast<uint4*>(&pk[0]);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (compute_local_warp == 0) {
                if (cute::elect_one_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
        // --- output warps
    } else if (warp_idx >= 4 && warp_idx <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 72;");
        { // epilogue_main
            int base_task_idx                = blockIdx.x;
            int base_seq_idx                 = sequence_idx;
            int base_head_idx                = base_task_idx % num_heads;
            long long base_bos               = sequence_begin;
            long long base_eos               = sequence_end;
            int seq_len_1                    = (int)(base_eos - base_bos);
            int num_chunks_1                 = (seq_len_1 + 32 - 1) / 32;
            int total_chunks_1               = num_chunks_1;
            int warp_id_in_role_1            = (warp_idx - 4);
            int epilogue_local_warp          = warp_id_in_role_1;
            int warp_in_wg_1                 = warp_idx % 4;
            const int tmem_row_base_1        = warp_in_wg_1 * 32 << 16;
            int state_row_1                  = warp_in_wg_1 * 32 + lane_idx;
            unsigned int epilogue_stage      = 0;
            unsigned int output_stage        = 0;
            int chunk_idx_1                  = 0;
            int head_idx_1                   = base_head_idx;
            long long bos_1                  = base_bos;
            long long eos_1                  = base_eos;
            unsigned int phase_final_ready   = 0;
            unsigned int phase_state_ready_1 = 0;
#pragma unroll 1
            for (int global_chunk_idx_1 = 0; global_chunk_idx_1 < total_chunks_1; global_chunk_idx_1++) {
                {
                    chunk_idx_1 = global_chunk_idx_1;
                }
                mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, phase_final_ready);
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 32) ? 1 : 0);
                if (Layouts::kFullChunks || chunk_is_full != 0) {
                    float tmem_load_8[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[0])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[1])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[2])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[3])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[4])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[5])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[6])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[7])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[8])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[9])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[10])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[11])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[12])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[13])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[14])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_8[15]))
                        : "r"(tmem_addr + 192 + (unsigned int)tmem_row_base_1));
                    float tmem_load_9[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[0])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[1])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[2])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[3])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[4])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[5])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[6])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[7])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[8])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[9])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[10])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[11])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[12])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[13])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[14])),
                          "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_9[15]))
                        : "r"(tmem_addr + 192 + (unsigned int)tmem_row_base_1 + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (cute::elect_one_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    unsigned int out_packed[8];
#pragma unroll
                    for (int lp = 0; lp < 8; lp++) {
                        BF162 bf2 = __float22bfloat162_rn(
                            make_float2(tmem_load_8[lp * 2 + 0], tmem_load_8[lp * 2 + 1 + 0]));
                        out_packed[lp] = *(uint32_t*)&bf2;
                    }
                    if (epilogue_local_warp == 0) {
                        if (global_chunk_idx_1 >= 1) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + output_stage * 8192;
#pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        if (dim_half != 0) {
#pragma unroll
                            for (int lp = 0; lp < 8; lp++) {
                                BF162 bf2 = __float22bfloat162_rn(
                                    make_float2(tmem_load_9[lp * 2 + 0], tmem_load_9[lp * 2 + 1 + 0]));
                                out_packed[lp] = *(uint32_t*)&bf2;
                            }
                        }
#pragma unroll
                        for (int token_group_1 = 0; token_group_1 < 2; token_group_1++) {
                            int mtx_idx      = lane_idx / 8;
                            int row_addr     = lane_idx & 7;
                            int dim_base     = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                            int token_base   = token_group_1 * 16 + mtx_idx / 2 * 8;
                            int token_addr   = token_base + row_addr;
                            int token_pair_1 = token_addr / 2;
                            int token_parity = token_addr & 1;
                            int raw_row      = token_pair_1 + dim_base / 64 * 16;
                            int raw_col      = (dim_base & 63 ^ (token_pair_1 & 3) << 4 ^ token_parity << 3) +
                                               token_parity * 64;
                            int stsm_offset  = (raw_row * 128 + raw_col) * 2;
                            const int pack_base = token_group_1 * 4;
                            uint32_t stmatrix_addr_0 =
                                static_cast<uint32_t>((unsigned long long)(out_stage_addr + stsm_offset));
                            asm volatile(
                                "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::
                                    "r"(stmatrix_addr_0),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base])),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 1])),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 2])),
                                "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (cute::elect_one_sync()) {
                            tma_store_4d(tma_store_out, 0, (int)(bos_1 + (long long)(chunk_idx_1 * 32)),
                                         head_idx_1, 0, smem_out_addr + output_stage * 8192);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                } else {
                    float tmem_load_10[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(tmem_load_10[0]), "=f"(tmem_load_10[1]), "=f"(tmem_load_10[2]),
                          "=f"(tmem_load_10[3]), "=f"(tmem_load_10[4]), "=f"(tmem_load_10[5]),
                          "=f"(tmem_load_10[6]), "=f"(tmem_load_10[7]), "=f"(tmem_load_10[8]),
                          "=f"(tmem_load_10[9]), "=f"(tmem_load_10[10]), "=f"(tmem_load_10[11]),
                          "=f"(tmem_load_10[12]), "=f"(tmem_load_10[13]), "=f"(tmem_load_10[14]),
                          "=f"(tmem_load_10[15]), "=f"(tmem_load_10[16]), "=f"(tmem_load_10[17]),
                          "=f"(tmem_load_10[18]), "=f"(tmem_load_10[19]), "=f"(tmem_load_10[20]),
                          "=f"(tmem_load_10[21]), "=f"(tmem_load_10[22]), "=f"(tmem_load_10[23]),
                          "=f"(tmem_load_10[24]), "=f"(tmem_load_10[25]), "=f"(tmem_load_10[26]),
                          "=f"(tmem_load_10[27]), "=f"(tmem_load_10[28]), "=f"(tmem_load_10[29]),
                          "=f"(tmem_load_10[30]), "=f"(tmem_load_10[31])
                        : "r"(tmem_addr + 192 + (unsigned int)tmem_row_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (cute::elect_one_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
#pragma unroll
                    for (int token_col = 0; token_col < 32; token_col++) {
                        long long out_token = bos_1 + (long long)(chunk_idx_1 * 32 + token_col);
                        if (out_token < eos_1) {
                            long long out_idx =
                                (out_token * (long long)num_heads + (long long)head_idx_1) * 128 +
                                (long long)state_row_1;
                            out_ptr[out_idx] = tmem_load_10[token_col];
                        }
                    }
                }

                epilogue_stage += 1;
                if (epilogue_stage == 5) {
                    epilogue_stage = 0;
                    phase_final_ready ^= 1;
                    phase_state_ready_1 ^= 1;
                }
            }
            if (epilogue_local_warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (epilogue_local_warp == 0) {
                if (cute::elect_one_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
        // ---- Role: idle ----
    } else if (warp_idx == 8 || warp_idx == 11) {
        // idle — no tasks assigned
        // ---- Role: mma ----
    } else if (warp_idx == 9) {
        { // mma_main
            long long mma_bos                   = sequence_begin;
            long long mma_eos                   = sequence_end;
            int mma_seq_len                     = (int)(mma_eos - mma_bos);
            int num_chunks_2                    = (mma_seq_len + 32 - 1) / 32;
            int total_chunks_2                  = num_chunks_2;
            unsigned int mma_stage              = 0;
            unsigned int phase_qk_full_1        = 0;
            unsigned int phase_state_inp_ready  = 0;
            unsigned int phase_out_empty_0      = 1;
            unsigned int phase_state_diag_ready = 0;
            unsigned int phase_u_inp_ready      = 0;
            unsigned int phase_u2_inp_ready     = 0;
#pragma unroll 1
            for (int global_chunk_idx = 0; global_chunk_idx < total_chunks_2; global_chunk_idx++) {
                mbarrier_wait(qk_full_addr + (mma_stage) * 8, phase_qk_full_1);
                mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, phase_state_inp_ready);
                {
                    mbarrier_wait(out_empty_addr, phase_out_empty_0);
                    phase_out_empty_0 ^= 1;
                }
                // Prediction: S[V,K] @ Kd[N,K]^T -> P[V,N].
                int mma_b_lo_0 = make_warp_uniform((((s_k_decayed_addr) >> 4) & 0x3FFF) + (mma_stage) * 2560);
                asm volatile("{\n\t"
                             ".reg .pred leader, p0, p1;\n\t"
                             ".reg .b32 dhi, blo, ta, id;\n\t"
                             ".reg .b64 db;\n\t"
                             "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                             "setp.ne.b32 p0, %3, 0;\n\t"
                             "setp.ne.b32 p1, 1, 0;\n\t"
                             ""
                             "mov.b32 dhi, 0x40004040;\n\t"
                             "mov.b32 id, 134743184;\n\t"
                             "mov.b32 ta, %2;\n\t"
                             "mov.b32 blo, %1;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 250;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "}\n" ::"r"(tmem_u_acc),
                             "r"(mma_b_lo_0), "r"(tmem_state_inp), "r"(0));
                elect_commit(old_out_ready_addr + (mma_stage) * 8);
                // History output: S @ Qd^T -> O[V,N].
                int mma_b_lo_1 = make_warp_uniform((((s_q_decayed_addr) >> 4) & 0x3FFF) + (mma_stage) * 2560);
                asm volatile("{\n\t"
                             ".reg .pred leader, p0, p1;\n\t"
                             ".reg .b32 dhi, blo, ta, id;\n\t"
                             ".reg .b64 db;\n\t"
                             "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                             "setp.ne.b32 p0, %3, 0;\n\t"
                             "setp.ne.b32 p1, 1, 0;\n\t"
                             ""
                             "mov.b32 dhi, 0x40004040;\n\t"
                             "mov.b32 id, 134743184;\n\t"
                             "mov.b32 ta, %2;\n\t"
                             "mov.b32 blo, %1;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 250;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 2;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "}\n" ::"r"(tmem_out),
                             "r"(mma_b_lo_1), "r"(tmem_state_inp), "r"(0));
                elect_commit(raw_inputs_free_addr + (mma_stage) * 8);
                mbarrier_wait(state_diag_ready_addr + (mma_stage) * 8, phase_state_diag_ready);
                // State decay, eight K=16 diagonal blocks.
                int mma_b_lo_2 =
                    make_warp_uniform(((((smem_state_diag0_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step(tmem_state, tmem_state_inp, mma_b_lo_2, 0xC0004010, 134546576, 0);
                int mma_b_lo_3 =
                    make_warp_uniform(((((smem_state_diag1_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_state + (16)), tmem_state_inp + 8, mma_b_lo_3, 0xC0004010, 134546576, 0);
                int mma_b_lo_4 =
                    make_warp_uniform(((((smem_state_diag2_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_state + (32)), tmem_state_inp + 16, mma_b_lo_4, 0xC0004010, 134546576, 0);
                int mma_b_lo_5 =
                    make_warp_uniform(((((smem_state_diag3_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_state + (48)), tmem_state_inp + 24, mma_b_lo_5, 0xC0004010, 134546576, 0);
                int mma_b_lo_6 =
                    make_warp_uniform(((((smem_state_diag4_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_state + (64)), tmem_state_inp + 32, mma_b_lo_6, 0xC0004010, 134546576, 0);
                int mma_b_lo_7 =
                    make_warp_uniform(((((smem_state_diag5_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_state + (80)), tmem_state_inp + 40, mma_b_lo_7, 0xC0004010, 134546576, 0);
                int mma_b_lo_8 =
                    make_warp_uniform(((((smem_state_diag6_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_state + (96)), tmem_state_inp + 48, mma_b_lo_8, 0xC0004010, 134546576, 0);
                int mma_b_lo_9 =
                    make_warp_uniform(((((smem_state_diag7_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_state + (112)), tmem_state_inp + 56, mma_b_lo_9, 0xC0004010, 134546576, 0);
                mbarrier_wait(u_inp_ready_addr + (mma_stage) * 8, phase_u_inp_ready);
                // U^T = [beta * (V - P)]^T @ INV^T.
                int mma_b_lo_10 = make_warp_uniform((((s_inverse_addr) >> 4) & 0x3FFF) + (mma_stage) * 2560);
                asm volatile("{\n\t"
                             ".reg .pred leader, p0, p1;\n\t"
                             ".reg .b32 dhi, blo, ta, id;\n\t"
                             ".reg .b64 db;\n\t"
                             "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                             "setp.ne.b32 p0, %3, 0;\n\t"
                             "setp.ne.b32 p1, 1, 0;\n\t"
                             ""
                             "mov.b32 dhi, 0xC0004010;\n\t"
                             "mov.b32 id, 134743184;\n\t"
                             "mov.b32 ta, %2;\n\t"
                             "mov.b32 blo, %1;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                             "add.u32 ta, ta, 8;\n\t"
                             "add.u32 blo, blo, 64;\n\t"
                             "mov.b64 db, {blo, dhi};\n\t"
                             "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                             "}\n" ::"r"(tmem_u2_acc),
                             "r"(mma_b_lo_10), "r"(tmem_u2_inp), "r"(0));
                elect_commit(u2_acc_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, phase_u2_inp_ready);
                {
                    // O += U^T @ Mqk^T; S += U^T @ Kr.
                    int mma_b_lo_13 = make_warp_uniform(((((s_mqk_trans_addr) >> 4) & 0x3FFF) | 0x1000000) +
                                                        (mma_stage) * 2560);
                    asm volatile("{\n\t"
                                 ".reg .pred leader, p0, p1;\n\t"
                                 ".reg .b32 dhi, blo, ta, id;\n\t"
                                 ".reg .b64 db;\n\t"
                                 "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                                 "setp.ne.b32 p0, %3, 0;\n\t"
                                 "setp.ne.b32 p1, 1, 0;\n\t"
                                 ""
                                 "mov.b32 dhi, 0x40004040;\n\t"
                                 "mov.b32 id, 134808720;\n\t"
                                 "mov.b32 ta, %2;\n\t"
                                 "mov.b32 blo, %1;\n\t"
                                 "mov.b64 db, {blo, dhi};\n\t"
                                 "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                                 "add.u32 ta, ta, 8;\n\t"
                                 "add.u32 blo, blo, 128;\n\t"
                                 "mov.b64 db, {blo, dhi};\n\t"
                                 "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                                 "}\n" ::"r"(tmem_out),
                                 "r"(mma_b_lo_13), "r"(tmem_u2_inp), "r"(1));
                    elect_commit(final_ready_addr + (mma_stage) * 8);
                    int mma_b_lo_14 = make_warp_uniform(
                        ((((s_k_restored_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_stage) * 2560);
                    asm volatile("{\n\t"
                                 ".reg .pred leader, p0, p1;\n\t"
                                 ".reg .b32 dhi, blo, ta, id;\n\t"
                                 ".reg .b64 db;\n\t"
                                 "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                                 "setp.ne.b32 p0, %3, 0;\n\t"
                                 "setp.ne.b32 p1, 1, 0;\n\t"
                                 ""
                                 "mov.b32 dhi, 0x40004040;\n\t"
                                 "mov.b32 id, 136381584;\n\t"
                                 "mov.b32 ta, %2;\n\t"
                                 "mov.b32 blo, %1;\n\t"
                                 "mov.b64 db, {blo, dhi};\n\t"
                                 "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                                 "add.u32 ta, ta, 8;\n\t"
                                 "add.u32 blo, blo, 128;\n\t"
                                 "mov.b64 db, {blo, dhi};\n\t"
                                 "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                                 "}\n" ::"r"(tmem_state),
                                 "r"(mma_b_lo_14), "r"(tmem_u2_inp), "r"(1));
                    elect_commit2(state_ready_addr + (mma_stage) * 8, smem_free_addr + (mma_stage) * 8);
                }
                mma_stage += 1;
                if (mma_stage == 5) {
                    mma_stage = 0;
                    phase_qk_full_1 ^= 1;
                    phase_state_inp_ready ^= 1;
                    phase_state_diag_ready ^= 1;
                    phase_u_inp_ready ^= 1;
                    phase_u2_inp_ready ^= 1;
                }
            }
            unsigned int phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, phase_tmem_dealloc_ready_0);
            phase_tmem_dealloc_ready_0 ^= 1;
            int tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_dealloc_addr),
                         "r"(512));
        }
        // ---- Role: load ----
    } else if (warp_idx == 10) {
        { // load_main
            int base_task_idx_1         = blockIdx.x;
            int base_head_idx_1         = base_task_idx_1 % num_heads;
            long long base_bos_1        = sequence_begin;
            long long base_eos_1        = sequence_end;
            int seq_len_2               = (int)(base_eos_1 - base_bos_1);
            int num_chunks_3            = (seq_len_2 + 32 - 1) / 32;
            int total_chunks_3          = num_chunks_3;
            int chunk_idx_2             = 0;
            int head_idx_2              = base_head_idx_1;
            long long bos_2             = base_bos_1;
            long long eos_2             = base_eos_1;
            unsigned int phase_v_free_0 = 1;
#pragma unroll 1
            for (int global_chunk_idx_2 = 0; global_chunk_idx_2 < total_chunks_3; global_chunk_idx_2++) {
                {
                    chunk_idx_2 = global_chunk_idx_2;
                }
                mbarrier_wait(v_free_addr, phase_v_free_0);
                phase_v_free_0 ^= 1;
                int chunk_is_full_1 = ((seq_len_2 >= (chunk_idx_2 + 1) * 32) ? 1 : 0);
                if (cute::elect_one_sync()) {
                    if (Layouts::kFullChunks || chunk_is_full_1 != 0) {
                        mbarrier_arrive_expect_tx(v_full_addr, 8192);
                        tma_4d_gmem2smem(smem_v_addr, tma_load_v, 0,
                                         (int)(bos_2 + (long long)(chunk_idx_2 * 32)), head_idx_2, 0,
                                         v_full_addr);
                    }
                }
                if (!Layouts::kFullChunks && chunk_is_full_1 == 0) {
#pragma unroll
                    for (int v_load_iter = 0; v_load_iter < 16; v_load_iter++) {
                        int v_item         = v_load_iter * 32 + lane_idx;
                        int row            = v_item / 16;
                        int segment        = v_item % 16;
                        long long token    = bos_2 + (long long)(chunk_idx_2 * 32 + row);
                        int token_valid    = ((token < eos_2) ? 1 : 0);
                        long long v_src    = (token * (long long)num_heads + (long long)head_idx_2) * 128 +
                                             (long long)(segment * 8);
                        int v_half         = segment / 8;
                        int v_half_segment = segment % 8;
                        int v_dst_row_addr =
                            smem_v_addr + (unsigned int)(v_half * 32 * 64 * 2) + (unsigned int)(row * 64 * 2);
                        int v_dst_addr =
                            (v_dst_row_addr + (v_half_segment * 8 * 2 ^ (v_dst_row_addr >> 7 & 7) << 4));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;" ::"r"(v_dst_addr),
                                     "l"(v_ptr + v_src), "r"((token_valid != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                }
                asm volatile("barrier.sync 8, 32;" ::: "memory");
                if (cute::elect_one_sync()) {
                    if (!Layouts::kFullChunks && chunk_is_full_1 == 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(v_full_addr);
                    }
                }
            }
        }
        // --- chunk preparation warps
    } else if (warp_idx >= 12 && warp_idx <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int base_task_idx_2         = blockIdx.x;
            int base_head_idx_2         = base_task_idx_2 % num_heads;
            long long base_bos_2        = sequence_begin;
            long long base_eos_2        = sequence_end;
            int seq_len_3               = (int)(base_eos_2 - base_bos_2);
            int num_chunks_4            = (seq_len_3 + 32 - 1) / 32;
            int total_chunks_4          = num_chunks_4;
            int instance_id             = (warp_idx - 12) / 4;
            int prep_instance           = instance_id;
            int warp_id_in_role_2       = (warp_idx - 12);
            int prep_local_warp         = warp_id_in_role_2 - prep_instance * 4;
            int prep_tid                = prep_local_warp * 32 + lane_idx;
            const float chunk_scale     = NumHeads ? Layouts::kScaleValue : scale;
            const float gate_bound      = NumHeads ? Layouts::kLowerBoundValue : lower_bound;
            int num_prep_iters          = (total_chunks_4 + 5 - 1 - prep_instance) / 5;
            unsigned int prep_stage     = (unsigned int)prep_instance;
            float gate_rate_task_cached = 0.0f;
            float gate_bias_task_cached = 0.0f;
            int chunk_idx_3             = prep_instance;
            int head_idx_3              = base_head_idx_2;
            long long bos_3             = base_bos_2;
            long long eos_3             = base_eos_2;
            {
                int cached_head_idx  = blockIdx.x % num_heads;
                float gate_rate_lane = 0.0f;
                if (lane_idx == 0) {
                    float expf_0   = __expf(A_log_ptr[cached_head_idx]);
                    gate_rate_lane = expf_0;
                }
                float shfl_0          = __shfl_sync(0xFFFFFFFF, gate_rate_lane, 0);
                gate_rate_task_cached = shfl_0;
                if (prep_tid < 128) {
                    gate_bias_task_cached = dt_bias_ptr[cached_head_idx * 128 + prep_tid];
                }
            }
            unsigned int phase_raw_inputs_free  = 1;
            unsigned int phase_smem_free        = 1;
            unsigned int phase_gate_raw_full    = 0;
            unsigned int phase_qk_raw_full      = 0;
            unsigned int phase_prep_diag_ready  = 0;
            unsigned int phase_prep_inv16_ready = 0;
#pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int global_chunk_idx_3 = prep_iter * 5 + prep_instance;
                {
                    chunk_idx_3 = global_chunk_idx_3;
                }
                float gate_rate_cached = gate_rate_task_cached;
                float gate_bias_cached = gate_bias_task_cached;
                int stage_f32          = prep_stage * 10240;
                int stage_bf16         = prep_stage * 20480;
                int chunk_is_full_2    = ((seq_len_3 >= (chunk_idx_3 + 1) * 32) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0      = 0.0f;
                mbarrier_wait(raw_inputs_free_addr + (prep_stage) * 8, phase_raw_inputs_free);
                if (Layouts::kFullChunks || chunk_is_full_2 != 0) {
                    if (prep_local_warp == 0) {
                        if (cute::elect_one_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + (prep_stage) * 8, 8192);
                            tma_3d_gmem2smem(smem_g_raw_addr + prep_stage * 40960, tma_load_g, 0, head_idx_3,
                                             (int)(bos_3 + (long long)(chunk_idx_3 * 32)),
                                             gate_raw_full_addr + (prep_stage) * 8);
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + (prep_stage) * 8, 16384);
                            tma_4d_gmem2smem(s_k_decayed_addr + prep_stage * 40960, tma_load_k, 0,
                                             (int)(bos_3 + (long long)(chunk_idx_3 * 32)), head_idx_3, 0,
                                             qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    if (prep_local_warp == 2 && lane_idx < 32) {
                        long long early_beta_token = bos_3 + (long long)(chunk_idx_3 * 32 + lane_idx);
                        float beta_logit =
                            (float)beta_ptr[early_beta_token * (long long)num_heads + (long long)head_idx_3];
                        float tanh_approx_0;
                        tanh_approx_0    = tanh_approx_f32(beta_logit * 0.5f);
                        early_beta_value = tanh_approx_0 * 0.5f + 0.5f;
                    }
                }
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, phase_smem_free);
                if (Layouts::kFullChunks || chunk_is_full_2 != 0) {
                    if (prep_local_warp == 0) {
                        if (cute::elect_one_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + prep_stage * 40960, tma_load_q, 0,
                                             (int)(bos_3 + (long long)(chunk_idx_3 * 32)), head_idx_3, 0,
                                             qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, phase_gate_raw_full);
                    if (prep_tid < 128) {
                        float early_gate_rate = gate_rate_cached;
                        float early_gate_bias = gate_bias_cached;
                        BF16 early_gate_raw   = smem_g_raw_all[stage_bf16 + prep_tid];
                        float cvt_f32_0       = __bfloat162float(early_gate_raw);
                        float early_gate_arg  = early_gate_rate * (cvt_f32_0 + early_gate_bias);
                        {
                            float tanh_approx_2;
                            tanh_approx_2            = tanh_approx_f32(early_gate_arg * 0.5f);
                            float early_gate_sigmoid = tanh_approx_2 * 0.5f + 0.5f;
                            early_gate0              = gate_bound * 1.4426950408889634f * early_gate_sigmoid;
                        }
                    }
                }
                if (!Layouts::kFullChunks && chunk_is_full_2 == 0) {
#pragma unroll
                    for (int gate_load_pass = 0; gate_load_pass < 4; gate_load_pass++) {
                        int gate_load_item        = gate_load_pass * 128 + prep_tid;
                        int gate_load_row         = gate_load_item / 16;
                        int gate_load_segment     = gate_load_item % 16;
                        long long gate_load_token = bos_3 + (long long)(chunk_idx_3 * 32 + gate_load_row);
                        long long gate_load_base =
                            (gate_load_token * (long long)num_heads + (long long)head_idx_3) * 128 +
                            (long long)(gate_load_segment * 8);
                        asm volatile(
                            "cp.async.cg.shared::cta.global [%0], [%1], 16, %2;" ::"r"(
                                smem_g_raw_addr + prep_stage * 40960 + (unsigned int)(gate_load_item * 16)),
                            "l"(g_ptr + gate_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                    }
                }
                if (!Layouts::kFullChunks && chunk_is_full_2 == 0) {
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" ::"r"(11 + prep_instance) : "memory");
                }
                if (prep_local_warp == 2 && lane_idx < 32) {
                    float beta_value = early_beta_value;
                    if (!Layouts::kFullChunks && chunk_is_full_2 == 0) {
                        long long beta_token = bos_3 + (long long)(chunk_idx_3 * 32 + lane_idx);
                        if (beta_token < eos_3) {
                            float beta_logit_1 =
                                (float)beta_ptr[beta_token * (long long)num_heads + (long long)head_idx_3];
                            float tanh_approx_3;
                            tanh_approx_3 = tanh_approx_f32(beta_logit_1 * 0.5f);
                            beta_value    = tanh_approx_3 * 0.5f + 0.5f;
                        }
                    }
                    early_beta_value = beta_value;
                }
                if (prep_tid < 128) {
                    int gate_col      = prep_tid;
                    float gate_rate   = gate_rate_cached;
                    float gate_bias   = gate_bias_cached;
                    float prefix_log2 = 0.0f;
                    if (Layouts::kFullChunks || chunk_is_full_2 != 0) {
                        prefix_log2                                                       = early_gate0;
                        smem_gate_all[stage_f32 + (gate_col & 99 | gate_col >> 1 & 12 |
                                                   (gate_col >> 1 ^ gate_col << 2) & 16)] = prefix_log2;
                        for (int gate_block_idx = 0; gate_block_idx < 4; gate_block_idx++) {
                            float gate_block[8];
                            for (int gate_row_in_block = 0; gate_row_in_block < 8; gate_row_in_block++) {
                                gate_block[gate_row_in_block] = 0.0f;
                                if (1 + gate_block_idx * 8 + gate_row_in_block < 32) {
                                    BF16 gate_raw =
                                        smem_g_raw_all[stage_bf16 +
                                                       (1 + gate_block_idx * 8 + gate_row_in_block) * 128 +
                                                       gate_col];
                                    float cvt_f32_1 = __bfloat162float(gate_raw);
                                    float gate_arg  = gate_rate * (cvt_f32_1 + gate_bias);
                                    float tanh_approx_4;
                                    tanh_approx_4   = tanh_approx_f32(gate_arg * 0.5f);
                                    float gate_tanh = tanh_approx_4;
                                    {
                                        gate_block[gate_row_in_block] =
                                            gate_bound * 1.4426950408889634f * (gate_tanh * 0.5f + 0.5f);
                                    }
                                }
                            }
                            for (int gate_row_in_block_1 = 0; gate_row_in_block_1 < 8;
                                 gate_row_in_block_1++) {
                                if (1 + gate_block_idx * 8 + gate_row_in_block_1 < 32) {
                                    prefix_log2 += gate_block[gate_row_in_block_1];
                                    smem_gate_all[stage_f32 +
                                                  (1 + gate_block_idx * 8 + gate_row_in_block_1) * 128 +
                                                  (gate_col & 99 | gate_col >> 1 & 12 |
                                                   (gate_col >> 1 ^ gate_col << 2) & 16)] = prefix_log2;
                                }
                            }
                        }
                    } else {
                        for (int gate_row = 0; gate_row < 32; gate_row++) {
                            long long gate_token = bos_3 + (long long)(chunk_idx_3 * 32 + gate_row);
                            float gate_log2      = 0.0f;
                            if (gate_token < eos_3) {
                                BF16 gate_raw_1  = smem_g_raw_all[stage_bf16 + gate_row * 128 + gate_col];
                                float cvt_f32_2  = __bfloat162float(gate_raw_1);
                                float gate_arg_1 = gate_rate * (cvt_f32_2 + gate_bias);
                                float tanh_approx_5;
                                tanh_approx_5      = tanh_approx_f32(gate_arg_1 * 0.5f);
                                float gate_sigmoid = tanh_approx_5 * 0.5f + 0.5f;
                                gate_log2          = gate_bound * 1.4426950408889634f * gate_sigmoid;
                            }
                            prefix_log2 += gate_log2;
                            smem_gate_all[stage_f32 + gate_row * 128 +
                                          (gate_col & 99 | gate_col >> 1 & 12 |
                                           (gate_col >> 1 ^ gate_col << 2) & 16)] = prefix_log2;
                        }
                    }
                }
                asm volatile("barrier.sync %0, 128;" ::"r"(11 + prep_instance) : "memory");
                if (Layouts::kFullChunks || chunk_is_full_2 != 0) {
                    mbarrier_wait(qk_raw_full_addr + (prep_stage) * 8, phase_qk_raw_full);
                }
#pragma unroll 1
                for (int work_pass = 0; work_pass < 4; work_pass++) {
                    int work_item       = work_pass * 128 + prep_tid;
                    int row_1           = work_item / 16;
                    int segment_1       = work_item % 16;
                    long long token_1   = bos_3 + (long long)(chunk_idx_3 * 32 + row_1);
                    int token_valid_1   = ((token_1 < eos_3) ? 1 : 0);
                    long long gmem_base = (token_1 * (long long)num_heads + (long long)head_idx_3) * 128 +
                                          (long long)(segment_1 * 8);
                    float q_raw_vec[8];
                    float k_raw_vec[8];
                    q_raw_vec[0] = 0.0f;
                    q_raw_vec[1] = 0.0f;
                    q_raw_vec[2] = 0.0f;
                    q_raw_vec[3] = 0.0f;
                    q_raw_vec[4] = 0.0f;
                    q_raw_vec[5] = 0.0f;
                    q_raw_vec[6] = 0.0f;
                    q_raw_vec[7] = 0.0f;
                    k_raw_vec[0] = 0.0f;
                    k_raw_vec[1] = 0.0f;
                    k_raw_vec[2] = 0.0f;
                    k_raw_vec[3] = 0.0f;
                    k_raw_vec[4] = 0.0f;
                    k_raw_vec[5] = 0.0f;
                    k_raw_vec[6] = 0.0f;
                    k_raw_vec[7] = 0.0f;
                    if (Layouts::kFullChunks || chunk_is_full_2 != 0) {
                        unsigned int packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                     : "r"((smem_q_raw_prefetch_addr + prep_stage * 40960 +
                                            (unsigned int)(segment_1 * 8 / 64 * 4096 + row_1 * 128 +
                                                               segment_1 * 8 % 64 * 2 ^
                                                           (segment_1 * 8 / 64 * 4096 + row_1 * 128 +
                                                                    segment_1 * 8 % 64 * 2 >>
                                                                7 &
                                                            7) << 4))));
                        float packed_f32[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_f32[pair * 2])[0]), "=f"((&packed_f32[pair * 2])[1])
                                         : "r"(packed[pair]));
                        }
#pragma unroll
                        for (int value_idx = 0; value_idx < 8; value_idx++) {
                            q_raw_vec[value_idx] = packed_f32[value_idx];
                        }
                        unsigned int packed_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                                     : "r"((s_k_decayed_addr + prep_stage * 40960 +
                                            (unsigned int)(segment_1 * 8 / 64 * 4096 + row_1 * 128 +
                                                               segment_1 * 8 % 64 * 2 ^
                                                           (segment_1 * 8 / 64 * 4096 + row_1 * 128 +
                                                                    segment_1 * 8 % 64 * 2 >>
                                                                7 &
                                                            7) << 4))));
                        float packed_0_f32[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_0_f32[pair * 2])[0]),
                                           "=f"((&packed_0_f32[pair * 2])[1])
                                         : "r"(packed_0[pair]));
                        }
#pragma unroll
                        for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                            k_raw_vec[value_idx_1] = packed_0_f32[value_idx_1];
                        }
                    } else if (token_valid_1 != 0) {
                        {
                            const uint4* vptr_0 = reinterpret_cast<const uint4*>(q_ptr + gmem_base);
                            uint4 vld_0[1];
#pragma unroll
                            for (int blk = 0; blk < 1; blk++) {
                                vld_0[blk]         = vptr_0[blk];
                                uint32_t* vpairs_0 = reinterpret_cast<uint32_t*>(&vld_0[blk]);
#pragma unroll
                                for (int pair = 0; pair < 4; pair++) {
                                    asm volatile("{\n\t"
                                                 "shl.b32 %0, %2, 16;\n\t"
                                                 "and.b32 %1, %2, 0xffff0000;\n\t"
                                                 "}\n"
                                                 : "=f"((&q_raw_vec[0 + blk * 8 + pair * 2])[0]),
                                                   "=f"((&q_raw_vec[0 + blk * 8 + pair * 2])[1])
                                                 : "r"(vpairs_0[pair]));
                                }
                            }
                        }
                        {
                            const uint4* vptr_1 = reinterpret_cast<const uint4*>(k_ptr + gmem_base);
                            uint4 vld_1[1];
#pragma unroll
                            for (int blk = 0; blk < 1; blk++) {
                                vld_1[blk]         = vptr_1[blk];
                                uint32_t* vpairs_1 = reinterpret_cast<uint32_t*>(&vld_1[blk]);
#pragma unroll
                                for (int pair = 0; pair < 4; pair++) {
                                    asm volatile("{\n\t"
                                                 "shl.b32 %0, %2, 16;\n\t"
                                                 "and.b32 %1, %2, 0xffff0000;\n\t"
                                                 "}\n"
                                                 : "=f"((&k_raw_vec[0 + blk * 8 + pair * 2])[0]),
                                                   "=f"((&k_raw_vec[0 + blk * 8 + pair * 2])[1])
                                                 : "r"(vpairs_1[pair]));
                                }
                            }
                        }
                    }
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                        float q_raw = q_raw_vec[elem_in_segment];
                        float k_raw = k_raw_vec[elem_in_segment];
                        float fma_1 = __fmaf_rn(q_raw, q_raw, q_sum);
                        q_sum       = fma_1;
                        float fma_2 = __fmaf_rn(k_raw, k_raw, k_sum);
                        k_sum       = fma_2;
                    }
                    float shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 8);
                    q_sum += shfl_xor_0;
                    float shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 8);
                    k_sum += shfl_xor_1;
                    float shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                    q_sum += shfl_xor_2;
                    float shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                    k_sum += shfl_xor_3;
                    float shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                    q_sum += shfl_xor_4;
                    float shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                    k_sum += shfl_xor_5;
                    float shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                    q_sum += shfl_xor_6;
                    float shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                    k_sum += shfl_xor_7;
                    float rsqrt_0         = rsqrtf(q_sum + 1e-06f);
                    float q_inv           = rsqrt_0;
                    float rsqrt_1         = rsqrtf(k_sum + 1e-06f);
                    float k_inv           = rsqrt_1;
                    const float2 scale2_2 = {q_inv, q_inv};
#pragma unroll
                    for (int ls = 0; ls < 4; ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_raw_vec)[ls], scale2_2);
                    const float2 scale2_3 = {k_inv, k_inv};
#pragma unroll
                    for (int ls = 0; ls < 4; ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(k_raw_vec)[ls], scale2_3);
                    float qd_vec[8];
                    float kd_vec[8];
                    float ki_vec[8];
                    float gate_prefix_lo[4];
                    float gate_prefix_hi[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[0])),
                                   "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[(0) + 1])),
                                   "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[(0) + 2])),
                                   "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[(0) + 3]))
                                 : "r"(smem_gate_all_addr +
                                       (unsigned int)((stage_f32 + row_1 * 128 +
                                                       (segment_1 * 8 & 99 | segment_1 * 8 >> 1 & 12 |
                                                        (segment_1 * 8 >> 1 ^ segment_1 * 8 << 2) & 16)) *
                                                      4)));
                    asm volatile(
                        "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[0])),
                          "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[(0) + 1])),
                          "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[(0) + 2])),
                          "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[(0) + 3]))
                        : "r"(smem_gate_all_addr +
                              (unsigned int)((stage_f32 + row_1 * 128 +
                                              (segment_1 * 8 + 4 & 99 | segment_1 * 8 + 4 >> 1 & 12 |
                                               (segment_1 * 8 + 4 >> 1 ^ segment_1 * 8 + 4 << 2) & 16)) *
                                             4)));
                    for (int elem_in_segment_1 = 0; elem_in_segment_1 < 8; elem_in_segment_1++) {
                        float prefix      = ((elem_in_segment_1 < 4) ? gate_prefix_lo[elem_in_segment_1]
                                                                     : gate_prefix_hi[elem_in_segment_1 - 4]);
                        float common_log2 = gate_bound * 1.4426950408889634f * 16.0f;
                        float decay_arg   = prefix - common_log2;
                        float exp2_0      = ex2_approx_ftz_f32(decay_arg);
                        float decay       = exp2_0;
                        float exp2_1      = ex2_approx_ftz_f32(-decay_arg);
                        float inverse_decay       = exp2_1;
                        qd_vec[elem_in_segment_1] = decay;
                        kd_vec[elem_in_segment_1] = decay;
                        ki_vec[elem_in_segment_1] = k_raw_vec[elem_in_segment_1] * inverse_decay;
                    }
#pragma unroll
                    for (int ls = 0; ls < 4; ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[ls],
                                          reinterpret_cast<const float2*>(q_raw_vec)[ls]);
                    const float2 scale2_4 = {chunk_scale, chunk_scale};
#pragma unroll
                    for (int ls = 0; ls < 4; ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[ls], scale2_4);
#pragma unroll
                    for (int ls = 0; ls < 4; ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(kd_vec)[ls],
                                          reinterpret_cast<const float2*>(k_raw_vec)[ls]);
                    unsigned int packed_1[4];
#pragma unroll
                    for (int lp = 0; lp < 4; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(qd_vec[lp * 2 + 0], qd_vec[lp * 2 + 1 + 0]));
                        packed_1[lp] = *(uint32_t*)&bf2;
                    }
                    asm volatile(
                        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                            (s_q_decayed_addr + prep_stage * 40960 +
                             (unsigned int)(segment_1 * 8 / 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^
                                            (segment_1 * 8 / 64 * 4096 + row_1 * 128 +
                                                     segment_1 * 8 % 64 * 2 >>
                                                 7 &
                                             7) << 4))),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1[0])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3])));
                    unsigned int packed_0_1[4];
#pragma unroll
                    for (int lp = 0; lp < 4; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(kd_vec[lp * 2 + 0], kd_vec[lp * 2 + 1 + 0]));
                        packed_0_1[lp] = *(uint32_t*)&bf2;
                    }
                    asm volatile(
                        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                            (s_k_decayed_addr + prep_stage * 40960 +
                             (unsigned int)(segment_1 * 8 / 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^
                                            (segment_1 * 8 / 64 * 4096 + row_1 * 128 +
                                                     segment_1 * 8 % 64 * 2 >>
                                                 7 &
                                             7) << 4))),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_0_1[0])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 1])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 2])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 3])));
                    unsigned int packed_1_1[4];
#pragma unroll
                    for (int lp = 0; lp < 4; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(ki_vec[lp * 2 + 0], ki_vec[lp * 2 + 1 + 0]));
                        packed_1_1[lp] = *(uint32_t*)&bf2;
                    }
                    asm volatile(
                        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                            (s_k_inverse_addr + prep_stage * 40960 +
                             (unsigned int)(segment_1 * 8 / 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^
                                            (segment_1 * 8 / 64 * 4096 + row_1 * 128 +
                                                     segment_1 * 8 % 64 * 2 >>
                                                 7 &
                                             7) << 4))),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1_1[0])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 1])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 2])),
                        "r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 3])));
                }
                asm volatile("barrier.sync %0, 128;" ::"r"(11 + prep_instance) : "memory");
                if (prep_local_warp == 2 && lane_idx < 32) {
                    smem_beta_all[stage_f32 + lane_idx] = early_beta_value;
                }
                asm volatile("barrier.sync %0, 128;" ::"r"(11 + prep_instance) : "memory");
                if (prep_tid < 128) {
                    float total_log2        = smem_gate_all[stage_f32 + 3968 +
                                                            (prep_tid & 99 | prep_tid >> 1 & 12 |
                                                             (prep_tid >> 1 ^ prep_tid << 2) & 16)];
                    int restore_publish_col = prep_tid;
                    restore_publish_col =
                        prep_tid & 99 | prep_tid >> 1 & 12 | (prep_tid >> 1 ^ prep_tid << 2) & 16;
                    float exp2_2 = ex2_approx_ftz_f32(total_log2 - gate_bound * 1.4426950408889634f * 16.0f);
                    float restore_factor_value                               = exp2_2;
                    smem_restore_factor_all[stage_f32 + restore_publish_col] = restore_factor_value;
                }
                if (prep_tid == 0) {
                    float exp2_3 = ex2_approx_ftz_f32(gate_bound * 1.4426950408889634f * 16.0f);
                    smem_restore_factor_all[stage_f32 + 272] = exp2_3;
                }
                int pair_row_base = prep_local_warp / 2 * 16;
                int pair_col_base = prep_local_warp % 2 * 16;
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float k_acc[8];
                float q_acc[8];
                if (pair_row_base >= pair_col_base) {
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                              (lane_idx / 16 % 8 * 16 ^ (pair_row_base + lane_idx % 16 & 7)
                                                                            << 4) /
                                                  16) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(
                            s_k_inverse_addr + prep_stage * 40960 +
                            (unsigned int)((lane_idx % 16 / 8 / 8 * 256 +
                                            (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) * 8 +
                                            (lane_idx % 16 / 8 % 8 * 16 ^
                                             (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 & 7) << 4) /
                                                16) *
                                           16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                 : "=f"(k_acc[0]), "=f"(k_acc[1]), "=f"(k_acc[2]), "=f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, "
                        "%7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(k_acc[4]), "=f"(k_acc[(4) + 1]), "=f"(k_acc[(4) + 2]), "=f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]),
                          "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                              (lane_idx / 16 % 8 * 16 ^ (pair_row_base + lane_idx % 16 & 7)
                                                                            << 4) /
                                                  16) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                 : "=f"(q_acc[0]), "=f"(q_acc[1]), "=f"(q_acc[2]), "=f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, "
                        "%7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(q_acc[4]), "=f"(q_acc[(4) + 1]), "=f"(q_acc[(4) + 2]), "=f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]),
                          "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                  (lane_idx / 16 % 8 * 16 ^
                                                   (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                      16 ^
                                              2) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(s_k_inverse_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx % 16 / 8 / 8 * 256 +
                                                   (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) * 8 +
                                                   (lane_idx % 16 / 8 % 8 * 16 ^
                                                    (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 & 7)
                                                        << 4) /
                                                       16 +
                                                   256 ^
                                               2) -
                                              256) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]),
                                   "+f"(k_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                  (lane_idx / 16 % 8 * 16 ^
                                                   (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                      16 ^
                                              2) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]),
                                   "+f"(q_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                  (lane_idx / 16 % 8 * 16 ^
                                                   (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                      16 ^
                                              2 ^ 6) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(s_k_inverse_addr + prep_stage * 40960 +
                              (unsigned int)((((lane_idx % 16 / 8 / 8 * 256 +
                                                    (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) * 8 +
                                                    (lane_idx % 16 / 8 % 8 * 16 ^
                                                     (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 & 7)
                                                         << 4) /
                                                        16 +
                                                    256 ^
                                                2) -
                                                   256 + 256 ^
                                               6) -
                                              256) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]),
                                   "+f"(k_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                  (lane_idx / 16 % 8 * 16 ^
                                                   (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                      16 ^
                                              2 ^ 6) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]),
                                   "+f"(q_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                  (lane_idx / 16 % 8 * 16 ^
                                                   (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                      16 ^
                                              2 ^ 6 ^ 2) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(
                            s_k_inverse_addr + prep_stage * 40960 +
                            (unsigned int)(((((lane_idx % 16 / 8 / 8 * 256 +
                                                   (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) * 8 +
                                                   (lane_idx % 16 / 8 % 8 * 16 ^
                                                    (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 & 7)
                                                        << 4) /
                                                       16 +
                                                   256 ^
                                               2) -
                                                  256 + 256 ^
                                              6) -
                                                 256 + 256 ^
                                             2) -
                                            256) *
                                           16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]),
                                   "+f"(k_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                  (lane_idx / 16 % 8 * 16 ^
                                                   (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                      16 ^
                                              2 ^ 6 ^ 2) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]),
                                   "+f"(q_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                              256) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(
                            s_k_inverse_addr + prep_stage * 40960 +
                            (unsigned int)((((((lane_idx % 16 / 8 / 8 * 256 +
                                                    (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) * 8 +
                                                    (lane_idx % 16 / 8 % 8 * 16 ^
                                                     (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 & 7)
                                                         << 4) /
                                                        16 +
                                                    256 ^
                                                2) -
                                                   256 + 256 ^
                                               6) -
                                                  256 + 256 ^
                                              2) -
                                                 256 + 256 ^
                                             6) +
                                            256 - 256) *
                                           16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]),
                                   "+f"(k_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                              256) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]),
                                   "+f"(q_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                                  256 ^
                                              2) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(s_k_inverse_addr + prep_stage * 40960 +
                              (unsigned int)(((((((lane_idx % 16 / 8 / 8 * 256 +
                                                       (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) *
                                                           8 +
                                                       (lane_idx % 16 / 8 % 8 * 16 ^
                                                        (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 &
                                                         7) << 4) /
                                                           16 +
                                                       256 ^
                                                   2) -
                                                      256 + 256 ^
                                                  6) -
                                                     256 + 256 ^
                                                 2) -
                                                    256 + 256 ^
                                                6) +
                                                   256 - 256 + 256 ^
                                               2) -
                                              256) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]),
                                   "+f"(k_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                                  256 ^
                                              2) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]),
                                   "+f"(q_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                                  256 ^
                                              2 ^ 6) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(s_k_inverse_addr + prep_stage * 40960 +
                              (unsigned int)((((((((lane_idx % 16 / 8 / 8 * 256 +
                                                        (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) *
                                                            8 +
                                                        (lane_idx % 16 / 8 % 8 * 16 ^
                                                         (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 &
                                                          7) << 4) /
                                                            16 +
                                                        256 ^
                                                    2) -
                                                       256 + 256 ^
                                                   6) -
                                                      256 + 256 ^
                                                  2) -
                                                     256 + 256 ^
                                                 6) +
                                                    256 - 256 + 256 ^
                                                2) -
                                                   256 + 256 ^
                                               6) -
                                              256) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]),
                                   "+f"(k_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                                  256 ^
                                              2 ^ 6) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]),
                                   "+f"(q_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_k_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                                  256 ^
                                              2 ^ 6 ^ 2) *
                                             16))
                        : "memory");
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(
                            s_k_inverse_addr + prep_stage * 40960 +
                            (unsigned int)(((((((((lane_idx % 16 / 8 / 8 * 256 +
                                                       (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8) *
                                                           8 +
                                                       (lane_idx % 16 / 8 % 8 * 16 ^
                                                        (pair_col_base + 8 * (lane_idx / 16) + lane_idx % 8 &
                                                         7) << 4) /
                                                           16 +
                                                       256 ^
                                                   2) -
                                                      256 + 256 ^
                                                  6) -
                                                     256 + 256 ^
                                                 2) -
                                                    256 + 256 ^
                                                6) +
                                                   256 - 256 + 256 ^
                                               2) -
                                                  256 + 256 ^
                                              6) -
                                                 256 + 256 ^
                                             2) -
                                            256) *
                                           16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]),
                                   "+f"(k_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(s_q_decayed_addr + prep_stage * 40960 +
                              (unsigned int)(((lane_idx / 16 / 8 * 256 + (pair_row_base + lane_idx % 16) * 8 +
                                                   (lane_idx / 16 % 8 * 16 ^
                                                    (pair_row_base + lane_idx % 16 & 7) << 4) /
                                                       16 ^
                                               2 ^ 6 ^ 2 ^ 6) +
                                                  256 ^
                                              2 ^ 6 ^ 2) *
                                             16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]),
                                   "+f"(q_acc[(4) + 3])
                                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                   "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    int row0      = pair_row_base + lane_idx / 4;
                    int row1      = row0 + 8;
                    int col0      = pair_col_base + lane_idx % 4 * 2;
                    float beta0   = smem_beta_all[stage_f32 + row0];
                    float beta1   = smem_beta_all[stage_f32 + row1];
                    float seed[8] = {};
                    if (row0 > col0) {
                        seed[0] = k_acc[0] * beta0;
                    }
                    if (row0 > col0 + 1) {
                        seed[1] = k_acc[1] * beta0;
                    }
                    if (row1 > col0) {
                        seed[2] = k_acc[2] * beta1;
                    }
                    if (row1 > col0 + 1) {
                        seed[3] = k_acc[3] * beta1;
                    }
                    if (row0 > col0 + 8) {
                        seed[4] = k_acc[4] * beta0;
                    }
                    if (row0 > col0 + 9) {
                        seed[5] = k_acc[5] * beta0;
                    }
                    if (row1 > col0 + 8) {
                        seed[6] = k_acc[6] * beta1;
                    }
                    if (row1 > col0 + 9) {
                        seed[7] = k_acc[7] * beta1;
                    }
                    unsigned int seed_packed[4];
#pragma unroll
                    for (int lp = 0; lp < 4; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(seed[lp * 2 + 0], seed[lp * 2 + 1 + 0]));
                        seed_packed[lp] = *(uint32_t*)&bf2;
                    }
                    int seed_lane_row = lane_idx % 16;
                    int seed_lane_col = lane_idx / 16 * 8;
                    int byte_off =
                        (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2;
                    int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                    int seed_addr    = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off;
                    uint32_t stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)seed_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                                     stmatrix_addr_5),
                                 "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                                 : "memory");
                } else {
                    q_acc[0] = 0.0f;
                    q_acc[1] = 0.0f;
                    q_acc[2] = 0.0f;
                    q_acc[3] = 0.0f;
                    q_acc[4] = 0.0f;
                    q_acc[5] = 0.0f;
                    q_acc[6] = 0.0f;
                    q_acc[7] = 0.0f;
                }
                int row0_1   = pair_row_base + lane_idx / 4;
                int row1_1   = row0_1 + 8;
                int col0_1   = pair_col_base + lane_idx % 4 * 2;
                float mqk[8] = {};
                if (row0_1 >= col0_1) {
                    mqk[0] = q_acc[0];
                }
                if (row0_1 >= col0_1 + 1) {
                    mqk[1] = q_acc[1];
                }
                if (row1_1 >= col0_1) {
                    mqk[2] = q_acc[2];
                }
                if (row1_1 >= col0_1 + 1) {
                    mqk[3] = q_acc[3];
                }
                if (row0_1 >= col0_1 + 8) {
                    mqk[4] = q_acc[4];
                }
                if (row0_1 >= col0_1 + 9) {
                    mqk[5] = q_acc[5];
                }
                if (row1_1 >= col0_1 + 8) {
                    mqk[6] = q_acc[6];
                }
                if (row1_1 >= col0_1 + 9) {
                    mqk[7] = q_acc[7];
                }
                unsigned int mqk_packed[4];
#pragma unroll
                for (int lp = 0; lp < 4; lp++) {
                    BF162 bf2      = __float22bfloat162_rn(make_float2(mqk[lp * 2 + 0], mqk[lp * 2 + 1 + 0]));
                    mqk_packed[lp] = *(uint32_t*)&bf2;
                }
#pragma unroll
                for (int publish_pair = 0; publish_pair < 2; publish_pair++) {
                    int publish_row          = pair_col_base + publish_pair * 8 + (lane_idx & 7);
                    int publish_col          = 128 + pair_row_base + lane_idx / 8 * 8;
                    uint32_t stmatrix_addr_6 = static_cast<uint32_t>(
                        (unsigned long long)(smem_final_trans_addr + prep_stage * 40960 +
                                             (unsigned int)(publish_col / 64 * 4096 + publish_row * 128 +
                                                                publish_col % 64 * 2 ^
                                                            (publish_col / 64 * 4096 + publish_row * 128 +
                                                                     publish_col % 64 * 2 >>
                                                                 7 &
                                                             7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n" ::"r"(
                                     stmatrix_addr_6),
                                 "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                                 : "memory");
                }

                asm volatile("barrier.sync %0, 128;" ::"r"(11 + prep_instance) : "memory");
                if (prep_tid < 128) {
                    float total_log2_1 = smem_gate_all[stage_f32 + 3968 +
                                                       (prep_tid & 99 | prep_tid >> 1 & 12 |
                                                        (prep_tid >> 1 ^ prep_tid << 2) & 16)];
                    float exp2_5       = ex2_approx_ftz_f32(total_log2_1);
                    s_g_total_all[stage_f32 + prep_tid] = exp2_5;
                }
                if (prep_local_warp >= 2) {
                    int stage_f32_0     = prep_stage * 10240;
                    float restore_scale = smem_restore_factor_all[stage_f32_0 + 272];
                    float restore_factor[8];
                    int restore_segment = lane_idx & 15;
                    {
                        int restore_half_id    = lane_idx >> 4;
                        int restore_vector_col = restore_segment * 8 + restore_half_id * 4 & 99 |
                                                 restore_segment * 8 + restore_half_id * 4 >> 1 & 12 |
                                                 (restore_segment * 8 + restore_half_id * 4 >> 1 ^
                                                  restore_segment * 8 + restore_half_id * 4 << 2) &
                                                     16;
                        float restore_factor_half[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half[0])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half[(0) + 1])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half[(0) + 2])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half[(0) + 3]))
                                     : "r"(smem_restore_factor_all_addr +
                                           (unsigned int)((stage_f32_0 + restore_vector_col) * 4)));
#pragma unroll
                        for (int restore_elem = 0; restore_elem < 4; restore_elem++) {
                            float shfl_xor_8 =
                                __shfl_xor_sync(0xFFFFFFFF, restore_factor_half[restore_elem], 16);
                            float restore_factor_partner = shfl_xor_8;
                            if (restore_half_id == 0) {
                                restore_factor[restore_elem]     = restore_factor_half[restore_elem];
                                restore_factor[restore_elem + 4] = restore_factor_partner;
                            } else {
                                restore_factor[restore_elem]     = restore_factor_partner;
                                restore_factor[restore_elem + 4] = restore_factor_half[restore_elem];
                            }
                        }
                    }
#pragma unroll 1
                    for (int restore_pass = 0; restore_pass < 6; restore_pass++) {
                        int restore_row = 8 + (prep_local_warp - 2) * 12 + restore_pass * 2 + (lane_idx >> 4);
                        float restore_qd_values[8];
                        float restore_kd_values[8];
                        float restore_ki_values[8];
                        unsigned int packed_2[4];
                        asm volatile(
                            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                            : "r"((s_q_decayed_addr + prep_stage * 40960 +
                                   (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                      restore_segment * 8 % 64 * 2 ^
                                                  (restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                           restore_segment * 8 % 64 * 2 >>
                                                       7 &
                                                   7) << 4))));
                        float packed_f32_1[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_f32_1[pair * 2])[0]),
                                           "=f"((&packed_f32_1[pair * 2])[1])
                                         : "r"(packed_2[pair]));
                        }
#pragma unroll
                        for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                            restore_qd_values[value_idx_2] = packed_f32_1[value_idx_2];
                        }
                        unsigned int packed_0_2[4];
                        asm volatile(
                            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[0])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 1])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 2])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 3]))
                            : "r"((s_k_decayed_addr + prep_stage * 40960 +
                                   (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                      restore_segment * 8 % 64 * 2 ^
                                                  (restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                           restore_segment * 8 % 64 * 2 >>
                                                       7 &
                                                   7) << 4))));
                        float packed_0_f32_1[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_0_f32_1[pair * 2])[0]),
                                           "=f"((&packed_0_f32_1[pair * 2])[1])
                                         : "r"(packed_0_2[pair]));
                        }
#pragma unroll
                        for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                            restore_kd_values[value_idx_3] = packed_0_f32_1[value_idx_3];
                        }
                        unsigned int packed_1_2[4];
                        asm volatile(
                            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[0])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 1])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 2])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 3]))
                            : "r"((s_k_inverse_addr + prep_stage * 40960 +
                                   (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                      restore_segment * 8 % 64 * 2 ^
                                                  (restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                           restore_segment * 8 % 64 * 2 >>
                                                       7 &
                                                   7) << 4))));
                        float packed_1_f32[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_1_f32[pair * 2])[0]),
                                           "=f"((&packed_1_f32[pair * 2])[1])
                                         : "r"(packed_1_2[pair]));
                        }
#pragma unroll
                        for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                            restore_ki_values[value_idx_4] = packed_1_f32[value_idx_4];
                        }
                        float restore_kr_values[8];
#pragma unroll
                        for (int restore_elem_1 = 0; restore_elem_1 < 8; restore_elem_1++) {
                            restore_kr_values[restore_elem_1] =
                                restore_ki_values[restore_elem_1] * restore_factor[restore_elem_1];
                        }
                        const float2 scale2_7 = {restore_scale, restore_scale};
#pragma unroll
                        for (int ls = 0; ls < 4; ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values)[ls], scale2_7);
                        const float2 scale2_8 = {restore_scale, restore_scale};
#pragma unroll
                        for (int ls = 0; ls < 4; ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values)[ls], scale2_8);
                        unsigned int packed_2_1[4];
#pragma unroll
                        for (int lp = 0; lp < 4; lp++) {
                            BF162 bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[lp * 2 + 0],
                                                                          restore_qd_values[lp * 2 + 1 + 0]));
                            packed_2_1[lp] = *(uint32_t*)&bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"((
                                         s_q_decayed_addr + prep_stage * 40960 +
                                         (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                            restore_segment * 8 % 64 * 2 ^
                                                        (restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                                 restore_segment * 8 % 64 * 2 >>
                                                             7 &
                                                         7) << 4))),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[0])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[(0) + 1])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[(0) + 2])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[(0) + 3])));
                        unsigned int packed_3[4];
#pragma unroll
                        for (int lp = 0; lp < 4; lp++) {
                            BF162 bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[lp * 2 + 0],
                                                                          restore_kd_values[lp * 2 + 1 + 0]));
                            packed_3[lp] = *(uint32_t*)&bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"((
                                         s_k_decayed_addr + prep_stage * 40960 +
                                         (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                            restore_segment * 8 % 64 * 2 ^
                                                        (restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                                 restore_segment * 8 % 64 * 2 >>
                                                             7 &
                                                         7) << 4))),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_3[0])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 1])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 2])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 3])));
                        unsigned int packed_4[4];
#pragma unroll
                        for (int lp = 0; lp < 4; lp++) {
                            BF162 bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[lp * 2 + 0],
                                                                          restore_kr_values[lp * 2 + 1 + 0]));
                            packed_4[lp] = *(uint32_t*)&bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"((
                                         s_k_restored_trans_addr + prep_stage * 40960 +
                                         (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                            restore_segment * 8 % 64 * 2 ^
                                                        (restore_segment * 8 / 64 * 4096 + restore_row * 128 +
                                                                 restore_segment * 8 % 64 * 2 >>
                                                             7 &
                                                         7) << 4))),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_4[0])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])),
                                     "r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3])));
                    }
                } else if (prep_local_warp == 1) {
                    int stage_f32_0_1     = prep_stage * 10240;
                    float restore_scale_1 = smem_restore_factor_all[stage_f32_0_1 + 272];
                    float restore_factor_1[8];
                    int restore_segment_1 = lane_idx & 15;
                    {
                        int restore_half_id_1    = lane_idx >> 4;
                        int restore_vector_col_1 = restore_segment_1 * 8 + restore_half_id_1 * 4 & 99 |
                                                   restore_segment_1 * 8 + restore_half_id_1 * 4 >> 1 & 12 |
                                                   (restore_segment_1 * 8 + restore_half_id_1 * 4 >> 1 ^
                                                    restore_segment_1 * 8 + restore_half_id_1 * 4 << 2) &
                                                       16;
                        float restore_factor_half_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half_1[0])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half_1[(0) + 1])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half_1[(0) + 2])),
                                       "=r"(*reinterpret_cast<uint32_t*>(&restore_factor_half_1[(0) + 3]))
                                     : "r"(smem_restore_factor_all_addr +
                                           (unsigned int)((stage_f32_0_1 + restore_vector_col_1) * 4)));
#pragma unroll
                        for (int restore_elem_2 = 0; restore_elem_2 < 4; restore_elem_2++) {
                            float shfl_xor_9 =
                                __shfl_xor_sync(0xFFFFFFFF, restore_factor_half_1[restore_elem_2], 16);
                            float restore_factor_partner_1 = shfl_xor_9;
                            if (restore_half_id_1 == 0) {
                                restore_factor_1[restore_elem_2]     = restore_factor_half_1[restore_elem_2];
                                restore_factor_1[restore_elem_2 + 4] = restore_factor_partner_1;
                            } else {
                                restore_factor_1[restore_elem_2]     = restore_factor_partner_1;
                                restore_factor_1[restore_elem_2 + 4] = restore_factor_half_1[restore_elem_2];
                            }
                        }
                    }
#pragma unroll 1
                    for (int restore_pass_1 = 0; restore_pass_1 < 4; restore_pass_1++) {
                        int restore_row_1 = restore_pass_1 * 2 + (lane_idx >> 4);
                        float restore_qd_values_1[8];
                        float restore_kd_values_1[8];
                        float restore_ki_values_1[8];
                        unsigned int packed_5[4];
                        asm volatile(
                            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
                            : "r"((s_q_decayed_addr + prep_stage * 40960 +
                                   (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                      restore_segment_1 * 8 % 64 * 2 ^
                                                  (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                           restore_segment_1 * 8 % 64 * 2 >>
                                                       7 &
                                                   7) << 4))));
                        float packed_f32_2[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_f32_2[pair * 2])[0]),
                                           "=f"((&packed_f32_2[pair * 2])[1])
                                         : "r"(packed_5[pair]));
                        }
#pragma unroll
                        for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                            restore_qd_values_1[value_idx_5] = packed_f32_2[value_idx_5];
                        }
                        unsigned int packed_0_3[4];
                        asm volatile(
                            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[0])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 1])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 2])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 3]))
                            : "r"((s_k_decayed_addr + prep_stage * 40960 +
                                   (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                      restore_segment_1 * 8 % 64 * 2 ^
                                                  (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                           restore_segment_1 * 8 % 64 * 2 >>
                                                       7 &
                                                   7) << 4))));
                        float packed_0_f32_2[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_0_f32_2[pair * 2])[0]),
                                           "=f"((&packed_0_f32_2[pair * 2])[1])
                                         : "r"(packed_0_3[pair]));
                        }
#pragma unroll
                        for (int value_idx_6 = 0; value_idx_6 < 8; value_idx_6++) {
                            restore_kd_values_1[value_idx_6] = packed_0_f32_2[value_idx_6];
                        }
                        unsigned int packed_1_3[4];
                        asm volatile(
                            "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[0])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 1])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 2])),
                              "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 3]))
                            : "r"((s_k_inverse_addr + prep_stage * 40960 +
                                   (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                      restore_segment_1 * 8 % 64 * 2 ^
                                                  (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                           restore_segment_1 * 8 % 64 * 2 >>
                                                       7 &
                                                   7) << 4))));
                        float packed_1_f32_1[8];
#pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile("{\n\t"
                                         "shl.b32 %0, %2, 16;\n\t"
                                         "and.b32 %1, %2, 0xffff0000;\n\t"
                                         "}\n"
                                         : "=f"((&packed_1_f32_1[pair * 2])[0]),
                                           "=f"((&packed_1_f32_1[pair * 2])[1])
                                         : "r"(packed_1_3[pair]));
                        }
#pragma unroll
                        for (int value_idx_7 = 0; value_idx_7 < 8; value_idx_7++) {
                            restore_ki_values_1[value_idx_7] = packed_1_f32_1[value_idx_7];
                        }
                        float restore_kr_values_1[8];
#pragma unroll
                        for (int restore_elem_3 = 0; restore_elem_3 < 8; restore_elem_3++) {
                            restore_kr_values_1[restore_elem_3] =
                                restore_ki_values_1[restore_elem_3] * restore_factor_1[restore_elem_3];
                        }
                        const float2 scale2_9 = {restore_scale_1, restore_scale_1};
#pragma unroll
                        for (int ls = 0; ls < 4; ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values_1)[ls], scale2_9);
                        const float2 scale2_10 = {restore_scale_1, restore_scale_1};
#pragma unroll
                        for (int ls = 0; ls < 4; ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values_1)[ls], scale2_10);
                        unsigned int packed_2_2[4];
#pragma unroll
                        for (int lp = 0; lp < 4; lp++) {
                            BF162 bf2      = __float22bfloat162_rn(make_float2(
                                restore_qd_values_1[lp * 2 + 0], restore_qd_values_1[lp * 2 + 1 + 0]));
                            packed_2_2[lp] = *(uint32_t*)&bf2;
                        }
                        asm volatile(
                            "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                                (s_q_decayed_addr + prep_stage * 40960 +
                                 (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                    restore_segment_1 * 8 % 64 * 2 ^
                                                (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                         restore_segment_1 * 8 % 64 * 2 >>
                                                     7 &
                                                 7) << 4))),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[0])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[(0) + 1])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[(0) + 2])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[(0) + 3])));
                        unsigned int packed_3_1[4];
#pragma unroll
                        for (int lp = 0; lp < 4; lp++) {
                            BF162 bf2      = __float22bfloat162_rn(make_float2(
                                restore_kd_values_1[lp * 2 + 0], restore_kd_values_1[lp * 2 + 1 + 0]));
                            packed_3_1[lp] = *(uint32_t*)&bf2;
                        }
                        asm volatile(
                            "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                                (s_k_decayed_addr + prep_stage * 40960 +
                                 (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                    restore_segment_1 * 8 % 64 * 2 ^
                                                (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                         restore_segment_1 * 8 % 64 * 2 >>
                                                     7 &
                                                 7) << 4))),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[0])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[(0) + 1])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[(0) + 2])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[(0) + 3])));
                        unsigned int packed_4_1[4];
#pragma unroll
                        for (int lp = 0; lp < 4; lp++) {
                            BF162 bf2      = __float22bfloat162_rn(make_float2(
                                restore_kr_values_1[lp * 2 + 0], restore_kr_values_1[lp * 2 + 1 + 0]));
                            packed_4_1[lp] = *(uint32_t*)&bf2;
                        }
                        asm volatile(
                            "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                                (s_k_restored_trans_addr + prep_stage * 40960 +
                                 (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                    restore_segment_1 * 8 % 64 * 2 ^
                                                (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 +
                                                         restore_segment_1 * 8 % 64 * 2 >>
                                                     7 &
                                                 7) << 4))),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[0])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 1])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 2])),
                            "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 3])));
                    }
                }
                if (prep_local_warp == 0) {
                    int inverse_row  = lane_idx;
                    int diag_block   = inverse_row / 8;
                    int lane_in_diag = lane_idx & 7;
                    float inv_row[8];
                    unsigned int packed_6[4];
                    int byte_off_1     = inverse_row * 128 + diag_block * 8 * 2;
                    int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                    asm volatile(
                        "ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_6[0])),
                          "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 1])),
                          "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 2])),
                          "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 3]))
                        : "r"(smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_1));
                    float packed_f32_3[8];
#pragma unroll
                    for (int pair = 0; pair < 4; pair++) {
                        asm volatile("{\n\t"
                                     "shl.b32 %0, %2, 16;\n\t"
                                     "and.b32 %1, %2, 0xffff0000;\n\t"
                                     "}\n"
                                     : "=f"((&packed_f32_3[pair * 2])[0]), "=f"((&packed_f32_3[pair * 2])[1])
                                     : "r"(packed_6[pair]));
                    }
#pragma unroll
                    for (int value_idx_8 = 0; value_idx_8 < 8; value_idx_8++) {
                        inv_row[value_idx_8] = packed_f32_3[value_idx_8];
                    }
#pragma unroll
                    for (int diag_elem = 0; diag_elem < 8; diag_elem++) {
                        if (lane_in_diag == diag_elem) {
                            inv_row[diag_elem] = 1.0f;
                        }
                    }
                    int diag_group_base = lane_idx - lane_in_diag;
#pragma unroll
                    for (int src_row = 0; src_row < 7; src_row++) {
                        float row_scale = -inv_row[src_row];
#pragma unroll
                        for (int prev_col = 0; prev_col < src_row; prev_col++) {
                            int pivot_lane = diag_group_base + src_row;
                            float shfl_2   = __shfl_sync(0xFFFFFFFF, inv_row[prev_col], pivot_lane);
                            float pivot    = shfl_2;
                            if (lane_in_diag > src_row) {
                                float fma_3       = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                                inv_row[prev_col] = fma_3;
                            }
                        }
                        if (lane_in_diag > src_row) {
                            inv_row[src_row] = row_scale;
                        }
                    }
                    unsigned int packed_0_4[4];
#pragma unroll
                    for (int lp = 0; lp < 4; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(inv_row[lp * 2 + 0], inv_row[lp * 2 + 1 + 0]));
                        packed_0_4[lp] = *(uint32_t*)&bf2;
                    }
                    int byte_off_1_1   = inverse_row * 128 + diag_block * 8 * 2;
                    int swizzled_off_2 = byte_off_1_1 ^ (byte_off_1_1 >> 7 & 7) << 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                                     smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_2),
                                 "r"(*reinterpret_cast<uint32_t*>(&packed_0_4[0])),
                                 "r"(*reinterpret_cast<uint32_t*>(&packed_0_4[(0) + 1])),
                                 "r"(*reinterpret_cast<uint32_t*>(&packed_0_4[(0) + 2])),
                                 "r"(*reinterpret_cast<uint32_t*>(&packed_0_4[(0) + 3])));
                }
                if (prep_local_warp < 2) {
                    if (cute::elect_one_sync()) {
                        mbarrier_arrive(prep_diag_ready_addr + (prep_stage) * 8);
                    }
                    mbarrier_wait(prep_diag_ready_addr + (prep_stage) * 8, phase_prep_diag_ready);
                }
                if (prep_local_warp < 2) {
                    int lane_row = lane_idx & 7;
                    int byte_off_2 =
                        (prep_local_warp * 16 + 8 + lane_row) * 128 + (prep_local_warp * 16 + 8) * 2;
                    int swizzled_off_3 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                    int d_addr     = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_3;
                    int byte_off_0 = (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_1_1 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                    int c_addr = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_1_1;
                    int byte_off_2_1     = (prep_local_warp * 16 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_3_1 = byte_off_2_1 ^ (byte_off_2_1 >> 7 & 7) << 4;
                    int a_addr = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_3_1;
                    unsigned int d_frag[2];
                    unsigned int c_frag[1];
                    float dc_acc[4];
                    unsigned int dc_bf16[2];
                    unsigned int inv_a_frag[1];
                    float o_acc[4];
                    unsigned int o_bf16[2];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                 : "=r"(d_frag[0])
                                 : "r"(d_addr)
                                 : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                 : "=r"(d_frag[1])
                                 : "r"(d_addr)
                                 : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                 : "=r"(c_frag[0])
                                 : "r"(c_addr)
                                 : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5}, {%6}, {%7, %8, %9, %10};\n"
                                 : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
                                 : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f), "f"(0.0f));
                    const float2 scale2_11 = {-1.0f, -1.0f};
#pragma unroll
                    for (int ls = 0; ls < 2; ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc)[ls], scale2_11);
#pragma unroll
                    for (int lp = 0; lp < 2; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(dc_acc[lp * 2 + 0], dc_acc[lp * 2 + 1 + 0]));
                        dc_bf16[lp] = *(uint32_t*)&bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                 : "=r"(inv_a_frag[0])
                                 : "r"(a_addr)
                                 : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5}, {%6}, {%7, %8, %9, %10};\n"
                                 : "=f"(o_acc[0]), "=f"(o_acc[1]), "=f"(o_acc[2]), "=f"(o_acc[3])
                                 : "r"(dc_bf16[0]), "r"(dc_bf16[1]), "r"(inv_a_frag[0]), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f), "f"(0.0f));
#pragma unroll
                    for (int lp = 0; lp < 2; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(o_acc[lp * 2 + 0], o_acc[lp * 2 + 1 + 0]));
                        o_bf16[lp] = *(uint32_t*)&bf2;
                    }
                    int byte_off_4 = (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_5 = byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4;
                    int o_addr = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_5;
                    uint32_t stmatrix_addr_12 = static_cast<uint32_t>((unsigned long long)o_addr);
                    asm volatile(
                        "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n" ::"r"(stmatrix_addr_12),
                        "r"(*reinterpret_cast<const uint32_t*>(&o_bf16[0]))
                        : "memory");
                    if (cute::elect_one_sync()) {
                        mbarrier_arrive(prep_inv16_ready_addr + (prep_stage) * 8);
                    }
                    mbarrier_wait(prep_inv16_ready_addr + (prep_stage) * 8, phase_prep_inv16_ready);
                }
                if (prep_local_warp == 0) {
                    int lane_row_1     = lane_idx % 16;
                    int lane_col       = lane_idx / 16 * 8;
                    int byte_off_3     = (16 + lane_row_1) * 128 + (16 + lane_col) * 2;
                    int swizzled_off_4 = byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4;
                    int d_addr_1     = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_4;
                    int byte_off_0_1 = (16 + lane_row_1) * 128 + lane_col * 2;
                    int swizzled_off_1_2 = byte_off_0_1 ^ (byte_off_0_1 >> 7 & 7) << 4;
                    int c_addr_1 = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_1_2;
                    int byte_off_2_2     = lane_row_1 * 128 + lane_col * 2;
                    int swizzled_off_3_2 = byte_off_2_2 ^ (byte_off_2_2 >> 7 & 7) << 4;
                    int a_addr_1 = smem_inv_work_addr + prep_stage * 40960 + (unsigned int)swizzled_off_3_2;
                    unsigned int d32_frag[4];
                    unsigned int c32_frag[4];
                    float dc32_acc[8];
                    unsigned int dc32_bf16[4];
                    unsigned int a32_frag[4];
                    float o32_acc[8];
                    unsigned int o32_bf16[4];
                    unsigned int zero32_bf16[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(d32_frag[0]), "=r"(d32_frag[1]), "=r"(d32_frag[2]), "=r"(d32_frag[3])
                                 : "r"(d_addr_1)
                                 : "memory");
                    int d_publish_addr =
                        (s_inverse_addr + prep_stage * 40960 +
                         (unsigned int)((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 +
                                            (16 + lane_col) % 16 * 2 ^
                                        ((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 +
                                                 (16 + lane_col) % 16 * 2 >>
                                             7 &
                                         1) << 4));
                    uint32_t stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)d_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                                     stmatrix_addr_13),
                                 "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[0])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[1])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[2])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[3]))
                                 : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(c32_frag[0]), "=r"(c32_frag[1]), "=r"(c32_frag[2]), "=r"(c32_frag[3])
                                 : "r"(c_addr_1)
                                 : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                 : "=f"(dc32_acc[0]), "=f"(dc32_acc[1]), "=f"(dc32_acc[2]), "=f"(dc32_acc[3])
                                 : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]),
                                   "r"(c32_frag[0]), "r"(c32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                 : "=f"(dc32_acc[4]), "=f"(dc32_acc[(4) + 1]), "=f"(dc32_acc[(4) + 2]),
                                   "=f"(dc32_acc[(4) + 3])
                                 : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]),
                                   "r"(c32_frag[2]), "r"(c32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f));
                    const float2 scale2_14 = {-1.0f, -1.0f};
#pragma unroll
                    for (int ls = 0; ls < 4; ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(dc32_acc)[ls], scale2_14);
#pragma unroll
                    for (int lp = 0; lp < 4; lp++) {
                        BF162 bf2 = __float22bfloat162_rn(
                            make_float2(dc32_acc[lp * 2 + 0], dc32_acc[lp * 2 + 1 + 0]));
                        dc32_bf16[lp] = *(uint32_t*)&bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(a32_frag[0]), "=r"(a32_frag[1]), "=r"(a32_frag[2]), "=r"(a32_frag[3])
                                 : "r"(a_addr_1)
                                 : "memory");
                    int a_publish_addr =
                        (s_inverse_addr + prep_stage * 40960 +
                         (unsigned int)(lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 ^
                                        (lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 >> 7 & 1)
                                            << 4));
                    uint32_t stmatrix_addr_15 = static_cast<uint32_t>((unsigned long long)a_publish_addr);
                    asm volatile(
                        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                            stmatrix_addr_15),
                        "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[0])),
                        "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[1])),
                        "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[2])),
                        "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[3]))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                 : "=f"(o32_acc[0]), "=f"(o32_acc[1]), "=f"(o32_acc[2]), "=f"(o32_acc[3])
                                 : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]),
                                   "r"(a32_frag[0]), "r"(a32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                                 "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                 : "=f"(o32_acc[4]), "=f"(o32_acc[(4) + 1]), "=f"(o32_acc[(4) + 2]),
                                   "=f"(o32_acc[(4) + 3])
                                 : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]),
                                   "r"(a32_frag[2]), "r"(a32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                                   "f"(0.0f));
#pragma unroll
                    for (int lp = 0; lp < 4; lp++) {
                        BF162 bf2 =
                            __float22bfloat162_rn(make_float2(o32_acc[lp * 2 + 0], o32_acc[lp * 2 + 1 + 0]));
                        o32_bf16[lp] = *(uint32_t*)&bf2;
                    }
                    int o_publish_addr =
                        (s_inverse_addr + prep_stage * 40960 +
                         (unsigned int)(lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 ^
                                        (lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 >>
                                             7 &
                                         1) << 4));
                    uint32_t stmatrix_addr_16 = static_cast<uint32_t>((unsigned long long)o_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                                     stmatrix_addr_16),
                                 "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[0])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[1])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[2])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[3]))
                                 : "memory");
#pragma unroll
                    for (int zero_word = 0; zero_word < 4; zero_word++) {
                        zero32_bf16[zero_word] = 0;
                    }
                    int zero_publish_addr = (s_inverse_addr + prep_stage * 40960 +
                                             (unsigned int)((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 +
                                                                (16 + lane_col) % 16 * 2 ^
                                                            ((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 +
                                                                     (16 + lane_col) % 16 * 2 >>
                                                                 7 &
                                                             1) << 4));
                    uint32_t stmatrix_addr_17 = static_cast<uint32_t>((unsigned long long)zero_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                                     stmatrix_addr_17),
                                 "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[0])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[1])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[2])),
                                 "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[3]))
                                 : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" ::"r"(11 + prep_instance) : "memory");
                if (prep_local_warp == 0) {
                    if (cute::elect_one_sync()) {
                        mbarrier_arrive(qk_full_addr + (prep_stage) * 8);
                    }
                }
                phase_raw_inputs_free ^= 1;
                phase_smem_free ^= 1;
                phase_gate_raw_full ^= 1;
                phase_qk_raw_full ^= 1;
                phase_prep_diag_ready ^= 1;
                phase_prep_inv16_ready ^= 1;
            }
        }
    }

    // Cleanup

#endif
}

} // namespace flash_kda::fused
