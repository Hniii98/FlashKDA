#pragma once

#include "fwd_config.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/stride.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/tfloat32.h>

#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor_impl.hpp"

#ifndef BLOCK_LEVEL_K1
#define BLOCK_LEVEL_K1 1
#endif

#ifndef BLOCK_LEVEL_K2
#define BLOCK_LEVEL_K2 1
#endif

__device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float tanh_approx_f32(float x) {
    float result;
    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float sigmoid_tanh_approx_f32(float x) {
    float th = tanh_approx_f32(x * 0.5f);
    return th * 0.5f + 0.5f;
}

__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {
    float result;
    asm("cvt.f32.bf16 %0, %1;\n" : "=f"(result) : "h"(x.storage));
    return result;
}

using namespace cute;

// Workspace per-tile byte sizes (all naturally 128-byte aligned)
template <int CHUNK, int D>
struct WorkspaceSizes {
    static_assert(CHUNK * D * 2 % 128 == 0);
    static_assert(D * 4 % 128 == 0);
    static_assert(CHUNK * CHUNK * 2 % 128 == 0);

    static constexpr int kKDecayed  = CHUNK * D * 2;        // 4096
    static constexpr int kQDecayed  = CHUNK * D * 2;        // 4096
    static constexpr int kKRestored = CHUNK * D * 2;        // 4096
    static constexpr int kGTotal    = D * 4;                 // 512
    static constexpr int kINV       = CHUNK * CHUNK * 2;     // 512
    static constexpr int kMqk       = CHUNK * CHUNK * 2;     // 512
    static constexpr int64_t kPerTile = kKDecayed + kQDecayed + kKRestored + kGTotal + kINV + kMqk;
};

enum class WarpRole {
    MMA,
    LOAD_QKG,
    STORE,
    NonParticipant,
};

template <int Stages>
CUTLASS_DEVICE
cutlass::PipelineTmaAsync<Stages> make_load_pipeline(
    typename cutlass::PipelineTmaAsync<Stages>::SharedStorage& storage,
    uint32_t transaction_bytes,
    WarpRole warp_role,
    uint32_t num_producers,
    uint32_t num_consumers
) {
    using Pipeline = cutlass::PipelineTmaAsync<Stages>;
    typename Pipeline::Params params;

    auto role = Pipeline::ThreadCategory::NonParticipant;
    bool is_leader = false;
    if (warp_role == WarpRole::LOAD_QKG) {
        role = Pipeline::ThreadCategory::Producer;
        is_leader = cute::elect_one_sync();
    } else if (warp_role == WarpRole::MMA) {
        role = Pipeline::ThreadCategory::Consumer;
    }

    params.transaction_bytes = transaction_bytes;
    params.role = role;
    params.is_leader = is_leader;
    params.num_consumers = num_consumers;
    params.num_producers = num_producers;

    Pipeline pipeline(storage, params, Shape<_1,_1>{});
    cutlass::pipeline_init_wait(1);
    return pipeline;
}

template <int Stages>
CUTLASS_DEVICE
cutlass::PipelineAsync<Stages> make_store_pipeline(
    typename cutlass::PipelineAsync<Stages>::SharedStorage& storage,
    WarpRole warp_role,
    uint32_t num_producers,
    uint32_t num_consumers
) {
    using Pipeline = cutlass::PipelineAsync<Stages>;
    typename Pipeline::Params params;

    auto role = Pipeline::ThreadCategory::NonParticipant;
    if (warp_role == WarpRole::MMA) {
        role = Pipeline::ThreadCategory::Producer;
    } else if (warp_role == WarpRole::STORE) {
        role = Pipeline::ThreadCategory::Consumer;
    }

    params.role = role;
    params.producer_arv_count = num_producers;
    params.consumer_arv_count = num_consumers;

    Pipeline pipeline(storage, params);
    cutlass::pipeline_init_wait(1);
    return pipeline;
}

template <class TensorA, class TensorB, class TensorC>
CUTLASS_DEVICE void mma_m16n16_bf16bf16bf16_1warp(
    TensorA const& A,
    TensorB const& B,
    TensorC& C,
    int mma_tid
) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );

    if (mma_tid >= int(size(mma))) return;

    using BF16 = cutlass::bfloat16_t;

    auto sC_store_op = [] __device__ (float x) { return BF16(x); };

    cooperative_gemm(mma_tid, mma, 1.0f, A, B, 0.0f, C, cute::identity{}, cute::identity{}, cute::identity{}, sC_store_op, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM90_U32x4_STSM_N{});
}

template <class TensorA, class TensorB, class TensorC>
CUTLASS_DEVICE void mma_m16n16_bf16bf16fp16_1warp(
    TensorA const& A,
    TensorB const& B,
    TensorC& C,
    int mma_tid
) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );

    if (mma_tid >= int(size(mma))) return;

    using FP16 = cutlass::half_t;

    auto sC_store_op = [] __device__ (float x) { return FP16(x); };

    cooperative_gemm(mma_tid, mma, 1.0f, A, B, 0.0f, C, cute::identity{}, cute::identity{}, cute::identity{}, sC_store_op, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM90_U32x4_STSM_N{});
}

// Same finite Neumann expansion as C16, tiled over the full CHUNK matrix.
// C/A register formats coincide for this FP16 atom; MOVM supplies B format.
template <int CHUNK, class TensorL, class TensorINV_fp16, class TensorINV_bf16>
CUTLASS_DEVICE void neumann_inv_fused_1warp(
    TensorL const& L_fp16, TensorINV_fp16 const& INV_fp16,
    TensorINV_bf16& INV_bf16_out, int tid, float inverse_rescale
) {
    using FP16 = cutlass::half_t;
    using BF16 = cutlass::bfloat16_t;
    constexpr int B = CHUNK / 16;
    static_assert(CHUNK >= 16 && (CHUNK & (CHUNK - 1)) == 0);
    if (tid >= 32) return;
    auto mma = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                             Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
    auto thr = mma.get_slice(tid);
    auto cp = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, FP16>{}, mma);
    auto tcp = cp.get_slice(tid);
    uint32_t power[B][B][4], inv[B][B][4], next[B][B][4], product[B][B][4];
    #pragma unroll
    for (int m = 0; m < B; ++m) {
        #pragma unroll
        for (int n = 0; n < B; ++n) {
            auto l = local_tile(L_fp16, make_shape(_16{}, _16{}), make_coord(m,n));
            auto v = local_tile(INV_fp16, make_shape(_16{}, _16{}), make_coord(m,n));
            auto f = thr.partition_fragment_A(l);
            auto tmp = make_fragment_like<FP16>(f);
            copy(cp, tcp.partition_S(l), tcp.retile_D(tmp));
            cute::transform(tmp, f, cute::identity{});
            #pragma unroll
            for (int r=0;r<4;++r) power[m][n][r]=reinterpret_cast<uint32_t*>(&f(0))[r];
            copy(cp, tcp.partition_S(v), tcp.retile_D(tmp));
            cute::transform(tmp, f, cute::identity{});
            #pragma unroll
            for (int r=0;r<4;++r) inv[m][n][r]=reinterpret_cast<uint32_t*>(&f(0))[r];
        }
    }
    auto multiply = [&](auto const& a, auto const& b, auto& dst) {
        #pragma unroll
        for (int m=0;m<B;++m) {
            #pragma unroll
            for (int n=0;n<B;++n) {
                #pragma unroll
                for (int r=0;r<4;++r) dst[m][n][r]=0;
                #pragma unroll
                for (int k=0;k<B;++k) {
                    uint32_t bt[4];
                    #pragma unroll
                    for (int r=0;r<4;++r) SM75_U32x1_MOVM_T::copy(b[k][n][r],bt[r]);
                    auto* d=dst[m][n]; auto const* x=a[m][k];
                    SM80_16x8x16_F16F16F16F16_TN::fma(d[0],d[1],x[0],x[1],x[2],x[3],bt[0],bt[1],d[0],d[1]);
                    SM80_16x8x16_F16F16F16F16_TN::fma(d[2],d[3],x[0],x[1],x[2],x[3],bt[2],bt[3],d[2],d[3]);
                }
            }
        }
    };
    // I-L -> S3 -> S7 -> S15 -> S31 (for CHUNK=32).
    #pragma unroll
    for (int p=2;p<CHUNK;p*=2) {
        multiply(power,power,next);
        multiply(inv,next,product);
        #pragma unroll
        for (int m=0;m<B;++m) {
            #pragma unroll
            for (int n=0;n<B;++n) {
                #pragma unroll
                for (int r=0;r<4;++r) {
                    union Pack { uint32_t u; __half2 h; } a,b;
                    a.u=inv[m][n][r]; b.u=product[m][n][r];
                    a.h=__hadd2(a.h,b.h); inv[m][n][r]=a.u;
                    power[m][n][r]=next[m][n][r];
                }
            }
        }
    }
    auto st = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto tst = st.get_slice(tid);
    #pragma unroll
    for (int m=0;m<B;++m) {
        #pragma unroll
        for (int n=0;n<B;++n) {
            auto tile=local_tile(INV_bf16_out,make_shape(_16{},_16{}),make_coord(m,n));
            auto f=thr.make_fragment_C(thr.partition_C(tile));
            #pragma unroll
            for (int r=0;r<4;++r) reinterpret_cast<uint32_t*>(&f(0))[r]=inv[m][n][r];
            // Undo similarity before the original final FP16 -> BF16 cast.
            if (inverse_rescale != 1.0f) {
                auto coord=thr.partition_C(make_identity_tensor(make_shape(_16{},_16{})));
                #pragma unroll
                for (int e=0;e<size(f);++e) {
                    int row=m*16+get<0>(coord(e)), col=n*16+get<1>(coord(e));
                    f(e)=f(e)*FP16(powf(inverse_rescale,float(row-col)));
                }
            }
            auto bf=make_fragment_like<BF16>(f);
            cute::transform(f,bf,[] __device__ (FP16 x) { return BF16(x); });
            copy(st,tst.retile_S(bf),tst.partition_D(tile));
        }
    }
}

// ==================== FP32 <-> BF16 state conversion in SMEM ====================
// Both FP32 (K_SW32) and BF16 (K_INTER) layouts resolve to the same 8x8 atom
// structure with Swizzle<0,0,3>. Conversion operates per-atom:
//   - Each warp handles one 8x8 atom (64 elements)
//   - Each thread converts 2 elements
//   - Warp-level iteration over all atoms in the D x D state

template <class FP32Layout, class BF16Layout, int D, int NumThreads>
__device__ void smem_cvt_fp32_to_bf16(
    float* __restrict__ fp32_smem,
    cutlass::bfloat16_t* __restrict__ bf16_smem,
    int tid
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int kBlock = 8;
    constexpr int kBlocksPerDim = D / kBlock;
    constexpr int kTotalBlocks = kBlocksPerDim * kBlocksPerDim;
    constexpr int kWarpSize = 32;

    auto fp32_view = make_tensor(make_smem_ptr(fp32_smem), FP32Layout{});
    auto bf16_view = make_tensor(make_smem_ptr(bf16_smem), BF16Layout{});

    int warp_id = tid / kWarpSize;
    int lane_id = tid % kWarpSize;
    int num_warps = NumThreads / kWarpSize;

    for (int blk = warp_id; blk < kTotalBlocks; blk += num_warps) {
        int br = (blk / kBlocksPerDim) * kBlock;
        int bc = (blk % kBlocksPerDim) * kBlock;
        int e0 = lane_id * 2;
        int e1 = lane_id * 2 + 1;
        int r0 = br + e0 / kBlock, c0 = bc + e0 % kBlock;
        int r1 = br + e1 / kBlock, c1 = bc + e1 % kBlock;
        bf16_view(r0, c0) = BF16(fp32_view(r0, c0));
        bf16_view(r1, c1) = BF16(fp32_view(r1, c1));
    }
}

template <class BF16Layout, class FP32Layout, int D, int NumThreads>
__device__ void smem_cvt_bf16_to_fp32(
    cutlass::bfloat16_t* __restrict__ bf16_smem,
    float* __restrict__ fp32_smem,
    int tid
) {
    constexpr int kBlock = 8;
    constexpr int kBlocksPerDim = D / kBlock;
    constexpr int kTotalBlocks = kBlocksPerDim * kBlocksPerDim;
    constexpr int kWarpSize = 32;

    auto bf16_view = make_tensor(make_smem_ptr(bf16_smem), BF16Layout{});
    auto fp32_view = make_tensor(make_smem_ptr(fp32_smem), FP32Layout{});

    int warp_id = tid / kWarpSize;
    int lane_id = tid % kWarpSize;
    int num_warps = NumThreads / kWarpSize;

    for (int blk = warp_id; blk < kTotalBlocks; blk += num_warps) {
        int br = (blk / kBlocksPerDim) * kBlock;
        int bc = (blk % kBlocksPerDim) * kBlock;
        int e0 = lane_id * 2;
        int e1 = lane_id * 2 + 1;
        int r0 = br + e0 / kBlock, c0 = bc + e0 % kBlock;
        int r1 = br + e1 / kBlock, c1 = bc + e1 % kBlock;
        fp32_view(r0, c0) = bf16_to_f32(bf16_view(r0, c0));
        fp32_view(r1, c1) = bf16_to_f32(bf16_view(r1, c1));
    }
}
