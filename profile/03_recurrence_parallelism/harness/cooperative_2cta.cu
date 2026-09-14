// Isolated CHUNK16 KDA GEMMs: single CTA, independent N split, cooperative cluster N split.
// Cooperative path loads A once per task and copies the peer fragment through DSM.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg = cooperative_groups;
__device__ __forceinline__ uint32_t smaddr(const void* p){return uint32_t(__cvta_generic_to_shared(p));}
// K-major 8x8 interleave, matching the canonical layout used by production helpers.
__host__ __device__ constexpr int offset(int r,int k,int K){return (r/8)*8*K+(k/8)*64+(r%8)*8+k%8;}
__device__ __forceinline__ uint64_t descriptor(const void* p,int K){
 return uint64_t(smaddr(p)>>4) | (uint64_t(128>>4)<<16) | (uint64_t((8*K*2)>>4)<<32) | (uint64_t(1)<<46);
}
__device__ __forceinline__ void load_a(uint32_t (&a)[4],const uint16_t* s,int m,int k,int K,int lane){
 uint32_t p=smaddr(s+offset(m+lane%16,k+(lane/16)*8,K));
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(p));
}
__device__ __forceinline__ void load_b(uint32_t (&b)[2],const uint16_t* s,int n,int k,int K,int lane){
 uint32_t p=smaddr(s+offset(n+lane%8,k+((lane/8)%2)*8,K));
 asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(b[0]),"=r"(b[1]):"r"(p));
}
__device__ __forceinline__ float hfloat(uint16_t x){return __half2float(__ushort_as_half(x));}



template<int M,int N,int K,bool Half,int Mode>
__global__ void kda_gemm(const uint16_t* A,const uint16_t* B,float* output) {
 constexpr int Parts=Mode==0?1:2;
 constexpr int LocalN=N/Parts;
 constexpr int Tiles=(M/16)*(LocalN/8);
 constexpr int MaxWarps=Mode==0?4:2;
 constexpr int Warps=Tiles<MaxWarps?Tiles:MaxWarps;
 __shared__ __align__(128) uint16_t sa[M*K];
 __shared__ __align__(128) uint16_t sb[LocalN*K];
 __shared__ float sc[M*(LocalN+1)];
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int rank=Mode==0?0:blockIdx.x%2;
 int job=blockIdx.x/Parts;
 int nbase=rank*LocalN;
 // One GEMM task per CTA (Mode0), pair of independent CTAs (Mode1), or cluster (Mode2).
 if constexpr(Mode==2) {
  auto cluster=cg::this_cluster();
  if(rank==0)for(int x=tid;x<M*K;x+=blockDim.x)sa[x]=A[job*M*K+x];
  for(int x=tid;x<LocalN*K;x+=blockDim.x)sb[x]=B[job*N*K+offset(nbase,0,K)+x];
  cluster.sync(); // source CTA is alive; A writes are visible before remote reads
  if(rank==1) {
   auto peer=cluster.map_shared_rank(sa,0);
   for(int x=tid;x<M*K;x+=blockDim.x)sa[x]=peer[x];
  }
  cluster.sync(); // remote reads complete before source CTA can exit
 } else {
  for(int x=tid;x<M*K;x+=blockDim.x)sa[x]=A[job*M*K+x];
  for(int x=tid;x<LocalN*K;x+=blockDim.x)sb[x]=B[job*N*K+offset(nbase,0,K)+x];
  __syncthreads();
 }
 for(int tile=warp;tile<Tiles;tile+=Warps) {
  int m=(tile/(LocalN/8))*16,n=(tile%(LocalN/8))*8;
  float d[4]={0};uint32_t hd[2]={0};
  #pragma unroll
  for(int k=0;k<K;k+=16) {
   uint32_t aa[4],bb[2];load_a(aa,sa,m,k,K,lane);load_b(bb,sb,n,k,K,lane);
   if constexpr(Half) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
     :"+r"(hd[0]),"+r"(hd[1]):"r"(aa[0]),"r"(aa[1]),"r"(aa[2]),"r"(aa[3]),"r"(bb[0]),"r"(bb[1]));
   } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
     :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(aa[0]),"r"(aa[1]),"r"(aa[2]),"r"(aa[3]),"r"(bb[0]),"r"(bb[1]));
   }
  }
  if constexpr(Half){d[0]=hfloat(hd[0]);d[1]=hfloat(hd[0]>>16);d[2]=hfloat(hd[1]);d[3]=hfloat(hd[1]>>16);}
  int row=m+lane/4,col=n+(lane%4)*2;
  sc[row*(LocalN+1)+col]=d[0];sc[row*(LocalN+1)+col+1]=d[1];
  sc[(row+8)*(LocalN+1)+col]=d[2];sc[(row+8)*(LocalN+1)+col+1]=d[3];
 }
 __syncthreads();
 for(int x=tid;x<M*LocalN;x+=blockDim.x)
  output[job*M*N+(x/LocalN)*N+nbase+x%LocalN]=sc[(x/LocalN)*(LocalN+1)+x%LocalN];
}

template<int M,int N,int K,bool Half,int Mode>
int launch(int batch,const uint16_t* a,const uint16_t* b,float* out,cudaStream_t stream) {
 constexpr int Parts=Mode==0?1:2;
 constexpr int Tiles=(M/16)*(N/Parts/8);
 constexpr int MaxWarps=Mode==0?4:2;
 constexpr int Threads=(Tiles<MaxWarps?Tiles:MaxWarps)*32;
 cudaLaunchConfig_t config{};
 config.gridDim=dim3(batch*Parts);config.blockDim=dim3(Threads);config.stream=stream;
 cudaLaunchAttribute attr{};
 if constexpr(Mode==2) {
  attr.id=cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x=2;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;
  config.attrs=&attr;config.numAttrs=1;
 }
 return int(cudaLaunchKernelEx(&config,kda_gemm<M,N,K,Half,Mode>,a,b,out));
}
template<int M,int N,int K,bool Half>
int dispatch(int mode,int batch,const uint16_t* a,const uint16_t* b,float* out,cudaStream_t stream) {
 if(mode==0)return launch<M,N,K,Half,0>(batch,a,b,out,stream);
 if(mode==1)return launch<M,N,K,Half,1>(batch,a,b,out,stream);
 if(mode==2)return launch<M,N,K,Half,2>(batch,a,b,out,stream);
 return int(cudaErrorInvalidValue);
}
extern "C" int run_cooperative(int shape,int mode,int batch,const uint16_t* a,const uint16_t* b,float* out,cudaStream_t stream) {
 if(batch<=0)return int(cudaErrorInvalidValue);
 switch(shape) {
 case 0:return dispatch<16,16,16,true>(mode,batch,a,b,out,stream);
 case 1:return dispatch<16,16,128,false>(mode,batch,a,b,out,stream);
 case 2:return dispatch<16,128,16,false>(mode,batch,a,b,out,stream);
 case 3:return dispatch<16,128,128,false>(mode,batch,a,b,out,stream);
 case 4:return dispatch<128,128,16,false>(mode,batch,a,b,out,stream);
 default:return int(cudaErrorInvalidValue);
 }
}
