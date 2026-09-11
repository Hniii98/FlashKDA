// Reorganized GEMM candidates; mathematical transpose and shared-B stacking are prepared by bench.py.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

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

template<int M,int N,int K,bool Half>
__global__ void legacy_mma(const uint16_t* A,const uint16_t* B,float* output){
 __shared__ __align__(128) uint16_t sa[M*K];
 __shared__ __align__(128) uint16_t sb[N*K];
 extern __shared__ float sc[];
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int x=tid;x<M*K;x+=blockDim.x)sa[x]=A[blockIdx.x*M*K+x];
 for(int x=tid;x<N*K;x+=blockDim.x)sb[x]=B[blockIdx.x*N*K+x];
 __syncthreads();
 // Same 16x16 output tiling and 32-column/warp partition as K2 for M16,N128.
 constexpr int Tiles=(M/16)*(N/16);
 constexpr int Warps=Tiles<4?Tiles:4;
 for(int q=0;q<((M/16)*(N/16))/Warps;++q){
  int tile=q*Warps+warp;
  int m=(tile/(N/16))*16,n=(tile%(N/16))*16;
  float d[2][4]={{0}};uint32_t hd[2][2]={{0}};
  #pragma unroll
  for(int k=0;k<K;k+=16){
   uint32_t a[4];load_a(a,sa,m,k,K,lane);
   #pragma unroll
   for(int j=0;j<2;++j){
    uint32_t b[2];load_b(b,sb,n+j*8,k,K,lane);
    if constexpr(Half){
     asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
      :"+r"(hd[j][0]),"+r"(hd[j][1]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }else{
     asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      :"+f"(d[j][0]),"+f"(d[j][1]),"+f"(d[j][2]),"+f"(d[j][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
   }
  }
  #pragma unroll
  for(int j=0;j<2;++j){
   if constexpr(Half){d[j][0]=hfloat(hd[j][0]);d[j][1]=hfloat(hd[j][0]>>16);d[j][2]=hfloat(hd[j][1]);d[j][3]=hfloat(hd[j][1]>>16);}
   int row=m+lane/4,col=n+j*8+(lane%4)*2;
   sc[row*(N+1)+col]=d[j][0];sc[row*(N+1)+col+1]=d[j][1];
   sc[(row+8)*(N+1)+col]=d[j][2];sc[(row+8)*(N+1)+col+1]=d[j][3];
  }
 }
 __syncthreads();
 for(int x=tid;x<M*N;x+=blockDim.x)output[blockIdx.x*M*N+x]=sc[(x/N)*(N+1)+x%N];
}

template<int M,int N,int K,bool Half,bool WS,bool Trans=false>
__global__ void tcgen_mma(const uint16_t* A,const uint16_t* B,float* output){
 constexpr int PM=M<64?(WS?32:64):M;
 constexpr int PN=WS?(N<64?64:N):N;
 constexpr int UsedCols=(WS && PM==32)?PN/4:PN;
 constexpr int Cols=UsedCols<32?32:UsedCols;
 __shared__ __align__(128) uint16_t sa[PM*K];
 __shared__ __align__(128) uint16_t sb[PN*K];
 extern __shared__ float sc[];
 __shared__ __align__(8) uint64_t barrier;
 __shared__ uint32_t tmem;
 int tid=threadIdx.x,warp=tid/32,lane=tid%32;
 // Padded packed arrays are prepared before timing. Their necessary reads count.
 for(int x=tid;x<PM*K;x+=128)sa[x]=A[blockIdx.x*PM*K+x];
 for(int x=tid;x<PN*K;x+=128)sb[x]=B[blockIdx.x*PN*K+x];
 if(tid==0)asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(smaddr(&barrier)):"memory");
 __syncthreads();
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 if(warp==0){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(smaddr(&tmem)),"r"(Cols):"memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;":::"memory");
 }
 __syncthreads();
 constexpr uint32_t IDesc=(Half?0u:((1u<<4)|(1u<<7)|(1u<<10)))|((PN/8)<<17)|((PM/16)<<24);
 if(tid==0){
  #pragma unroll
  for(int k=0;k<K;k+=16){
   uint64_t ad=descriptor(sa+offset(0,k,K),K),bd=descriptor(sb+offset(0,k,K),K);
   if constexpr(WS){
    asm volatile("{.reg .pred p; setp.ne.u32 p,%4,0; tcgen05.mma.ws.cta_group::1.kind::f16 [%0],%1,%2,%3,p;}"
     ::"r"(tmem),"l"(ad),"l"(bd),"r"(IDesc),"r"(k):"memory");
   }else{
    asm volatile("{.reg .pred p; setp.ne.u32 p,%4,0; tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;}"
     ::"r"(tmem),"l"(ad),"l"(bd),"r"(IDesc),"r"(k):"memory");
   }
  }
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(smaddr(&barrier)):"memory");
 }
 asm volatile("{.reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0; @!p bra WAIT;}"::"r"(smaddr(&barrier)):"memory");
 asm volatile("tcgen05.fence::after_thread_sync;":::"memory");
 constexpr int LoadN=(PM==32)?PN/4:PN;
 #pragma unroll
 for(int n=0;n<LoadN;n+=8){
  uint32_t value[8],addr=tmem+n;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
    :"=r"(value[0]),"=r"(value[1]),"=r"(value[2]),"=r"(value[3]),"=r"(value[4]),"=r"(value[5]),"=r"(value[6]),"=r"(value[7]):"r"(addr):"memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
  int row=(PM==32)?lane:(PM==64?warp*16+lane:tid);
  #pragma unroll
  for(int j=0;j<8;++j){
   int col=(PM==32)?warp*(PN/4)+n+j:n+j;
   bool valid=(PM!=64 || lane<16) && row<M && col<N;
   if(valid)sc[row*(N+1)+col]=Half?hfloat(value[j]):__uint_as_float(value[j]);
  }
 }
 __syncthreads();
 for(int x=tid;x<M*N;x+=128){
  if constexpr(Trans) output[blockIdx.x*M*N+x]=sc[(x%M)*(N+1)+x/M];
  else output[blockIdx.x*M*N+x]=sc[(x/N)*(N+1)+x%N];
 }
 __syncthreads();
 if(warp==0)asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(tmem),"r"(Cols):"memory");
 if(tid==0)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(smaddr(&barrier)):"memory");
}

template<int M,int N,int K,bool Half>
void launch(int mode,int batch,const uint16_t* a,const uint16_t* b,float* out,cudaStream_t stream){
 constexpr int bytes=M*(N+1)*4;
 if(mode==0){
  cudaFuncSetAttribute(legacy_mma<M,N,K,Half>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  legacy_mma<M,N,K,Half><<<batch,((M/16)*(N/16)<4)?32*((M/16)*(N/16)):128,bytes,stream>>>(a,b,out);
 }else if(mode==1){
  cudaFuncSetAttribute(tcgen_mma<M,N,K,Half,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  tcgen_mma<M,N,K,Half,false><<<batch,128,bytes,stream>>>(a,b,out);
 }else{
  cudaFuncSetAttribute(tcgen_mma<M,N,K,Half,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  tcgen_mma<M,N,K,Half,true><<<batch,128,bytes,stream>>>(a,b,out);
 }
}

extern "C" int run_mma(int shape,int mode,int batch,const uint16_t* a,const uint16_t* b,float* out,cudaStream_t stream){
 switch(shape){
 case 0:launch<16,16,16,true>(mode,batch,a,b,out,stream);break;
 case 1:launch<16,16,128,false>(mode,batch,a,b,out,stream);break;
 case 2:launch<16,128,16,false>(mode,batch,a,b,out,stream);break;
 case 3:launch<16,128,128,false>(mode,batch,a,b,out,stream);break;
 case 4:launch<128,128,16,false>(mode,batch,a,b,out,stream);break;
 case 5:launch<32,16,128,false>(mode,batch,a,b,out,stream);break;
 case 6:launch<32,128,128,false>(mode,batch,a,b,out,stream);break;
 case 7:tcgen_mma<128,16,16,false,false,true><<<batch,128,128*17*4,stream>>>(a,b,out);break;
 case 8:tcgen_mma<128,16,128,false,false,true><<<batch,128,128*17*4,stream>>>(a,b,out);break;
 case 9:launch<32,16,16,true>(mode,batch,a,b,out,stream);break;
 default:return -1;
 }return int(cudaGetLastError());
}
