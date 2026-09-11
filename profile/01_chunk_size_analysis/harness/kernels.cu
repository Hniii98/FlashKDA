// Independent diagnostic CUDA kernels. Not a replacement for production FlashKDA.
// BF16/F16 WMMA compiles to the legacy HMMA path; verify this in retained NCU SASS.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>
using BF = __nv_bfloat16;
namespace w = nvcuda::wmma;
__device__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x)); return y; }
template<class T> __device__ float cv(T x) { return float(x); }
template<class T> __device__ T cast(float x) { return T(x); }

// All pointers shared, row-major. WARPS output tiles distributed round-robin.
// Resident keeps all assigned output fragments live; streaming stores each tile.
template<int M,int N,int K,class Input,class Acc,int WARPS,bool Resident>
__device__ void product(const Input* a,const Input* b,Acc* out) {
    constexpr int Tiles=(M/16)*(N/16), Per=(Tiles+WARPS-1)/WARPS;
    int warp=threadIdx.x/32;
    using Frag=w::fragment<w::accumulator,16,16,16,Acc>;
    w::fragment<w::matrix_a,16,16,16,Input,w::row_major> af;
    w::fragment<w::matrix_b,16,16,16,Input,w::row_major> bf;
    if constexpr(Resident) {
        Frag acc[Per];
        #pragma unroll
        for(int t=0;t<Per;++t) w::fill_fragment(acc[t],0.0f);
        #pragma unroll
        for(int kk=0;kk<K;kk+=16) {
            #pragma unroll
            for(int t=0;t<Per;++t) {
                int tile=t*WARPS+warp;
                if(tile<Tiles) {
                    int m=(tile/(N/16))*16,n=(tile%(N/16))*16;
                    w::load_matrix_sync(af,a+m*K+kk,K);
                    w::load_matrix_sync(bf,b+kk*N+n,N);
                    w::mma_sync(acc[t],af,bf,acc[t]);
                }
            }
        }
        #pragma unroll
        for(int t=0;t<Per;++t) {
            int tile=t*WARPS+warp;
            if(tile<Tiles) w::store_matrix_sync(out+(tile/(N/16))*16*N+(tile%(N/16))*16,acc[t],N,w::mem_row_major);
        }
    } else {
        for(int tile=warp;tile<Tiles;tile+=WARPS) {
            Frag acc; w::fill_fragment(acc,0.0f);
            int m=(tile/(N/16))*16,n=(tile%(N/16))*16;
            #pragma unroll
            for(int kk=0;kk<K;kk+=16) {
                w::load_matrix_sync(af,a+m*K+kk,K);
                w::load_matrix_sync(bf,b+kk*N+n,N);
                w::mma_sync(acc,af,bf,acc);
            }
            w::store_matrix_sync(out+m*N+n,acc,N,w::mem_row_major);
        }
    }
    if constexpr(WARPS==1) __syncwarp(); else __syncthreads();
}

template<int C>
__global__ void range_probe(const float* decay,const BF* kval,float* out) {
    int batch=blockIdx.x,col=threadIdx.x;
    float gs[C],g=0.f;
    #pragma unroll
    for(int i=0;i<C;++i) {g+=-decay[(batch*C+i)*128+col]*1.4426950408889634f;gs[i]=g;}
    BF total=BF(ex2(g));
    for(int i=0;i<C;++i) {
        BF en=BF(ex2(gs[i])),ep=BF(ex2(-gs[i])),k=kval[(batch*C+i)*128+col];
        BF ki=k*ep,kr=ki*total,stable=k*BF(ex2(g-gs[i]));
        int idx=((batch*C+i)*128+col)*5;
        out[idx]=float(en);out[idx+1]=float(ep);out[idx+2]=float(ki);out[idx+3]=float(kr);out[idx+4]=float(stable);
    }
}

template<int C,class Acc,bool Resident>
__global__ void neumann_kernel(const half* input,half* output,BF* outbf,float* trace) {
    __shared__ __align__(32) half p[C*C],inv[C*C],nextp[C*C];
    __shared__ __align__(32) Acc tmp[C*C];
    int offset=blockIdx.x*C*C;
    for(int x=threadIdx.x;x<C*C;x+=32) {p[x]=input[offset+x];inv[x]=__hsub(half(x/C==x%C?1.f:0.f),p[x]);}
    __syncwarp();
    int stage=0;
    #pragma unroll
    for(int power=2;power<C;power*=2) {
        product<C,C,C,half,Acc,1,Resident>(p,p,tmp);
        for(int x=threadIdx.x;x<C*C;x+=32) nextp[x]=half(float(tmp[x]));
        __syncwarp();
        if(trace) for(int x=threadIdx.x;x<C*C;x+=32) trace[((blockIdx.x*5+stage)*2)*C*C+x]=float(nextp[x]);
        product<C,C,C,half,Acc,1,Resident>(inv,nextp,tmp);
        // Round GEMM output to fp16, then fp16 addition, as in original algorithm.
        for(int x=threadIdx.x;x<C*C;x+=32) {inv[x]=__hadd(inv[x],half(float(tmp[x])));p[x]=nextp[x];}
        __syncwarp();
        if(trace) for(int x=threadIdx.x;x<C*C;x+=32) trace[((blockIdx.x*5+stage)*2+1)*C*C+x]=float(inv[x]);
        ++stage;
    }
    for(int x=threadIdx.x;x<C*C;x+=32) {output[offset+x]=inv[x];if(outbf)outbf[offset+x]=BF(float(inv[x]));}
}

// Stable reference/fallback on GPU: independent fp32 forward solve per RHS column.
template<int C>
__global__ void triangular_kernel(const half* input,half* output,BF* outbf) {
    __shared__ half L[C*C];
    for(int x=threadIdx.x;x<C*C;x+=32)L[x]=input[blockIdx.x*C*C+x];
    __syncwarp();
    for(int col=threadIdx.x;col<C;col+=32) {
        float values[C];
        #pragma unroll
        for(int i=0;i<C;++i) {
            float z=i==col?1.f:0.f;
            for(int j=0;j<i;++j) z=fmaf(-float(L[i*C+j]),values[j],z);
            values[i]=z;
            int x=blockIdx.x*C*C+i*C+col;output[x]=half(z);if(outbf)outbf[x]=BF(z);
        }
    }
}

template<int M,int N,int K,bool Resident>
__global__ void mma_probe(const BF* inputa,const BF* inputb,float* output) {
    extern __shared__ __align__(32) unsigned char storage[];
    BF* a=(BF*)storage;BF* b=a+M*K;float* tmp=(float*)(b+K*N);
    for(int i=threadIdx.x;i<M*K;i+=32)a[i]=inputa[blockIdx.x*M*K+i];
    for(int i=threadIdx.x;i<K*N;i+=32)b[i]=inputb[blockIdx.x*K*N+i];
    __syncwarp();
    product<M,N,K,BF,float,1,Resident>(a,b,tmp);
    for(int i=threadIdx.x;i<M*N;i+=32)output[blockIdx.x*M*N+i]=tmp[i];
}

// Input [head,T,D], activated gate a>=0 and beta. No activation/L2 normalization here.
// Workspace [head,chunk,...]. Stable mode uses direct exponent differences for L/M.
template<int C,bool Stable>
__global__ void prepare_kernel(const BF* q,const BF* k,const float* a,const float* beta,
    BF* kd,BF* qd,BF* kr,float* gt,half* L,BF* M,int T) {
    constexpr int D=128;
    extern __shared__ __align__(32) unsigned char storage[];
    BF* skd=(BF*)storage;BF* sqd=skd+C*D;BF* ski=sqd+C*D;
    float* G=(float*)(ski+D*C);float* tmp=G+C*D;
    int tile=blockIdx.x,base=tile*C*D;
    int col=threadIdx.x;
    float g=0;
    for(int i=0;i<C;++i){g+=-a[base+i*D+col]*1.4426950408889634f;G[i*D+col]=g;}
    gt[tile*D+col]=ex2(g);
    for(int i=0;i<C;++i) {
        int x=i*D+col;
        BF en=BF(ex2(G[x])),ep=BF(ex2(-G[x]));
        skd[x]=k[base+x]*en;sqd[x]=(q[base+x]*en)*BF(0.08838834764831845f);
        ski[col*C+i]=k[base+x]*ep;
        kd[base+x]=skd[x];qd[base+x]=sqd[x];
        kr[base+x]=Stable?k[base+x]*BF(ex2(g-G[x])):ski[col*C+i]*BF(ex2(g));
    }
    __syncthreads();
    if constexpr(!Stable) {
        product<C,C,D,BF,float,4,false>(skd,ski,tmp);
        for(int x=threadIdx.x;x<C*C;x+=128) L[tile*C*C+x]=x/C>x%C?__hmul(half(tmp[x]),half(beta[tile*C+x/C])):half(0.f);
        __syncthreads();
        product<C,C,D,BF,float,4,false>(sqd,ski,tmp);
        for(int x=threadIdx.x;x<C*C;x+=128)M[tile*C*C+x]=x/C>=x%C?BF(tmp[x]):BF(0.f);
    } else {
        for(int x=threadIdx.x;x<C*C;x+=128) {
            int i=x/C,j=x%C;float l=0,m=0;
            if(i>=j)for(int d=0;d<D;++d){float w=ex2(G[i*D+d]-G[j*D+d]);float kj=float(k[base+j*D+d]);l=fmaf(float(k[base+i*D+d])*w,kj,l);m=fmaf(float(q[base+i*D+d])*w,kj,m);}
            L[tile*C*C+x]=i>j?__hmul(half(l),half(beta[tile*C+i])):half(0.f);
            M[tile*C*C+x]=BF(m*0.08838834764831845f);
        }
    }
}

template<int C>
__global__ void recurrence_kernel(const BF* kd,const BF* qd,const BF* kr,const float* gt,
    const BF* inv,const BF* mqk,const BF* v,const float* beta,const BF* initial,BF* final,BF* out,int T) {
    constexpr int D=128;
    extern __shared__ __align__(32) unsigned char smem[];
    BF* state=(BF*)smem;
    BF* a=state+D*D;BF* b=a+C*D;BF* u=b+C*D;BF* aux=u+C*D;BF* square=aux+C*D;
    float* tmp=(float*)(square+C*C);
    // 2 scalar-matrix output fragments saved as bf16 in aux; scratch reused by state update.
    int head=blockIdx.x;
    for(int x=threadIdx.x;x<D*D;x+=128)state[x]=initial[head*D*D+x];
    __syncthreads();
    for(int t=0;t<T/C;++t) {
        int tile=head*(T/C)+t,base=tile*C*D;
        for(int x=threadIdx.x;x<C*D;x+=128)a[x]=kd[base+x];
        __syncthreads();
        product<C,D,D,BF,float,4,false>(a,state,tmp);
        for(int x=threadIdx.x;x<C*D;x+=128)b[x]=(v[base+x]-BF(tmp[x]))*BF(beta[tile*C+x/D]);
        for(int x=threadIdx.x;x<C*C;x+=128)square[x]=inv[tile*C*C+x];
        __syncthreads();
        product<C,D,C,BF,float,4,false>(square,b,tmp);
        for(int x=threadIdx.x;x<C*D;x+=128){u[x]=BF(tmp[x]);a[x]=qd[base+x];}
        __syncthreads();
        product<C,D,D,BF,float,4,false>(a,state,tmp);
        for(int x=threadIdx.x;x<C*D;x+=128)aux[x]=BF(tmp[x]);
        for(int x=threadIdx.x;x<C*C;x+=128)square[x]=mqk[tile*C*C+x];
        __syncthreads();
        product<C,D,C,BF,float,4,false>(square,u,tmp);
        for(int x=threadIdx.x;x<C*D;x+=128)out[base+x]=aux[x]+BF(tmp[x]);
        // transpose k_restored for state update
        for(int x=threadIdx.x;x<C*D;x+=128)a[(x%D)*C+x/D]=kr[base+x];
        __syncthreads();
        product<D,D,C,BF,float,4,false>(a,u,tmp);
        for(int x=threadIdx.x;x<D*D;x+=128)state[x]=BF(fmaf(float(state[x]),gt[tile*D+x/D],tmp[x]));
        __syncthreads();
    }
    for(int x=threadIdx.x;x<D*D;x+=128)final[head*D*D+x]=state[x];
}

extern "C" int probe_range(int C,int batch,const float* a,const BF* k,float* out,cudaStream_t s) {
    #define GO(CV) range_probe<CV><<<batch,128,0,s>>>(a,k,out)
    if(C==16){GO(16);}else if(C==32){GO(32);}else if(C==64){GO(64);}else return -1;
    #undef GO
    return cudaGetLastError();
}
extern "C" int inverse(int C,int batch,int method,int resident,const half* in,half* out,BF* outbf,float* trace,cudaStream_t s) {
    #define GO(CV) if(method==2)triangular_kernel<CV><<<batch,32,0,s>>>(in,out,outbf);else if(method==0){if(resident)neumann_kernel<CV,half,true><<<batch,32,0,s>>>(in,out,outbf,trace);else neumann_kernel<CV,half,false><<<batch,32,0,s>>>(in,out,outbf,trace);}else{if(resident)neumann_kernel<CV,float,true><<<batch,32,0,s>>>(in,out,outbf,trace);else neumann_kernel<CV,float,false><<<batch,32,0,s>>>(in,out,outbf,trace);}
    if(C==16){GO(16);}else if(C==32){GO(32);}else if(C==64){GO(64);}else return -1;
    #undef GO
    return cudaGetLastError();
}
template<int M,int N,int K,bool R> int launch_mma_shape(int batch,const BF* a,const BF* b,float* o,cudaStream_t s){
    int bytes=2*(M*K+K*N)+4*M*N;
    auto err=cudaFuncSetAttribute(mma_probe<M,N,K,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess)return err;
    mma_probe<M,N,K,R><<<batch,32,bytes,s>>>(a,b,o);return cudaGetLastError();
}
template<int C> int launch_mma(int op,int batch,int resident,const BF* a,const BF* b,float* o,cudaStream_t s) {
    #define GO(M,N,K) return resident ? launch_mma_shape<M,N,K,true>(batch,a,b,o,s) : launch_mma_shape<M,N,K,false>(batch,a,b,o,s)
    if(op==0){GO(C,C,128);}else if(op==1){GO(C,128,128);}else if(op==2){GO(C,128,C);}else if(op==3){GO(128,128,C);}else return -1;
    #undef GO
    return cudaGetLastError();
}
extern "C" int probe_mma(int C,int op,int batch,int resident,const BF* a,const BF* b,float* out,cudaStream_t s) {
    if(C==16)return launch_mma<16>(op,batch,resident,a,b,out,s);
    if(C==32)return launch_mma<32>(op,batch,resident,a,b,out,s);
    if(C==64)return launch_mma<64>(op,batch,resident,a,b,out,s);return -1;
}
template<int C,bool Stable> int launch_prepare(int heads,int T,const BF* q,const BF* k,const float* a,const float* beta,BF* kd,BF* qd,BF* kr,float* gt,half* L,BF* M,cudaStream_t s){
    int bytes=10*C*128+4*C*C;
    auto err=cudaFuncSetAttribute(prepare_kernel<C,Stable>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess)return err;
    prepare_kernel<C,Stable><<<heads*T/C,128,bytes,s>>>(q,k,a,beta,kd,qd,kr,gt,L,M,T);return cudaGetLastError();
}
extern "C" int prepare(int C,int heads,int T,int stable,const BF* q,const BF* k,const float* a,const float* beta,BF* kd,BF* qd,BF* kr,float* gt,half* L,BF* M,cudaStream_t s) {
    #define GO(CV) if(stable)return launch_prepare<CV,true>(heads,T,q,k,a,beta,kd,qd,kr,gt,L,M,s);else return launch_prepare<CV,false>(heads,T,q,k,a,beta,kd,qd,kr,gt,L,M,s)
    if(C==16){GO(16);}else if(C==32){GO(32);}else if(C==64){GO(64);}else return -1;
    #undef GO
    return cudaGetLastError();
}
template<int C> int launch_recur(int heads,int T,const BF* kd,const BF* qd,const BF* kr,const float* gt,const BF* inv,const BF* mqk,const BF* v,const float* beta,const BF* initial,BF* final,BF* out,cudaStream_t s){
    int bytes=2*(128*128+4*C*128+C*C)+4*128*128;
    cudaError_t e=cudaFuncSetAttribute(recurrence_kernel<C>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(e!=cudaSuccess)return e;
    recurrence_kernel<C><<<heads,128,bytes,s>>>(kd,qd,kr,gt,inv,mqk,v,beta,initial,final,out,T);
    return cudaGetLastError();
}
extern "C" int recur(int C,int heads,int T,const BF* kd,const BF* qd,const BF* kr,const float* gt,const BF* inv,const BF* mqk,const BF* v,const float* beta,const BF* initial,BF* final,BF* out,cudaStream_t s){
    #define GO(CV) return launch_recur<CV>(heads,T,kd,qd,kr,gt,inv,mqk,v,beta,initial,final,out,s)
    if(C==16){GO(16);}else if(C==32){GO(32);}else if(C==64){GO(64);}return -1;
    #undef GO
}
