// Host-only ABI bridge. All GPU kernels and their launch code are the original csrc.
#include "fwd.h"
extern "C" int forward(void* q,void* k,void* v,void* g,void* beta,void* si,void* so,void* out,void* ws,void* al,void* dt,int N,int T,int H,cudaStream_t stream){
 using B=cutlass::bfloat16_t;
 launch_fwd<128,true,true,false,false>((B*)q,(B*)k,(B*)v,(B*)g,(B*)beta,si,1.f/11.313708498984761f,so,(B*)out,ws,N*((T+15)/16),N*T,H,N,nullptr,(float*)al,(float*)dt,-5.f*1.4426950408889634f,stream);
 return int(cudaGetLastError());
}
