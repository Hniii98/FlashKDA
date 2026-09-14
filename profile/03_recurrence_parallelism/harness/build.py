"""Generate isolated K2 variants from production source; production csrc is never edited."""
from pathlib import Path
import subprocess,json,hashlib,concurrent.futures,sys
ROOT=Path(__file__).resolve().parents[3]
GROUP='persistent' if '--persistent' in sys.argv else 'multi_head' if '--heads' in sys.argv else 'column_split'
OUT=ROOT/'profile/03_recurrence_parallelism'/GROUP
OUT.mkdir(parents=True,exist_ok=True)
B=ROOT/'profile/03_recurrence_parallelism/harness'/('build_'+GROUP);B.mkdir(exist_ok=True)
# launch_fwd expects a base-2 gate scale, matching flash_kda.cpp for lower_bound=-5.
files=['fwd.h','smxx/utils.cuh','smxx/fwd_kernel1.cuh','smxx/fwd_kernel2.cuh','smxx/fwd_launch.cu']
orig={n:(ROOT/'csrc'/n).read_text() for n in files}
for n in files:(B/Path(n).name).write_text(orig[n])
if '--persistent' in sys.argv:
    s=orig['smxx/fwd_kernel2.cuh']
    s=s.replace('    int total_tiles\n) {','    int total_tiles, unsigned int* queue\n) {')
    marker='    // --- warp specialization'
    s=s.replace(marker,'''    __shared__ unsigned int next_job;
    unsigned int job = blockIdx.x;
    while(job < unsigned(N*H)) {
'''+marker)
    s=s.replace('int seq_idx  = blockIdx.x;', 'int seq_idx = job % N;').replace('int head_idx = blockIdx.y;', 'int head_idx = job / N;')
    end=s.rfind('    __syncthreads();\n#endif\n}')
    assert end>=0
    s=s[:end]+'''#if TOPIC_SCHED != 0
    if(warp_role == WarpRole::STORE && lane_predicate) tma_store_wait<0>();
#endif
'''+s[end:]
    end=s.rfind('\n}')
    s=s[:end]+'''
#if TOPIC_SCHED == 0
    break;
#else
    if(threadIdx.x==0){
        // All consumers and TMA writes finished before the next task reuses storage.
        for(int i=0;i<InputStages;++i){
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(uint32_t(__cvta_generic_to_shared(&shared_storage.load_pipeline.full_barrier_[i]))):"memory");
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(uint32_t(__cvta_generic_to_shared(&shared_storage.load_pipeline.empty_barrier_[i]))):"memory");
        }
        for(int i=0;i<OutputStages;++i){
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(uint32_t(__cvta_generic_to_shared(&shared_storage.store_pipeline.full_barrier_[i]))):"memory");
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(uint32_t(__cvta_generic_to_shared(&shared_storage.store_pipeline.empty_barrier_[i]))):"memory");
        }
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(uint32_t(__cvta_generic_to_shared(&shared_storage.state_acc_tma_barrier))):"memory");
#if TOPIC_SCHED == 2
        next_job=gridDim.x+atomicAdd(queue,1u);
#else
        next_job=job+gridDim.x;
#endif
    }
    __syncthreads();
    job=next_job;
#endif
    }
'''+s[end:]
    (B/'fwd_kernel2.cuh').write_text(s)
    s=orig['smxx/fwd_launch.cu'];s=s[:s.index('// Explicit instantiations')]
    s='static int topic_phase=0; static int topic_workers=148; static unsigned int* topic_queue=nullptr;\n'+s
    s=s.replace('#if BLOCK_LEVEL_K1 >= 0\n    {','#if BLOCK_LEVEL_K1 >= 0\n    if(topic_phase!=2) {').replace('#if BLOCK_LEVEL_K2 >= 0\n    {','#if BLOCK_LEVEL_K2 >= 0\n    if(topic_phase!=1) {')
    s=s.replace('dim3 grid_k2(N, H);','''#if TOPIC_SCHED == 0
        dim3 grid_k2(N*H);
#else
        dim3 grid_k2(std::min(N*H,topic_workers));
#if TOPIC_SCHED == 2
        cudaMemsetAsync(topic_queue,0,sizeof(unsigned int),stream);
#endif
#endif''')
    s=s.replace('out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles','out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles, topic_queue')
    wrapper='extern "C" int run(int phase,void* q,void* k,void* v,void* g,void* beta,void* si,void* so,void* out,void* ws,void* cu,void* al,void* dt,int tiles,int T,int H,int N,cudaStream_t stream){\n topic_phase=phase;using B=cutlass::bfloat16_t;\n launch_fwd<128,true,true,false,true>((B*)q,(B*)k,(B*)v,(B*)g,(B*)beta,si,1.f/11.3137085f,so,(B*)out,ws,tiles,T,H,N,(int64_t*)cu,(float*)al,(float*)dt,-5.f*1.4426950408889634f,stream);\n return int(cudaGetLastError());\n}\nextern "C" int shared_bytes(){return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>);}\n'
    (B/'fwd_launch.cu').write_text(s+wrapper+'\nextern "C" void configure(int workers,void* queue){topic_workers=workers;topic_queue=(unsigned int*)queue;}\n')
    flags=['nvcc','-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-gencode','arch=compute_103a,code=sm_103a','-Xcompiler=-fPIC','-shared','-I'+str(B),'-I'+str(ROOT/'cutlass/include'),'-I'+str(ROOT/'cutlass/tools/util/include'),'-I'+str(ROOT/'cutlass/examples/common')]
    def compile_sched(v):
        name=['original','static','dynamic'][v]
        subprocess.run(flags+[f'-DTOPIC_SCHED={v}',str(B/'fwd_launch.cu'),'-o',str(B/(name+'.so'))],check=True)
        print('built',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(compile_sched,[0,1,2]))
    (OUT/'SOURCE.json').write_text(json.dumps(dict(source_sha256={n:hashlib.sha256(v.encode()).hexdigest() for n,v in orig.items()},compile_flags=flags),indent=2)+'\n')
    sys.exit(0)
if '--heads' in sys.argv:
    s=orig['smxx/fwd_kernel2.cuh']
    s=s.replace('__launch_bounds__(NumThreads)', '__launch_bounds__(NumThreads * TOPIC_HEADS)')
    s=s.replace('threadIdx.x','local_tid')
    s=s.replace(') {\n    using BF16 = cutlass::bfloat16_t;', ') {\n    int local_tid = int(threadIdx.x) % NumThreads;\n    int head_slot = int(threadIdx.x) / NumThreads;\n    using BF16 = cutlass::bfloat16_t;',1)
    s=s.replace('*reinterpret_cast<SharedStorageT*>(shared_mem)', 'reinterpret_cast<SharedStorageT*>(shared_mem)[head_slot]')
    s=s.replace('int head_idx = blockIdx.y;', 'int head_idx = blockIdx.y * TOPIC_HEADS + head_slot;')
    s=s.replace('compute_barrier(kComputeThreads, 0)', 'compute_barrier(kComputeThreads, head_slot)')
    (B/'fwd_kernel2.cuh').write_text(s)
    u=orig['smxx/utils.cuh'].replace('typename Pipeline::Params params;', 'typename Pipeline::Params params;\n    params.initializing_warp = (int(threadIdx.x)/192)*6;')
    (B/'utils.cuh').write_text(u)
    s=orig['smxx/fwd_launch.cu'];s=s[:s.index('// Explicit instantiations')]
    s='static int topic_phase=0;\n'+s
    s=s.replace('#if BLOCK_LEVEL_K1 >= 0\n    {','#if BLOCK_LEVEL_K1 >= 0\n    if(topic_phase!=2) {').replace('#if BLOCK_LEVEL_K2 >= 0\n    {','#if BLOCK_LEVEL_K2 >= 0\n    if(topic_phase!=1) {')
    s=s.replace('int smem_size_k2 = sizeof(SharedStorageK2T);','int smem_size_k2 = sizeof(SharedStorageK2T) * TOPIC_HEADS;')
    s=s.replace('dim3 grid_k2(N, H);','dim3 grid_k2(N, H/TOPIC_HEADS);').replace('dim3 block_k2(kK2Threads);','dim3 block_k2(kK2Threads * TOPIC_HEADS);')
    wrapper='extern "C" int run(int phase,void* q,void* k,void* v,void* g,void* beta,void* si,void* so,void* out,void* ws,void* cu,void* al,void* dt,int tiles,int T,int H,int N,cudaStream_t stream){\n topic_phase=phase;using B=cutlass::bfloat16_t;\n launch_fwd<128,true,true,false,true>((B*)q,(B*)k,(B*)v,(B*)g,(B*)beta,si,1.f/11.3137085f,so,(B*)out,ws,tiles,T,H,N,(int64_t*)cu,(float*)al,(float*)dt,-5.f*1.4426950408889634f,stream);\n return int(cudaGetLastError());\n}\nextern "C" int shared_bytes(){return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>);}\n'
    wrapper=wrapper.replace('return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>);','return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>)*TOPIC_HEADS;').replace('topic_phase=phase;', 'if(H%TOPIC_HEADS) return -2; topic_phase=phase;')
    (B/'fwd_launch.cu').write_text(s+wrapper)
    flags=['nvcc','-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-gencode','arch=compute_103a,code=sm_103a','-Xcompiler=-fPIC','-shared','-I'+str(B),'-I'+str(ROOT/'cutlass/include'),'-I'+str(ROOT/'cutlass/tools/util/include'),'-I'+str(ROOT/'cutlass/examples/common')]
    def compile_heads(n):
        name='original' if n==1 else 'heads2'
        subprocess.run(flags+[f'-DTOPIC_HEADS={n}',str(B/'fwd_launch.cu'),'-o',str(B/(name+'.so'))],check=True)
        print('built',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(compile_heads,[1,2]))
    (OUT/'SOURCE.json').write_text(json.dumps(dict(source_sha256={n:hashlib.sha256(v.encode()).hexdigest() for n,v in orig.items()},compile_flags=flags),indent=2)+'\n')
    sys.exit(0)
s=orig['smxx/fwd_kernel2.cuh']
s=s.replace('cutlass::bfloat16_t* out_raw_ptr,','cutlass::bfloat16_t* out_raw_ptr, cutlass::bfloat16_t* final_raw_ptr,')
s=s.replace('constexpr int kComputeThreads = 128;','constexpr int kComputeThreads = 128 / TOPIC_SPLIT;')
s=s.replace('const int warp_id = compute_tid / 32;','const int warp_id = compute_tid / 32 + int(blockIdx.z)*(4/TOPIC_SPLIT);')
start=s.index('    if (warp_role == WarpRole::STORE && lane_predicate) {',s.index('load_pipeline.consumer_release(load_read)'))
end=s.index('    __syncthreads();\n#endif\n}',start)
old=s[start:end]
manual='''
    if (warp_role == WarpRole::STORE) {
        StorePipelineState out_read;
        for(int t=0;t<t_tiles;++t){
            if(lane_predicate) store_pipeline.consumer_wait(out_read);
            __syncwarp();
            Tensor so=make_tensor(make_smem_ptr(shared_storage.output[out_read.index()].out.begin()),VOLayout{});
            int len=min(CHUNK,seq_len-t*CHUNK);
            constexpr int W=D/TOPIC_SPLIT;
            for(int x=threadIdx.x%32;x<len*W;x+=32){
                int row=x/W,col=int(blockIdx.z)*W+x%W;
                out_raw_ptr[(bos+t*CHUNK+row)*H*D+head_idx*D+col]=so(row,col);
            }
            __syncwarp();
            if(lane_predicate) store_pipeline.consumer_release(out_read);
            ++out_read;
        }
    }
    __syncthreads();
    if constexpr(HasStateOut){
        Tensor ss=make_tensor(make_smem_ptr(shared_storage.state_acc.begin()),TransposedStateSmemLayout{});
        constexpr int W=D/TOPIC_SPLIT;
        for(int x=threadIdx.x;x<D*W;x+=NumThreads){
            int row=x/W,col=int(blockIdx.z)*W+x%W;
            final_raw_ptr[(seq_idx*H+head_idx)*D*D+col*D+row]=ss(row,col);
        }
    }
'''
s=s[:start]+'#if TOPIC_MANUAL\n'+manual+'#else\n'+old+'#endif\n'+s[end:]
(B/'fwd_kernel2.cuh').write_text(s)
s=orig['smxx/fwd_launch.cu'];s=s[:s.index('// Explicit instantiations')]
s='static int topic_phase=0;\n'+s
s=s.replace('#if BLOCK_LEVEL_K1 >= 0\n    {','#if BLOCK_LEVEL_K1 >= 0\n    if(topic_phase!=2) {').replace('#if BLOCK_LEVEL_K2 >= 0\n    {','#if BLOCK_LEVEL_K2 >= 0\n    if(topic_phase!=1) {')
s=s.replace('constexpr int kK2Threads = 32 * 2 + 128;','constexpr int kK2Threads = 32 * 2 + 128/TOPIC_SPLIT;').replace('dim3 grid_k2(N, H);','dim3 grid_k2(N, H, TOPIC_SPLIT);').replace('out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles','out_ptr, static_cast<BF16*>(final_state_ptr), T_total, H, N, cu_seqlens_ptr, total_tiles')
s+='''
extern "C" int run(int phase,void* q,void* k,void* v,void* g,void* beta,void* si,void* so,void* out,void* ws,void* cu,void* al,void* dt,int tiles,int T,int H,int N,cudaStream_t stream){
 topic_phase=phase;using B=cutlass::bfloat16_t;
 launch_fwd<128,true,true,false,true>((B*)q,(B*)k,(B*)v,(B*)g,(B*)beta,si,1.f/11.3137085f,so,(B*)out,ws,tiles,T,H,N,(int64_t*)cu,(float*)al,(float*)dt,-5.f*1.4426950408889634f,stream);
 return int(cudaGetLastError());
}
extern "C" int shared_bytes(){return sizeof(SharedStorageK2<K2Layouts<128,16>,3,2>);}
'''
(B/'fwd_launch.cu').write_text(s)
flags=['nvcc','-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-gencode','arch=compute_103a,code=sm_103a','-Xcompiler=-fPIC','-shared','-I'+str(B),'-I'+str(ROOT/'cutlass/include'),'-I'+str(ROOT/'cutlass/tools/util/include'),'-I'+str(ROOT/'cutlass/examples/common')]
def compile(v):
 split,manual=v;name='original' if not manual else f'split{split}'
 subprocess.run(flags+[f'-DTOPIC_SPLIT={split}',f'-DTOPIC_MANUAL={manual}',str(B/'fwd_launch.cu'),'-o',str(B/(name+'.so'))],check=True)
 print('built',name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(compile,[(1,0),(1,1),(2,1),(4,1)]))
(OUT/'SOURCE.json').write_text(json.dumps(dict(source_sha256={n:hashlib.sha256(v.encode()).hexdigest() for n,v in orig.items()},compile_flags=flags),indent=2)+'\n')
