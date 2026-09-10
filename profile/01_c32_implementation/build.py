"""Build only this candidate; reuse the existing environment and CUTLASS headers."""
import os
from pathlib import Path
from torch.utils.cpp_extension import load
root = Path(__file__).resolve().parents[2]
cutlass = Path('/home/lcpu/60990375/FlashKDA/cutlass')
build = root / 'profile/01_c32_implementation/build'
build.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MAX_JOBS', '2')
load(name='flash_kda_C', sources=[str(root/'csrc/flash_kda.cpp'),str(root/'csrc/smxx/fwd_launch.cu')],
     extra_include_paths=['/home/lcpu/60990375/topic7-envs/python-include', '/home/lcpu/60990375/topic7-envs/python-include/python3.12', '/home/lcpu/60990375/topic7-envs/python-include/x86_64-linux-gnu/python3.12', str(root/'csrc'),str(cutlass/'include'),str(cutlass/'tools/util/include'),str(cutlass/'examples/common')],
     extra_cflags=['-O3','-Wno-psabi'],
     extra_cuda_cflags=['-O3','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
       '-U__CUDA_NO_HALF2_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
       '--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math',
       '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills','-lineinfo','--threads','2',
       '-gencode=arch=compute_103a,code=sm_103a'],
     build_directory=str(build),verbose=True)
