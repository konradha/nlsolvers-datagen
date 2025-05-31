
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /scratch/tmp.24755961.konradha/torchinductor_konradha/rq/crq3m3grnclfckvkksxrf2jg73hagwieolxo3rlgu2z2v663afl3.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '3c9eac74f04023b38fe37317f0afe4f78f5f0ccf49418abd41d59a08f8a919b2'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0)
    x0 = xindex % ks0
    x2 = xindex
    tmp26 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (x0 + (299*ks0)), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 299, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 299, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tl.full([1], 298, tl.int32)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp9 == tmp10
    tmp12 = tmp8 & tmp8
    tmp13 = tl.load(in_ptr0 + (ks0 + x0), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.load(in_ptr0 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.load(in_ptr0 + (x0 + (298*ks0)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tmp1 == tmp10
    tmp23 = tl.load(in_ptr0 + (ks0 + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp8, tmp23, tmp24)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp29 = tl.where(tmp22, tmp27, tmp28)
    tmp30 = tl.where(tmp8, tmp21, tmp29)
    tmp31 = tmp0 == tmp10
    tmp33 = tl.where(tmp31, tmp27, tmp32)
    tmp34 = tl.where(tmp2, tmp30, tmp33)
    tl.store(out_ptr0 + (x2), tmp34, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /scratch/tmp.24755961.konradha/torchinductor_konradha/uv/cuvi747mgngom33pwxvrie4po5isymwvermlnz5tybblus7yhtl5.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': ['out_ptr1'], 'no_x_dim': False, 'backend_hash': '3c9eac74f04023b38fe37317f0afe4f78f5f0ccf49418abd41d59a08f8a919b2'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    x2 = xindex
    tmp6 = tl.load(in_ptr0 + (1 + (ks0*x1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (298 + (ks0*x1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 299, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 298, tl.int32)
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 == tmp4
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tmp12 = tl.where(tmp2, tmp8, tmp11)
    tl.store(out_ptr1 + (x2), tmp12, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.triton_heuristics.CachingAutotuner):
            kernel.cuda_kernel_saved = False
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s0, s0), (s0, 1), torch.float64)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_0_xnumel = s0*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg1_1, buf0, s0, triton_poi_fused_0_xnumel, grid=grid(triton_poi_fused_0_xnumel), stream=stream0)
        torch.cuda.synchronize()
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1_xnumel = s0*s0
        triton_poi_fused_1.run(buf0, arg1_1, s0, triton_poi_fused_1_xnumel, grid=grid(triton_poi_fused_1_xnumel), stream=stream0)
        del buf0
        torch.cuda.synchronize()

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.triton_heuristics.CachingAutotuner):
            if not kernel.cuda_kernel_saved:
                if len(kernel.launchers) == 0:
                    kernel.precompile()
                kernel.save_cuda_kernel(
                    grid=(0, 0, 0),   # use dummy grid
                    stream="stream",  # use dummy stream
                    launcher=kernel.launchers[0],
                )
    return (arg1_1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 300
    arg1_1 = rand_strided((300, 300), (300, 1), device='cuda:0', dtype=torch.float64)
    arg2_1 = 300
    arg3_1 = 300
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
