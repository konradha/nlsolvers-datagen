
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = False
torch._inductor.config.debug = True
torch._inductor.config.static_weight_shapes = False
torch._inductor.config.size_asserts = False
torch._inductor.config.triton.debug_sync_kernel = True
torch._inductor.config.triton.store_cubin = True
torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.3.1+cu121
# torch cuda version: 12.1
# torch git version: d44533f9d073df13895333e70b66f81c513c1889


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Mon_Apr__3_17:16:06_PDT_2023 
# Cuda compilation tools, release 12.1, V12.1.105 
# Build cuda_12.1.r12.1/compiler.32688072_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 2080 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        select = torch.ops.aten.select.int(arg1_1, 0, 1)
        slice_1 = torch.ops.aten.slice.Tensor(select, 0, 1, 299);  select = None
        select_1 = torch.ops.aten.select.int(arg1_1, 0, 0)
        slice_2 = torch.ops.aten.slice.Tensor(select_1, 0, 1, 299);  select_1 = None
        copy = torch.ops.aten.copy.default(slice_2, slice_1);  slice_2 = slice_1 = None
        select_2 = torch.ops.aten.select.int(arg1_1, 0, 0)
        slice_scatter = torch.ops.aten.slice_scatter.default(select_2, copy, 0, 1, 299);  select_2 = copy = None
        select_scatter = torch.ops.aten.select_scatter.default(arg1_1, slice_scatter, 0, 0);  slice_scatter = None
        select_6 = torch.ops.aten.select.int(select_scatter, 0, 299)
        slice_6 = torch.ops.aten.slice.Tensor(select_6, 0, 1, 299);  select_6 = None
        select_7 = torch.ops.aten.select.int(select_scatter, 0, 298)
        slice_7 = torch.ops.aten.slice.Tensor(select_7, 0, 1, 299);  select_7 = None
        copy_1 = torch.ops.aten.copy.default(slice_6, slice_7);  slice_6 = slice_7 = None
        select_8 = torch.ops.aten.select.int(select_scatter, 0, 299)
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(select_8, copy_1, 0, 1, 299);  select_8 = copy_1 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(select_scatter, slice_scatter_1, 0, 299);  select_scatter = slice_scatter_1 = None
        slice_11 = torch.ops.aten.slice.Tensor(select_scatter_1, 0, 0, 9223372036854775807)
        select_12 = torch.ops.aten.select.int(slice_11, 1, 0);  slice_11 = None
        slice_12 = torch.ops.aten.slice.Tensor(select_scatter_1, 0, 0, 9223372036854775807)
        select_13 = torch.ops.aten.select.int(slice_12, 1, 1);  slice_12 = None
        copy_2 = torch.ops.aten.copy.default(select_12, select_13);  select_12 = select_13 = None
        slice_13 = torch.ops.aten.slice.Tensor(select_scatter_1, 0, 0, 9223372036854775807)
        select_scatter_2 = torch.ops.aten.select_scatter.default(slice_13, copy_2, 1, 0);  slice_13 = copy_2 = None
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(select_scatter_1, select_scatter_2, 0, 0, 9223372036854775807);  select_scatter_1 = select_scatter_2 = None
        slice_17 = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
        select_17 = torch.ops.aten.select.int(slice_17, 1, 299);  slice_17 = None
        slice_18 = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
        select_18 = torch.ops.aten.select.int(slice_18, 1, 298);  slice_18 = None
        copy_3 = torch.ops.aten.copy.default(select_17, select_18);  select_17 = select_18 = None
        slice_19 = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
        select_scatter_3 = torch.ops.aten.select_scatter.default(slice_19, copy_3, 1, 299);  slice_19 = copy_3 = None
        slice_scatter_3 = torch.ops.aten.slice_scatter.default(slice_scatter_2, select_scatter_3, 0, 0, 9223372036854775807);  slice_scatter_2 = select_scatter_3 = None
        copy_ = torch.ops.aten.copy_.default(arg1_1, slice_scatter_3);  arg1_1 = slice_scatter_3 = None
        return (copy_,)
        
def load_args(reader):
    reader.symint(300)  # arg0_1
    buf0 = reader.storage(None, 2392, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (s0, s0), dtype=torch.float64, is_leaf=True)  # arg1_1
    reader.symint(300)  # arg2_1
    reader.symint(300)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
