import torch
import torch._dynamo as dynamo
from torch._inductor import config

import os

config.debug = True
config.trace.enabled = True
config.trace.output_code = True
config.trace.fx_graph = True
config.trace.ir_post_fusion = True

config.triton.store_cubin = True
config.triton.debug_sync_kernel = True
config.static_weight_shapes = False 
config.size_asserts = False

def neumann_bc(u: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    assert len(u.shape) == 2 
    # no corners
    u[0, 1:ny-1]    = u[1, 1:ny-1]
    u[nx-1, 1:ny-1] = u[nx-2, 1:ny-1]

    # with corners
    u[:, 0]    = u[:, 1]
    u[:, ny-1] = u[:, ny-2] 
    return u

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
        

    import torch._inductor.config as inductor_config
    inductor_config.freezing = False
    inductor_config.triton.store_cubin = True
    
    compiled_fn = torch.compile(
        neumann_bc,
        backend="inductor",
        dynamic=True,  # Important for handling different sizes
        fullgraph=True,  # Try to compile the whole graph
        options={
          "debug": True,
          "trace.enabled": True,
          "trace.output_code": True,
          "triton.store_cubin": True,
          "static_weight_shapes": False
        },
    )


    nx = ny = 300
    test_tensor = torch.rand((nx, ny), dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")

    result = compiled_fn(test_tensor, nx, ny)
    explanation = dynamo.explain(compiled_fn)(test_tensor, nx, ny)
    print(explanation)

    print("\nLook for CUDA code in the debug directory:")
    os.system("find . -name 'torch_compile_debug' -type d | xargs ls -la")
