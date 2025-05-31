class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "f64[s0, s0]", arg2_1: "Sym(300)", arg3_1: "Sym(300)"):
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:21 in neumann_bc, code: u[0, 1:ny-1]    = u[1, 1:ny-1]
        select_2: "f64[s0]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        select_1: "f64[s0]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        slice_2: "f64[298]" = torch.ops.aten.slice.Tensor(select_1, 0, 1, 299);  select_1 = None
        select: "f64[s0]" = torch.ops.aten.select.int(arg1_1, 0, 1)
        slice_1: "f64[298]" = torch.ops.aten.slice.Tensor(select, 0, 1, 299);  select = None
        
        # No stacktrace found for following nodes
        select_int: "f64[s0]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        slice_scatter_default: "f64[s0]" = torch.ops.aten.slice_scatter.default(select_int, slice_1, 0, 1, 299);  select_int = slice_1 = None
        select_scatter_default: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(arg1_1, slice_scatter_default, 0, 0);  slice_scatter_default = None
        
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:22 in neumann_bc, code: u[nx-1, 1:ny-1] = u[nx-2, 1:ny-1]
        select_8: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default, 0, 299)
        select_6: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default, 0, 299)
        slice_6: "f64[298]" = torch.ops.aten.slice.Tensor(select_6, 0, 1, 299);  select_6 = None
        select_7: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default, 0, 298)
        slice_7: "f64[298]" = torch.ops.aten.slice.Tensor(select_7, 0, 1, 299);  select_7 = None
        
        # No stacktrace found for following nodes
        select_int_1: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default, 0, 299)
        slice_scatter_default_1: "f64[s0]" = torch.ops.aten.slice_scatter.default(select_int_1, slice_7, 0, 1, 299);  select_int_1 = slice_7 = None
        select_scatter_default_1: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(select_scatter_default, slice_scatter_default_1, 0, 299);  select_scatter_default = slice_scatter_default_1 = None
        
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:25 in neumann_bc, code: u[:, 0]    = u[:, 1]
        select_12: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default_1, 1, 0)
        select_13: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default_1, 1, 1)
        
        # No stacktrace found for following nodes
        select_scatter_default_2: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(select_scatter_default_1, select_13, 1, 0);  select_scatter_default_1 = select_13 = None
        
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:26 in neumann_bc, code: u[:, ny-1] = u[:, ny-2]
        select_17: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default_2, 1, 299)
        select_18: "f64[s0]" = torch.ops.aten.select.int(select_scatter_default_2, 1, 298)
        
        # No stacktrace found for following nodes
        select_scatter_default_3: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(select_scatter_default_2, select_18, 1, 299);  select_scatter_default_2 = select_18 = None
        
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:26 in neumann_bc, code: u[:, ny-1] = u[:, ny-2]
        copy_: "f64[s0, s0]" = torch.ops.aten.copy_.default(arg1_1, select_scatter_default_3);  arg1_1 = select_scatter_default_3 = None
        return (copy_,)
        