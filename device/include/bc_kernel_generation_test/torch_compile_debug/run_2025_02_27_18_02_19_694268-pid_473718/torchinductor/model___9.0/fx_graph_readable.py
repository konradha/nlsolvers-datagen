class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "f64[s0, s0]", arg2_1: "Sym(300)", arg3_1: "Sym(300)"):
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:21 in neumann_bc, code: u[0, 1:ny-1]    = u[1, 1:ny-1]
        select: "f64[s0]" = torch.ops.aten.select.int(arg1_1, 0, 1)
        slice_1: "f64[298]" = torch.ops.aten.slice.Tensor(select, 0, 1, 299);  select = None
        select_1: "f64[s0]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        slice_2: "f64[298]" = torch.ops.aten.slice.Tensor(select_1, 0, 1, 299);  select_1 = None
        copy: "f64[298]" = torch.ops.aten.copy.default(slice_2, slice_1);  slice_2 = slice_1 = None
        select_2: "f64[s0]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        slice_scatter: "f64[s0]" = torch.ops.aten.slice_scatter.default(select_2, copy, 0, 1, 299);  select_2 = copy = None
        select_scatter: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(arg1_1, slice_scatter, 0, 0);  slice_scatter = None
        
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:22 in neumann_bc, code: u[nx-1, 1:ny-1] = u[nx-2, 1:ny-1]
        select_6: "f64[s0]" = torch.ops.aten.select.int(select_scatter, 0, 299)
        slice_6: "f64[298]" = torch.ops.aten.slice.Tensor(select_6, 0, 1, 299);  select_6 = None
        select_7: "f64[s0]" = torch.ops.aten.select.int(select_scatter, 0, 298)
        slice_7: "f64[298]" = torch.ops.aten.slice.Tensor(select_7, 0, 1, 299);  select_7 = None
        copy_1: "f64[298]" = torch.ops.aten.copy.default(slice_6, slice_7);  slice_6 = slice_7 = None
        select_8: "f64[s0]" = torch.ops.aten.select.int(select_scatter, 0, 299)
        slice_scatter_1: "f64[s0]" = torch.ops.aten.slice_scatter.default(select_8, copy_1, 0, 1, 299);  select_8 = copy_1 = None
        select_scatter_1: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(select_scatter, slice_scatter_1, 0, 299);  select_scatter = slice_scatter_1 = None
        
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:25 in neumann_bc, code: u[:, 0]    = u[:, 1]
        slice_11: "f64[s0, s0]" = torch.ops.aten.slice.Tensor(select_scatter_1, 0, 0, 9223372036854775807)
        select_12: "f64[s0]" = torch.ops.aten.select.int(slice_11, 1, 0);  slice_11 = None
        slice_12: "f64[s0, s0]" = torch.ops.aten.slice.Tensor(select_scatter_1, 0, 0, 9223372036854775807)
        select_13: "f64[s0]" = torch.ops.aten.select.int(slice_12, 1, 1);  slice_12 = None
        copy_2: "f64[s0]" = torch.ops.aten.copy.default(select_12, select_13);  select_12 = select_13 = None
        slice_13: "f64[s0, s0]" = torch.ops.aten.slice.Tensor(select_scatter_1, 0, 0, 9223372036854775807)
        select_scatter_2: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(slice_13, copy_2, 1, 0);  slice_13 = copy_2 = None
        slice_scatter_2: "f64[s0, s0]" = torch.ops.aten.slice_scatter.default(select_scatter_1, select_scatter_2, 0, 0, 9223372036854775807);  select_scatter_1 = select_scatter_2 = None
        
        # File: /cluster/home/konradha/nonlinear-solvers/nlsolvers/device/include/bc_update_kernel_fusion.py:26 in neumann_bc, code: u[:, ny-1] = u[:, ny-2]
        slice_17: "f64[s0, s0]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
        select_17: "f64[s0]" = torch.ops.aten.select.int(slice_17, 1, 299);  slice_17 = None
        slice_18: "f64[s0, s0]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
        select_18: "f64[s0]" = torch.ops.aten.select.int(slice_18, 1, 298);  slice_18 = None
        copy_3: "f64[s0]" = torch.ops.aten.copy.default(select_17, select_18);  select_17 = select_18 = None
        slice_19: "f64[s0, s0]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807)
        select_scatter_3: "f64[s0, s0]" = torch.ops.aten.select_scatter.default(slice_19, copy_3, 1, 299);  slice_19 = copy_3 = None
        slice_scatter_3: "f64[s0, s0]" = torch.ops.aten.slice_scatter.default(slice_scatter_2, select_scatter_3, 0, 0, 9223372036854775807);  slice_scatter_2 = select_scatter_3 = None
        copy_: "f64[s0, s0]" = torch.ops.aten.copy_.default(arg1_1, slice_scatter_3);  arg1_1 = slice_scatter_3 = None
        return (copy_,)
        