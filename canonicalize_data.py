import netCDF4 as nc
import numpy as np
import yaml
import sys
from pathlib import Path
import re

def dump_header(ds, filepath):
    print(f"netcdf {filepath.stem} {{")
    print("dimensions:")
    for dim_name, dim in ds.dimensions.items():
        size = len(dim) if not dim.isunlimited() else "UNLIMITED"
        print(f"    {dim_name} = {size} ;")
    
    print("\nvariables:")
    for var_name, var in ds.variables.items():
        dims_str = ', '.join(var.dimensions)
        print(f"    {var.dtype} {var_name}({dims_str}) ;")
        for attr_name in var.ncattrs():
            attr_val = ds.attrs[attr_name] if hasattr(ds, 'attrs') else getattr(var, attr_name)
            print(f"        {var_name}:{attr_name} = {attr_val} ;")
    
    print("\n// global attributes:")
    for attr_name in ds.ncattrs():
        if hasattr(ds, 'attrs'):
            attr_val = ds.attrs[attr_name]
        else:
            attr_val = getattr(ds, attr_name)
        print(f"    :{attr_name} = {attr_val} ;")
    
    print("}")

def parse_shape_from_filename(fname):
    match = re.search(r'shape_(\d+)x(\d+)(?:x(\d+))?', str(fname))
    if match:
        dims = [int(match.group(i)) for i in range(1, 4) if match.group(i)]
        return tuple(dims)
    return None

def check_and_reformat(input_file, config, dry_run=True):
    print(f"Processing: {input_file}")
    print(f"Using config from: {config}\n")
    
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Configuration:")
    print(f"  PDE Type: {cfg['pde_type']}")
    print(f"  Boundary Condition: {cfg['boundary_condition']}")
    print(f"  Trajectories: {cfg['num_trajectories']}")
    print(f"  Snapshots: {cfg['num_snapshots']}")
    print(f"  Grid: {cfg['grid']}")
    print(f"  Domain: {cfg['domain']}")
    if 'm_classes' in cfg:
        print(f"  m_classes: {cfg['m_classes']}")
    if 'c_classes' in cfg:
        print(f"  c_classes: {cfg['c_classes']}")
    print()
    
    with nc.Dataset(input_file, 'r') as ds_in:
        n_traj = int(ds_in.num_trajectories)
        n_snap = int(ds_in.num_snapshots)
        spatial_shape = eval(ds_in.spatial_shape)
        dims = int(ds_in.dims)
        is_complex = bool(ds_in.is_complex)
        
        Lx = float(ds_in.Lx)
        Ly = float(ds_in.Ly)
        Lz = float(ds_in.Lz) if hasattr(ds_in, 'Lz') else None
        
        filename_shape = parse_shape_from_filename(input_file)
        if filename_shape:
            nx, ny = filename_shape[0], filename_shape[1]
            nz = filename_shape[2] if len(filename_shape) == 3 else None
        else:
            nx, ny = spatial_shape[0], spatial_shape[1]
            nz = spatial_shape[2] if dims == 3 else None
        
        nt = cfg.get('nt', n_snap)
        T = cfg['domain']['T'][1]
        
        print(f"Dataset info:")
        print(f"  Stored spatial shape: {spatial_shape} (downsampled from original)")
        print(f"  Target spatial shape: {nx}x{ny}" + (f"x{nz}" if nz else ""))
        print(f"  Stored snapshots: {n_snap} (downsampled from nt={nt})")
        print(f"  Temporal extent: T={T}")
        print()
        
        print(f"Changes to be made:")
        print(f"  1. Add coordinate arrays: member, time, x, y" + (", z" if nz else ""))
        print(f"  2. Reshape data from flattened to (member, time, x, y" + (", z)" if nz else ")"))
        print(f"  3. Rename variables: re_u->u_real, im_u->u_imag (if complex) or keep u, v (if real)")
        print(f"  4. Add enhanced global attributes from config")
        print()
        
        if dry_run:
            print("DRY RUN: Generating future header\n")
            print("="*60)
            
            class DummyDim:
                def __init__(self, size):
                    self.size = size
                def __len__(self):
                    return self.size
                def isunlimited(self):
                    return False
            
            class DummyVar:
                def __init__(self, dtype, dims):
                    self.dtype = dtype
                    self.dimensions = dims
                    self.attrs = {}
                def ncattrs(self):
                    return list(self.attrs.keys())
            
            class DummyDataset:
                def __init__(self):
                    self.dimensions = {}
                    self.variables = {}
                    self.attrs = {}
                def ncattrs(self):
                    return list(self.attrs.keys())
            
            ds_dummy = DummyDataset()
            ds_dummy.dimensions['member'] = DummyDim(n_traj)
            ds_dummy.dimensions['time'] = DummyDim(n_snap)
            ds_dummy.dimensions['x'] = DummyDim(spatial_shape[0])
            ds_dummy.dimensions['y'] = DummyDim(spatial_shape[1])
            if dims == 3:
                ds_dummy.dimensions['z'] = DummyDim(spatial_shape[2])
            
            ds_dummy.variables['member'] = DummyVar('int32', ('member',))
            ds_dummy.variables['time'] = DummyVar('float32', ('time',))
            ds_dummy.variables['x'] = DummyVar('float32', ('x',))
            ds_dummy.variables['y'] = DummyVar('float32', ('y',))
            if dims == 3:
                ds_dummy.variables['z'] = DummyVar('float32', ('z',))
            
            shape_dims = ('member', 'time', 'x', 'y', 'z') if dims == 3 else ('member', 'time', 'x', 'y')
            
            if is_complex:
                ds_dummy.variables['u_real'] = DummyVar('float32', shape_dims)
                ds_dummy.variables['u_imag'] = DummyVar('float32', shape_dims)
            else:
                ds_dummy.variables['u'] = DummyVar('float32', shape_dims)
                ds_dummy.variables['v'] = DummyVar('float32', shape_dims)
            
            param_dims = ('member', 'x', 'y', 'z') if dims == 3 else ('member', 'x', 'y')
            ds_dummy.variables['m'] = DummyVar('float32', param_dims)
            ds_dummy.variables['c'] = DummyVar('float32', param_dims)
            
            ds_dummy.attrs['pde_type'] = cfg['pde_type']
            ds_dummy.attrs['boundary_condition'] = cfg['boundary_condition']
            ds_dummy.attrs['num_trajectories'] = n_traj
            ds_dummy.attrs['num_snapshots'] = n_snap
            ds_dummy.attrs['nt'] = nt
            ds_dummy.attrs['spatial_shape'] = str(spatial_shape)
            ds_dummy.attrs['dims'] = dims
            ds_dummy.attrs['is_complex'] = is_complex
            ds_dummy.attrs['Lx'] = Lx
            ds_dummy.attrs['Ly'] = Ly
            if Lz:
                ds_dummy.attrs['Lz'] = Lz
            ds_dummy.attrs['T'] = T
            if 'm_classes' in cfg:
                ds_dummy.attrs['m_classes'] = ', '.join(cfg['m_classes'])
            if 'c_classes' in cfg:
                ds_dummy.attrs['c_classes'] = ', '.join(cfg['c_classes'])
            
            dump_header(ds_dummy, Path(input_file))
            print("="*60)
            return
        
        output_file = str(input_file).replace('.nc', '_standard.nc')
        print(f"Writing to: {output_file}")
        
        spat_size = int(np.prod(spatial_shape))
        
        with nc.Dataset(output_file, 'w') as ds_out:
            ds_out.createDimension('member', n_traj)
            ds_out.createDimension('time', n_snap)
            ds_out.createDimension('x', spatial_shape[0])
            ds_out.createDimension('y', spatial_shape[1])
            if dims == 3:
                ds_out.createDimension('z', spatial_shape[2])
            
            member_var = ds_out.createVariable('member', 'i4', ('member',))
            time_var = ds_out.createVariable('time', 'f4', ('time',))
            x_var = ds_out.createVariable('x', 'f4', ('x',))
            y_var = ds_out.createVariable('y', 'f4', ('y',))
            if dims == 3:
                z_var = ds_out.createVariable('z', 'f4', ('z',))
            
            shape_dims = ('member', 'time', 'x', 'y', 'z') if dims == 3 else ('member', 'time', 'x', 'y')
            
            if is_complex:
                u_re = ds_out.createVariable('u_real', 'f4', shape_dims, chunksizes=(1, n_snap, spatial_shape[0], spatial_shape[1], spatial_shape[2]) if dims == 3 else (1, n_snap, spatial_shape[0], spatial_shape[1]))
                u_im = ds_out.createVariable('u_imag', 'f4', shape_dims, chunksizes=(1, n_snap, spatial_shape[0], spatial_shape[1], spatial_shape[2]) if dims == 3 else (1, n_snap, spatial_shape[0], spatial_shape[1]))
            else:
                u_var = ds_out.createVariable('u', 'f4', shape_dims, chunksizes=(1, n_snap, spatial_shape[0], spatial_shape[1], spatial_shape[2]) if dims == 3 else (1, n_snap, spatial_shape[0], spatial_shape[1]))
                v_var = ds_out.createVariable('v', 'f4', shape_dims, chunksizes=(1, n_snap, spatial_shape[0], spatial_shape[1], spatial_shape[2]) if dims == 3 else (1, n_snap, spatial_shape[0], spatial_shape[1]))
            
            param_dims = ('member', 'x', 'y', 'z') if dims == 3 else ('member', 'x', 'y')
            m_var = ds_out.createVariable('m', 'f4', param_dims)
            c_var = ds_out.createVariable('c', 'f4', param_dims)
            
            member_var[:] = np.arange(n_traj)
            time_var[:] = np.linspace(0.0, T, n_snap)
            x_var[:] = np.linspace(-Lx, Lx, spatial_shape[0])
            y_var[:] = np.linspace(-Ly, Ly, spatial_shape[1])
            if dims == 3:
                z_var[:] = np.linspace(-Lz, Lz, spatial_shape[2])
            
            for traj_idx in range(n_traj):
                offset_temp = traj_idx * n_snap * spat_size
                offset_spat = traj_idx * spat_size
                
                if is_complex:
                    re_data = ds_in.variables['re_u'][offset_temp:offset_temp + n_snap * spat_size]
                    im_data = ds_in.variables['im_u'][offset_temp:offset_temp + n_snap * spat_size]
                    
                    re_reshaped = re_data.reshape((n_snap,) + spatial_shape)
                    im_reshaped = im_data.reshape((n_snap,) + spatial_shape)
                    
                    u_re[traj_idx, :] = re_reshaped
                    u_im[traj_idx, :] = im_reshaped
                else:
                    u_data = ds_in.variables['u'][offset_temp:offset_temp + n_snap * spat_size]
                    v_data = ds_in.variables['v'][offset_temp:offset_temp + n_snap * spat_size]
                    
                    u_reshaped = u_data.reshape((n_snap,) + spatial_shape)
                    v_reshaped = v_data.reshape((n_snap,) + spatial_shape)
                    
                    u_var[traj_idx, :] = u_reshaped
                    v_var[traj_idx, :] = v_reshaped
                
                m_data = ds_in.variables['m'][offset_spat:offset_spat + spat_size]
                c_data = ds_in.variables['c'][offset_spat:offset_spat + spat_size]
                
                m_var[traj_idx, :] = m_data.reshape(spatial_shape)
                c_var[traj_idx, :] = c_data.reshape(spatial_shape)
                
                if (traj_idx + 1) % 100 == 0:
                    print(f"  Processed {traj_idx + 1}/{n_traj} trajectories")
            
            ds_out.pde_type = cfg['pde_type']
            ds_out.boundary_condition = cfg['boundary_condition']
            ds_out.num_trajectories = int(n_traj)
            ds_out.num_snapshots = int(n_snap)
            ds_out.nt = nt
            ds_out.spatial_shape = str(spatial_shape)
            ds_out.dims = dims
            ds_out.is_complex = int(is_complex)
            ds_out.Lx = Lx
            ds_out.Ly = Ly
            if Lz:
                ds_out.Lz = Lz
            ds_out.T = T
            if 'm_classes' in cfg:
                ds_out.m_classes = ', '.join(cfg['m_classes'])
            if 'c_classes' in cfg:
                ds_out.c_classes = ', '.join(cfg['c_classes'])
        
        print(f"Complete: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python canonicalize_dataset.py <netcdf_file> <config_yaml> [--dry-run]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    config_file = Path(sys.argv[2])
    dry_run = '--dry-run' in sys.argv
    
    if not input_file.exists():
        print(f"Error: NetCDF file not found: {input_file}")
        sys.exit(1)
    
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    if dry_run:
        print("DRY RUN MODE: No files will be modified\n")
    
    check_and_reformat(input_file, config_file, dry_run)
