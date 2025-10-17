import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import re

dirs = [
    "/cluster/scratch/konradha/full_kge_2d/class_2D_shape_128x128_ns_100_real.nc",
    "/cluster/scratch/konradha/full_kge_2d/class_2D_shape_128x128_ns_50_real.nc",
    "/cluster/scratch/konradha/full_kge_3d/class_3D_shape_128x128x128_ns_40_real.nc",
    "/cluster/scratch/konradha/full_nlse_2d/class_2D_shape_128x128_ns_128_complex.nc",
]

def parse_shape_from_filename(fname):
    match = re.search(r'shape_(\d+)x(\d+)(?:x(\d+))?', fname)
    if match:
        dims = [int(match.group(i)) for i in range(1, 4) if match.group(i)]
        return tuple(dims)
    return None

def get_trajectory_info(dataset):
    n_spatial = dataset.num_trajectories * np.prod(eval(dataset.spatial_shape))
    n_temporal = dataset.num_trajectories * dataset.num_snapshots * np.prod(eval(dataset.spatial_shape))
    n_snapshots = dataset.num_snapshots
    n_trajectories = dataset.num_trajectories
    spatial_shape = eval(dataset.spatial_shape)
    is_complex = bool(dataset.is_complex)
    
    if is_complex:
        temporal_vars = ['re_u', 'im_u']
        spatial_vars = ['re_u0', 'im_u0']
    else:
        temporal_vars = ['u', 'v']
        spatial_vars = ['u0', 'v0']
    
    return temporal_vars, spatial_vars, n_trajectories, n_snapshots, spatial_shape, is_complex

def check_trajectory(dataset, traj_idx, spatial_shape, n_snapshots, temporal_vars):
    spat_size = int(np.prod(spatial_shape))
    traj_offset = traj_idx * n_snapshots * spat_size
    traj_size = n_snapshots * spat_size
    
    for var_name in temporal_vars:
        data = dataset.variables[var_name][traj_offset:traj_offset + traj_size]
        if np.any(np.isnan(data)):
            return False, f"NaN in {var_name}, traj {traj_idx}"
        if np.all(data == data[0]):
            return False, f"Constant {var_name}, traj {traj_idx}"
    
    return True, "OK"

def compare_trajectories(dataset, traj_a, traj_b, spatial_shape, n_snapshots, temporal_vars, sample_size=1000):
    spat_size = int(np.prod(spatial_shape))
    traj_size = n_snapshots * spat_size
    
    offset_a = traj_a * traj_size
    offset_b = traj_b * traj_size
    
    indices = np.random.choice(traj_size, min(sample_size, traj_size), replace=False)
    
    for var_name in temporal_vars:
        data_a = dataset.variables[var_name][offset_a + indices]
        data_b = dataset.variables[var_name][offset_b + indices]
        if np.allclose(data_a, data_b):
            return False, f"Identical {var_name}, traj {traj_a} vs {traj_b}"
    
    return True, "Different"

def plot_snapshots(dataset, traj_idx, spatial_shape, n_snapshots, temporal_vars, is_complex, output_file):
    spat_size = int(np.prod(spatial_shape))
    traj_offset = traj_idx * n_snapshots * spat_size
    
    snapshot_indices = [0, n_snapshots // 2, n_snapshots - 1]
    
    if is_complex:
        n_vars = 1
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.reshape(1, -1)
        
        for j, snap_idx in enumerate(snapshot_indices):
            offset = traj_offset + snap_idx * spat_size
            re_data = dataset.variables['re_u'][offset:offset + spat_size]
            im_data = dataset.variables['im_u'][offset:offset + spat_size]
            data = np.sqrt(re_data**2 + im_data**2)
            
            data = data.reshape(spatial_shape)
            
            if len(spatial_shape) == 3:
                data = data[:, :, spatial_shape[2] // 2]
            
            axes[0, j].imshow(data, cmap='viridis')
            axes[0, j].set_title(f"|u| t={snap_idx}")
            axes[0, j].axis('off')
    else:
        n_vars = len(temporal_vars)
        fig, axes = plt.subplots(n_vars, 3, figsize=(12, 4 * n_vars))
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_name in enumerate(temporal_vars):
            for j, snap_idx in enumerate(snapshot_indices):
                offset = traj_offset + snap_idx * spat_size
                data = dataset.variables[var_name][offset:offset + spat_size]
                
                data = data.reshape(spatial_shape)
                
                if len(spatial_shape) == 3:
                    data = data[:, :, spatial_shape[2] // 2]
                
                axes[i, j].imshow(data, cmap='viridis')
                axes[i, j].set_title(f"{var_name} t={snap_idx}")
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

for fname in dirs:
    print(f"\n{fname}", flush=True)
    
    try:
        dataset = nc.Dataset(fname, 'r')
    except Exception as e:
        print(f"  ERROR: Could not open file: {e}", flush=True)
        continue
    
    temporal_vars, spatial_vars, n_trajectories, n_snapshots, spatial_shape, is_complex = get_trajectory_info(dataset)
    
    print(f"Trajectories: {n_trajectories}, Snapshots: {n_snapshots}, Spatial shape: {spatial_shape}", flush=True)
    print(f"Temporal vars: {temporal_vars}, Spatial vars: {spatial_vars}, Complex: {is_complex}", flush=True)
    
    check_trajs = min(10, n_trajectories)
    sampled_traj_indices = np.random.choice(n_trajectories, check_trajs, replace=False)
    sampled_traj_indices.sort()
    
    for traj_idx in sampled_traj_indices:
        valid, msg = check_trajectory(dataset, traj_idx, spatial_shape, n_snapshots, temporal_vars)
        if not valid:
            print(f"  FAIL: {msg}", flush=True)
        else:
            print(f"  Traj {traj_idx}: {msg}", flush=True)
    
    if n_trajectories > 1:
        pairs = min(5, n_trajectories - 1)
        pair_indices = np.random.choice(n_trajectories - 1, pairs, replace=False)
        
        for i in pair_indices:
            different, msg = compare_trajectories(dataset, i, i + 1, spatial_shape, n_snapshots, temporal_vars)
            if not different:
                print(f"  FAIL: {msg}", flush=True)
            else:
                print(f"  Traj {i} vs {i+1}: {msg}", flush=True)
    
    plot_trajs = min(3, n_trajectories)
    plot_traj_indices = np.random.choice(n_trajectories, plot_trajs, replace=False)
    plot_traj_indices.sort()
    
    for traj_idx in plot_traj_indices:
        output_file = fname.replace('.nc', f'_traj{traj_idx}_snapshots.png')
        try:
            plot_snapshots(dataset, traj_idx, spatial_shape, n_snapshots, temporal_vars, is_complex, output_file)
            print(f"  Plot saved: {output_file}", flush=True)
        except Exception as e:
            print(f"  ERROR plotting traj {traj_idx}: {e}", flush=True)
    
    dataset.close()
    print(f"Finished {fname}", flush=True)

print("\nAll files processed", flush=True)
