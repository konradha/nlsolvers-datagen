import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse

def get_dataset_info(dataset):
    n_trajectories = dataset.dimensions['member'].size
    n_snapshots = dataset.dimensions['time'].size
    
    spatial_dims = []
    for dim in ['x', 'y', 'z']:
        if dim in dataset.dimensions:
            spatial_dims.append(dataset.dimensions[dim].size)
    spatial_shape = tuple(spatial_dims)
    
    is_complex = 're_u' in dataset.variables
    
    if is_complex:
        temporal_vars = ['re_u', 'im_u']
    else:
        temporal_vars = ['u', 'v']
    
    return temporal_vars, n_trajectories, n_snapshots, spatial_shape, is_complex

def check_trajectory(dataset, member_idx, spatial_shape, n_snapshots, temporal_vars):
    for var_name in temporal_vars:
        var = dataset.variables[var_name]
        data = var[member_idx, :, ...]
        
        if np.any(np.isnan(data)):
            return False, f"NaN in {var_name}, member {member_idx}"
        if np.all(data == data.flat[0]):
            return False, f"Constant {var_name}, member {member_idx}"
    
    return True, "OK"

def compare_members(dataset, member_a, member_b, temporal_vars, sample_size=1000):
    for var_name in temporal_vars:
        var = dataset.variables[var_name]
        data_a = var[member_a, :, ...].flatten()
        data_b = var[member_b, :, ...].flatten()
        
        n_elements = len(data_a)
        indices = np.random.choice(n_elements, min(sample_size, n_elements), replace=False)
        
        if np.allclose(data_a[indices], data_b[indices]):
            return False, f"Identical {var_name}, member {member_a} vs {member_b}"
    
    return True, "Different"

def plot_snapshots(dataset, member_idx, spatial_shape, n_snapshots, temporal_vars, is_complex, output_file):
    snapshot_indices = [0, n_snapshots // 2, n_snapshots - 1]
    
    if is_complex:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = axes.reshape(1, -1)
        
        for j, snap_idx in enumerate(snapshot_indices):
            re_data = dataset.variables['re_u'][member_idx, snap_idx, ...]
            im_data = dataset.variables['im_u'][member_idx, snap_idx, ...]
            data = np.sqrt(re_data**2 + im_data**2)
            
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
                data = dataset.variables[var_name][member_idx, snap_idx, ...]
                
                if len(spatial_shape) == 3:
                    data = data[:, :, spatial_shape[2] // 2]
                
                axes[i, j].imshow(data, cmap='viridis')
                axes[i, j].set_title(f"{var_name} t={snap_idx}")
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Sanity check for NetCDF datasets')
    parser.add_argument('file', type=str, help='Path to NetCDF file')
    args = parser.parse_args()
    
    fname = args.file
    print(f"\n{fname}", flush=True)
    
    try:
        dataset = nc.Dataset(fname, 'r')
    except Exception as e:
        print(f"  ERROR: Could not open file: {e}", flush=True)
        sys.exit(1)
    
    temporal_vars, n_trajectories, n_snapshots, spatial_shape, is_complex = get_dataset_info(dataset)
    
    print(f"Members: {n_trajectories}, Snapshots: {n_snapshots}, Spatial shape: {spatial_shape}", flush=True)
    print(f"Temporal vars: {temporal_vars}, Complex: {is_complex}", flush=True)
    
    check_members = min(10, n_trajectories)
    sampled_member_indices = np.random.choice(n_trajectories, check_members, replace=False)
    sampled_member_indices.sort()
    
    for member_idx in sampled_member_indices:
        valid, msg = check_trajectory(dataset, member_idx, spatial_shape, n_snapshots, temporal_vars)
        if not valid:
            print(f"  FAIL: {msg}", flush=True)
        else:
            print(f"  Member {member_idx}: {msg}", flush=True)
    
    if n_trajectories > 1:
        pairs = min(5, n_trajectories - 1)
        pair_indices = np.random.choice(n_trajectories - 1, pairs, replace=False)
        
        for i in pair_indices:
            different, msg = compare_members(dataset, i, i + 1, temporal_vars)
            if not different:
                print(f"  FAIL: {msg}", flush=True)
            else:
                print(f"  Member {i} vs {i+1}: {msg}", flush=True)
    
    plot_members = min(3, n_trajectories)
    plot_member_indices = np.random.choice(n_trajectories, plot_members, replace=False)
    plot_member_indices.sort()
    
    for member_idx in plot_member_indices:
        output_file = fname.replace('.nc', f'_member{member_idx}_snapshots.png')
        try:
            plot_snapshots(dataset, member_idx, spatial_shape, n_snapshots, temporal_vars, is_complex, output_file)
            print(f"  Plot saved: {output_file}", flush=True)
        except Exception as e:
            print(f"  ERROR plotting member {member_idx}: {e}", flush=True)
    
    dataset.close()
    print(f"Finished {fname}", flush=True)

if __name__ == '__main__':
    main()
