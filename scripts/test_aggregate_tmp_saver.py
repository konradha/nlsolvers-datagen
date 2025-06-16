import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import re
import sys
from mpi4py import MPI
import time
import os

from downsampling import downsample_interpolation, downsample_interpolation_3d

sys.stdout.reconfigure(line_buffering=True)

def find_h5_files(base_dir, pde_type):
    base_path = Path(base_dir)
    pattern = f"**/{pde_type}/**/*.h5"
    found = list(base_path.glob(pattern))
    found = sorted(list(set(found)))
    return found

def extract_run_params_from_h5(h5_file):
    params = {}
    
    with h5py.File(h5_file, 'r') as f:
        if 'metadata' in f:
            meta = f['metadata']
            for key in meta.attrs.keys():
                params[key] = meta.attrs[key]
        
        if 'grid' in f:
            grid = f['grid']
            for key in grid.attrs.keys():
                params[f'grid_{key}'] = grid.attrs[key]
        
        if 'time' in f:
            time_grp = f['time']
            for key in time_grp.attrs.keys():
                params[f'time_{key}'] = time_grp.attrs[key]
        
        if 'focusing' in f:
            focusing = f['focusing']
            for key in focusing.attrs.keys():
                params[f'focusing_{key}'] = focusing.attrs[key]
    
    path = Path(h5_file)
    run_match = re.search(r'run_([a-f0-9]+)_(\d+)', path.name)
    if run_match:
        params['run_id'] = run_match.group(1)
        params['run_sequence'] = int(run_match.group(2))
    
    parts = path.parts
    for part in parts:
        if part.endswith('_2d') or part.endswith('_3d'):
            params['pde_class'] = part
            break
    
    for part in parts:
        if part.startswith('c_') or part.startswith('m_'):
            params['method'] = part
        elif 'breather' in part or 'soliton' in part or 'ring' in part:
            params['phenomenon_dir'] = part
    
    params['filepath'] = str(h5_file)
    return params

def get_data_info_from_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        has_u = 'u' in f
        has_v = 'v' in f
        has_u0 = 'initial_condition' in f and 'u0' in f['initial_condition']
        has_v0 = 'initial_condition' in f and 'v0' in f['initial_condition']
        
        m_field = None
        c_field = None
        
        m_options = ['focusing/m', 'm']
        for loc in m_options:
            if loc in f:
                m_field = loc
                break
        
        c_options = ['anisotropy/c', 'focusing/c', 'c']
        for loc in c_options:
            if loc in f:
                c_field = loc
                break
        
        if not has_u:
            raise ValueError(f"No 'u' dataset found in {h5_file}")
        
        u_shape = f['u'].shape
        n_snapshots = u_shape[0]
        spatial_dims = len(u_shape) - 1
        spatial_shape = u_shape[1:]
        
        u_sample = f['u'][0] if n_snapshots > 0 else f['u'][:]
        is_complex = np.iscomplexobj(u_sample)
        
        domain_size = None
        if 'grid' in f:
            grid = f['grid']
            if 'Lx' in grid.attrs:
                if spatial_dims == 2:
                    domain_size = (float(grid.attrs['Lx']), float(grid.attrs['Ly']))
                elif spatial_dims == 3:
                    domain_size = (float(grid.attrs['Lx']), float(grid.attrs['Ly']), float(grid.attrs['Lz']))
        
        return {
            'n_snapshots': n_snapshots,
            'spatial_dims': spatial_dims,
            'spatial_shape': spatial_shape,
            'is_complex': is_complex,
            'has_v': has_v,
            'has_u0': has_u0,
            'has_v0': has_v0,
            'u_shape': u_shape,
            'm_field': m_field,
            'c_field': c_field,
            'domain_size': domain_size
        }

def load_trajectory_data_from_h5(h5_file, info):
    with h5py.File(h5_file, 'r') as f:
        u_data = f['u'][:]
        v_data = f['v'][:] if info['has_v'] else None
        
        u0 = None
        v0 = None
        if 'initial_condition' in f:
            ic_group = f['initial_condition']
            if 'u0' in ic_group:
                u0 = ic_group['u0'][:]
            if 'v0' in ic_group:
                v0 = ic_group['v0'][:]
        
        m_data = None
        c_data = None
        
        if info['m_field'] and info['m_field'] in f:
            m_data = f[info['m_field']][:]
        
        if info['c_field'] and info['c_field'] in f:
            c_data = f[info['c_field']][:]
        
        if info['is_complex']:
            u_out = u_data.real
            v_out = u_data.imag
            is_complex_flag = True
        else:
            u_out = u_data
            v_out = v_data
            is_complex_flag = False
        
        target_shape = info['spatial_shape']
        
        if info['domain_size']:
            spatial_dims = info['spatial_dims']
            domain_size = info['domain_size']
            
            if spatial_dims == 2:
                Lx, Ly = domain_size
                if u0 is not None and u0.shape != target_shape:
                    u0_3d = u0[np.newaxis, :, :]
                    u0_down = downsample_interpolation(u0_3d, target_shape, Lx, Ly)
                    u0 = u0_down[0, :, :]
                if v0 is not None and v0.shape != target_shape:
                    v0_3d = v0[np.newaxis, :, :]
                    v0_down = downsample_interpolation(v0_3d, target_shape, Lx, Ly)
                    v0 = v0_down[0, :, :]
                if m_data is not None and m_data.shape != target_shape:
                    m_3d = m_data[np.newaxis, :, :]
                    m_down = downsample_interpolation(m_3d, target_shape, Lx, Ly)
                    m_data = m_down[0, :, :]
                if c_data is not None and c_data.shape != target_shape:
                    c_3d = c_data[np.newaxis, :, :]
                    c_down = downsample_interpolation(c_3d, target_shape, Lx, Ly)
                    c_data = c_down[0, :, :]
            elif spatial_dims == 3:
                Lx, Ly, Lz = domain_size
                if u0 is not None and u0.shape != target_shape:
                    u0_4d = u0[np.newaxis, :, :, :]
                    u0_down = downsample_interpolation_3d(u0_4d, target_shape, Lx, Ly, Lz)
                    u0 = u0_down[0, :, :, :]
                if v0 is not None and v0.shape != target_shape:
                    v0_4d = v0[np.newaxis, :, :, :]
                    v0_down = downsample_interpolation_3d(v0_4d, target_shape, Lx, Ly, Lz)
                    v0 = v0_down[0, :, :, :]
                if m_data is not None and m_data.shape != target_shape:
                    m_4d = m_data[np.newaxis, :, :, :]
                    m_down = downsample_interpolation_3d(m_4d, target_shape, Lx, Ly, Lz)
                    m_data = m_down[0, :, :, :]
                if c_data is not None and c_data.shape != target_shape:
                    c_4d = c_data[np.newaxis, :, :, :]
                    c_down = downsample_interpolation_3d(c_4d, target_shape, Lx, Ly, Lz)
                    c_data = c_down[0, :, :, :]
        
        return u_out, v_out, u0, v0, m_data, c_data, is_complex_flag

def has_nan_values(*arrays):
    for arr in arrays:
        if arr is not None and np.any(np.isnan(arr)):
            return True
    return False

def compute_trajectory_stats(u_data, v_data=None):
    spatial_axes = tuple(range(1, len(u_data.shape)))
    
    u_abs = np.abs(u_data)
    u_max = np.max(u_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
    u_min = np.min(u_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
    u_mean = np.mean(u_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
    u_std = np.std(u_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
    
    stats = {
        'max_u': u_max,
        'min_u': u_min,
        'mean_u': u_mean,
        'std_u': u_std
    }
    
    if v_data is not None:
        v_abs = np.abs(v_data)
        v_max = np.max(v_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
        v_min = np.min(v_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
        v_mean = np.mean(v_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
        v_std = np.std(v_abs, axis=spatial_axes, keepdims=True).reshape(-1, 1)
        
        stats.update({
            'max_v': v_max,
            'min_v': v_min,
            'mean_v': v_mean,
            'std_v': v_std
        })
    
    return stats

def compute_global_stats(u_data, v_data=None):
    u_abs = np.abs(u_data)
    stats = {
        'global_min_u': float(np.min(u_abs)),
        'global_max_u': float(np.max(u_abs)),
        'global_mean_u': float(np.mean(u_abs)),
        'global_std_u': float(np.std(u_abs))
    }
    
    if v_data is not None:
        v_abs = np.abs(v_data)
        stats.update({
            'global_min_v': float(np.min(v_abs)),
            'global_max_v': float(np.max(v_abs)),
            'global_mean_v': float(np.mean(v_abs)),
            'global_std_v': float(np.std(v_abs))
        })
    
    return stats

def create_group_key(info, pde_class):
    return (
        pde_class,
        info['n_snapshots'], 
        info['spatial_dims'], 
        info['spatial_shape'],
        info['is_complex'],
        info['has_v']
    )

def process_and_write_rank_file(files_batch, scratch_file, group_key):
    pde_class, n_snapshots, spatial_dims, spatial_shape, is_complex, has_v = group_key
    
    valid_trajectories = []
    
    for h5_file, info, params in files_batch:
        try:
            t = -time.time()
            u_data, v_data, u0, v0, m_data, c_data, is_complex_flag = load_trajectory_data_from_h5(h5_file, info)
            
            if has_nan_values(u_data, v_data, u0, v0, m_data, c_data):
                continue
            
            stats = compute_trajectory_stats(u_data, v_data)
            
            valid_trajectories.append({
                'u_data': np.ascontiguousarray(u_data),
                'v_data': np.ascontiguousarray(v_data) if v_data is not None else None,
                'u0': np.ascontiguousarray(u0) if u0 is not None else None,
                'v0': np.ascontiguousarray(v0) if v0 is not None else None,
                'm_data': np.ascontiguousarray(m_data) if m_data is not None else None,
                'c_data': np.ascontiguousarray(c_data) if c_data is not None else None,
                'is_complex': is_complex_flag,
                'stats': stats,
                'params': params
            })
            t += time.time()
            print(h5_file, f"{t:.2f}s")
            
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue
    
    if not valid_trajectories:
        return 0
    
    u_stack = [traj['u_data'] for traj in valid_trajectories]
    u_array = np.stack(u_stack, axis=0)
    
    v_array = None
    v_stack = [traj['v_data'] for traj in valid_trajectories if traj['v_data'] is not None]
    if v_stack:
        v_array = np.stack(v_stack, axis=0)
    
    u0_stack = [traj['u0'] for traj in valid_trajectories if traj['u0'] is not None]
    u0_array = np.stack(u0_stack, axis=0) if u0_stack else None
    
    v0_stack = [traj['v0'] for traj in valid_trajectories if traj['v0'] is not None]
    v0_array = np.stack(v0_stack, axis=0) if v0_stack else None
    
    m_stack = [traj['m_data'] for traj in valid_trajectories if traj['m_data'] is not None]
    m_array = np.stack(m_stack, axis=0) if m_stack else None
    
    c_stack = [traj['c_data'] for traj in valid_trajectories if traj['c_data'] is not None]
    c_array = np.stack(c_stack, axis=0) if c_stack else None
    
    is_complex_array = np.array([traj['is_complex'] for traj in valid_trajectories])
    
    max_u_array = np.stack([traj['stats']['max_u'] for traj in valid_trajectories], axis=0)
    min_u_array = np.stack([traj['stats']['min_u'] for traj in valid_trajectories], axis=0)
    mean_u_array = np.stack([traj['stats']['mean_u'] for traj in valid_trajectories], axis=0)
    std_u_array = np.stack([traj['stats']['std_u'] for traj in valid_trajectories], axis=0)
    
    max_v_array = None
    min_v_array = None
    mean_v_array = None
    std_v_array = None
    
    if v_array is not None:
        max_v_array = np.stack([traj['stats']['max_v'] for traj in valid_trajectories if 'max_v' in traj['stats']], axis=0)
        min_v_array = np.stack([traj['stats']['min_v'] for traj in valid_trajectories if 'min_v' in traj['stats']], axis=0)
        mean_v_array = np.stack([traj['stats']['mean_v'] for traj in valid_trajectories if 'mean_v' in traj['stats']], axis=0)
        std_v_array = np.stack([traj['stats']['std_v'] for traj in valid_trajectories if 'std_v' in traj['stats']], axis=0)
    
    global_stats = compute_global_stats(u_array, v_array)
    params_json = [json.dumps(traj['params'], default=str) for traj in valid_trajectories]
    
    with h5py.File(scratch_file, 'w') as f:
        f.create_dataset('u', data=u_array, compression='gzip', compression_opts=9)
        if v_array is not None:
            f.create_dataset('v', data=v_array, compression='gzip', compression_opts=9)
        
        f.create_dataset('is_complex', data=is_complex_array, compression='gzip', compression_opts=9)
        
        f.create_dataset('max_u', data=max_u_array, compression='gzip', compression_opts=9)
        f.create_dataset('min_u', data=min_u_array, compression='gzip', compression_opts=9)
        f.create_dataset('mean_u', data=mean_u_array, compression='gzip', compression_opts=9)
        f.create_dataset('std_u', data=std_u_array, compression='gzip', compression_opts=9)
        
        if max_v_array is not None:
            f.create_dataset('max_v', data=max_v_array, compression='gzip', compression_opts=9)
            f.create_dataset('min_v', data=min_v_array, compression='gzip', compression_opts=9)
            f.create_dataset('mean_v', data=mean_v_array, compression='gzip', compression_opts=9)
            f.create_dataset('std_v', data=std_v_array, compression='gzip', compression_opts=9)
        
        f.create_dataset('params', data=params_json, dtype=h5py.string_dtype(encoding='utf-8'))
        
        if u0_array is not None:
            f.create_dataset('u0', data=u0_array, compression='gzip', compression_opts=9)
        
        if v0_array is not None:
            f.create_dataset('v0', data=v0_array, compression='gzip', compression_opts=9)
        
        if m_array is not None:
            f.create_dataset('m', data=m_array, compression='gzip', compression_opts=9)
        
        if c_array is not None:
            f.create_dataset('c', data=c_array, compression='gzip', compression_opts=9)
        
        f.attrs['num_trajectories'] = len(valid_trajectories)
        f.attrs['rank_global_stats'] = json.dumps(global_stats)
    
    return len(valid_trajectories)

def merge_rank_files(scratch_files, output_file, group_key):
    pde_class, n_snapshots, spatial_dims, spatial_shape, is_complex, has_v = group_key
    
    all_datasets = defaultdict(list)
    all_global_stats = []
    total_trajectories = 0
    
    for scratch_file in scratch_files:
        if not os.path.exists(scratch_file):
            continue
            
        with h5py.File(scratch_file, 'r') as f:
            if f.attrs['num_trajectories'] == 0:
                continue
                
            total_trajectories += f.attrs['num_trajectories']
            
            rank_stats = json.loads(f.attrs['rank_global_stats'])
            all_global_stats.append(rank_stats)
            
            for key in f.keys():
                if key in f:
                    all_datasets[key].append(f[key][:])
    
    if total_trajectories == 0:
        return
    
    merged_global_stats = {}
    if all_global_stats:
        all_u_data = []
        all_v_data = []
        for stats in all_global_stats:
            all_u_data.extend([stats['global_min_u'], stats['global_max_u']])
            if 'global_min_v' in stats:
                all_v_data.extend([stats['global_min_v'], stats['global_max_v']])
        
        merged_global_stats['global_min_u'] = float(min(all_u_data))
        merged_global_stats['global_max_u'] = float(max(all_u_data))
        if all_v_data:
            merged_global_stats['global_min_v'] = float(min(all_v_data))
            merged_global_stats['global_max_v'] = float(max(all_v_data))
    
    with h5py.File(output_file, 'w') as f:
        for key, arrays in all_datasets.items():
            if arrays:
                merged_array = np.concatenate(arrays, axis=0)
                f.create_dataset(key, data=merged_array, compression='gzip', compression_opts=9)
        
        f.attrs['num_trajectories'] = total_trajectories
        f.attrs['num_snapshots'] = n_snapshots
        f.attrs['spatial_dims'] = spatial_dims
        f.attrs['spatial_shape'] = spatial_shape
        f.attrs['pde_class'] = pde_class
        
        for key, value in merged_global_stats.items():
            f.attrs[key] = value

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description='Memory-efficient MPI PDE trajectory aggregator')
    parser.add_argument('--base_path', required=True, help='Base path containing PDE data')
    parser.add_argument('--pde_type', required=True, help='PDE type (e.g., nlse_2d)')
    parser.add_argument('--output_file', required=True, help='Output HDF5 file path')
    parser.add_argument('--scratch_dir', required=True, help='Scratch directory for temporary files')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process only a few files')

    # master gathers initial setup for all ranks to treat different trajectories
    if rank == 0:
        args = parser.parse_args()
        if not Path(args.base_path).exists():
            print(f"Error: Base path {args.base_path} does not exist", file=sys.stderr)
            comm.Abort(1)
        if not Path(args.scratch_dir).exists():
            os.makedirs(args.scratch_dir, exist_ok=True)
            print(f"Created scratch directory: {args.scratch_dir}")

        start_time = time.time()
        all_h5_files = find_h5_files(args.base_path, args.pde_type)

        if args.debug:
            all_h5_files = np.random.choice(all_h5_files, size, replace=False)

        if all_h5_files.size == 0:
            print(f"No .h5 files found for PDE type: {args.pde_type}", file=sys.stderr)
            comm.Abort(1)

        print(f"Found {len(all_h5_files)} total files. Analyzing to find consistent groups...")
        trajectory_groups = defaultdict(list)
        for h5_file in all_h5_files:
            try:
                info = get_data_info_from_h5(h5_file)
                params = extract_run_params_from_h5(h5_file)
                pde_class = params.get('pde_class', args.pde_type)
                group_key = create_group_key(info, pde_class)
                trajectory_groups[group_key].append(h5_file)
            except Exception as e:
                print(f"Warning: Could not analyze {h5_file}: {e}", file=sys.stderr)
        
        if not trajectory_groups:
            print("No valid trajectory groups found.", file=sys.stderr)
            comm.Abort(1)
        
        print(f"Found {len(trajectory_groups)} distinct trajectory groups.")
        largest_group = max(trajectory_groups.items(), key=lambda item: len(item[1]))
        group_key, files_to_process = largest_group

        if args.debug:
            files_to_process = files_to_process[:size*2]
            print(f"DEBUG MODE: Processing only {len(files_to_process)} files.")

        pde_class, n_snapshots, spatial_dims, spatial_shape, _, _ = group_key
        print(f"Processing largest group: {pde_class}, shape {spatial_shape}, {len(files_to_process)} files")

        files_per_rank = [files_to_process[i::size] for i in range(size)]

    else:
        args = None
        group_key = None
        files_per_rank = None
    args = comm.bcast(args, root=0)
    group_key = comm.bcast(group_key, root=0)
    my_files = comm.scatter(files_per_rank, root=0)
    my_file_infos = []
    for h5_file in my_files:
        try:
            info = get_data_info_from_h5(h5_file)
            params = extract_run_params_from_h5(h5_file)
            my_file_infos.append((h5_file, info, params))
        except Exception as e:
            print(f"Rank {rank} skipping {h5_file} due to error: {e}", file=sys.stderr)

    scratch_file = Path(args.scratch_dir) / f"rank_{rank:04d}.h5"
    num_processed = process_and_write_rank_file(my_file_infos, scratch_file, group_key)
    
    all_counts = comm.gather(num_processed, root=0)
    comm.Barrier()

    if rank == 0:
        total_processed = sum(all_counts)
        print(f"All ranks processed {total_processed} trajectories total.")

        if total_processed > 0:
            all_scratch_files = [Path(args.scratch_dir) / f"rank_{r:04d}.h5" for r in range(size)]
            merge_rank_files(all_scratch_files, args.output_file, group_key)
            end_time = time.time()
            print(f"Merged into {args.output_file} in {end_time - start_time:.2f}s")
        else:
            print("No valid trajectories were processed.")
        for scratch_f in all_scratch_files:
            if os.path.exists(scratch_f):
                os.remove(scratch_f)
        print("Cleaned up scratch files.")

    comm.Barrier()

if __name__ == '__main__':
    main()
