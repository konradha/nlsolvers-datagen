import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import re
import sys
import time
import os

from downsampling import downsample_interpolation, downsample_interpolation_3d

sys.stdout.reconfigure(line_buffering=True)
TIMESTEP_CHUNK_SIZE = 10

def create_group_key(params, info):
    pde_class = params.get('pde_class', 'unknown_pde')
    n_snapshots = info['n_snapshots']
    spatial_shape = tuple(info['spatial_shape'])
    return (pde_class, n_snapshots, spatial_shape)

def discover_and_group_files(base_path, pde_type):
    print(f"Starting discovery in {base_path} for PDE type {pde_type}...", flush=True)
    source_files = list(Path(base_path).glob(f"**/{pde_type}/**/*.h5"))
    groups = defaultdict(list)
    total_files = len(source_files)
    for i, file_path in enumerate(source_files):
        if (i + 1) % 100 == 0:
            print(f"  ..scanned {i+1}/{total_files} files...", flush=True)
        try:
            with h5py.File(file_path, 'r') as f:
                info = get_data_info_from_h5(f)
                params = extract_run_params_from_h5(f, file_path)
                group_key = create_group_key(params, info)
                groups[group_key].append(str(file_path))
        except Exception as e:
            print(f"Warning: Could not read metadata from {file_path}. Skipping. Error: {e}", file=sys.stderr, flush=True)
    print(f"Discovery complete. Found {len(groups)} distinct simulation groups.", flush=True)
    return groups

def get_data_info_from_h5(h5_file_handle):
    f = h5_file_handle
    info = {'n_snapshots': f['u'].shape[0], 'spatial_dims': f['u'].ndim - 1, 'spatial_shape': f['u'].shape[1:]}
    info['is_complex'] = np.iscomplexobj(f['u'][0])
    info['m_field'] = next((loc for loc in ['focusing/m', 'm'] if loc in f), None)
    info['c_field'] = next((loc for loc in ['anisotropy/c', 'focusing/c', 'c'] if loc in f), None)
    return info

def extract_run_params_from_h5(h5_file_handle, file_path):
    f, path = h5_file_handle, Path(file_path)
    params = {'filepath': str(file_path)}
    if 'metadata' in f: params.update(f['metadata'].attrs)
    if 'grid' in f: params.update({f'grid_{k}': v for k,v in f['grid'].attrs.items()})
    if 'time' in f: params.update({f'time_{k}': v for k,v in f['time'].attrs.items()})
    run_match = re.search(r'run_([a-f0-9]+)_(\d+)', path.name)
    if run_match: params['run_id'], params['run_sequence'] = run_match.group(1), int(run_match.group(2))
    params['pde_class'] = next((p for p in path.parts if p.endswith(('_2d', '_3d'))), None)
    return params

def maybe_downsample(data, target_shape, domain_size):
    if data is None or data.shape == target_shape:
        return data
    if not all(d is not None for d in domain_size):
        return data
    if len(data.shape) == 2:
        data_b = data[np.newaxis, :, :]
        Lx, Ly = domain_size[0], domain_size[1]
        downsampled_b = downsample_interpolation(data_b, target_shape, Lx, Ly)
    elif len(data.shape) == 3:
        data_b = data[np.newaxis, :, :, :]
        Lx, Ly, Lz = domain_size[0], domain_size[1], domain_size[2]
        downsampled_b = downsample_interpolation_3d(data_b, target_shape, Lx, Ly, Lz)
    else:
        return data
    return downsampled_b[0]

def append_trajectory_to_slice(agg_file_handle, source_path, group_key, write_index):
    _, n_snapshots, target_spatial_shape = group_key
    f_out = agg_file_handle

    with h5py.File(source_path, 'r') as f_in:
        params = extract_run_params_from_h5(f_in, source_path)
        info = get_data_info_from_h5(f_in)
        
        f_out['params'][write_index] = json.dumps(params, default=str)
        
        traj_stats = {
            'max_u': np.full(n_snapshots, -np.inf, dtype='f4'), 'min_u': np.full(n_snapshots, np.inf, dtype='f4'),
            'mean_u': np.zeros(n_snapshots, dtype='f4'), 'std_u': np.zeros(n_snapshots, dtype='f4'),
            'max_v': np.full(n_snapshots, -np.inf, dtype='f4'), 'min_v': np.full(n_snapshots, np.inf, dtype='f4'),
            'mean_v': np.zeros(n_snapshots, dtype='f4'), 'std_v': np.zeros(n_snapshots, dtype='f4'),
        }

        u_source = f_in['u']
        is_complex = info['is_complex']

        for i in range(0, n_snapshots, TIMESTEP_CHUNK_SIZE):
            end = min(i + TIMESTEP_CHUNK_SIZE, n_snapshots)
            chunk = u_source[i:end]
            
            if is_complex:
                u_chunk, v_chunk = chunk.real.astype('f4'), chunk.imag.astype('f4')
            else:
                u_chunk = chunk.astype('f4')
                v_chunk = f_in['v'][i:end] if 'v' in f_in else None
            
            f_out['u'][write_index, i:end] = u_chunk
            if v_chunk is not None:
                f_out['v'][write_index, i:end] = v_chunk
            
            spatial_axes = tuple(range(1, u_chunk.ndim))
            traj_stats['max_u'][i:end] = np.max(np.abs(u_chunk), axis=spatial_axes)
            traj_stats['min_u'][i:end] = np.min(np.abs(u_chunk), axis=spatial_axes)
            traj_stats['mean_u'][i:end] = np.mean(np.abs(u_chunk), axis=spatial_axes)
            traj_stats['std_u'][i:end] = np.std(np.abs(u_chunk), axis=spatial_axes)
            if v_chunk is not None:
                traj_stats['max_v'][i:end] = np.max(np.abs(v_chunk), axis=spatial_axes)
                traj_stats['min_v'][i:end] = np.min(np.abs(v_chunk), axis=spatial_axes)
                traj_stats['mean_v'][i:end] = np.mean(np.abs(v_chunk), axis=spatial_axes)
                traj_stats['std_v'][i:end] = np.std(np.abs(v_chunk), axis=spatial_axes)

        for stat_name, stat_data in traj_stats.items():
            f_out[stat_name][write_index] = stat_data.reshape(-1, 1)

        domain_size = params.get('grid_Lx'), params.get('grid_Ly'), params.get('grid_Lz')
        ic_group = f_in.get('initial_condition', {})
        
        data_to_process = {
            'u0': ic_group.get('u0'), 'v0': ic_group.get('v0'),
            'm': f_in.get(info.get('m_field')), 'c': f_in.get(info.get('c_field')),
        }

        for name, dset_in in data_to_process.items():
            if dset_in is not None:
                data = dset_in[:]
                downsampled_data = maybe_downsample(data, target_spatial_shape, domain_size)
                if downsampled_data is not None:
                    f_out[name][write_index] = downsampled_data.astype('f4')

def main():
    parser = argparse.ArgumentParser(description='Scalable, sequential HDF5 aggregator.')
    parser.add_argument('--base_path', required=True)
    parser.add_argument('--pde_type', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_files', type=int, default=None, help='Subsample a specific number of files for a quick sanity check.')
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    groups = discover_and_group_files(args.base_path, args.pde_type)
    
    if args.num_files is not None:
        print(f"\n--- Subsampling to a total of {args.num_files} files for sanity check ---", flush=True)
        all_files = [file for files_list in groups.values() for file in files_list]
        if len(all_files) > args.num_files:
            subsampled_files = set(np.random.choice(all_files, args.num_files, replace=False))
            
            new_groups = defaultdict(list)
            for key, file_list in groups.items():
                filtered_list = [f for f in file_list if f in subsampled_files]
                if filtered_list:
                    new_groups[key] = filtered_list
            groups = new_groups
        else:
            print(f"Requested {args.num_files} but only {len(all_files)} found. Processing all files.", flush=True)

    print("\n--- Groups to be Processed ---", flush=True)
    for i, (key, files) in enumerate(groups.items()):
        pde, snaps, shape = key
        print(f"Group {i+1}: PDE={pde}, Snapshots={snaps}, Shape={shape}, Files={len(files)}", flush=True)
    print("----------------------------\n", flush=True)

    for group_idx, (group_key, file_list) in enumerate(groups.items()):
        pde, n_snapshots, spatial_shape = group_key
        
        subsample_tag = f"_n{args.num_files}" if args.num_files is not None else ""
        final_filename = f"{pde}_s{n_snapshots}_sh{'x'.join(map(str,shape))}{subsample_tag}.h5"
        final_output_path = Path(args.output_dir) / final_filename
        
        print(f"--- Processing Group {group_idx+1}/{len(groups)} into {final_output_path} ---", flush=True)
        start_time = time.time()
        
        num_trajectories_in_group = len(file_list)

        with h5py.File(final_output_path, 'w') as f_agg:
            f_agg.attrs['pde_class'] = pde
            f_agg.attrs['num_snapshots'] = n_snapshots
            f_agg.attrs['spatial_shape'] = str(spatial_shape)
            
            f_agg.create_dataset('u', shape=(num_trajectories_in_group, n_snapshots, *spatial_shape), dtype='f4', chunks=True)
            f_agg.create_dataset('v', shape=(num_trajectories_in_group, n_snapshots, *spatial_shape), dtype='f4', chunks=True)
            for stat in ['max', 'min', 'mean', 'std']:
                f_agg.create_dataset(f'{stat}_u', shape=(num_trajectories_in_group, n_snapshots, 1), dtype='f4', chunks=True)
                f_agg.create_dataset(f'{stat}_v', shape=(num_trajectories_in_group, n_snapshots, 1), dtype='f4', chunks=True)
            f_agg.create_dataset('u0', shape=(num_trajectories_in_group, *spatial_shape), dtype='f4', chunks=True)
            f_agg.create_dataset('v0', shape=(num_trajectories_in_group, *spatial_shape), dtype='f4', chunks=True)
            f_agg.create_dataset('m', shape=(num_trajectories_in_group, *spatial_shape), dtype='f4', chunks=True)
            f_agg.create_dataset('c', shape=(num_trajectories_in_group, *spatial_shape), dtype='f4', chunks=True)
            f_agg.create_dataset('params', shape=(num_trajectories_in_group,), dtype=h5py.string_dtype(encoding='utf-8'), chunks=True)

            processed_count = 0
            for i, source_path in enumerate(file_list):
                try:
                    append_trajectory_to_slice(f_agg, source_path, group_key, write_index=i)
                    processed_count += 1
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i+1}/{len(file_list)} files...", flush=True)
                except Exception as e:
                    print(f"ERROR processing {source_path}: {e}", file=sys.stderr, flush=True)
            
            f_agg.attrs['num_trajectories'] = processed_count
            if processed_count != num_trajectories_in_group:
                print(f"Warning: Expected {num_trajectories_in_group} trajectories but only processed {processed_count}. Trimming file.", flush=True)
                for dset in f_agg.values():
                    dset.resize(processed_count, axis=0)

        end_time = time.time()
        print(f"Finished group in {end_time - start_time:.2f}s.\n", flush=True)

    print("All groups processed. Aggregation complete.", flush=True)

if __name__ == '__main__':
    main()
