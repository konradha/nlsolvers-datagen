#!/usr/bin/env python3

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
import pickle
import gc
from mpi4py import MPI

from downsampling import downsample_interpolation, downsample_interpolation_3d

sys.stdout.reconfigure(line_buffering=True)

SPATIAL_CHUNK_SIZE = 32
TEMPORAL_CHUNK_SIZE = 5
STREAM_BUFFER_SIZE = 10

def create_group_key(params, info):
    pde_class = params.get('pde_class', 'unknown_pde')
    spatial_shape = tuple(info['spatial_shape'])
    return (pde_class, spatial_shape)

def create_temporal_subkey(n_snapshots):
    return n_snapshots

def discover_and_group_files(base_path, pde_type, comm, rank):
    if rank == 0:
        print(f"Phase 1: Starting discovery in {base_path} for PDE type {pde_type}...", flush=True)
        source_files = list(Path(base_path).glob(f"**/{pde_type}/**/*.h5"))

        nested_groups = defaultdict(lambda: defaultdict(list))
        total_files = len(source_files)

        for i, file_path in enumerate(source_files):
            if (i + 1) % 100 == 0:
                print(f"  Scanned {i+1}/{total_files} files...", flush=True)
            try:
                with h5py.File(file_path, 'r') as f:
                    info = get_data_info_from_h5(f)
                    params = extract_run_params_from_h5(f, file_path)
                    group_key = create_group_key(params, info)
                    temporal_key = create_temporal_subkey(info['n_snapshots'])

                    file_metadata = {
                        'path': str(file_path),
                        'info': info,
                        'params': params
                    }
                    nested_groups[group_key][temporal_key].append(file_metadata)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr, flush=True)

        print(f"Discovery complete. Found {len(nested_groups)} spatial groups.", flush=True)
        return dict(nested_groups)
    else:
        return None

def get_data_info_from_h5(h5_file_handle):
    f = h5_file_handle
    info = {
        'n_snapshots': f['u'].shape[0],
        'spatial_dims': f['u'].ndim - 1,
        'spatial_shape': f['u'].shape[1:]
    }
    info['is_complex'] = np.iscomplexobj(f['u'][0])
    info['m_field'] = next((loc for loc in ['focusing/m', 'm'] if loc in f), None)
    info['c_field'] = next((loc for loc in ['anisotropy/c', 'focusing/c', 'c'] if loc in f), None)
    return info

def extract_run_params_from_h5(h5_file_handle, file_path):
    f, path = h5_file_handle, Path(file_path)
    params = {'filepath': str(file_path)}

    if 'metadata' in f:
        params.update(f['metadata'].attrs)
    if 'grid' in f:
        params.update({f'grid_{k}': v for k, v in f['grid'].attrs.items()})
    if 'time' in f:
        params.update({f'time_{k}': v for k, v in f['time'].attrs.items()})

    run_match = re.search(r'run_([a-f0-9]+)_(\d+)', path.name)
    if run_match:
        params['run_id'] = run_match.group(1)
        params['run_sequence'] = int(run_match.group(2))

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

def create_rank_output_file(output_dir, rank, group_key, temporal_key, num_trajectories, target_spatial_shape):
    pde_class, spatial_shape = group_key
    n_snapshots = temporal_key

    filename = f"rank_{rank}_{pde_class}_s{n_snapshots}_sh{'x'.join(map(str, spatial_shape))}.h5"
    filepath = Path(output_dir) / "streaming" / filename
    filepath.parent.mkdir(exist_ok=True, parents=True)

    spatial_chunks = tuple(min(SPATIAL_CHUNK_SIZE, dim) for dim in target_spatial_shape)
    temporal_chunks = min(TEMPORAL_CHUNK_SIZE, n_snapshots)
    traj_chunks = min(5, num_trajectories)

    with h5py.File(filepath, 'w') as f:
        f.attrs['rank'] = rank
        f.attrs['pde_class'] = pde_class
        f.attrs['num_snapshots'] = n_snapshots
        f.attrs['spatial_shape'] = str(spatial_shape)
        f.attrs['target_spatial_shape'] = str(target_spatial_shape)
        f.attrs['num_trajectories'] = num_trajectories

        f.create_dataset('u', shape=(num_trajectories, n_snapshots, *target_spatial_shape),
                        dtype='f4', chunks=(traj_chunks, temporal_chunks, *spatial_chunks))
        f.create_dataset('v', shape=(num_trajectories, n_snapshots, *target_spatial_shape),
                        dtype='f4', chunks=(traj_chunks, temporal_chunks, *spatial_chunks))

        for stat in ['max', 'min', 'mean', 'std']:
            f.create_dataset(f'{stat}_u', shape=(num_trajectories, n_snapshots),
                           dtype='f4', chunks=(traj_chunks, temporal_chunks))
            f.create_dataset(f'{stat}_v', shape=(num_trajectories, n_snapshots),
                           dtype='f4', chunks=(traj_chunks, temporal_chunks))

        f.create_dataset('u0', shape=(num_trajectories, *target_spatial_shape),
                        dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f.create_dataset('v0', shape=(num_trajectories, *target_spatial_shape),
                        dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f.create_dataset('m', shape=(num_trajectories, *target_spatial_shape),
                        dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f.create_dataset('c', shape=(num_trajectories, *target_spatial_shape),
                        dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f.create_dataset('params', shape=(num_trajectories,),
                        dtype=h5py.string_dtype(encoding='utf-8'), chunks=(traj_chunks,))

    return filepath

def process_single_trajectory_true_streaming(f_out, file_metadata, write_index, target_spatial_shape):
    source_path = file_metadata['path']
    info = file_metadata['info']
    params = file_metadata['params']
    n_snapshots = info['n_snapshots']

    with h5py.File(source_path, 'r') as f_in:
        f_out['params'][write_index] = json.dumps(params, default=str)
        
        u_source = f_in['u']
        is_complex = info['is_complex']
        
        for t in range(n_snapshots):
            u_frame = u_source[t]

            if is_complex:
                u_data = u_frame.real.astype('f4')
                v_data = u_frame.imag.astype('f4')
            else:
                u_data = u_frame.astype('f4')
                v_data = f_in['v'][t].astype('f4') if 'v' in f_in else np.zeros_like(u_data)

            f_out['u'][write_index, t] = u_data
            f_out['v'][write_index, t] = v_data

            u_abs = np.abs(u_data)
            v_abs = np.abs(v_data)

            f_out['max_u'][write_index, t] = np.max(u_abs)
            f_out['min_u'][write_index, t] = np.min(u_abs)
            f_out['mean_u'][write_index, t] = np.mean(u_abs)
            f_out['std_u'][write_index, t] = np.std(u_abs)

            f_out['max_v'][write_index, t] = np.max(v_abs)
            f_out['min_v'][write_index, t] = np.min(v_abs)
            f_out['mean_v'][write_index, t] = np.mean(v_abs)
            f_out['std_v'][write_index, t] = np.std(v_abs)

            del u_data, v_data, u_abs, v_abs, u_frame
            gc.collect()

        domain_size = (params.get('grid_Lx'), params.get('grid_Ly'), params.get('grid_Lz'))
        ic_group = f_in.get('initial_condition', {})

        data_to_process = {
            'u0': ic_group.get('u0'),
            'v0': ic_group.get('v0'),
            'm': f_in.get(info.get('m_field')) if info.get('m_field') else None,
            'c': f_in.get(info.get('c_field')) if info.get('c_field') else None,
        }

        for name, dset_in in data_to_process.items():
            if dset_in is not None:
                data = dset_in[:]
                downsampled_data = maybe_downsample(data, target_spatial_shape, domain_size)
                if downsampled_data is not None:
                    f_out[name][write_index] = downsampled_data.astype('f4')
                del data
                if downsampled_data is not None:
                    del downsampled_data

def distribute_files_across_ranks(nested_groups, comm, rank, size):
    if rank == 0:
        all_assignments = []
        
        for group_key, temporal_groups in nested_groups.items():
            for temporal_key, file_list in temporal_groups.items():
                num_files = len(file_list)
                
                if num_files >= size:
                    files_per_rank = num_files // size
                    remainder = num_files % size
                    
                    start_idx = 0
                    for r in range(size):
                        chunk_size = files_per_rank + (1 if r < remainder else 0)
                        if chunk_size > 0:
                            chunk_files = file_list[start_idx:start_idx + chunk_size]
                            assignment = {
                                'group_key': group_key,
                                'temporal_key': temporal_key,
                                'files': chunk_files
                            }
                            all_assignments.append((r, assignment))
                            start_idx += chunk_size
                else:
                    for i, file_metadata in enumerate(file_list):
                        target_rank = i % size
                        assignment = {
                            'group_key': group_key,
                            'temporal_key': temporal_key,
                            'files': [file_metadata]
                        }
                        all_assignments.append((target_rank, assignment))
        
        rank_assignments = [[] for _ in range(size)]
        for target_rank, assignment in all_assignments:
            rank_assignments[target_rank].append(assignment)
            
        print(f"Distribution: {[len(assignments) for assignments in rank_assignments]} tasks per rank", flush=True)
    else:
        rank_assignments = None
    
    my_assignments = comm.scatter(rank_assignments, root=0)
    return my_assignments if my_assignments is not None else []

def streaming_processing_phase(my_assignments, output_dir, rank):
    print(f"Phase 2: Rank {rank} starting true streaming processing...", flush=True)
    
    rank_files = []
    
    for assignment in my_assignments:
        group_key = assignment['group_key']
        temporal_key = assignment['temporal_key']
        files = assignment['files']
        target_spatial_shape = files[0]['info']['spatial_shape']
        num_trajectories = len(files)
        
        pde_class, spatial_shape = group_key
        print(f"Rank {rank}: Processing {num_trajectories} trajectories for {pde_class}, {temporal_key} snapshots", flush=True)
        
        rank_file_path = create_rank_output_file(output_dir, rank, group_key, temporal_key, num_trajectories, target_spatial_shape)
        
        with h5py.File(rank_file_path, 'a') as f_out:
            for i, file_metadata in enumerate(files):
                try:
                    process_single_trajectory_true_streaming(f_out, file_metadata, i, target_spatial_shape)
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Rank {rank}: Processed {i+1}/{num_trajectories} trajectories", flush=True)
                        
                except Exception as e:
                    print(f"Rank {rank} ERROR processing {file_metadata['path']}: {e}", file=sys.stderr, flush=True)
        
        rank_files.append({
            'filepath': str(rank_file_path),
            'group_key': group_key,
            'temporal_key': temporal_key,
            'processed_count': num_trajectories
        })
    
    print(f"Rank {rank}: Completed streaming processing", flush=True)
    return rank_files

def streaming_final_aggregation(output_dir, rank):
    if rank != 0:
        return

    print("Phase 3: Starting streaming aggregation...", flush=True)

    streaming_dir = Path(output_dir) / "streaming"
    rank_files = list(streaming_dir.glob("rank_*.h5"))

    print(f"Found {len(rank_files)} rank files to aggregate", flush=True)

    groups_to_aggregate = defaultdict(lambda: defaultdict(list))

    for rank_file_path in rank_files:
        with h5py.File(rank_file_path, 'r') as f:
            pde_class = f.attrs['pde_class']
            spatial_shape_str = f.attrs['spatial_shape']
            spatial_shape = eval(spatial_shape_str)
            n_snapshots = f.attrs['num_snapshots']

            group_key = (pde_class, spatial_shape)
            temporal_key = n_snapshots

            groups_to_aggregate[group_key][temporal_key].append(rank_file_path)

    for group_key, temporal_groups in groups_to_aggregate.items():
        for temporal_key, rank_file_paths in temporal_groups.items():
            streaming_aggregate_group(group_key, temporal_key, rank_file_paths, output_dir)

    print("Phase 3: Cleaning up streaming files...", flush=True)
    try:
        import shutil
        shutil.rmtree(streaming_dir)
        print("Phase 3: Successfully removed streaming directory.", flush=True)
    except Exception as e:
        print(f"Phase 3: Warning - Could not remove streaming directory: {e}", flush=True)

def streaming_aggregate_group(group_key, temporal_key, rank_file_paths, output_dir):
    pde_class, spatial_shape = group_key
    n_snapshots = temporal_key

    final_filename = f"{pde_class}_s{n_snapshots}_sh{'x'.join(map(str, spatial_shape))}.h5"
    final_output_path = Path(output_dir) / final_filename

    total_trajectories = 0
    target_spatial_shape = None

    for rank_file_path in rank_file_paths:
        with h5py.File(rank_file_path, 'r') as f:
            total_trajectories += f['u'].shape[0]
            if target_spatial_shape is None:
                target_spatial_shape = eval(f.attrs['target_spatial_shape'])

    print(f"Streaming aggregation: {len(rank_file_paths)} files -> {final_output_path} ({total_trajectories} trajectories)", flush=True)

    if total_trajectories == 0:
        return

    spatial_chunks = tuple(min(SPATIAL_CHUNK_SIZE, dim) for dim in target_spatial_shape)
    temporal_chunks = min(TEMPORAL_CHUNK_SIZE, n_snapshots)
    traj_chunks = min(STREAM_BUFFER_SIZE, total_trajectories)

    with h5py.File(final_output_path, 'w') as f_final:
        f_final.attrs['pde_class'] = pde_class
        f_final.attrs['num_snapshots'] = n_snapshots
        f_final.attrs['spatial_shape'] = str(spatial_shape)
        f_final.attrs['num_trajectories'] = total_trajectories

        f_final.create_dataset('u', shape=(total_trajectories, n_snapshots, *target_spatial_shape),
                              dtype='f4', chunks=(traj_chunks, temporal_chunks, *spatial_chunks))
        f_final.create_dataset('v', shape=(total_trajectories, n_snapshots, *target_spatial_shape),
                              dtype='f4', chunks=(traj_chunks, temporal_chunks, *spatial_chunks))

        for stat in ['max', 'min', 'mean', 'std']:
            f_final.create_dataset(f'{stat}_u', shape=(total_trajectories, n_snapshots),
                                 dtype='f4', chunks=(traj_chunks, temporal_chunks))
            f_final.create_dataset(f'{stat}_v', shape=(total_trajectories, n_snapshots),
                                 dtype='f4', chunks=(traj_chunks, temporal_chunks))

        f_final.create_dataset('u0', shape=(total_trajectories, *target_spatial_shape),
                              dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f_final.create_dataset('v0', shape=(total_trajectories, *target_spatial_shape),
                              dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f_final.create_dataset('m', shape=(total_trajectories, *target_spatial_shape),
                              dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f_final.create_dataset('c', shape=(total_trajectories, *target_spatial_shape),
                              dtype='f4', chunks=(traj_chunks, *spatial_chunks))
        f_final.create_dataset('params', shape=(total_trajectories,),
                              dtype=h5py.string_dtype(encoding='utf-8'), chunks=(traj_chunks,))

        write_offset = 0

        for rank_file_path in rank_file_paths:
            with h5py.File(rank_file_path, 'r') as f_rank:
                actual_count = f_rank['u'].shape[0]
                
                if actual_count == 0:
                    continue

                end_offset = write_offset + actual_count

                for traj_batch_start in range(0, actual_count, STREAM_BUFFER_SIZE):
                    traj_batch_end = min(traj_batch_start + STREAM_BUFFER_SIZE, actual_count)
                    dest_start = write_offset + traj_batch_start
                    dest_end = write_offset + traj_batch_end

                    for dataset_name in ['u', 'v', 'max_u', 'min_u', 'mean_u', 'std_u', 'max_v', 'min_v', 'mean_v', 'std_v', 'u0', 'v0', 'm', 'c', 'params']:
                        if dataset_name in f_rank and dataset_name in f_final:
                            f_final[dataset_name][dest_start:dest_end] = f_rank[dataset_name][traj_batch_start:traj_batch_end]

                write_offset = end_offset

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description='True streaming HDF5 aggregator.')
    parser.add_argument('--base_path', required=True)
    parser.add_argument('--pde_type', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_files', type=int, default=None)
    parser.add_argument('--debug', action='store_true', help='Debug mode: process only num_ranks files for quick validation')

    args = parser.parse_args()

    if rank == 0:
        print(f"Running true streaming aggregation with {size} ranks", flush=True)
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    comm.Barrier()

    nested_groups = discover_and_group_files(args.base_path, args.pde_type, comm, rank)
    nested_groups = comm.bcast(nested_groups, root=0)

    if args.debug and rank == 0:
        print(f"Debug mode: Limiting to {size} files total for quick validation...", flush=True)

        all_files = []
        for group_data in nested_groups.values():
            for file_list in group_data.values():
                all_files.extend(file_list)

        if len(all_files) > size:
            np.random.seed(42)
            selected_indices = set(np.random.choice(len(all_files), size, replace=False))
            selected_files = {all_files[i]['path'] for i in selected_indices}

            filtered_groups = defaultdict(lambda: defaultdict(list))
            for group_key, temporal_groups in nested_groups.items():
                for temporal_key, file_list in temporal_groups.items():
                    filtered_list = [f for f in file_list if f['path'] in selected_files]
                    if filtered_list:
                        filtered_groups[group_key][temporal_key] = filtered_list

            nested_groups = dict(filtered_groups)

    elif args.num_files is not None and rank == 0:
        print(f"Subsampling to {args.num_files} files total...", flush=True)

        all_files = []
        for group_data in nested_groups.values():
            for file_list in group_data.values():
                all_files.extend(file_list)

        if len(all_files) > args.num_files:
            np.random.seed(42)
            selected_indices = set(np.random.choice(len(all_files), args.num_files, replace=False))
            selected_files = {all_files[i]['path'] for i in selected_indices}

            filtered_groups = defaultdict(lambda: defaultdict(list))
            for group_key, temporal_groups in nested_groups.items():
                for temporal_key, file_list in temporal_groups.items():
                    filtered_list = [f for f in file_list if f['path'] in selected_files]
                    if filtered_list:
                        filtered_groups[group_key][temporal_key] = filtered_list

            nested_groups = dict(filtered_groups)

    nested_groups = comm.bcast(nested_groups, root=0)

    my_assignments = distribute_files_across_ranks(nested_groups, comm, rank, size)

    comm.Barrier()

    rank_files = streaming_processing_phase(my_assignments, args.output_dir, rank)

    comm.Barrier()

    streaming_final_aggregation(args.output_dir, rank)

    if rank == 0:
        print("True streaming aggregation complete. All data consolidated efficiently.", flush=True)

if __name__ == '__main__':
    main()
