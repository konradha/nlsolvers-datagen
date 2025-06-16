import argparse
import h5py
import numpy as np
import json
import sys
from pathlib import Path
from itertools import combinations

def print_header(title):
    print("\n" + f"--- {title.upper()} ---")

def chunked_distance(dset, idx1, idx2, chunk_size=10):
    shape = dset.shape
    n_snapshots = shape[1]
    
    sum_sq_diff = 0.0
    for i in range(0, n_snapshots, chunk_size):
        end = min(i + chunk_size, n_snapshots)
        chunk1 = dset[idx1, i:end]
        chunk2 = dset[idx2, i:end]
        sum_sq_diff += np.sum((chunk1 - chunk2)**2)
        
    return np.sqrt(sum_sq_diff)

def check_file(filepath, expected_system_type, check_diversity, samples, tolerance):
    if not Path(filepath).is_file():
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(filepath, 'r') as f:
        print_header(f"Checking File: {filepath}")

        print_header("1. System Verification & Attributes")
        try:
            file_system_type = f.attrs.get('pde_class', 'N/A')
            print(f"Expected System Type: {expected_system_type}")
            print(f"Found System Type:    {file_system_type}")

            if file_system_type != expected_system_type:
                print(f"   [FAIL] Mismatch! Expected '{expected_system_type}' but file contains '{file_system_type}'.", flush=True)
                sys.exit(1)
            else:
                print("   [OK] System type matches the file's metadata.", flush=True)

            num_traj = f.attrs['num_trajectories']
            print(f"\nTotal Trajectories: {num_traj}", flush=True)
            for key, value in f.attrs.items():
                if key not in ['num_trajectories', 'pde_class']:
                    print(f"{key.replace('_', ' ').title()}: {value}", flush=True)
        except KeyError as e:
            print(f"Missing essential attribute: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

        print_header("2. Dataset & Metadata Integrity")
        dsets = sorted(list(f.keys()))
        for key in dsets:
            shape = f[key].shape
            print(f"  - {key:<12} | Shape: {shape}", flush=True)
            if key != 'params' and shape[0] != num_traj:
                print(f"    [WARNING] First dimension ({shape[0]}) does not match num_trajectories ({num_traj})", flush=True)

        print("\nVerifying parameter metadata...", flush=True)
        if 'params' in f and num_traj > 1:
            try:
                path1 = Path(json.loads(f['params'][0]).get('filepath', 'N/A')).name
                path2 = Path(json.loads(f['params'][-1]).get('filepath', 'N/A')).name
                print(f"   Source file of first trajectory: ...{path1}", flush=True)
                print(f"   Source file of last trajectory:  ...{path2}", flush=True)
                if path1 == path2:
                    print("   [FAIL] Metadata for first and last trajectories is identical.", flush=True)
                else:
                    print("   [OK] Metadata appears distinct.", flush=True)
            except Exception as e:
                print(f"   [FAIL] Could not parse or inspect metadata: {e}", file=sys.stderr, flush=True)

        if check_diversity:
            print_header("3. Definitive Data Diversity Check")
            if 'u' not in f:
                 print("   [FAIL] 'u' dataset not found. Cannot perform diversity check.", flush=True)
                 sys.exit(1)
            if num_traj < 2:
                print("   [INFO] Not enough trajectories (<2) to perform a diversity check.", flush=True)
                return

            num_to_sample = min(samples, num_traj)
            if num_to_sample < 2:
                print(f"   [INFO] Sample count ({num_to_sample}) is too low for a pairwise check.", flush=True)
                return

            print(f"Randomly sampling {num_to_sample} of {num_traj} trajectories...", flush=True)
            np.random.seed(42)
            indices = np.random.choice(num_traj, size=num_to_sample, replace=False)
            
            print(f"Checking pairs from indices: {sorted(indices.tolist())}", flush=True)

            min_distance_found = float('inf')
            redundant_pair_found = False
            for i, j in combinations(indices, 2):
                distance = chunked_distance(f['u'], i, j)
                min_distance_found = min(min_distance_found, distance)

                if distance < tolerance:
                    print(f"\n   [CRITICAL FAIL] Trajectories at index {i} and {j} are nearly identical.", flush=True)
                    print(f"   L2 Distance: {distance:.4e} | Tolerance: {tolerance:.1e}", flush=True)
                    redundant_pair_found = True
                    break

            if not redundant_pair_found:
                num_pairs = len(list(combinations(range(num_to_sample), 2)))
                print(f"\n   [OK] All {num_pairs} sampled pairs are numerically distinct.", flush=True)
                print(f"   Minimum distance found among pairs: {min_distance_found:.4e}", flush=True)
            else:
                 print("\n   [OVERALL FAIL] Redundant data detected. Aggregation may have errors.", flush=True)
                 sys.exit(1)

        print("\n" + "="*55)
        print("Sanity check complete.")
        print("="*55)

def main():
    parser = argparse.ArgumentParser(
        description="A high-scrutiny sanity check for aggregated HDF5 trajectory files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("filepath", type=str, help="Path to the HDF5 file to check.")
    parser.add_argument("system_type", type=str, help="Expected system type (e.g., 'nlse_2d').")
    parser.add_argument(
        "--check-diversity", action="store_true",
        help="Perform a definitive (but slower) pairwise distance check on a sample of trajectories."
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Number of trajectories to sample for the diversity check (default: 5)."
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6,
        help="The minimum L2 distance for two trajectories to be considered different (default: 1e-6)."
    )
    args = parser.parse_args()
    if args.samples > 5:
        print("Warning: Limiting pairwise comparison to a maximum of 5 samples.", file=sys.stderr, flush=True)
        args.samples = 5
        
    check_file(args.filepath, args.system_type, args.check_diversity, args.samples, args.tolerance)

if __name__ == "__main__":
    main()
