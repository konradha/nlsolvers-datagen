import argparse
import os
import time
import uuid
import datetime
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from m_fields_3d import generate_m_fields
from c_fields_3d import generate_c_fields

from downsampling import downsample_fft_3d, downsample_interpolation_3d
from real_sampler import RealWaveSampler3d
from valid_spaces_real import get_parameter_spaces_3d as get_parameter_spaces

sys.stdout.reconfigure(line_buffering=True)
class Launcher:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.setup_directories()
        self.configure_grid()
        # restriction to play nice with hand-rolled C++ integrators
        assert args.nx == args.ny and np.abs(args.Lx - args.Ly) < 1e-8
        self.sampler = RealWaveSampler3d(args.nx, args.ny, args.nz, args.Lx)
        self.system_type = args.system_type

    def setup_directories(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.ic_dir = self.output_dir / "initial_conditions"
        self.ic_dir.mkdir(exist_ok=True)

        self.traj_dir = self.output_dir / "trajectories"
        self.traj_dir.mkdir(exist_ok=True)

        self.focusing_dir = self.output_dir / "focusing"
        self.focusing_dir.mkdir(exist_ok=True)

        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.anisotropy_dir = self.output_dir / "anisotropy"
        self.anisotropy_dir.mkdir(exist_ok=True)


        self.h5_dir = self.output_dir / "hdf5"
        self.h5_dir.mkdir(exist_ok=True)

        self.save_parameter_file()

    def save_parameter_file(self):
        params_file = self.output_dir / f"params_{self.run_id}.txt"
        with open(params_file, "w") as f:
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Grid: {self.args.nx}x{self.args.ny}\n")
            f.write(f"Domain: {self.args.Lx}x{self.args.Ly}\n")
            f.write(
                f"Time: T={self.args.T}, steps={self.args.nt}, snapshots={self.args.snapshots}\n")
            f.write(f"Phenomenon: {self.args.phenomenon}\n")
            f.write(f"Amplification: {self.args.m_type}\n")
            f.write(f"Executable: {self.args.exe}\n")

    def configure_grid(self):
        self.x = np.linspace(-self.args.Lx, self.args.Lx, self.args.nx)
        self.y = np.linspace(-self.args.Ly, self.args.Ly, self.args.ny)
        self.z = np.linspace(-self.args.Lz, self.args.Lz, self.args.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.dx = 2 * self.args.Lx / (self.args.nx - 1)
        self.dy = 2 * self.args.Ly / (self.args.ny - 1)
        self.dz = 2 * self.args.Lz / (self.args.nz - 1)
        self.dV = self.dx * self.dy

    def generate_initial_condition(self, run_idx, phenomenon_params=None):
        if phenomenon_params is None:
            phenomenon_params = self.sample_phenomenon_params()
        print(f"run {run_idx + 1} parameters:", phenomenon_params)
        u0, v0 = self.sampler.generate_initial_condition(
            phenomenon_type=self.args.phenomenon, **phenomenon_params)

        if isinstance(u0, torch.Tensor):
            u0 = u0.detach().numpy()
        if isinstance(v0, torch.Tensor):
            v0 = v0.detach().numpy()

        return (u0, v0), phenomenon_params

    def sample_phenomenon_params(self):
        parameter_spaces = get_parameter_spaces(self.args.Lx)
        if self.args.phenomenon in parameter_spaces:
            space = parameter_spaces[self.args.phenomenon]
            params = {}

            for key, values in space.items():
                if isinstance(values[0], tuple) or isinstance(values[0], list):
                    l = len(values)
                    idx = np.random.randint(0, l)
                    params[key] = values[idx]
                else:
                    params[key] = np.random.choice(values)

            params["system_type"] = self.system_type
            params["velocity_type"] = self.args.velocity_type
            return params
        else:
            raise ValueError(
                f"Unknown phenomenon type: {self.args.phenomenon}")

    def generate_spatial_amplification(self, run_idx, c=None):     
        field_list, params = generate_m_fields(self.args.nx, self.args.Lx, c_field=c, num_fields=1, field_types=[self.args.m_type])
        print(f"run {run_idx + 1} m(x) parameters:", params) 
        m = field_list[0]
        return m.astype(np.float64)  # safety first

    def generate_anisotropy(self, run_idx):
        # needs to be called before mass to ensure -- if m depends on c -- that it gets generated correctly!
        field_list, params = generate_c_fields(self.args.nx, self.args.Lx, num_fields=1,field_types=[self.args.anisotropy_type])
        print(f"run {run_idx + 1} c(x) parameters:", params)
        c = field_list[0] 
        return c.astype(np.float64)


    def run_simulation(self, run_idx, u0, v0, m, c):
        u0_file = self.ic_dir / f"u0_{self.run_id}_{run_idx:04d}.npy"
        np.save(u0_file, u0)

        v0_file = self.ic_dir / f"v0_{self.run_id}_{run_idx:04d}.npy"
        np.save(v0_file, v0)

        m_file = self.focusing_dir / f"m_{self.run_id}_{run_idx:04d}.npy"
        np.save(m_file, m)

        c_file = self.anisotropy_dir / f"c_{self.run_id}_{run_idx:04d}.npy" 
        np.save(c_file, c)

        traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"
        vel_file = self.traj_dir / f"vel_{self.run_id}_{run_idx:04d}.npy"

        exe_path = Path(self.args.exe)
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable {self.args.exe} not found")

        cmd = [
            str(exe_path),
            str(self.args.nx),
            str(self.args.ny),
            str(self.args.nz),
            str(self.args.Lx),
            str(self.args.Ly),
            str(self.args.Lz),
            str(u0_file),
            str(v0_file),
            str(traj_file),
            str(vel_file),
            str(self.args.T),
            str(self.args.nt),
            str(self.args.snapshots),
            str(m_file),
            str(c_file)
        ]

        start_time = time.time()
        import subprocess
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True)
        if result.stderr:
            print("Command used:", " ".join(cmd))
            print(f"Warnings/Errors: {result.stderr}")

        end_time = time.time()
        walltime = end_time - start_time

        traj_data = np.load(traj_file)
        vel_data = np.load(vel_file)
        return (traj_data, vel_data), walltime

    def downsample_trajectory(self, traj_data):
        if self.args.dr_strategy == 'none':
            return traj_data

        elif self.args.dr_strategy == 'FFT':
            return downsample_fft_3d(traj_data,
                                  target_shape=(self.args.dr_x, self.args.dr_y, self.args.dr_z))

        elif self.args.dr_strategy == 'interpolation':
            return downsample_interpolation_3d(
                traj_data,
                target_shape=(self.args.dr_x, self.args.dr_y, self.args.dr_z),
                Lx=self.args.Lx,
                Ly=self.args.Ly,
                Lz=self.args.Lz
            )
        else:
            raise ValueError(
                f"Unknown downsampling strategy: {self.args.dr_strategy}")

    def save_to_hdf5(self, run_idx, u0, v0, m, c, traj_data, vel_data,
                     phenomenon_params, elapsed_time):
        h5_file = self.h5_dir / f"run_{self.run_id}_{run_idx:04d}.h5"
        with h5py.File(h5_file, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['problem_type'] = self.system_type
            meta.attrs['boundary_condition'] = 'noflux'
            meta.attrs['run_id'] = self.run_id
            meta.attrs['run_index'] = run_idx
            meta.attrs['timestamp'] = str(datetime.datetime.now())
            meta.attrs['elapsed_time'] = elapsed_time
            meta.attrs['phenomenon'] = self.args.phenomenon

            for key, value in phenomenon_params.items():
                if isinstance(value, (list, tuple)) and len(value) <= 10:
                    meta.attrs[f'phenomenon_{key}'] = str(value)
                else:
                    meta.attrs[f'phenomenon_{key}'] = str(value)

            grid = f.create_group('grid')
            grid.attrs['nx'] = self.args.nx
            grid.attrs['ny'] = self.args.ny
            grid.attrs['nz'] = self.args.nz
            grid.attrs['Lx'] = self.args.Lx
            grid.attrs['Ly'] = self.args.Ly
            grid.attrs['Lz'] = self.args.Lz

            time_grp = f.create_group('time')
            time_grp.attrs['T'] = self.args.T
            time_grp.attrs['nt'] = self.args.nt
            time_grp.attrs['num_snapshots'] = self.args.snapshots

            ic_grp = f.create_group('initial_condition')
            ic_grp.create_dataset('u0', data=u0)
            ic_grp.create_dataset('v0', data=v0)

            m_grp = f.create_group('focusing')
            m_grp.attrs['type'] = self.args.m_type

            m_grp.create_dataset('m', data=m)
            m_grp.create_dataset('c', data=c)
            f.create_dataset('u', data=traj_data)
            f.create_dataset('v', data=vel_data)
            f.create_dataset('X', data=self.X)
            f.create_dataset('Y', data=self.Y)

        return h5_file

    def cleanup(self, run_idx):
        if self.args.delete_intermediates:
            u0_file = self.ic_dir / f"u0_{self.run_id}_{run_idx:04d}.npy"
            v0_file = self.ic_dir / f"v0_{self.run_id}_{run_idx:04d}.npy"
            m_file = self.focusing_dir / f"m_{self.run_id}_{run_idx:04d}.npy"
            c_file = self.anisotropy_dir / f"c_{self.run_id}_{run_idx:04d}.npy"
            traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"
            vel_file = self.traj_dir / f"vel_{self.run_id}_{run_idx:04d}.npy"

            for file in [u0_file, v0_file, m_file, c_file, traj_file, vel_file]:
                if file.exists():
                    os.unlink(file)

    def run(self):
        dt = self.args.T / self.args.nt
        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)
        dz = 2 * self.args.Lz / (self.args.nz - 1)

        sols = dict()
        for i in range(self.args.num_runs):
            try:
                pre_start = time.time()
                (u0, v0), phenomenon_params = self.generate_initial_condition(i) 
                c = self.generate_anisotropy(i)
                m = self.generate_spatial_amplification(i, c=c)
                pre_end = time.time()

                (traj_data, vel_data), walltime = self.run_simulation(i, u0, v0, m, c)
                post_start = time.time()
                traj_data = self.downsample_trajectory(traj_data)
                vel_data = self.downsample_trajectory(vel_data)
                sols[f"{self.run_id}_{i}"] = (traj_data, vel_data)
                # save downsampled (u, v) trajectory only to perform analysis
                self.save_to_hdf5(
                    i, u0, v0, m, c, traj_data, vel_data,
                    phenomenon_params, walltime)
                if self.args.delete_intermediates:
                    self.cleanup(i)
                post_end = time.time()
                total_t = walltime + \
                    abs(pre_end - pre_start) + abs(post_end - post_start)
                part_pre = abs(pre_end - pre_start) / total_t
                part_run = walltime / total_t
                part_pst = abs(post_end - post_start) / total_t
                print(
                    f"walltime: {total_t:.2f}s, pre: {part_pre * 100:.1f}%, run: {part_run * 100:.1f}%, post: {part_pst*100:.1f}%")

            except Exception as e:
                print(f"Error in run {i+1}: {e}")
                # import traceback as t
                # import pdb
                # t.print_exc()
                # pdb.set_trace()
                continue


        if self.args.delete_intermediates:
            params_file = self.output_dir / f"params_{self.run_id}.txt"
            if params_file.exists():
                os.unlink(params_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real solver launcher")

    parser.add_argument("--system-type", type=str, default="sine_gordon",
                        choices=["sine_gordon", 
                                 "phi4",
                                 "klein_gordon"])

    parser.add_argument("--phenomenon", type=str, default="kink_field",
                        choices=["kink_field"],
                        help="Phenomenon type to simulate")

    parser.add_argument("--velocity-type", type=str, default="zero",
                        choices=["zero",  "random"])

    parser.add_argument("--anisotropy-type", type=str, default="constant", choices=['constant',
        'piecewise_constant', 'sign_changing_mass', 'layered', 'waveguide', 'quasiperiodic', 'turbulent'])


    parser.add_argument("--m_type", type=str, default="constant",
                        choices=['constant', 'piecewise', 'gradient',
                                 'topological', 'defects', 'quasiperiodic', 'multiscale'],
                        help="Type of spatial amplification function m(x,y,z)")

    parser.add_argument("--nx", type=int, default=128, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in y")
    parser.add_argument("--nz", type=int, default=128, help="Grid points in z")
    parser.add_argument(
        "--Lx",
        type=float,
        default=3.0,
        help="Domain half-width in x")
    parser.add_argument(
        "--Ly",
        type=float,
        default=3.,
        help="Domain half-width in y")
    parser.add_argument(
        "--Lz",
        type=float,
        default=3.,
        help="Domain half-width in z")
    parser.add_argument("--T", type=float, default=1.5, help="Simulation time")
    parser.add_argument(
        "--nt",
        type=int,
        default=500,
        help="Number of time steps")
    parser.add_argument(
        "--snapshots",
        type=int,
        default=100,
        help="Number of snapshots")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory")
    parser.add_argument(
        "--exe",
        type=str,
        required=True,
        help="Path to executable")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs to perform")

    parser.add_argument("--dr-x", type=int, default=128,
                        help="Number of gridpoints to sample down for in x-direction")
    parser.add_argument("--dr-y", type=int, default=128,
                        help="Number of gridpoints to sample down for in y-direction")
    parser.add_argument("--dr-z", type=int, default=128,
                        help="Number of gridpoints to sample down for in z-direction")
    parser.add_argument("--dr-strategy", choices=["FFT", "interpolation", "none"],
                        default="interpolation",
                        help="Downsampling strategy: Default is interpolation due to " +
                             "non-periodic boundary conditions. Choose 'none' to keep the resolution")

    parser.add_argument("--delete-intermediates", action="store_true", default=True,
                        help="Removing intermediate files (mostly npy to be recovered from hdf5)")
    
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))

    return args


def main():
    args = parse_args()
    launcher = Launcher(args)
    launcher.run()


if __name__ == "__main__":
    sys.exit(main())
