import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import time
import uuid
import subprocess
from pathlib import Path
import pandas as pd
import sys
import json

from real_sampler import RealWaveSampler
from m_fields_2d import generate_m_fields
from c_fields_2d import generate_c_fields
from downsampling import downsample_fft, downsample_interpolation

mpl.rcParams.update({
    'font.size': 15,
    'font.family': 'serif',
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
    'savefig.dpi': 300,
    'figure.titlesize': 20,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.color': 'grey',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    'xtick.major.pad': 5,
    'ytick.major.pad': 5,
    'figure.subplot.hspace': 0.3,
    'figure.subplot.wspace': 0.25,
})

mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['figure.titleweight'] = 'normal'
mpl.rcParams['axes.labelweight'] = 'normal'

def plot_initial_fields(u0_np, v0_np, c_field_np, m_field_np, Lx, Ly, output_path, study_id, ic_type, anisotropy_type, m_type):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # fig.suptitle(f"ICs: (IC: {ic_type.replace('_', ' ').title()})", fontsize=mpl.rcParams['figure.titlesize'] + 2)

    fields = [u0_np, v0_np, c_field_np, m_field_np]
    titles = [r'$u_0 $', r'$v_0$',
              rf'$c $',
              rf'$m $']
    cmaps = ['coolwarm', 'coolwarm', 'viridis', 'cividis']

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            idx = i * 2 + j
            if idx < len(fields):
                field_data = fields[idx]
                title_text = titles[idx]
                cmap_name = cmaps[idx]

                vmin, vmax = np.nanmin(field_data), np.nanmax(field_data)
                if not (np.isfinite(vmin) and np.isfinite(vmax)):
                    vmin, vmax = -1, 1
                if vmin == vmax:
                    vmin -= 0.5
                    vmax += 0.5
                if titles[idx] in [r'$u_0 \bm {x}$', r'$v_0 \bm {x}$'] and vmin < 0 < vmax:
                    abs_max = max(abs(vmin), abs(vmax))
                    norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
                    im = ax.imshow(field_data, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=cmap_name, aspect='auto', norm=norm)
                else:
                    im = ax.imshow(field_data, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=cmap_name, aspect='auto', vmin=vmin, vmax=vmax)

                ax.set_title(title_text)
                ax.set_xlabel("$x$")
                ax.set_ylabel("$y$")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_aspect('equal')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filename = f"initial_fields_snapshot_{study_id}.png"
    full_output_path = output_path.parent / filename

    fig.savefig(full_output_path)
    plt.close(fig)
    print(f"Saved initial fields snapshot: {full_output_path}")


def downsample_2d_field(field_2d, target_nx, target_ny, Lx_domain, Ly_domain, strategy='interpolation'):
    original_ny, original_nx = field_2d.shape
    if original_nx == target_nx and original_ny == target_ny:
        return field_2d.copy()
    temp_traj = field_2d[np.newaxis, :, :]
    if strategy == 'FFT':
        downsampled_traj = downsample_fft(temp_traj, target_shape=(target_ny, target_nx))
    elif strategy == 'interpolation':
        downsampled_traj = downsample_interpolation(temp_traj, target_shape=(target_ny, target_nx), Lx=Lx_domain, Ly=Ly_domain)
    else:
        print(f"Warning: Unknown downsampling strategy '{strategy}'. Using basic slicing.", file=sys.stderr)
        step_y = max(1, original_ny // target_ny)
        step_x = max(1, original_nx // target_nx)
        return field_2d[::step_y, ::step_x][:target_ny, :target_nx].copy()
    return downsampled_traj[0]

def compute_gradient_sq_norm(u_2d, dx, dy, c_2d=None):
    if c_2d is None:
        c_2d_inner = 1.0
    else:
        c_2d_inner = c_2d[1:-1, 1:-1]

    ux_inner = (u_2d[1:-1, 2:] - u_2d[1:-1, :-2]) / (2. * dx)
    uy_inner = (u_2d[2:, 1:-1] - u_2d[:-2, 1:-1]) / (2. * dy)

    energy_grad_density = c_2d_inner * (ux_inner**2 + uy_inner**2)
    return np.sum(energy_grad_density) * dx * dy

def compute_mass(u_2d, dx, dy):
    return np.sum(u_2d**2) * dx * dy

def compute_hamiltonian_kge_u_cubed(u_2d, ut_2d, dx, dy, m_2d, c_2d):
    energy_ut = 0.5 * np.sum(ut_2d**2) * dx * dy
    energy_grad = 0.5 * compute_gradient_sq_norm(u_2d, dx, dy, c_2d=c_2d)
    energy_pot = np.sum(m_2d * (0.25 * u_2d**4)) * dx * dy
    return energy_ut, energy_grad, energy_pot

class WaveIntegratorStudy:
    def __init__(self, args_launcher_ns, study_params_config_ns):
        self.args_launcher = args_launcher_ns
        self.study_params = study_params_config_ns
        self.run_id = str(uuid.uuid4())[:8]
        self.system_name = f"GNWE $f(u) = u^3$"
        self.equation_name_file_safe = "KGE_u_cubed"
        self.dtype = np.float64
        self.hamiltonian_func = compute_hamiltonian_kge_u_cubed

        self.base_output_dir = Path(self.args_launcher.output_dir)
        self.study_output_dir = self.base_output_dir / f"{self.equation_name_file_safe}_study_{self.run_id}"
        self.study_output_dir.mkdir(exist_ok=True, parents=True)

        self.plots_dir = self.study_output_dir / "plots"; self.plots_dir.mkdir(exist_ok=True)
        self.data_dir = self.study_output_dir / "data"; self.data_dir.mkdir(exist_ok=True)
        self.temp_dir_base = self.study_output_dir / "temp_runs"; self.temp_dir_base.mkdir(exist_ok=True)

        self.u0_high_res, self.v0_high_res = None, None
        self.m_field_high_res, self.c_field_high_res = None, None
        self.high_res_nx = max(self.study_params['nx_values_study'])
        self.high_res_ny = self.high_res_nx
        self._prepare_high_resolution_inputs()
        if self.u0_high_res is not None and \
           self.v0_high_res is not None and \
           self.c_field_high_res is not None and \
           self.m_field_high_res is not None:

            plot_initial_fields(
                u0_np=self.u0_high_res,
                v0_np=self.v0_high_res,
                c_field_np=self.c_field_high_res,
                m_field_np=self.m_field_high_res,
                Lx=self.study_params['L_study'],
                Ly=self.study_params['L_study'], #[-L,L]²
                output_path=self.plots_dir / f"initial_fields_snapshot_{self.run_id}.png",
                study_id=self.run_id,
                ic_type=self.study_params['ic_type_study'],
                anisotropy_type=self.study_params['anisotropy_type_study'],
                m_type=self.study_params['m_type_study']
            )

        self.max_snapshots = self.study_params.get('max_snapshots_study', 50)
        self.all_trajectory_data_for_comparison = []


    def _prepare_high_resolution_inputs(self):
        print(f"Preparing high-resolution inputs (Nx={self.high_res_nx}) for {self.equation_name_file_safe}...")
        Lx_domain = self.study_params['L_study']
        ic_type = self.study_params['ic_type_study']
        phenom_params = json.loads(self.study_params.get('phenomenon_params_override_json_study', '{}'))

        sampler_high_res = RealWaveSampler(self.high_res_nx, self.high_res_ny, Lx_domain)
        u0_hr, v0_hr = sampler_high_res.generate_initial_condition(phenomenon_type=ic_type, **phenom_params)

        # FROM: http://dx.doi.org/10.1016/j.apnum.2021.07.026
        # Adapted here: nonlinear term + anisotropy are different. We take it as a guide.
        def f(x, y):
            e1 = np.exp(
                    -(x + 2) ** 2 - y ** 2
                    )
            e2 = np.exp(
                    -(x - 2) ** 2 - y ** 2
                    )
            return e1 + e2

        def g(x, y):
            return np.exp(-(x ** 2 + y ** 2))

        x, y = np.meshgrid(
                np.linspace(-Lx_domain, Lx_domain,self.high_res_nx), 
                np.linspace(-Lx_domain, Lx_domain,self.high_res_nx),
                )

        u0_hr = f(x, y) 
        v0_hr = g(x, y)

        self.u0_high_res = u0_hr.numpy() if hasattr(u0_hr, 'numpy') else u0_hr
        self.v0_high_res = v0_hr.numpy() if hasattr(v0_hr, 'numpy') else v0_hr
        self.u0_high_res = self.u0_high_res.squeeze().astype(self.dtype)
        self.v0_high_res = self.v0_high_res.squeeze().astype(self.dtype)

        c_field_list_hr, _ = generate_c_fields(self.high_res_nx,
                Lx_domain, num_fields=1, field_types=[self.study_params['anisotropy_type_study']])
        self.c_field_high_res = c_field_list_hr[0].astype(self.dtype)

        m_field_list_hr, _ = generate_m_fields(self.high_res_nx, Lx_domain,
                c_field=self.c_field_high_res, num_fields=1, field_types=[self.study_params['m_type_study']])
        self.m_field_high_res = m_field_list_hr[0].astype(self.dtype)
        print("High-resolution inputs prepared.")
 
    def _get_or_create_downsampled_inputs_for_nx(self, target_nx, target_ny, base_temp_dir_for_nx):
        u0_path = base_temp_dir_for_nx / "u0.npy"; v0_path = base_temp_dir_for_nx / "v0.npy"
        m_path = base_temp_dir_for_nx / "m.npy"; c_path = base_temp_dir_for_nx / "c.npy"

        if u0_path.exists() and v0_path.exists() and m_path.exists() and c_path.exists() and not self.args_launcher.force_regenerate_ics:
            print(f"Found existing downsampled inputs in {base_temp_dir_for_nx}")
            return u0_path, v0_path, m_path, c_path

        print(f"Creating downsampled inputs for nx={target_nx} in {base_temp_dir_for_nx}")
        Lx_domain = self.study_params['L_study']
        strategy = self.study_params.get('downsampling_strategy_study', 'interpolation')

        u0_ds = downsample_2d_field(self.u0_high_res, target_nx, target_ny, Lx_domain, Lx_domain, strategy=strategy)
        v0_ds = downsample_2d_field(self.v0_high_res, target_nx, target_ny, Lx_domain, Lx_domain, strategy=strategy)
        m_ds = downsample_2d_field(self.m_field_high_res, target_nx, target_ny, Lx_domain, Lx_domain, strategy=strategy)
        c_ds = downsample_2d_field(self.c_field_high_res, target_nx, target_ny, Lx_domain, Lx_domain, strategy=strategy)

        expected_shape = (target_ny, target_nx)
        if u0_ds.shape != expected_shape:
            raise ValueError(f"u0 downsample shape error for nx={target_nx}, expected {expected_shape}, got {u0_ds.shape}")
        if v0_ds.shape != expected_shape:
            raise ValueError(f"v0 downsample shape error for nx={target_nx}, expected {expected_shape}, got {v0_ds.shape}")
        if m_ds.shape != expected_shape:
            raise ValueError(f"m downsample shape error for nx={target_nx}, expected {expected_shape}, got {m_ds.shape}")
        if c_ds.shape != expected_shape:
            raise ValueError(f"c downsample shape error for nx={target_nx}, expected {expected_shape}, got {c_ds.shape}")

        np.save(u0_path, u0_ds.astype(self.dtype)); np.save(v0_path, v0_ds.astype(self.dtype))
        np.save(m_path, m_ds.astype(self.dtype)); np.save(c_path, c_ds.astype(self.dtype))
        return u0_path, v0_path, m_path, c_path

    def _build_command(self, exe_path, sim_args, u0_file, v0_file, m_file, c_file, traj_output_file, vel_output_file):
        return [
            str(exe_path),
            str(sim_args['nx']), str(sim_args['ny']),
            str(sim_args['Lx']), str(sim_args['Ly']),
            str(u0_file), str(v0_file),
            str(traj_output_file), str(vel_output_file),
            str(sim_args['T_sim_actual']),
            str(sim_args['nt_sim_actual']),
            str(sim_args['num_snapshots_actual']),
            str(m_file), str(c_file)
        ]

    def _run_single_simulation(self, integrator_name, exe_path, sim_args, u0_file, v0_file, m_file, c_file, temp_dir_for_dt_run):
        traj_file = temp_dir_for_dt_run / f"traj_{integrator_name}.npy"
        vel_file = temp_dir_for_dt_run / f"vel_{integrator_name}.npy"

        command = self._build_command(exe_path, sim_args, u0_file, v0_file, m_file, c_file, traj_file, vel_file)

        actual_dt = sim_args['T_sim_actual'] / sim_args['nt_sim_actual']
        print(f"Running {integrator_name} (nx={sim_args['nx']}, dt={actual_dt:.2e}): {' '.join(command)}")

        start_time = time.time()
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if result.stderr:
                print(f"STDERR ({integrator_name}, nx={sim_args['nx']}, dt={actual_dt:.2e}):\n{result.stderr}\n---------", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {integrator_name} (nx={sim_args['nx']}, dt={actual_dt:.2e}):"
                  f"\nCmd: {' '.join(map(str,e.cmd))}\nReturn Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}", file=sys.stderr)
            raise
        wall_time = time.time() - start_time
        print(f"{integrator_name} (nx={sim_args['nx']}, dt={actual_dt:.2e}) finished in {wall_time:.2f}s.")
        return traj_file, vel_file, wall_time

    def _analyze_trajectory(self, traj_file, vel_file, sim_args, m_field_np, c_field_np):
        traj_data = np.load(traj_file).astype(self.dtype)
        vel_data = np.load(vel_file).astype(self.dtype)

        num_snapshots = traj_data.shape[0]
        mass_ts = np.zeros(num_snapshots, dtype=self.dtype)
        h_kinetic_ts = np.zeros(num_snapshots, dtype=self.dtype)
        h_gradient_ts = np.zeros(num_snapshots, dtype=self.dtype)
        h_potential_ts = np.zeros(num_snapshots, dtype=self.dtype)
        h_total_ts = np.zeros(num_snapshots, dtype=self.dtype)

        dx = (2 * sim_args['Lx']) / (sim_args['nx'] - 1) 
        dy = (2 * sim_args['Ly']) / (sim_args['ny'] - 1)

        simulation_stable = True
        for i in range(num_snapshots):
            u_snapshot, ut_snapshot = traj_data[i], vel_data[i]
            if not (np.all(np.isfinite(u_snapshot)) and np.all(np.isfinite(ut_snapshot))):
                print(f"WARNING: Non-finite values detected in snapshot {i} for nx={sim_args['nx']}, "
                      f"dt={(sim_args['T_sim_actual']/sim_args['nt_sim_actual']):.2e}. Analysis might be compromised.", file=sys.stderr)
                simulation_stable = False
                mass_ts[i:] = np.nan
                h_kinetic_ts[i:] = np.nan; h_gradient_ts[i:]=np.nan; h_potential_ts[i:]=np.nan; h_total_ts[i:]=np.nan
                break

            mass_ts[i] = compute_mass(u_snapshot, dx, dy)
            e_k, e_g, e_p = self.hamiltonian_func(u_snapshot, ut_snapshot, dx, dy, m_field_np, c_field_np)
            h_kinetic_ts[i], h_gradient_ts[i], h_potential_ts[i] = e_k, e_g, e_p
            h_total_ts[i] = e_k + e_g + e_p
            if not (np.isfinite(mass_ts[i]) and np.isfinite(h_total_ts[i])):
                 simulation_stable = False

        h0, m0 = h_total_ts[0], mass_ts[0]
        h_log10_rel_err_ts = np.full_like(h_total_ts, np.nan)
        m_log10_rel_err_ts = np.full_like(mass_ts, np.nan)
        max_abs_h_rel_err = np.nan

        if not (np.isfinite(h0) and np.isfinite(m0)):
            print(f"WARNING: Initial Hamiltonian H0 ({h0}) or Mass M0 ({m0}) is non-finite for "
                  f"nx={sim_args['nx']}, dt={(sim_args['T_sim_actual']/sim_args['nt_sim_actual']):.2e}. Error metrics will be NaN.",
                  file=sys.stderr)
            simulation_stable = False
        elif simulation_stable:
            abs_h_err_t = np.abs(h_total_ts - h0)
            abs_m_err_t = np.abs(mass_ts - m0)

            with np.errstate(divide='ignore', invalid='ignore'):
                if np.abs(h0) > 1e-15: raw_h_rel_err_t = abs_h_err_t / np.abs(h0)
                else: raw_h_rel_err_t = np.where(abs_h_err_t < 1e-15, 0.0, np.inf)

                if np.abs(m0) > 1e-15: raw_m_rel_err_t = abs_m_err_t / np.abs(m0)
                else: raw_m_rel_err_t = np.where(abs_m_err_t < 1e-15, 0.0, np.inf)

                h_log10_rel_err_ts[1:] = np.log10(np.maximum(raw_h_rel_err_t[1:], 1e-16))
                m_log10_rel_err_ts[1:] = np.log10(np.maximum(raw_m_rel_err_t[1:], 1e-16))

                max_abs_h_rel_err = np.nanmax(raw_h_rel_err_t[1:]) if len(raw_h_rel_err_t[1:]) > 0 else np.nan

        if not simulation_stable and np.isfinite(max_abs_h_rel_err) :
            max_abs_h_rel_err = np.nan

        metrics = {
            'time_points': np.linspace(0, sim_args['T_sim_actual'], num_snapshots),
            'mass': mass_ts, 'mass_log10_rel_error': m_log10_rel_err_ts,
            'hamiltonian_total': h_total_ts, 'hamiltonian_log10_rel_error': h_log10_rel_err_ts,
            'max_abs_hamiltonian_rel_error': max_abs_h_rel_err,
            'hamiltonian_kinetic': h_kinetic_ts, 'hamiltonian_gradient': h_gradient_ts, 'hamiltonian_potential': h_potential_ts,
            'simulation_stable': simulation_stable
        }
        return metrics, traj_data

    def _plot_generic(self, setup_fn, plot_fn, suptitle_text, fname_base, set_equal=False, rect_layout=[0.05, 0.05, 0.95, 0.92]):
        fig, axes = setup_fn()
        if set_equal:
            axes_dims = axes.shape
            for i in np.ndindex(axes_dims):
                axes[i].set_aspect('equal')

        plot_fn(fig, axes)
        if suptitle_text:
            fig.suptitle(suptitle_text)
        plt.tight_layout(rect=rect_layout)
        plot_fpath = self.plots_dir / f"{fname_base}_{self.equation_name_file_safe}_{self.run_id}.png"
        fig.savefig(plot_fpath); plt.close(fig); print(f"Saved plot: {plot_fpath}")

    def _plot_convergence(self, df_results, metric_key, y_label, title_suffix, file_suffix):
        integrators = df_results['integrator'].unique()
        nx_values = sorted(df_results['nx'].unique())

        if not nx_values: print(f"No data to plot for convergence: {title_suffix}"); return
        
        num_cols = len(nx_values)
        fig_width = 4.5 * num_cols
        fig_height = 3.9

        def setup():
            return plt.subplots(1, num_cols, figsize=(fig_width, fig_height), sharey=True, squeeze=False)

        def plot_data(fig, axes_list_2d):
            axes_list = axes_list_2d[0,:]
            for i, nx_val in enumerate(nx_values):
                ax = axes_list[i]
                for integrator_name in integrators:
                    subset = df_results[(df_results['integrator'] == integrator_name) & (df_results['nx'] == nx_val)]
                    if not subset.empty:
                        valid_points = subset.dropna(subset=['dt', metric_key])
                        if not valid_points.empty:
                             ax.plot(valid_points['dt'], valid_points[metric_key], marker='o', linestyle='-', label=integrator_name)
                ax.set_xlabel(r"$\tau$")
                ax.set_title(f"$N_x = {nx_val}$")
                if i == 0: ax.set_ylabel(y_label)
                ax.set_xscale('log')
                ax.legend()
                ax.grid(True, which="both", ls=":", alpha=0.7)

        suptitle_text = f"{self.system_name}: {title_suffix} vs. $\\tau$"
        self._plot_generic(setup, plot_data, suptitle_text, f"convergence_{file_suffix}")

    def _plot_work_precision(self, df_results, error_metric_key, error_label, title_suffix, file_suffix, dt_label_symbol=r"\tau"):
        integrators = df_results['integrator'].unique()
        nx_values = sorted(df_results['nx'].unique())
        
        dt_markers_available = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'H', 'd', 'p']
        marker_char_to_unicode = {
            'o': '\u25CF',  # ● 
            's': '\u25A0',  # ■ 
            '^': '\u25B2',  # ▲ 
            'D': '\u25C6',  # ◆ 
            'v': '\u25BC',  # ▼ 
            'P': '\u271A',  # ✚ 
            'X': '\u2716',  # ✖ 
            '*': '\u2605',  # ★ 
            'h': '\u2B22',  # ⬢ 
            '<': '\u25C0',  # ◀ 
            '>': '\u25B6',  # ▶ 
            'H': '\u2B22',  # ⬢ 
            'd': '\u25C7',  # ◇ 
            'p': '\u2B1F',  # ⬟ 
        }

        if df_results.empty:
            print(f"No data to plot for work-precision: {title_suffix}")
            return

        fig_width = 8
        fig_height = 7

        def setup():
            return plt.subplots(figsize=(fig_width, fig_height))

        def plot_data(fig, ax):
            all_dt_values = sorted(df_results['dt'].unique())
            
            if len(all_dt_values) > len(dt_markers_available):
                print(f"Warning: More unique dt values ({len(all_dt_values)}) "
                      f"than available distinct markers ({len(dt_markers_available)}). "
                      f"Markers will repeat.")
            
            dt_to_marker_char_map = {
                dt_val: dt_markers_available[i % len(dt_markers_available)] 
                for i, dt_val in enumerate(all_dt_values)
            }
            
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors_cycle = prop_cycle.by_key()['color']
            color_series_idx = 0

            for nx_val in nx_values:
                for integrator_name in integrators:
                    subset = df_results[
                        (df_results['integrator'] == integrator_name) & 
                        (df_results['nx'] == nx_val)
                    ]
                    if not subset.empty:
                        valid_points = subset[subset[error_metric_key] > 0].dropna(
                            subset=['walltime', error_metric_key, 'dt']
                        )
                        if not valid_points.empty:
                            series_color = colors_cycle[color_series_idx % len(colors_cycle)]
                            
                            ax.plot(valid_points['walltime'], valid_points[error_metric_key],
                                    linestyle='-',
                                    color=series_color,
                                    label=f"{integrator_name}, $N_x={nx_val}$",
                                    marker='None') 
                            
                            for _, point_data in valid_points.iterrows():
                                marker_char = dt_to_marker_char_map[point_data['dt']]
                                ax.plot(point_data['walltime'], point_data[error_metric_key],
                                        marker=marker_char,
                                        color=series_color,
                                        linestyle='None',
                                        markersize=7) 
                            color_series_idx += 1
            
            dt_marker_legend_parts = []
            for dt_val, marker_char_code in dt_to_marker_char_map.items():
                visual_marker_repr = marker_char_to_unicode.get(marker_char_code, marker_char_code)
                dt_fmtd = f"{dt_val:g}"
                dt_marker_legend_parts.append(f"{visual_marker_repr}: ${dt_label_symbol}={dt_fmtd}$")
            
            title_line1 = f"{self.system_name}: {title_suffix}"
            title_line2 = ""
            
            if dt_marker_legend_parts:
                title_line2 = ", ".join(dt_marker_legend_parts)
            
            if title_line2:
                full_plot_title = f"{title_line1}\n{title_line2}"
            else:
                full_plot_title = title_line1
            
            ax.set_xlabel("Walltime / [1]")
            ax.set_ylabel(error_label)
            ax.set_title(full_plot_title)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(loc='best')
            ax.grid(True, which="both", ls=":", alpha=0.7)

        self._plot_generic(setup, plot_data, "", f"work_precision_{file_suffix}", rect_layout=[0.1, 0.1, 0.95, 0.93])


    def _plot_energy_component_evolution(self, df_full_timeseries_results):
        integrators = df_full_timeseries_results['integrator'].unique()
        if not integrators.size: print("No full timeseries data to plot energy components."); return

        example_runs = []
        for name in integrators:
            stable_runs = df_full_timeseries_results[(df_full_timeseries_results['integrator'] == name) & (df_full_timeseries_results['simulation_stable'] == True)]
            if not stable_runs.empty:
                example_runs.append(stable_runs.loc[stable_runs['dt'].idxmin()])
            elif not df_full_timeseries_results[df_full_timeseries_results['integrator'] == name].empty:
                all_runs_for_integrator = df_full_timeseries_results[df_full_timeseries_results['integrator'] == name]
                example_runs.append(all_runs_for_integrator.loc[all_runs_for_integrator['dt'].idxmin()])


        if not example_runs: print("No suitable example runs found for energy component plot."); return
        num_examples = len(example_runs)
        energy_keys = ['hamiltonian_kinetic', 'hamiltonian_gradient', 'hamiltonian_potential']
        energy_labels = ['$\int |v|$', '$\int |\\nabla u|$', '$\int 1/4 |u|^4$']
        
        fig_width = 7
        fig_height = 3.3 * num_examples

        def setup(): return plt.subplots(num_examples, 1, figsize=(fig_width, fig_height), sharex=True, squeeze=False)

        def plot_data(fig, axes_list_2d):
            axes_list = axes_list_2d[:,0]
            for i, run_data in enumerate(example_runs):
                ax = axes_list[i]
                time_points = run_data['metrics']['time_points']
                for key, label in zip(energy_keys, energy_labels):
                    ax.plot(time_points, run_data['metrics'][key], label=label)
                ax.plot(time_points, run_data['metrics']['hamiltonian_total'], label='$H$', linestyle='--', color='black')
                ax.set_ylabel("$E$")
                stability_str = "" if run_data['simulation_stable'] else " (Unstable)"
                title = (f"{run_data['integrator']}\n"
                         f"($N_x={run_data['nx']}$, $\\tau={run_data['dt']:.1e}$){stability_str}")
                ax.set_title(title, fontsize=mpl.rcParams['axes.titlesize']-2)
                ax.legend(loc='best'); ax.grid(True, ls=":", alpha=0.7)
            if num_examples > 0:
                axes_list[-1].set_xlabel("$t$")

        suptitle_text = f"Energy Component Evolution"
        self._plot_generic(setup, plot_data, suptitle_text, "energy_components_evolution", rect_layout=[0.08, 0.05, 0.98, 0.93])


    def _plot_solution_snapshots(self, trajectory_snapshots_dict):
        if not trajectory_snapshots_dict: print("No trajectory data for snapshots."); return

        for snapshot_key, snapshot_info in trajectory_snapshots_dict.items():
            trajectory_data = snapshot_info['traj']
            sim_args = snapshot_info['args']
            integrator_name = snapshot_key.split('_nx')[0]

            num_total_snapshots = trajectory_data.shape[0]
            T_simulation = sim_args['T_sim_actual']
            Lx, Ly = sim_args['Lx'], sim_args['Ly']

            snapshot_abs_times = [T_simulation / 4, T_simulation / 2, 3 * T_simulation / 4, T_simulation * 0.98]

            snapshot_indices = []
            actual_times_for_snapshots = []
            if num_total_snapshots > 1:
                time_per_snapshot_idx = T_simulation / (num_total_snapshots - 1)
                for t_abs in snapshot_abs_times:
                    idx = min(max(0, int(round(t_abs / time_per_snapshot_idx))), num_total_snapshots - 1)
                    snapshot_indices.append(idx)
                    actual_times_for_snapshots.append(idx * time_per_snapshot_idx)
            elif num_total_snapshots == 1:
                snapshot_indices = [0]
                actual_times_for_snapshots = [0.0]

            if not snapshot_indices: continue
            
            num_snap_plots = len(snapshot_indices)
            fig_width = 4.1 * num_snap_plots
            fig_height = 4.0


            def setup(): return plt.subplots(1, num_snap_plots, figsize=(fig_width, fig_height), squeeze=False, sharey=True)

            def plot_data(fig, axes_list_2d):
                axes_list = axes_list_2d[0,:]
                for i, snap_idx in enumerate(snapshot_indices):
                    ax = axes_list[i]
                    field_snapshot = trajectory_data[snap_idx]

                    vmin, vmax = np.nanmin(field_snapshot), np.nanmax(field_snapshot)
                    if not (np.isfinite(vmin) and np.isfinite(vmax)):
                        vmin, vmax = -1, 1
                    if vmin == vmax:
                        vmin = vmax - .25
                        vmax = vmax + .25


                    im = ax.imshow(field_snapshot, extent=[-Lx, Lx, -Ly, Ly],
                            origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')
                    ax.set_title(f"$t = {actual_times_for_snapshots[i]:.2f}$")
                    ax.set_xlabel("$x$")
                    if i == 0: ax.set_ylabel("$y$")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            actual_dt_calc = T_simulation / sim_args['nt_sim_actual']
            ic_name_formatted = self.study_params['ic_type_study']
            suptitle_text = (f"Snapshots: {integrator_name}\n"
                        f"($N_x={sim_args['nx']}$, $\\tau={actual_dt_calc:.1e}$)")
            self._plot_generic(setup, plot_data, suptitle_text, f"snapshots_{snapshot_key}", set_equal=True,
                    rect_layout=[0.05, 0.05, 0.98, 0.88])

    def _plot_solution_differences(self, trajectory_comparison_data):
        if not trajectory_comparison_data or len(trajectory_comparison_data) < 2:
            print("Not enough trajectory data for solution difference plot.")
            return

        df_comp_list = []
        grouped_runs = {}
        for run_info in trajectory_comparison_data:
            key = (run_info['nx'], run_info['dt'])
            if key not in grouped_runs: grouped_runs[key] = []
            grouped_runs[key].append(run_info)

        name1 = self.args_launcher.name1
        name2 = self.args_launcher.name2
        if not name1 or not name2 :
            print("Both integrator names must be specified for difference plotting.")
            return

        for (nx_val, dt_val), runs in grouped_runs.items():
            run1_data = next((r for r in runs if r['name'] == name1), None)
            run2_data = next((r for r in runs if r['name'] == name2), None)
            if not run1_data or not run2_data: continue

            try:
                traj1 = np.load(run1_data['traj_file'])
                traj2 = np.load(run2_data['traj_file'])
            except Exception: continue
            if traj1.shape != traj2.shape: continue

            num_total_snapshots = traj1.shape[0]
            T_simulation = run1_data['T_sim']
            timepoints_to_compare = [T_simulation / 4, T_simulation / 2, 3 * T_simulation / 4]

            for t_compare_abs in timepoints_to_compare:
                snap_idx = 0
                if num_total_snapshots > 1:
                    time_per_snapshot_idx = T_simulation / (num_total_snapshots - 1)
                    snap_idx = min(max(0, int(round(t_compare_abs / time_per_snapshot_idx))), num_total_snapshots - 1)
                actual_time_for_snap = snap_idx * (T_simulation / (num_total_snapshots -1)) if num_total_snapshots > 1 else 0.0
                
                sol1, sol2 = traj1[snap_idx], traj2[snap_idx]
                if np.all(np.isfinite(sol1)) and np.all(np.isfinite(sol2)):
                    diff_l2_norm = np.linalg.norm((sol1 - sol2).ravel())
                    norm_sol1 = np.linalg.norm(sol1.ravel())
                    rel_diff_l2_norm = diff_l2_norm / norm_sol1 if norm_sol1 > 1e-15 else (0.0 if diff_l2_norm < 1e-15 else np.inf)
                    df_comp_list.append({'nx': nx_val, 'dt': dt_val, 'time_approx': actual_time_for_snap, 'rel_l2_diff_norm': rel_diff_l2_norm})
        
        if not df_comp_list:
            print("No valid solution differences computed for plotting.")
            return
        
        df_plot = pd.DataFrame(df_comp_list)
        nx_values = sorted(df_plot['nx'].unique())
        num_cols = len(nx_values)
        fig_width = 5 * num_cols
        fig_height = 4.

        def setup():
            return plt.subplots(1, num_cols, figsize=(fig_width, fig_height), sharey=True, squeeze=False)

        def plot_data_fn(fig, axes_list_2d):
            axes_list = axes_list_2d[0,:]
            time_labels = {
                self.study_params['T_sim_study'] / 4: '$T/4$',
                self.study_params['T_sim_study'] / 2: '$T/2$',
                3 * self.study_params['T_sim_study'] / 4: '$3T/4$'
            }
            approx_times_for_legend = sorted(df_plot['time_approx'].unique())
            
            actual_time_labels = {np.round(k,5): v for k,v in time_labels.items()}

            for i, nx_val in enumerate(nx_values):
                ax = axes_list[i]
                sub_df_nx = df_plot[df_plot['nx'] == nx_val]
                for t_approx in approx_times_for_legend:
                    sub_df_nx_t = sub_df_nx[np.isclose(sub_df_nx['time_approx'], t_approx)]
                    if not sub_df_nx_t.empty:
                        label_t = actual_time_labels.get(np.round(t_approx,5), f'$t \\approx {t_approx:.2f}$')
                        ax.plot(sub_df_nx_t['dt'], sub_df_nx_t['rel_l2_diff_norm'], marker='o', linestyle='-', label=label_t)
                ax.set_xlabel(r"$\tau$")
                ax.set_title(f"$N_x = {nx_val}$")
                if i == 0: ax.set_ylabel(f"$L^2$ / [1]")
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.legend(title="$t$ / [1]", loc='best')
                ax.grid(True, which="both", ls=":", alpha=0.7)
        
        suptitle_text = f"Relative $L^2$ Distance: ({name1} vs {name2})"
        self._plot_generic(setup, plot_data_fn, suptitle_text, "solution_L2_differences")


    def execute(self):
        print(f"Starting {self.system_name} Integrator Study (ID: {self.run_id})")
        all_results_summary_list = []
        all_results_full_timeseries_list = []
        trajectories_for_snapshots = {}
        self.all_trajectory_data_for_comparison = []

        integrators_to_run = []
        if self.args_launcher.exe1 and self.args_launcher.name1:
            integrators_to_run.append({'name': self.args_launcher.name1, 'exe_path': Path(self.args_launcher.exe1)})
        if self.args_launcher.exe2 and self.args_launcher.name2:
            integrators_to_run.append({'name': self.args_launcher.name2, 'exe_path': Path(self.args_launcher.exe2)})

        if not integrators_to_run:
            print("No integrators specified to run. Exiting.", file=sys.stderr); return

        for integrator_info in integrators_to_run:
            integrator_name, exe_path = integrator_info['name'], integrator_info['exe_path']
            for nx_val in self.study_params['nx_values_study']:
                sim_args = {
                    'nx': nx_val, 'ny': nx_val,
                    'Lx': self.study_params['L_study'], 'Ly': self.study_params['L_study'],
                    'T_sim_actual': self.study_params['T_sim_study']
                }
                temp_dir_for_nx = self.temp_dir_base / f"nx{nx_val}_{integrator_name}"
                temp_dir_for_nx.mkdir(exist_ok=True, parents=True)

                try:
                    u0_f, v0_f, m_f, c_f = self._get_or_create_downsampled_inputs_for_nx(nx_val, nx_val, temp_dir_for_nx)
                    m_field_np, c_field_np = np.load(m_f), np.load(c_f)
                except Exception as e_ic:
                    print(f"Failed to get/create initial conditions for {integrator_name}, nx={nx_val}. Skipping. Error: {e_ic}", file=sys.stderr)
                    continue

                for dt_val in self.study_params['dt_values_study']:
                    sim_args['nt_sim_actual'] = int(round(sim_args['T_sim_actual'] / dt_val))
                    if sim_args['nt_sim_actual'] == 0:
                        print(f"Skipping {integrator_name}, nx={nx_val}, dt={dt_val} due to nt_sim_actual=0.", file=sys.stderr)
                        continue

                    sim_args['num_snapshots_actual'] = min(sim_args['nt_sim_actual'] + 1, self.max_snapshots)

                    temp_dir_for_dt_run = temp_dir_for_nx / f"dt{dt_val:.3e}"
                    temp_dir_for_dt_run.mkdir(exist_ok=True, parents=True)

                    try:
                        traj_file, vel_file, wall_time = self._run_single_simulation(
                            integrator_name, exe_path, sim_args, u0_f, v0_f, m_f, c_f, temp_dir_for_dt_run
                        )
                        self.all_trajectory_data_for_comparison.append({
                            'name': integrator_name, 'nx': nx_val, 'dt': dt_val,
                            'T_sim': sim_args['T_sim_actual'],
                            'num_snaps': sim_args['num_snapshots_actual'],
                            'traj_file': traj_file
                        })

                        metrics, trajectory_data = self._analyze_trajectory(traj_file, vel_file, sim_args, m_field_np, c_field_np)

                        summary_data = {
                            'integrator': integrator_name, 'nx': nx_val, 'dt': dt_val,
                            'T_sim': sim_args['T_sim_actual'], 'walltime': wall_time,
                            'final_mass_log10_rel_error': metrics['mass_log10_rel_error'][-1],
                            'final_hamiltonian_log10_rel_error': metrics['hamiltonian_log10_rel_error'][-1],
                            'max_abs_hamiltonian_rel_error': metrics['max_abs_hamiltonian_rel_error'],
                            'simulation_stable': metrics['simulation_stable']
                        }
                        all_results_summary_list.append(summary_data)

                        if nx_val == max(self.study_params['nx_values_study']):
                             all_results_full_timeseries_list.append({**summary_data, 'metrics': metrics})

                        if dt_val == min(self.study_params['dt_values_study']):
                            trajectories_for_snapshots[f"{integrator_name}_nx{nx_val}"] = {'traj': trajectory_data, 'args': sim_args.copy()}

                    except Exception as e_run:
                        print(f"Failed run or analysis: {integrator_name}, nx={nx_val}, dt={dt_val}. Error: {e_run}", file=sys.stderr)
                        failed_summary = {
                            'integrator': integrator_name, 'nx': nx_val, 'dt': dt_val,
                            'T_sim': sim_args['T_sim_actual'], 'walltime': np.nan,
                            'final_mass_log10_rel_error': np.nan, 'final_hamiltonian_log10_rel_error': np.nan,
                            'max_abs_hamiltonian_rel_error': np.nan, 'simulation_stable': False
                        }
                        all_results_summary_list.append(failed_summary)
                        continue
                    finally:
                        if not self.args_launcher.keep_temps:
                            try:
                                for temp_f in temp_dir_for_dt_run.glob('*'):
                                    if temp_f.name != traj_file.name:
                                        if temp_f.exists(): os.unlink(temp_f)
                            except OSError as e_rm:
                                print(f"Error during selective temp file removal in {temp_dir_for_dt_run}: {e_rm}", file=sys.stderr)

        if len(integrators_to_run) >= 2:
             self._plot_solution_differences(self.all_trajectory_data_for_comparison)


        if not all_results_summary_list:
            print("No simulation results to process. Exiting.", file=sys.stderr); return

        df_summary = pd.DataFrame(all_results_summary_list)

        self._plot_convergence(df_summary, 'final_mass_log10_rel_error',
                               r"$\log(|m(T)-m_0|/|m_0|)$", "Mass Conservation Error", "mass_error_log10")
        self._plot_convergence(df_summary, 'final_hamiltonian_log10_rel_error',
                               r"$\log(|H-H_0|/|H_0|)$", "$H$ Conservation Error", "hamiltonian_error_log10")

        self._plot_work_precision(df_summary, 'max_abs_hamiltonian_rel_error',
                                  r"$\max_t |(E(t)-E_0)/E_0|$",
                                  "$\max_t \quad H$ Error", "hamiltonian_error_abs_rel_vs_walltime") 

        if all_results_full_timeseries_list:
            df_full_ts = pd.DataFrame(all_results_full_timeseries_list)
            self._plot_energy_component_evolution(df_full_ts)

        self._plot_solution_snapshots(trajectories_for_snapshots)

        if not self.args_launcher.keep_temps:
            print("Cleaning up temporary run files...")
            for integrator_info in integrators_to_run:
                integrator_name = integrator_info['name']
                for nx_val in self.study_params['nx_values_study']:
                    temp_dir_for_nx = self.temp_dir_base / f"nx{nx_val}_{integrator_name}"
                    for dt_val in self.study_params['dt_values_study']:
                        temp_dir_for_dt_run = temp_dir_for_nx / f"dt{dt_val:.3e}"
                        if temp_dir_for_dt_run.exists():
                            try:
                                for f_path in temp_dir_for_dt_run.glob('*'): os.unlink(f_path)
                                os.rmdir(temp_dir_for_dt_run)
                            except OSError as e_rm: print(f"Error final cleanup {temp_dir_for_dt_run}: {e_rm}", file=sys.stderr)
                    if temp_dir_for_nx.exists():
                        try:
                            if not any(temp_dir_for_nx.iterdir()):
                                for ic_p_name in ["u0.npy", "v0.npy", "m.npy", "c.npy"]:
                                    ic_f = temp_dir_for_nx / ic_p_name
                                    if ic_f.exists(): os.unlink(ic_f)
                                os.rmdir(temp_dir_for_nx)
                        except OSError as e_rm: print(f"Error final cleanup {temp_dir_for_nx}: {e_rm}", file=sys.stderr)

            if self.temp_dir_base.exists() and not any(self.temp_dir_base.iterdir()):
                try:
                    os.rmdir(self.temp_dir_base)
                except OSError as e_rm: print(f"Error final cleanup {self.temp_dir_base}: {e_rm}", file=sys.stderr)
        elif self.args_launcher.keep_temps:
            print(f"Temporary files kept in {self.temp_dir_base}")

        print(f"Study {self.run_id} finished. Outputs in {self.study_output_dir}")
