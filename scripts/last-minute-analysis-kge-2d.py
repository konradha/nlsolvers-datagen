import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal import windows
import argparse
import h5py
import os
from pathlib import Path

def plot_2d_data_on_ax(ax, data, x_coords, y_coords, title, xlabel="x", ylabel="y", cmap="viridis", aspect='auto', cbar_label="Value"):
    im = ax.imshow(data.T, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                   origin='lower', aspect=aspect, cmap=cmap)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def load_kge_data_from_hdf5(filepath):
    data_dict = {}
    with h5py.File(filepath, 'r') as f:
        data_dict['Nx'] = f['grid'].attrs['nx']
        data_dict['Ny'] = f['grid'].attrs['ny']
        data_dict['Lx'] = f['grid'].attrs['Lx']
        data_dict['Ly'] = f['grid'].attrs['Ly']
        data_dict['T_final'] = f['time'].attrs['T']
        data_dict['num_snapshots'] = f['time'].attrs['num_snapshots']
        data_dict['u_all_times'] = f['u'][:]
        data_dict['v_all_times'] = f['v'][:]
        data_dict['m_xy'] = f['focusing/m'][:]
    expected_u_shape = (data_dict['num_snapshots'], data_dict['Nx'], data_dict['Ny'])

    data_dict['x_coords'] = np.linspace(-data_dict['Lx'], data_dict['Lx'], data_dict['Nx'])
    data_dict['y_coords'] = np.linspace(-data_dict['Ly'], data_dict['Ly'], data_dict['Ny'])
    data_dict['t_coords'] = np.linspace(0, data_dict['T_final'], data_dict['num_snapshots'])
    return data_dict

class KGEAnalyzer:
    def __init__(self, u_data, v_data, m_xy_data, x_coords, y_coords, t_coords, output_dir, base_filename):
        self.u_data = u_data
        self.v_data = v_data
        self.m_xy = m_xy_data
        self.x = x_coords
        self.y = y_coords
        self.t = t_coords
        self.Nx = len(x_coords)
        self.Ny = len(y_coords)
        self.num_snapshots = len(t_coords)
        self.output_dir = Path(output_dir)
        self.base_filename = base_filename
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.Nx > 1:
            self.dx = self.x[1] - self.x[0]
        else:
            self.dx = 1.0
        if self.Ny > 1:
            self.dy = self.y[1] - self.y[0]
        else:
            self.dy = 1.0
        self.kx_full = fftshift(fftfreq(self.Nx, d=self.dx) * 2 * np.pi)
        self.ky_full = fftshift(fftfreq(self.Ny, d=self.dy) * 2 * np.pi)

    def get_fields_at_time_index(self, t_idx):
        if not (0 <= t_idx < self.num_snapshots):
            raise ValueError(f"Time index {t_idx} out of bounds [0, {self.num_snapshots-1}]")
        u_field = self.u_data[t_idx, :, :]
        v_field = self.v_data[t_idx, :, :]
        return u_field, v_field

    def compute_observables(self, u_field, v_field):
        grad_u_y, grad_u_x = np.gradient(u_field, self.dy, self.dx)
        kinetic_energy_dens = 0.5 * v_field**2
        gradient_energy_dens = 0.5 * (grad_u_x**2 + grad_u_y**2)
        potential_energy_dens = 0.25 *  u_field**4
        total_energy_dens = kinetic_energy_dens + gradient_energy_dens + potential_energy_dens
        return u_field, v_field, kinetic_energy_dens, gradient_energy_dens, potential_energy_dens, total_energy_dens

    def compute_local_k_spectrum(self, field_to_analyze, x_idx_center, y_idx_center,
                                 window_size_x, window_size_y):
        half_wx = window_size_x // 2
        half_wy = window_size_y // 2
        x_start = max(0, x_idx_center - half_wx)
        x_end = min(self.Nx, x_idx_center + half_wx + (window_size_x % 2))
        y_start = max(0, y_idx_center - half_wy)
        y_end = min(self.Ny, y_idx_center + half_wy + (window_size_y % 2))
        sub_field = field_to_analyze[x_start:x_end, y_start:y_end]
        actual_ws_x = sub_field.shape[0]
        actual_ws_y = sub_field.shape[1]
        if actual_ws_x == 0 or actual_ws_y == 0:
            print(f"Warning: Window at ({x_idx_center},{y_idx_center}) with size ({window_size_x},{window_size_y}) resulted in empty sub-region. Skipping.")
            return np.zeros((self.Nx, self.Ny)), self.kx_full, self.ky_full
        win_x = windows.tukey(actual_ws_x, alpha=0.25)
        win_y = windows.tukey(actual_ws_y, alpha=0.25)
        tukey_window_2d = np.outer(win_x, win_y)
        sub_region_times_window = sub_field * tukey_window_2d
        padded_for_fft = np.zeros((self.Nx, self.Ny), dtype=sub_field.dtype)
        pad_x_start = (self.Nx - actual_ws_x) // 2
        pad_y_start = (self.Ny - actual_ws_y) // 2
        padded_for_fft[pad_x_start : pad_x_start+actual_ws_x,
                       pad_y_start : pad_y_start+actual_ws_y] = sub_region_times_window
        local_Fk = fftshift(fft2(padded_for_fft))
        local_k_spectrum = np.abs(local_Fk)**2
        return local_k_spectrum, self.kx_full, self.ky_full

    def plot_m_xy_data(self):
        if self.m_xy is not None:
            fig_m, ax_m = plt.subplots(figsize=(8,6))
            plot_2d_data_on_ax(ax_m, self.m_xy, self.x, self.y, title="$m(x,y)$ KGE Coefficient")
            fig_m.savefig(self.output_dir /
                          f"{self.base_filename}_m_xy_profile.png", dpi=300)
            plt.close(fig_m)
        else:
            print("m(x,y) data not available.")

    def analyze_timestep_data(self, t_idx, plot_basics=True,
                              plot_energy_comp=True, plot_energy_ratio=True,
                              plot_radial_spec=True, local_k_points_frac=None,
                              window_frac=0.25):
        u_t, v_t = self.get_fields_at_time_index(t_idx)
        time_val = self.t[t_idx]
        filename_suffix = f"_t_idx_{t_idx}.png"
        print(f"\n--- Analysis for KGE at t = {time_val:.3f} (snapshot index {t_idx}) ---")
        u_plot, v_plot, kin_e_dens, grad_e_dens, pot_e_dens, total_e_dens = self.compute_observables(u_t, v_t)

        if plot_basics:
            fig_u, ax_u = plt.subplots(figsize=(8,6))
            plot_2d_data_on_ax(ax_u, u_plot, self.x, self.y, f"Field $u(x,y)$ at t={time_val:.3f}", cmap="RdBu_r", cbar_label="$u$")
            fig_u.savefig(self.output_dir /
                          f"{self.base_filename}_u_field{filename_suffix}",
                          dpi=300)
            plt.close(fig_u)
            fig_v, ax_v = plt.subplots(figsize=(8,6))
            plot_2d_data_on_ax(ax_v, v_plot, self.x, self.y, f"Velocity $v(x,y) = u_t$ at t={time_val:.3f}", cmap="RdBu_r", cbar_label="$v$")
            fig_v.savefig(self.output_dir /
                          f"{self.base_filename}_v_field{filename_suffix}",
                          dpi=300)
            plt.close(fig_v)
            fig_etot, ax_etot = plt.subplots(figsize=(8,6))
            plot_2d_data_on_ax(ax_etot, total_e_dens, self.x, self.y, f"Total Energy Density $E(x,y)$ at t={time_val:.3f}", cbar_label="$E$")
            fig_etot.savefig(self.output_dir /
                             f"{self.base_filename}_total_energy_dens{filename_suffix}",
                             dpi=300)
            plt.close(fig_etot)

        if plot_energy_comp:
            fig_ec, axes_ec = plt.subplots(1, 3, figsize=(22, 6))
            plot_2d_data_on_ax(axes_ec[0], kin_e_dens, self.x, self.y, f"$0.5 v^2$", cbar_label="E dens", cmap="inferno")
            plot_2d_data_on_ax(axes_ec[1], grad_e_dens, self.x, self.y, f"$0.5 |\\nabla u|^2$", cbar_label="E dens", cmap="inferno")
            plot_2d_data_on_ax(axes_ec[2], pot_e_dens, self.x, self.y, f"$0.25 m u^4$", cbar_label="E dens", cmap="inferno")
            fig_ec.suptitle(f"Energy Component Comparison at t={time_val:.3f}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_ec.savefig(self.output_dir /
                           f"{self.base_filename}_energy_components{filename_suffix}",
                           dpi=300)
            plt.close(fig_ec)

        if plot_energy_ratio:
            fig_er, ax_er = plt.subplots(figsize=(8, 6))
            linear_energy_dens = kin_e_dens + grad_e_dens
            ratio_val = pot_e_dens / (linear_energy_dens + 1e-12)
            plot_2d_data_on_ax(ax_er, np.log10(ratio_val + 1e-12), self.x, self.y, f"Log10 (Nonlinear / Linear Energy Ratio) at t={time_val:.3f}", cbar_label="Log10 (Ratio)", cmap="coolwarm")
            fig_er.savefig(self.output_dir /
                           f"{self.base_filename}_energy_ratio{filename_suffix}",
                           dpi=300)
            plt.close(fig_er)

        if plot_radial_spec:
            fig_rs, ax_rs = plt.subplots(figsize=(8, 6))
            Nx_loc, Ny_loc = u_t.shape
            Fk_u_loc = fftshift(fft2(u_t))
            power_spectrum_u_loc = np.abs(Fk_u_loc)**2
            kx_vals_loc = fftshift(fftfreq(Nx_loc, d=self.dx) * 2 * np.pi)
            ky_vals_loc = fftshift(fftfreq(Ny_loc, d=self.dy) * 2 * np.pi)
            KX_loc, KY_loc = np.meshgrid(kx_vals_loc, ky_vals_loc, indexing='ij')
            k_radial_loc = np.sqrt(KX_loc**2 + KY_loc**2)
            dk_loc = min(kx_vals_loc[1]-kx_vals_loc[0] if Nx_loc > 1 else np.inf, ky_vals_loc[1]-ky_vals_loc[0] if Ny_loc > 1 else np.inf)
            if dk_loc == np.inf : dk_loc = 1.0
            k_bins_loc = np.arange(0, k_radial_loc.max() + dk_loc, dk_loc)
            k_bin_centers_loc = (k_bins_loc[:-1] + k_bins_loc[1:]) / 2
            radial_ps_loc = np.zeros(len(k_bin_centers_loc))
            for i_bin in range(len(k_bin_centers_loc)):
                mask_loc = (k_radial_loc >= k_bins_loc[i_bin]) & (k_radial_loc < k_bins_loc[i_bin+1])
                if np.any(mask_loc):
                    radial_ps_loc[i_bin] = power_spectrum_u_loc[mask_loc].mean()
                else:
                    radial_ps_loc[i_bin] = 0
            ax_rs.plot(k_bin_centers_loc, radial_ps_loc)
            ax_rs.set_xlabel("Wavenumber k"); ax_rs.set_ylabel("Radially Averaged Power P(k)")
            ax_rs.set_title(f"Radially Averaged Power Spectrum of u at t={time_val:.3f}")
            ax_rs.set_yscale('log'); ax_rs.grid(True, which="both", ls="-", alpha=0.5)
            fig_rs.savefig(self.output_dir /
                           f"{self.base_filename}_radial_spectrum_u{filename_suffix}",
                           dpi=300)
            plt.close(fig_rs)

        if local_k_points_frac:
            ws_x = int(self.Nx * window_frac); ws_y = int(self.Ny * window_frac)
            if ws_x % 2 == 0: ws_x = max(1, ws_x -1)
            if ws_y % 2 == 0: ws_y = max(1, ws_y -1)
            ws_x = min(ws_x, self.Nx); ws_y = min(ws_y, self.Ny)
            for x_frac, y_frac in local_k_points_frac:
                x_idx_c = int(x_frac * (self.Nx -1)); y_idx_c = int(y_frac * (self.Ny -1))
                x_val = self.x[x_idx_c]; y_val = self.y[y_idx_c]
                point_filename_suffix = f"_x{x_val:.2f}_y{y_val:.2f}_t_idx_{t_idx}.png".replace(".","_")
                print(f"Computing local k-spectrum of u at (x={x_val:.2f}, y={y_val:.2f}) [idx {x_idx_c},{y_idx_c}] with window ({ws_x},{ws_y})")
                spectrum, kx_coords, ky_coords = self.compute_local_k_spectrum(u_t, x_idx_c, y_idx_c, ws_x, ws_y)
                fig_k, ax_k = plt.subplots(figsize=(8,6)); plot_2d_data_on_ax(ax_k, np.log10(spectrum + 1e-12), kx_coords, ky_coords,
                        f"Log Local k-spectrum of u at (x={x_val:.2f}, y={y_val:.2f}), t={time_val:.3f}",
                        xlabel="$k_x$", ylabel="$k_y$", aspect=1.0, cbar_label="Log Power")
                fig_k.savefig(self.output_dir /
                              f"{self.base_filename}_local_k_spectrum_u{point_filename_suffix}",
                              dpi=300)
                plt.close(fig_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze KGE simulation data from HDF5.")
    parser.add_argument("hdf5_filepath", type=str, help="Path to the HDF5 file containing KGE simulation data.")
    parser.add_argument("--output_dir", type=str, default="kge_analysis_plots", help="Directory to save plots.")
    parser.add_argument("--t_indices", type=str, default="0,-1", help="Comma-separated list of snapshot indices to analyze (e.g., 0,10,20 or 0,-1 for first and last).")
    parser.add_argument("--k_points", type=str, default="0.5,0.5", help="Semicolon-separated (x_frac,y_frac) pairs for local k-spectrum.")
    parser.add_argument("--window_frac", type=float, default=0.25, help="Fraction of domain size for local k-spectrum window.")
    parser.add_argument("--no_basics", action="store_true", help="Skip basic u, v, total energy plots.")
    parser.add_argument("--no_energy_comp", action="store_true", help="Skip energy component comparison plot.")
    parser.add_argument("--no_energy_ratio", action="store_true", help="Skip nonlinear/linear energy ratio plot.")
    parser.add_argument("--no_radial_spec", action="store_true", help="Skip radially averaged power spectrum plot.")

    args = parser.parse_args()
    base_fname_for_plots = Path(args.hdf5_filepath).stem
    loaded_data = load_kge_data_from_hdf5(args.hdf5_filepath)
    analyzer = KGEAnalyzer(
        u_data=loaded_data['u_all_times'], v_data=loaded_data['v_all_times'], m_xy_data=loaded_data['m_xy'],
        x_coords=loaded_data['x_coords'], y_coords=loaded_data['y_coords'], t_coords=loaded_data['t_coords'],
        output_dir=args.output_dir, base_filename=base_fname_for_plots
    )
    analyzer.plot_m_xy_data()
    time_indices_to_analyze = []
    num_total_snapshots = loaded_data['num_snapshots']
    if args.t_indices:
        idx_strs = args.t_indices.split(',')
        for idx_str in idx_strs:
            try: idx = int(idx_str)
            except ValueError: print(f"Warning: Could not parse snapshot index '{idx_str}'. Skipping."); continue
            if idx < 0: idx = num_total_snapshots + idx
            if 0 <= idx < num_total_snapshots: time_indices_to_analyze.append(idx)
            else: print(f"Warning: Snapshot index {idx_str} (parsed as {idx}) is out of bounds for num_snapshots={num_total_snapshots}. Skipping.")
    if not time_indices_to_analyze:
         time_indices_to_analyze = [0]
         if num_total_snapshots > 1: time_indices_to_analyze.extend([num_total_snapshots // 2, num_total_snapshots - 1])
         time_indices_to_analyze = sorted(list(set(time_indices_to_analyze)))
    k_points_list_frac = []
    if args.k_points:
        point_pairs_str = args.k_points.split(';')
        for pair_str in point_pairs_str:
            try: x_f_str, y_f_str = pair_str.split(','); k_points_list_frac.append((float(x_f_str), float(y_f_str)))
            except ValueError: print(f"Warning: Could not parse k_point pair '{pair_str}'. Expected format 'x_frac,y_frac'. Skipping.")
    for t_idx in time_indices_to_analyze:
        analyzer.analyze_timestep_data(t_idx,
                                     plot_basics=not args.no_basics,
                                     plot_energy_comp=not args.no_energy_comp,
                                     plot_energy_ratio=not args.no_energy_ratio,
                                     plot_radial_spec=not args.no_radial_spec,
                                     local_k_points_frac=k_points_list_frac,
                                     window_frac=args.window_frac)
    print(f"\nKGE Analysis complete. Plots saved to: {Path(args.output_dir).resolve()}")
