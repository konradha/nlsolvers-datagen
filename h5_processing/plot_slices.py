import argparse
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "mathtext.fontset": "cm",
})


def format_params(attrs):
    lines = []
    lines.append(r"\textbf{--- System \& Grid ---}")
    problem_type_raw = attrs.get('problem_type', 'N/A')
    problem_type_latex = problem_type_raw
    lines.append(f"Type: {problem_type_latex}")
    lines.append(f"Grid: {attrs.get('nx', '?')}x{attrs.get('ny', '?')}x{attrs.get('nz', '?')}")
    lines.append(f"Domain: $[-{attrs.get('Lx', '?'):.1f}, {attrs.get('Lx', '?'):.1f}]^3$")
    lines.append(f"BC: {attrs.get('boundary_condition', 'N/A')}")

    lines.append(r"\textbf{\\--- Time ---}")
    lines.append(f"$T = {attrs.get('T', '?'):.2f}$")
    lines.append(f"Steps ($n_t$): {attrs.get('nt', '?')}")
    lines.append(f"Snapshots: {attrs.get('num_snapshots', '?')}")
    lines.append(f"Walltime: {attrs.get('elapsed_time', -1):.2f} s")

    lines.append(r"\textbf{\\--- Coefficients ---}")
    parent_dir_name = Path(attrs.get('filename_hdf5', '')).parent.parent.name

    c_m_pair_inferred = 'N/A'
    prefixes = ["kge_kink_field_", "nlse_multi_soliton_state_", "nlse_skyrmion_tube_"]
    for prefix in prefixes:
        if parent_dir_name.startswith(prefix):
            potential_pair = parent_dir_name[len(prefix):]
            if potential_pair.endswith("_with_visuals"):
                 potential_pair = potential_pair[:-len("_with_visuals")]
            c_m_pair_inferred = potential_pair
            break

    c_m_pair_latex = c_m_pair_inferred
    lines.append(f"c, m Pair: {c_m_pair_latex}")

    lines.append(r"\textbf{\\--- Initial Condition ---}")
    phenomenon_raw = attrs.get('phenomenon', 'N/A')
    phenomenon_latex = phenomenon_raw
    lines.append(f"Phenomenon: {phenomenon_latex}")
    phenom_keys = sorted([k for k in attrs if k.startswith('phenomenon_')])
    for k in phenom_keys:
        key_raw = k.replace('phenomenon_', '')
        key_short_latex = key_raw
        val_raw = attrs[k]
        val_str = str(val_raw)

        if isinstance(val_raw, bytes):
            val_str = val_raw.decode('utf-8')

        if isinstance(val_str, str) and val_str.startswith('(') and val_str.endswith(')'):
             try:
                 eval_val = eval(val_str)
                 if isinstance(eval_val, tuple) and len(eval_val) == 2:
                     val_str = f"({eval_val[0]:.2g}, {eval_val[1]:.2g})"
             except:
                 pass

        if key_short_latex == 'velocity\\_type':
            lines.append(f"{key_short_latex}: {val_str}")
        else:
            lines.append(f"{key_short_latex}: {val_str}")


    return "\n".join(lines)


def calculate_energy_terms(u_snap, v_snap, c, m, dx, dy, dz, problem_type):
    if len(u_snap.shape) == 2:
        dV = dx * dy
        grad_u_x, grad_u_y = np.gradient(u_snap, dx, dy, axis=(0, 1))
        grad_term_integrand = (np.abs(grad_u_x)**2 + np.abs(grad_u_y)**2)
    elif len(u_snap.shape) == 3:
        dV = dx * dy * dz
        grad_u_x, grad_u_y, grad_u_z = np.gradient(u_snap, dx, dy, dz, axis=(0, 1, 2))
        grad_term_integrand = (np.abs(grad_u_x)**2 + np.abs(grad_u_y)**2 + np.abs(grad_u_z)**2)
    else:
        raise NotImplemented

    if problem_type == 'klein_gordon':
        kinetic_term = 0.5 * np.sum(v_snap**2) * dV
        gradient_term = 0.5 * np.sum(grad_term_integrand) * dV
        potential_term = 0.5 * np.sum(u_snap**4) * dV
        total_energy = kinetic_term + gradient_term + potential_term
        return total_energy, kinetic_term, gradient_term, potential_term

    elif problem_type == 'cubic':
        gradient_term = np.sum(grad_term_integrand) * dV
        potential_term = -0.5 * np.sum((np.abs(u_snap)**4)) * dV
        total_energy = gradient_term + potential_term
        return total_energy, np.nan, gradient_term, potential_term
    else:
        return np.nan, np.nan, np.nan, np.nan

def plot_analysis_figure(times, energies, kinetic_energies, gradient_energies, potential_energies, u_data, u0, output_filename, problem_type):
    num_snapshots = len(times)
    is_complex = np.iscomplexobj(u_data)
    eps = 1e-18

    max_amp = np.zeros(num_snapshots)
    max_amp_diff_rel = np.zeros(num_snapshots)
    energy_rel_diff = np.zeros(num_snapshots)

    if is_complex:
        u0_max = np.max(np.abs(u0)) + eps
    else:
        u0_max = np.max(np.abs(u0)) + eps

    E0 = energies[0]

    for i in range(num_snapshots):
        u_t = u_data[i]
        if is_complex:
            max_amp[i] = np.max(np.abs(u_t))
        else:
            max_amp[i] = np.max(np.abs(u_t))

        if i == 0:
            max_amp_diff_rel[i] = np.nan
            energy_rel_diff[i] = np.nan
        else:
            max_amp_diff_rel[i] = np.max(np.abs(u_t - u_data[0])) / u0_max
            energy_rel_diff[i] = np.abs(energies[i] - E0) / (np.abs(E0) + eps)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(times, np.log10(energy_rel_diff), label=r'$\log_{10}(|E(t)-E(0)| / |E(0)|)$', color='black', linestyle='-', marker='.')
    ax1.plot(times, np.log10(max_amp_diff_rel), label=r'$\log_{10}(\max|u(t)-u(0)| / \max|u(0)|)$', color='red', linestyle='--', marker='+')
    ax1.set_ylabel(r'$\log_{10}$ Relative Difference')
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=':')
    ax1.set_title('Relative Energy and Max Amplitude Deviation')

    ax2 = axes[1]
    if problem_type == 'klein_gordon':
        ax2.plot(times, kinetic_energies, label=r'$E_{\mathrm{kin}}(t)$', color='blue', linestyle='-')
        ax2.plot(times, gradient_energies, label=r'$E_{\mathrm{grad}}(t)$', color='green', linestyle='--')
        ax2.plot(times, potential_energies, label=r'$E_{\mathrm{pot}}(t)$', color='purple', linestyle=':')
        ax2.plot(times, energies, label=r'$E_{\mathrm{tot}}(t)$', color='black', linestyle='-', linewidth=1.5)
    elif problem_type == 'cubic':
        ax2.plot(times, gradient_energies, label=r'$E_{\mathrm{grad}}(t)$', color='green', linestyle='--')
        ax2.plot(times, potential_energies, label=r'$E_{\mathrm{pot}}(t)$', color='purple', linestyle=':')
        ax2.plot(times, energies, label=r'$E_{\mathrm{tot}}(t)$', color='black', linestyle='-', linewidth=1.5)
    else:
        raise NotImplemented
    ax2.set_ylabel('Energy Components')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':')
    ax2.set_title('Energy Term Evolution (Absolute Values)')

    ax3 = axes[2]
    ax3.plot(times, max_amp, label=r'$\max |u(t)|$', color='darkorange', marker='x', markersize=4, linestyle='-')
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel('Max Amplitude')
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=':')
    ax3.set_title('Maximum Amplitude Evolution')
    try:
        min_val = np.min(max_amp)
        max_val = np.max(max_amp)
        if not np.isclose(min_val, max_val):
           ax3.set_ylim(bottom=min_val - 0.05 * (max_val-min_val+eps))
    except:
        pass

    plot_stem = Path(output_filename).stem.replace("_analysis", "")
    plot_title_latex = plot_stem
    plt.suptitle(f'Analysis: {plot_title_latex}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    print(f"Analysis plot saved to {output_filename}")

def plot_trajectory_slices(h5_filepath, output_dir):
    filepath = Path(h5_filepath)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_filename_slices = output_path / f"{filepath.stem}_slices.png"
    plot_filename_analysis = output_path / f"{filepath.stem}_analysis.png"

    try:
        with h5py.File(filepath, 'r') as f:
            metadata = dict(f['metadata'].attrs)
            metadata['filename_hdf5'] = str(h5_filepath)
            grid_attrs = dict(f['grid'].attrs)
            time_attrs = dict(f['time'].attrs)
            metadata.update(grid_attrs)
            metadata.update(time_attrs)

            problem_type = metadata.get('problem_type', 'unknown')
            is_kge = (problem_type == 'klein_gordon')
            is_nlse_cubic = (problem_type == 'cubic') 

            u = f['u'][()]
            u0 = f['initial_condition/u0'][()]
            is_2d = len(u.shape) == 3  
            is_3d = len(u.shape) == 4  

            if not is_2d and not is_3d: raise Exception("Dims?")

            if is_kge and 'v' in f:
                v = f['v'][()]
                v0 = f['initial_condition/v0'][()]
            elif is_nlse_cubic:
                v = None
                v0 = None
            else:
                raise NotImplemented

            # what happens when you don't carefully define data schemata ...
            c = f['anisotropy/c'][()] if 'anisotropy/c' in f else (f['focusing/c'][()] if 'focusing/c' in f else f['c'][()])
            m = f['focusing/m'][()] if 'focusing/m' in f else f['m'][()]
            # c = f['anisotropy/c'][()] if 'anisotropy/c' in f else f['c'][()]
            # m = f['focusing/m'][()] if 'focusing/m' in f else f['m'][()]
            # c = f['anisotropy/c'][()]
            # m = f['focusing/m'][()]

            nx = metadata['nx']
            ny = metadata['ny']
            nz = metadata.get('nz', 1)
            Lx = metadata['Lx']
            Ly = metadata['Ly']
            Lz = metadata.get('Lz', 0)

            dx = 2 * Lx / (nx - 1) if nx > 1 else 2 * Lx
            dy = 2 * Ly / (ny - 1) if ny > 1 else 2 * Ly
            dz = 2 * Lz / (nz - 1) if nz > 1 else 2 * Lz

            T = metadata['T']
            num_snapshots = metadata['num_snapshots']

            if is_3d:
                slice_idx_z = nz // 2
                c_slice = c[:, :, slice_idx_z]
                m_slice = m[:, :, slice_idx_z]
            else:
                c_slice = c
                m_slice = m

            times_to_plot = [0, T/4, T/2, 3*T/4, T]
            snapshot_indices = np.linspace(0, num_snapshots - 1, num_snapshots, dtype=int)
            time_points = np.linspace(0, T, num_snapshots)

            plot_indices = []
            actual_times = []
            for t_target in times_to_plot:
                idx = np.argmin(np.abs(time_points - t_target))
                plot_indices.append(snapshot_indices[idx])
                actual_times.append(time_points[idx])

            if is_3d:
                u_slices = u[plot_indices, :, :, slice_idx_z]
                if v is not None:
                    v_slices = v[plot_indices, :, :, slice_idx_z]
                else:
                    v_slices = None
            else:
                u_slices = u[plot_indices]
                if v is not None:
                    v_slices = v[plot_indices]
                else:
                    v_slices = None

            is_complex = np.iscomplexobj(u)

            if is_complex:
                u_mag_min = np.min(np.abs(u_slices))
                u_mag_max = np.max(np.abs(u_slices))
                u_phase_min = -np.pi
                u_phase_max = np.pi
                cmap_u_mag = 'viridis'
                cmap_u_phase = 'hsv'
            else:
                u_min = np.min(u_slices)
                u_max = np.max(u_slices)
                if np.isclose(u_min, u_max):
                    u_min -= 0.1
                    u_max += 0.1
                cmap_u = 'coolwarm'

                if v_slices is not None:
                    v_min = np.min(v_slices)
                    v_max = np.max(v_slices)
                    if np.isclose(v_min, v_max):
                        v_min -= 0.1
                        v_max += 0.1
                    cmap_v = 'coolwarm'

            c_min, c_max = np.min(c), np.max(c)
            m_min, m_max = np.min(m), np.max(m)
            m_cmap = 'viridis'
            m_vmin, m_vmax = m_min, m_max
            if m_min < -1e-6 and m_max > 1e-6:
                m_abs_max = max(abs(m_min), abs(m_max))
                m_vmin, m_vmax = -m_abs_max, m_abs_max
                m_cmap = 'coolwarm'

            num_rows = 5 if is_kge else 4
            num_cols_u = 2 if not is_complex else 4

            fig = plt.figure(figsize=(5 * num_cols_u, 3.5 * num_rows + 0.5))
            gs = gridspec.GridSpec(num_rows, num_cols_u + 1, figure=fig, width_ratios=[1]*num_cols_u + [0.8])

            ax_text = fig.add_subplot(gs[:, -1])
            ax_text.axis('off')

            z_label = r'$z_0$' if is_3d else ''

            ax_c = fig.add_subplot(gs[0, 0])
            im_c = ax_c.imshow(c_slice.T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap='viridis',
                               vmin=c_min, vmax=c_max, interpolation='nearest')
            title_c = r'$c(x, y, ' + z_label + r')$' if is_3d else r'$c(x, y)$'
            ax_c.set_title(title_c)
            ax_c.set_xlabel(r'$x$'); ax_c.set_ylabel(r'$y$'); ax_c.set_aspect('equal')
            fig.colorbar(im_c, ax=ax_c, shrink=0.8)

            ax_m = fig.add_subplot(gs[0, 1])
            im_m = ax_m.imshow(m_slice.T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=m_cmap,
                              vmin=m_vmin, vmax=m_vmax, interpolation='nearest')
            title_m = r'$m(x, y, ' + z_label + r')$' if is_3d else r'$m(x, y)$'
            ax_m.set_title(title_m)
            ax_m.set_xlabel(r'$x$'); ax_m.set_ylabel(r'$y$'); ax_m.set_aspect('equal')
            fig.colorbar(im_m, ax=ax_m, shrink=0.8)

            plot_idx = 0
            for r in range(1, num_rows):
                for c_u in range(2):
                    if plot_idx >= len(plot_indices):
                        break

                    t_val = actual_times[plot_idx]
                    u_slice = u_slices[plot_idx]

                    if is_complex:
                        col_start_u = c_u * 2
                        ax_u_mag = fig.add_subplot(gs[r, col_start_u])
                        im_u_mag = ax_u_mag.imshow(np.abs(u_slice).T, extent=[-Lx, Lx, -Ly, Ly],
                                                  origin='lower', cmap=cmap_u_mag,
                                                  vmin=u_mag_min, vmax=u_mag_max, interpolation='nearest')
                        ax_u_mag.set_title(fr'$|u(t={t_val:.2f})|$')
                        ax_u_mag.set_xlabel(r'$x$'); ax_u_mag.set_ylabel(r'$y$'); ax_u_mag.set_aspect('equal')
                        fig.colorbar(im_u_mag, ax=ax_u_mag, shrink=0.8)

                        ax_u_phase = fig.add_subplot(gs[r, col_start_u + 1])
                        im_u_phase = ax_u_phase.imshow(np.angle(u_slice).T, extent=[-Lx, Lx, -Ly, Ly],
                                                      origin='lower', cmap=cmap_u_phase,
                                                      vmin=u_phase_min, vmax=u_phase_max, interpolation='nearest')
                        ax_u_phase.set_title(fr'$\arg(u(t={t_val:.2f}))$')
                        ax_u_phase.set_xlabel(r'$x$'); ax_u_phase.set_ylabel(r'$y$'); ax_u_phase.set_aspect('equal')
                        fig.colorbar(im_u_phase, ax=ax_u_phase, ticks=[-np.pi, 0, np.pi], format='%.2f', shrink=0.8)
                    else:
                        col_start_u = c_u * 1
                        if r == num_rows - 1 and is_kge and c_u == 1:
                            if v_slices is not None:
                                v_slice = v_slices[plot_idx]
                                ax_v = fig.add_subplot(gs[r, col_start_u])
                                im_v = ax_v.imshow(v_slice.T, extent=[-Lx, Lx, -Ly, Ly],
                                                  origin='lower', cmap=cmap_v,
                                                  vmin=v_min, vmax=v_max, interpolation='nearest')
                                ax_v.set_title(fr'$u_t(t={t_val:.2f})$')
                                ax_v.set_xlabel(r'$x$'); ax_v.set_ylabel(r'$y$'); ax_v.set_aspect('equal')
                                fig.colorbar(im_v, ax=ax_v, shrink=0.8)
                            else:
                                ax_empty = fig.add_subplot(gs[r, col_start_u])
                                ax_empty.axis('off')
                                ax_empty.set_title(r'$u_t$ N/A')
                        else:
                            ax_u = fig.add_subplot(gs[r, col_start_u])
                            im_u = ax_u.imshow(u_slice.T, extent=[-Lx, Lx, -Ly, Ly],
                                              origin='lower', cmap=cmap_u,
                                              vmin=u_min, vmax=u_max, interpolation='nearest')
                            ax_u.set_title(fr'$u(t={t_val:.2f})$')
                            ax_u.set_xlabel(r'$x$'); ax_u.set_ylabel(r'$y$'); ax_u.set_aspect('equal')
                            fig.colorbar(im_u, ax=ax_u, shrink=0.8)

                    plot_idx += 1
                if plot_idx >= len(plot_indices):
                    break

            param_text = format_params(metadata)
            ax_text.text(0.02, 0.98, param_text, transform=ax_text.transAxes,
                         fontsize=9, verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.4', fc='ivory', alpha=0.6))

            plt.tight_layout(rect=[0, 0, 0.88, 0.97])
            plot_stem = filepath.stem
            plot_title_latex = plot_stem
            dimensionality = "2D" if is_2d else "3D"
            plt.suptitle(f"Run Slices ({dimensionality}): {plot_title_latex}", fontsize=14, y=0.995)
            plt.savefig(plot_filename_slices, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Slices plot saved to {plot_filename_slices}")

            energies = np.zeros(num_snapshots)
            kinetic_energies = np.zeros(num_snapshots)
            gradient_energies = np.zeros(num_snapshots)
            potential_energies = np.zeros(num_snapshots)

            for i in range(num_snapshots):
                u_snap = u[i]
                v_snap = v[i] if v is not None else None
                energies[i], kinetic_energies[i], gradient_energies[i], potential_energies[i] = calculate_energy_terms(
                    u_snap, v_snap, c, m, dx, dy, dz, problem_type)

            plot_analysis_figure(time_points, energies, kinetic_energies, gradient_energies,
                                potential_energies, u, u0, plot_filename_analysis, problem_type)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot slices and analysis from simulation HDF5 file.")
    parser.add_argument("h5_file", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output plots.")

    if len(sys.argv) < 3:
         parser.print_help()
         sys.exit(1)

    args = parser.parse_args()
    if not os.path.isfile(args.h5_file):
        print(f"Error: Input file not found: {args.h5_file}", file=sys.stderr)
        sys.exit(1)

    plot_trajectory_slices(args.h5_file, args.output_dir)
