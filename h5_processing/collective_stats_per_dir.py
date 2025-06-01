import os
import sys
import glob
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpi4py import MPI
from mpl_toolkits.axes_grid1 import make_axes_locatable

def find_h5_files(base_dir, pattern="**/*.h5"):
    base_path = Path(base_dir)
    found = list(base_path.glob(pattern))
    found = sorted(list(set(found)))
    return found

def extract_metadata(h5_file):
    try:
        with h5py.File(h5_file, 'r') as f:
            metadata = dict(f['metadata'].attrs) if 'metadata' in f else {}
            if 'grid' in f:
                metadata.update(dict(f['grid'].attrs))
            if 'time' in f:
                metadata.update(dict(f['time'].attrs))
            
            metadata['filename'] = str(h5_file)
            
            problem_type = metadata.get('problem_type', 'unknown')
            
            dims = 0
            if 'u' in f:
                u_shape = f['u'].shape
                if len(u_shape) == 3:  
                    dims = 2
                elif len(u_shape) == 4:  
                    dims = 3
                else:
                    dims = -1
            
            metadata['dims'] = dims
            
            return metadata
    except Exception as e:
        return None

def calculate_energy_terms(u_snap, v_snap, c, m, dx, dy, dz=None, problem_type='unknown'):
    if len(u_snap.shape) == 2:
        dV = dx * dy
        grad_u_x, grad_u_y = np.gradient(u_snap, dx, dy, axis=(0, 1))
        grad_term_integrand = (np.abs(grad_u_x)**2 + np.abs(grad_u_y)**2)
    elif len(u_snap.shape) == 3:
        dV = dx * dy * dz
        grad_u_x, grad_u_y, grad_u_z = np.gradient(u_snap, dx, dy, dz, axis=(0, 1, 2))
        grad_term_integrand = (np.abs(grad_u_x)**2 + np.abs(grad_u_y)**2 + np.abs(grad_u_z)**2)
    else:
        return np.nan, np.nan, np.nan, np.nan

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
    
    elif problem_type == 'sine_gordon':
        kinetic_term = 0.5 * np.sum(v_snap**2) * dV
        gradient_term = 0.5 * np.sum(grad_term_integrand) * dV
        potential_term = np.sum((1 - np.cos(u_snap))) * dV
        total_energy = kinetic_term + gradient_term + potential_term
        return total_energy, kinetic_term, gradient_term, potential_term

    elif problem_type == 'phi4':
        kinetic_term = 0.5 * np.sum(v_snap**2) * dV
        gradient_term = 0.5 * np.sum(grad_term_integrand) * dV
        potential_term = np.sum((u**2 - u**4)) * dV
        total_energy = kinetic_term + gradient_term + potential_term
        return total_energy, kinetic_term, gradient_term, potential_term
    
    else:
        return np.nan, np.nan, np.nan, np.nan

def analyze_file_energy(h5_file, return_timeseries=False):
    try:
        with h5py.File(h5_file, 'r') as f:
            metadata = dict(f['metadata'].attrs) if 'metadata' in f else {}
            if 'grid' in f:
                metadata.update(dict(f['grid'].attrs))
            if 'time' in f:
                metadata.update(dict(f['time'].attrs))
            
            problem_type = metadata.get('problem_type', 'unknown')
            is_kge = (problem_type == 'klein_gordon')
            is_nlse_cubic = (problem_type == 'cubic')
            is_sge = (problem_type == 'sine_gordon')
            
            u = f['u'][()]
            is_2d = len(u.shape) == 3  
            is_3d = len(u.shape) == 4
            
            if 'v' in f and (is_kge or is_sge):
                v = f['v'][()]
            else:
                v = None
            
            c_location_options = ['anisotropy/c', 'focusing/c', 'c']
            m_location_options = ['focusing/m', 'm']
            
            c = None
            for loc in c_location_options:
                if loc in f:
                    c = f[loc][()]
                    break
            
            m = None    
            for loc in m_location_options:
                if loc in f:
                    m = f[loc][()]
                    break
            
            if c is None or m is None:
                raise ValueError(f"Could not find c or m in {h5_file}")
            
            nx = metadata['nx']
            ny = metadata['ny']
            nz = metadata.get('nz', 1)
            Lx = metadata['Lx']
            Ly = metadata['Ly']
            Lz = metadata.get('Lz', 0)
            
            dx = 2 * Lx / (nx - 1) if nx > 1 else np.nan
            dy = 2 * Ly / (ny - 1) if ny > 1 else np.nan
            dz = 2 * Lz / (nz - 1) if nz > 1 else np.nan
            
            T = metadata['T']
            num_snapshots = metadata['num_snapshots']
            time_points = np.linspace(0, T, num_snapshots)
            
            energies = np.zeros(num_snapshots)
            kinetic_energies = np.zeros(num_snapshots)
            gradient_energies = np.zeros(num_snapshots)
            potential_energies = np.zeros(num_snapshots)
            max_amplitudes = np.zeros(num_snapshots)

            energy_ratios = np.zeros(num_snapshots)
            energy_ratios[0] = 1.

            has_nan = np.sum(np.isnan(u)) > 1
            
            for i in range(num_snapshots):
                u_snap = u[i]
                v_snap = v[i] if v is not None else None
                energies[i], kinetic_energies[i], gradient_energies[i], potential_energies[i] = calculate_energy_terms(
                    u_snap, v_snap, c, m, dx, dy, dz, problem_type)
                if i == 0: E0 = energies[i]
                if i > 0:
                    energy_ratios[i] = energies[i] / E0 if E0 != 0 else np.nan
                
                if np.iscomplexobj(u_snap):
                    max_amplitudes[i] = np.max(np.abs(u_snap))
                else:
                    max_amplitudes[i] = np.max(np.abs(u_snap))
            
            if energies[0] != 0:
                energy_conservation = np.abs((energies - energies[0]) / energies[0])
            else:
                energy_conservation = np.abs(energies - energies[0])
            
            max_energy_deviation = np.max(energy_conservation)
            mean_energy_deviation = np.mean(energy_conservation)
            
            results = {
                'filename': str(h5_file),
                'problem_type': problem_type,
                'dims': 2 if is_2d else 3 if is_3d else -1,
                'initial_energy': energies[0],
                'final_energy': energies[-1],
                'max_energy_deviation': max_energy_deviation,
                'mean_energy_deviation': mean_energy_deviation,
                'initial_amplitude': max_amplitudes[0],
                'final_amplitude': max_amplitudes[-1],
                'amplitude_ratio': max_amplitudes[-1] / max_amplitudes[0] if max_amplitudes[0] > 0 else np.nan,
                'T': T,
                'nx': nx,
                'ny': ny,
                'nz': nz if is_3d else 1,
                'has_nan': has_nan,
                'max_energy_ration': np.max(energy_ratios)
            }
            
            if return_timeseries:
                results['times'] = time_points
                results['energies'] = energies
                results['kinetic_energies'] = kinetic_energies
                results['gradient_energies'] = gradient_energies
                results['potential_energies'] = potential_energies
                results['max_amplitudes'] = max_amplitudes
                results['energy_conservation'] = energy_conservation
            
            return results
    except Exception as e:
        return None

def get_group_key(h5_file):
    metadata = extract_metadata(h5_file)
    if metadata:
        key = (metadata.get('dims', -1), metadata.get('problem_type', 'unknown'))
        return key
    return None

def process_file_batch(files, rank, return_timeseries=False):
    # print(rank, "gets files", files)
    results = []
    for file_path in files:
        file_result = analyze_file_energy(file_path, return_timeseries)
        if file_result:
            results.append((file_path, file_result))
    return results

def generate_collective_stats(all_results, output_dir, group_key):
    dims, problem_type = group_key
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_results = []
    detailed_results = []
    
    for _, result in all_results:
        summary_results.append({k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
        detailed_results.append(result)
    
    if not summary_results:
        return
    
    stats_filename = output_path / f"energy_stats_{dims}D_{problem_type}.csv"
    with open(stats_filename, 'w') as f:
        headers = list(summary_results[0].keys())
        f.write(','.join(headers) + '\n')
        for result in summary_results:
            values = [str(result.get(h, '')) for h in headers]
            f.write(','.join(values) + '\n')
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.8,
        'axes.labelpad': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })
    
    colors = plt.cm.viridis(np.linspace(0, 1, min(20, len(detailed_results))))
    
    fig = plt.figure(figsize=(12, 10))
   
    ax1 = plt.subplot(2, 2, 1)
    for i, result in enumerate(detailed_results):
        if i < 20:
            ax1.semilogy(result['times'], result['energy_conservation'], color=colors[i], 
                        linewidth=1.2, alpha=0.9)
    
    ax1.set_title(f'Energy Conservation ({dims}D {problem_type})', fontsize=12)
    ax1.set_xlabel('$t$', fontsize=11)
    ax1.set_ylabel(r'$|E(t) - E(0)|/|E(0)|$', fontsize=11)
    ax1.grid(True, which='both', linestyle=':', alpha=0.3)
    ax1.tick_params(which='both', direction='in')
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='major', length=5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = plt.subplot(2, 2, 2)
    energy_deviations = [result['max_energy_deviation'] for result in summary_results]
    bins = np.linspace(min(energy_deviations), max(energy_deviations), 
                      min(20, max(10, len(energy_deviations)//5)))
    
    n, bins, patches = ax2.hist(energy_deviations, bins=bins, color='steelblue', 
                               alpha=0.8, edgecolor='black', linewidth=0.5)
    
    stats_text = (f"Mean: {np.mean(energy_deviations):.2e}\n"
                 f"Median: {np.median(energy_deviations):.2e}\n"
                 f"Max: {np.max(energy_deviations):.2e}\n"
                 f"Count: {len(energy_deviations)}")
    
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title('Distribution of Maximum Energy Deviation', fontsize=12)
    ax2.set_xlabel(r'$\max |E(t) - E(0)|/|E(0)|$', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.tick_params(which='both', direction='in')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax3 = plt.subplot(2, 2, 3)
    
    all_times = detailed_results[0]['times']
    all_amplitudes = np.zeros((len(detailed_results), len(all_times)))
    
    for i, result in enumerate(detailed_results):
        if len(result['times']) == len(all_times):
            norm_amp = result['max_amplitudes'] / result['max_amplitudes'][0]
            all_amplitudes[i, :] = norm_amp
   
    for i, result in enumerate(detailed_results):
        if i < 20:
            norm_amp = result['max_amplitudes'] / result['max_amplitudes'][0]
            ax3.plot(result['times'], norm_amp, color=colors[i], 
                    linewidth=0.8, alpha=0.6)
    
    if all_amplitudes.shape[0] > 5:
        p25 = np.percentile(all_amplitudes, 25, axis=0)
        p50 = np.percentile(all_amplitudes, 50, axis=0)
        p75 = np.percentile(all_amplitudes, 75, axis=0)
        
        ax3.plot(all_times, p50, 'k-', linewidth=2, label='Median')
        ax3.fill_between(all_times, p25, p75, color='gray', alpha=0.3, label='25-75 percentile')
        ax3.legend(loc='best', fontsize=9)
    
    ax3.set_title('Normalized Maximum Amplitude', fontsize=12)
    ax3.set_xlabel('$t$', fontsize=11)
    ax3.set_ylabel(r'$\max|u(t)|/\max|u(0)|$', fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.3)
    ax3.tick_params(which='both', direction='in')
    ax3.set_ylim(bottom=min(0.4, np.min(all_amplitudes)*0.9), 
                top=max(1.6, np.max(all_amplitudes)*1.1))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    ax4 = plt.subplot(2, 2, 4)

     
    if problem_type in ['klein_gordon', 'sine_gordon']:
        avg_kinetic = np.zeros_like(detailed_results[0]['times'])
        avg_gradient = np.zeros_like(detailed_results[0]['times'])
        avg_potential = np.zeros_like(detailed_results[0]['times'])
        
        for result in detailed_results:
            avg_kinetic += result['kinetic_energies']
            avg_gradient += result['gradient_energies']
            avg_potential += result['potential_energies']
        
        avg_kinetic /= len(detailed_results)
        avg_gradient /= len(detailed_results)
        avg_potential /= len(detailed_results)
        
        ax4.plot(detailed_results[0]['times'], avg_kinetic, 'b-', 
                linewidth=2, label=r'$\langle E_{\mathrm{kin}} \rangle$')
        ax4.plot(detailed_results[0]['times'], avg_gradient, 'g-', 
                linewidth=2, label=r'$\langle E_{\mathrm{grad}} \rangle$')
        ax4.plot(detailed_results[0]['times'], avg_potential, 'r-', 
                linewidth=2, label=r'$\langle E_{\mathrm{pot}} \rangle$')
        ax4.plot(detailed_results[0]['times'], avg_kinetic + avg_gradient + avg_potential, 
                'k--', linewidth=1.5, label=r'$\langle E_{\mathrm{total}} \rangle$')
       
        for i, result in enumerate(detailed_results):
            if i < 5:
                ax4.plot(result['times'], result['kinetic_energies'], 'b-', 
                        linewidth=0.7, alpha=0.2)
                ax4.plot(result['times'], result['gradient_energies'], 'g-', 
                        linewidth=0.7, alpha=0.2)
                ax4.plot(result['times'], result['potential_energies'], 'r-', 
                        linewidth=0.7, alpha=0.2)
        
    elif problem_type == 'cubic':
        avg_gradient = np.zeros_like(detailed_results[0]['times'])
        avg_potential = np.zeros_like(detailed_results[0]['times'])
        
        for result in detailed_results:
            avg_gradient += result['gradient_energies']
            avg_potential += result['potential_energies']
        
        avg_gradient /= len(detailed_results)
        avg_potential /= len(detailed_results)
        
        ax4.plot(detailed_results[0]['times'], avg_gradient, 'g-', 
                linewidth=2, label=r'$\langle E_{\mathrm{grad}} \rangle$')
        ax4.plot(detailed_results[0]['times'], avg_potential, 'r-', 
                linewidth=2, label=r'$\langle E_{\mathrm{pot}} \rangle$')
        ax4.plot(detailed_results[0]['times'], avg_gradient + avg_potential, 
                'k--', linewidth=1.5, label=r'$\langle E_{\mathrm{total}} \rangle$')
        
        for i, result in enumerate(detailed_results):
            if i < 5:
                ax4.plot(result['times'], result['gradient_energies'], 'g-', 
                        linewidth=0.7, alpha=0.2)
                ax4.plot(result['times'], result['potential_energies'], 'r-', 
                        linewidth=0.7, alpha=0.2)
    
    ax4.set_title('Energy Components', fontsize=12)
    ax4.set_xlabel('$t$', fontsize=11)
    ax4.set_ylabel('Energy', fontsize=11)
    ax4.grid(True, linestyle=':', alpha=0.3)
    ax4.tick_params(which='both', direction='in')
    ax4.legend(loc='best', fontsize=9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig_energy_filename = output_path / f"energy_plots_{dims}D_{problem_type}.png"
    plt.savefig(fig_energy_filename, dpi=300, bbox_inches='tight')
    plt.close()

    nan_rate = 0
    for r in summary_results: nan_rate += int(r['has_nan']) 
    
    summary_stats = {
        'mean_energy_deviation': np.mean([r['mean_energy_deviation'] for r in summary_results]),
        'median_energy_deviation': np.median([r['mean_energy_deviation'] for r in summary_results]),
        'max_energy_deviation': np.max([r['max_energy_deviation'] for r in summary_results]),
        'mean_amplitude_ratio': np.mean([r['amplitude_ratio'] for r in summary_results if not np.isnan(r['amplitude_ratio'])]),
        'min_amplitude_ratio': np.min([r['amplitude_ratio'] for r in summary_results if not np.isnan(r['amplitude_ratio'])]),
        'max_amplitude_ratio': np.max([r['amplitude_ratio'] for r in summary_results if not np.isnan(r['amplitude_ratio'])]),
        'num_simulations': len(summary_results),
        'nan_rate': nan_rate / len(summary_results),
    }
    
    summary_filename = output_path / f"summary_stats_{dims}D_{problem_type}.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"Summary Statistics for {dims}D {problem_type} ({len(summary_results)} simulations)\n")
        f.write("=" * 50 + "\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value:.6e}\n" if isinstance(value, float) else f"{key}: {value}\n")

def generate_collective_stats(all_results, output_dir, group_key):
    dims, problem_type = group_key
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_results = []
    detailed_results = []

    for _, result in all_results:
        summary_results.append({k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
        detailed_results.append(result)

    if not summary_results:
        return

    stats_filename = output_path / f"energy_stats_{dims}D_{problem_type}.csv"
    with open(stats_filename, 'w') as f:
        headers = list(summary_results[0].keys())
        f.write(','.join(headers) + '\n')
        for result in summary_results:
            values = [str(result.get(h, '')) for h in headers]
            f.write(','.join(values) + '\n')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.8,
        'axes.labelpad': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    colors = plt.cm.viridis(np.linspace(0, 1, min(20, len(detailed_results))))

    fig_conservation = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    for i, result in enumerate(detailed_results):
        if i < 20:
            ax1.semilogy(result['times'], result['energy_conservation'], color=colors[i],
                        linewidth=1.2, alpha=0.9)

    ax1.set_title(f'Energy Conservation ({dims}D {problem_type})', fontsize=12)
    ax1.set_xlabel('$t$', fontsize=11)
    ax1.set_ylabel(r'$|E(t) - E(0)|/|E(0)|$', fontsize=11)
    ax1.grid(True, which='both', linestyle=':', alpha=0.3)
    ax1.tick_params(which='both', direction='in')
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='major', length=5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = plt.subplot(2, 2, 2)
    energy_deviations = [result['max_energy_deviation'] for result in summary_results]
    bins = np.linspace(min(energy_deviations), max(energy_deviations),
                      min(20, max(10, len(energy_deviations)//5)))

    n, bins, patches = ax2.hist(energy_deviations, bins=bins, color='steelblue',
                               alpha=0.8, edgecolor='black', linewidth=0.5)

    stats_text = (f"Mean: {np.mean(energy_deviations):.2e}\n"
                 f"Median: {np.median(energy_deviations):.2e}\n"
                 f"Max: {np.max(energy_deviations):.2e}\n"
                 f"Count: {len(energy_deviations)}")

    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_title('Distribution of Maximum Energy Deviation', fontsize=12)
    ax2.set_xlabel(r'$\max |E(t) - E(0)|/|E(0)|$', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.tick_params(which='both', direction='in')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    time_structure_groups = {}
    for result in detailed_results:
        time_key = (len(result['times']), round(result['times'][-1], 2))
        if time_key not in time_structure_groups:
            time_structure_groups[time_key] = []
        time_structure_groups[time_key].append(result)

    dominant_groups = sorted(time_structure_groups.items(), key=lambda x: len(x[1]), reverse=True)

    amplitudes_fig = plt.figure(figsize=(15, 10))
    ax_amp_main = amplitudes_fig.add_subplot(2, 2, 1)

    dominant_key, dominant_results = dominant_groups[0]
    dominant_times = dominant_results[0]['times']

    all_amplitudes = np.zeros((len(dominant_results), len(dominant_times)))
    for i, result in enumerate(dominant_results):
        norm_amp = result['max_amplitudes'] / (result['max_amplitudes'][0] + 1e-10)
        all_amplitudes[i, :] = norm_amp

    for i, result in enumerate(dominant_results):
        if i < 10:
            norm_amp = result['max_amplitudes'] / (result['max_amplitudes'][0] + 1e-10)
            ax_amp_main.plot(result['times'], norm_amp, color=colors[i],
                    linewidth=0.8, alpha=0.6)

    if len(dominant_results) > 5:
        p25 = np.percentile(all_amplitudes, 25, axis=0)
        p50 = np.percentile(all_amplitudes, 50, axis=0)
        p75 = np.percentile(all_amplitudes, 75, axis=0)

        ax_amp_main.plot(dominant_times, p50, 'k-', linewidth=2, label='Median')
        ax_amp_main.fill_between(dominant_times, p25, p75, color='gray', alpha=0.3, label='25-75 percentile')
        ax_amp_main.legend(loc='best', fontsize=9)

    ax_amp_main.set_title(f'Normalized Maximum Amplitude (Main Group: {len(dominant_results)} runs)', fontsize=12)
    ax_amp_main.set_xlabel('$t$', fontsize=11)
    ax_amp_main.set_ylabel(r'$\max|u(t)|/\max|u(0)|$', fontsize=11)
    ax_amp_main.grid(True, linestyle=':', alpha=0.3)
    ax_amp_main.tick_params(which='both', direction='in')
    ax_amp_main.set_ylim(bottom=min(0.4, np.min(all_amplitudes)*0.9),
                top=max(1.6, np.max(all_amplitudes)*1.1))
    ax_amp_main.spines['top'].set_visible(False)
    ax_amp_main.spines['right'].set_visible(False)

    for idx, (time_key, results) in enumerate(dominant_groups[1:3]):
        pos = idx + 2
        if len(results) < 2:
            continue

        ax_amp = amplitudes_fig.add_subplot(2, 2, pos)
        n_steps, t_max = time_key

        for i, result in enumerate(results):
            if i < 10:
                norm_amp = result['max_amplitudes'] / result['max_amplitudes'][0]
                ax_amp.plot(result['times'], norm_amp, color=colors[i],
                        linewidth=0.8, alpha=0.6)

        if len(results) > 5:
            all_amps = np.zeros((len(results), n_steps))
            for i, result in enumerate(results):
                norm_amp = result['max_amplitudes'] / result['max_amplitudes'][0]
                all_amps[i, :] = norm_amp

            p25 = np.percentile(all_amps, 25, axis=0)
            p50 = np.percentile(all_amps, 50, axis=0)
            p75 = np.percentile(all_amps, 75, axis=0)

            ax_amp.plot(results[0]['times'], p50, 'k-', linewidth=2, label='Median')
            ax_amp.fill_between(results[0]['times'], p25, p75, color='gray', alpha=0.3, label='25-75 percentile')
            ax_amp.legend(loc='best', fontsize=9)

        ax_amp.set_title(f'Group {pos}: {n_steps} steps, T={t_max} ({len(results)} runs)', fontsize=11)
        ax_amp.set_xlabel('$t$', fontsize=11)
        ax_amp.set_ylabel(r'$\max|u(t)|/\max|u(0)|$', fontsize=11)
        ax_amp.grid(True, linestyle=':', alpha=0.3)
        ax_amp.tick_params(which='both', direction='in')
        ax_amp.spines['top'].set_visible(False)
        ax_amp.spines['right'].set_visible(False)

    plt.tight_layout()
    amplitudes_fig.savefig(output_path / f"amplitude_plots_{dims}D_{problem_type}.png", dpi=300, bbox_inches='tight')
    plt.close(amplitudes_fig)

    energy_figs = []
    for group_idx, (time_key, results) in enumerate(dominant_groups[:min(3, len(dominant_groups))]):
        if len(results) < 2:
            continue

        n_steps, t_max = time_key
        group_times = results[0]['times']

        energy_fig = plt.figure(figsize=(12, 10))
        energy_figs.append(energy_fig)

        if problem_type in ['klein_gordon', 'sine_gordon']:
            avg_kinetic = np.zeros(n_steps)
            avg_gradient = np.zeros(n_steps)
            avg_potential = np.zeros(n_steps)

            for result in results:
                avg_kinetic += result['kinetic_energies']
                avg_gradient += result['gradient_energies']
                avg_potential += result['potential_energies']

            avg_kinetic /= len(results)
            avg_gradient /= len(results)
            avg_potential /= len(results)

            ax_e = energy_fig.add_subplot(111)
            ax_e.plot(group_times, avg_kinetic, 'b-',
                    linewidth=2, label=r'$\langle E_{\mathrm{kin}} \rangle$')
            ax_e.plot(group_times, avg_gradient, 'g-',
                    linewidth=2, label=r'$\langle E_{\mathrm{grad}} \rangle$')
            ax_e.plot(group_times, avg_potential, 'r-',
                    linewidth=2, label=r'$\langle E_{\mathrm{pot}} \rangle$')
            ax_e.plot(group_times, avg_kinetic + avg_gradient + avg_potential,
                    'k--', linewidth=1.5, label=r'$\langle E_{\mathrm{total}} \rangle$')

            for i, result in enumerate(results):
                if i < 5:
                    ax_e.plot(result['times'], result['kinetic_energies'], 'b-',
                            linewidth=0.7, alpha=0.2)
                    ax_e.plot(result['times'], result['gradient_energies'], 'g-',
                            linewidth=0.7, alpha=0.2)
                    ax_e.plot(result['times'], result['potential_energies'], 'r-',
                            linewidth=0.7, alpha=0.2)

        elif problem_type == 'cubic':
            avg_gradient = np.zeros(n_steps)
            avg_potential = np.zeros(n_steps)

            for result in results:
                avg_gradient += result['gradient_energies']
                avg_potential += result['potential_energies']

            avg_gradient /= len(results)
            avg_potential /= len(results)

            ax_e = energy_fig.add_subplot(111)
            ax_e.plot(group_times, avg_gradient, 'g-',
                    linewidth=2, label=r'$\langle E_{\mathrm{grad}} \rangle$')
            ax_e.plot(group_times, avg_potential, 'r-',
                    linewidth=2, label=r'$\langle E_{\mathrm{pot}} \rangle$')
            ax_e.plot(group_times, avg_gradient + avg_potential,
                    'k--', linewidth=1.5, label=r'$\langle E_{\mathrm{total}} \rangle$')

            for i, result in enumerate(results):
                if i < 5:
                    ax_e.plot(result['times'], result['gradient_energies'], 'g-',
                            linewidth=0.7, alpha=0.2)
                    ax_e.plot(result['times'], result['potential_energies'], 'r-',
                            linewidth=0.7, alpha=0.2)

        group_label = "Main" if group_idx == 0 else f"Group {group_idx+1}"
        ax_e.set_title(f'Energy Components - {group_label}: {n_steps} steps, T={t_max} ({len(results)} runs)', fontsize=12)
        ax_e.set_xlabel('$t$', fontsize=11)
        ax_e.set_ylabel('Energy', fontsize=11)
        ax_e.grid(True, linestyle=':', alpha=0.3)
        ax_e.tick_params(which='both', direction='in')
        ax_e.legend(loc='best', fontsize=9)
        ax_e.spines['top'].set_visible(False)
        ax_e.spines['right'].set_visible(False)

        plt.tight_layout()
        energy_fig.savefig(output_path / f"energy_plots_{dims}D_{problem_type}_group{group_idx+1}.png", dpi=300, bbox_inches='tight')
        plt.close(energy_fig)

    for fig in energy_figs:
        plt.close(fig)

    plt.figure(figsize=(12, 8))

    nan_rate = 0
    for r in summary_results:
        nan_rate += int(r['has_nan'])

    summary_stats = {
        'mean_energy_deviation': np.mean([r['mean_energy_deviation'] for r in summary_results]),
        'median_energy_deviation': np.median([r['mean_energy_deviation'] for r in summary_results]),
        'max_energy_deviation': np.max([r['max_energy_deviation'] for r in summary_results]),
        'mean_amplitude_ratio': np.mean([r['amplitude_ratio'] for r in summary_results if not np.isnan(r['amplitude_ratio'])]),
        'min_amplitude_ratio': np.min([r['amplitude_ratio'] for r in summary_results if not np.isnan(r['amplitude_ratio'])]),
        'max_amplitude_ratio': np.max([r['amplitude_ratio'] for r in summary_results if not np.isnan(r['amplitude_ratio'])]),
        'num_simulations': len(summary_results),
        'nan_rate': nan_rate / len(summary_results),
    }

    group_stats = []
    for idx, (time_key, results) in enumerate(dominant_groups[:min(5, len(dominant_groups))]):
        n_steps, t_max = time_key
        group_stats.append({
            'group_name': f"Group {idx+1}",
            'n_steps': n_steps,
            'T_max': t_max,
            'count': len(results),
            'mean_energy_deviation': np.mean([r['mean_energy_deviation'] for r in results]),
            'max_energy_deviation': np.max([r['max_energy_deviation'] for r in results]),
        })

    summary_filename = output_path / f"summary_stats_{dims}D_{problem_type}.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"Summary Statistics for {dims}D {problem_type} ({len(summary_results)} simulations)\n")
        f.write("=" * 50 + "\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value:.6e}\n" if isinstance(value, float) else f"{key}: {value}\n")

        f.write("\nTime Structure Groups:\n")
        f.write("=" * 50 + "\n")
        for gs in group_stats:
            f.write(f"{gs['group_name']}: {gs['n_steps']} steps, T_max={gs['T_max']}, {gs['count']} runs\n")
            f.write(f"  Mean Energy Deviation: {gs['mean_energy_deviation']:.6e}\n")
            f.write(f"  Max Energy Deviation: {gs['max_energy_deviation']:.6e}\n")

def plot_field_info(all_results, output_dir, group_key):
    dims, problem_type = group_key
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    detailed_results = []
    for _, result in all_results:
        detailed_results.append(result)
    
    if len(detailed_results) < 3:
        print(f"Not enough simulations for representative analysis ({len(detailed_results)} found, need at least 3)")
        return
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })
    
    energy_metrics = []
    for i, result in enumerate(detailed_results):
        conservation_metric = np.trapz(result['energy_conservation'], result['times']) / result['times'][-1]
        energy_diff = np.diff(result['energies']) / np.diff(result['times'])
        explosiveness_metric = np.max(np.abs(energy_diff)) / np.abs(result['energies'][0]) if result['energies'][0] != 0 else 0
        final_deviation = result['energy_conservation'][-1]
        
        energy_metrics.append({
            'index': i,
            'filename': result['filename'],
            'conservation_metric': conservation_metric,
            'explosiveness_metric': explosiveness_metric,
            'final_deviation': final_deviation
        })
    
    conservation_values = [m['conservation_metric'] for m in energy_metrics]
    median_idx = np.argmin([abs(m['conservation_metric'] - np.median(conservation_values)) for m in energy_metrics])
    explosive_idx = np.argmax([m['explosiveness_metric'] for m in energy_metrics])
    deterioration_idx = np.argmax([m['final_deviation'] for m in energy_metrics])
    
    median_case = detailed_results[energy_metrics[median_idx]['index']]
    explosive_case = detailed_results[energy_metrics[explosive_idx]['index']]
    deterioration_case = detailed_results[energy_metrics[deterioration_idx]['index']]
    
    median_filename = energy_metrics[median_idx]['filename']
    explosive_filename = energy_metrics[explosive_idx]['filename']
    deterioration_filename = energy_metrics[deterioration_idx]['filename']
    
    fig = plt.figure(figsize=(10, 5))
    gs = plt.GridSpec(2, 3, figure=fig, height_ratios=[1, 2], width_ratios=[1, 1, 1])
    
    time_points = median_case['times']
    T_max = time_points[-1]
    snapshot_times = [0, T_max/2, T_max]
    snapshot_indices = [np.argmin(np.abs(time_points - t)) for t in snapshot_times]
    
    ax_c = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])
    
    with h5py.File(median_filename, 'r') as f:
        c_location_options = ['anisotropy/c', 'focusing/c', 'c']
        m_location_options = ['focusing/m', 'm']
        
        c_field = None
        for loc in c_location_options:
            if loc in f:
                c_field = f[loc][()]
                break
                
        m_field = None
        for loc in m_location_options:
            if loc in f:
                m_field = f[loc][()]
                break
                
        if 'metadata' in f:
            metadata = dict(f['metadata'].attrs)
        else:
            metadata = {}
        if 'grid' in f:
            metadata.update(dict(f['grid'].attrs))
        if 'time' in f:
            metadata.update(dict(f['time'].attrs))
    
    is_3d = len(c_field.shape) == 3
    
    if is_3d:
        mid_z = c_field.shape[-1] // 2
        c_slice = c_field[:, :, mid_z]
        m_slice = m_field[:, :, mid_z]
    else:
        c_slice = c_field
        m_slice = m_field
    
    im_c = ax_c.imshow(c_slice.T, origin='lower', cmap='viridis')
    ax_c.set_title('$c(x,y)$', fontsize=12)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    
    m_min, m_max = np.min(m_slice), np.max(m_slice)
    m_cmap = 'viridis'
    if m_min < -1e-6 and m_max > 1e-6:
        m_abs_max = max(abs(m_min), abs(m_max))
        m_vmin, m_vmax = -m_abs_max, m_abs_max
        m_cmap = 'coolwarm'
    else:
        m_vmin, m_vmax = m_min, m_max
    
    im_m = ax_m.imshow(m_slice.T, origin='lower', cmap=m_cmap, vmin=m_vmin, vmax=m_vmax)
    ax_m.set_title('$m(x,y)$', fontsize=12)
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
    
    ax_text.axis('off')

    if is_3d:
        s = "×" + str(metadata.get('nz', '?')) 
    else:
        s = ""
    
    metadata_text = (
        f"Problem Type: {problem_type}\n"
        f"Domain: {dims}D, {metadata.get('nx', '?')}×{metadata.get('ny', '?')}{s}\n"
        f"Time: T = {metadata.get('T', '?')}, Steps = {metadata.get('nt', '?')}\n"
        f"Energy Metrics:\n"
        f"  Median: {Path(median_filename).stem}\n"
        f"  Most Explosive: {Path(explosive_filename).stem}\n"
        f"  Worst Conservation: {Path(deterioration_filename).stem}"
    )
    
    ax_text.text(0.05, 0.95, metadata_text, transform=ax_text.transAxes,
                va='top', ha='left', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax_energy = fig.add_subplot(gs[1, :])
    
    ax_energy.semilogy(median_case['times'], median_case['energy_conservation'], 'k-', 
                      label=f'Median', linewidth=2)
    ax_energy.semilogy(explosive_case['times'], explosive_case['energy_conservation'], 'r-', 
                      label=f'Most Explosive', linewidth=2)
    ax_energy.semilogy(deterioration_case['times'], deterioration_case['energy_conservation'], 'b-', 
                      label=f'Worst Conservation', linewidth=2)
    
    ax_energy.set_title(f'Energy Conservation ({dims}D {problem_type})', fontsize=12)
    ax_energy.set_xlabel('$t$', fontsize=11)
    ax_energy.set_ylabel(r'$|E(t) - E(0)|/|E(0)|$', fontsize=11)
    ax_energy.grid(True, which='both', linestyle=':', alpha=0.3)
    ax_energy.legend(loc='best', fontsize=10)
    ax_energy.spines['top'].set_visible(False)
    ax_energy.spines['right'].set_visible(False)
    
    for i, idx in enumerate(snapshot_indices):
        t = time_points[idx]
        ax_energy.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
        ax_energy.text(t, ax_energy.get_ylim()[0]*1.1, f'$t_{i+1}$', 
                      ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_filename = output_path / f"field_info_{dims}D_{problem_type}.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return median_filename, explosive_filename, deterioration_filename

def plot_case_snapshots(all_results, output_dir, group_key):
    dims, problem_type = group_key
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    detailed_results = []
    for _, result in all_results:
        detailed_results.append(result)
    
    if len(detailed_results) < 3:
        print(f"Not enough simulations for representative analysis ({len(detailed_results)} found, need at least 3)")
        return
    
    energy_metrics = []
    for i, result in enumerate(detailed_results):
        conservation_metric = np.trapz(result['energy_conservation'], result['times']) / result['times'][-1]
        energy_diff = np.diff(result['energies']) / np.diff(result['times'])
        explosiveness_metric = np.max(np.abs(energy_diff)) / np.abs(result['energies'][0]) if result['energies'][0] != 0 else 0
        final_deviation = result['energy_conservation'][-1]
        
        energy_metrics.append({
            'index': i,
            'filename': result['filename'],
            'conservation_metric': conservation_metric,
            'explosiveness_metric': explosiveness_metric,
            'final_deviation': final_deviation
        })
    
    conservation_values = [m['conservation_metric'] for m in energy_metrics]
    median_idx = np.argmin([abs(m['conservation_metric'] - np.median(conservation_values)) for m in energy_metrics])
    explosive_idx = np.argmax([m['explosiveness_metric'] for m in energy_metrics])
    deterioration_idx = np.argmax([m['final_deviation'] for m in energy_metrics])
    
    median_filename = energy_metrics[median_idx]['filename']
    explosive_filename = energy_metrics[explosive_idx]['filename']
    deterioration_filename = energy_metrics[deterioration_idx]['filename']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })
    
    fig = plt.figure(figsize=(9, 8))
    gs = plt.GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.1])
    
    time_points = detailed_results[0]['times']
    T_max = time_points[-1]
    snapshot_times = [0, T_max/2, T_max]
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    case_labels = ['Median', 'Most Explosive', 'Worst Conservation']
    case_files = [median_filename, explosive_filename, deterioration_filename]
    
    u_data = {}
    u_min, u_max = float('inf'), float('-inf')
    is_complex = False
    for case, filename in zip(case_labels, case_files):
        u_data[case] = []
        with h5py.File(filename, 'r') as f:
            u_dataset = f['u']
            u_shape = u_dataset.shape
            if len(u_shape) == 4:
                nt, nx, ny, nz = u_shape
            elif len(u_shape) == 3:
                nt, nx, ny = u_shape
            else:
                raise Exception("Dataset ill-formed")

            snapshot_indices = [0, nt // 2, -1]
            mid_z = nz // 2 if len(u_shape) == 4 else None
            for idx in snapshot_indices:
                if len(u_shape) == 4: 
                    snapshot = u_dataset[idx, :, :, mid_z]
                else:
                    snapshot = u_dataset[idx, :, :]
                u_data[case].append(snapshot)

                if np.iscomplexobj(snapshot):
                    is_complex = True
                    u_min = min(u_min, 0)
                    u_max = max(u_max, np.max(np.abs(snapshot)))
                else:
                    u_min = min(u_min, np.min(snapshot))
                    u_max = max(u_max, np.max(snapshot))
                
    cmap = 'viridis' if is_complex else 'coolwarm'
    
    for row, case in enumerate(case_labels):
        for col, (t, snap_idx) in enumerate(zip(snapshot_times, range(len(snapshot_indices)))):
            ax = fig.add_subplot(gs[row, col])
            
            data = u_data[case][snap_idx]
            if is_complex:
                im = ax.imshow(np.abs(data).T, origin='lower', cmap=cmap, vmin=u_min, vmax=u_max)
            else:
                im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=u_min, vmax=u_max)
            
            if row == 0:
                ax.set_title(f'$t = {t:.2f}$', fontsize=11)
            
            if col == 0:
                ax.set_ylabel(case, fontsize=11)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    cax = fig.add_subplot(gs[3, :])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    
    plt.tight_layout()
    fig_filename = output_path / f"case_snapshots_{dims}D_{problem_type}.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(output_path / f"representative_cases_{dims}D_{problem_type}.txt", 'w') as f:
        f.write(f"Representative Cases for {dims}D {problem_type}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Median Case: {median_filename}\n")
        f.write(f"  Conservation Metric: {energy_metrics[median_idx]['conservation_metric']:.6e}\n")
        f.write(f"  Explosiveness Metric: {energy_metrics[median_idx]['explosiveness_metric']:.6e}\n")
        f.write(f"  Final Deviation: {energy_metrics[median_idx]['final_deviation']:.6e}\n\n")
        
        f.write(f"Most Explosive Case: {explosive_filename}\n")
        f.write(f"  Conservation Metric: {energy_metrics[explosive_idx]['conservation_metric']:.6e}\n")
        f.write(f"  Explosiveness Metric: {energy_metrics[explosive_idx]['explosiveness_metric']:.6e}\n")
        f.write(f"  Final Deviation: {energy_metrics[explosive_idx]['final_deviation']:.6e}\n\n")
        
        f.write(f"Worst Conservation Case: {deterioration_filename}\n")
        f.write(f"  Conservation Metric: {energy_metrics[deterioration_idx]['conservation_metric']:.6e}\n")
        f.write(f"  Explosiveness Metric: {energy_metrics[deterioration_idx]['explosiveness_metric']:.6e}\n")
        f.write(f"  Final Deviation: {energy_metrics[deterioration_idx]['final_deviation']:.6e}\n")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser(description="Collect energy statistics from simulation HDF5 files (MPI version).")
    parser.add_argument("base_dir", type=str, help="Base directory containing HDF5 files.")
    parser.add_argument("output_dir", type=str, help="Directory to save output statistics and plots.")
    parser.add_argument("--pattern", type=str, default="**/*.h5", help="Glob pattern for finding HDF5 files.")
    
    if rank == 0:
        args = parser.parse_args()
        if not os.path.isdir(args.base_dir):
            print(f"Error: Base directory not found: {args.base_dir}")
            comm.Abort(1)
        
        all_h5_files = find_h5_files(args.base_dir, args.pattern)
        # print(all_h5_files)
        
        if not all_h5_files:
            print("No HDF5 files found. Exiting.")
            comm.Abort(1)
        
        files_per_rank = [[] for _ in range(size)]
        for i, file_path in enumerate(all_h5_files):
            files_per_rank[i % size].append(file_path)
    else:
        args = None
        files_per_rank = None
    
    args = comm.bcast(args, root=0)
    files_per_rank = comm.scatter(files_per_rank, root=0)
    
    my_results = process_file_batch(files_per_rank, rank, return_timeseries=True) 
    all_results = comm.gather(my_results, root=0)
    
    if rank == 0:
        grouped_results = defaultdict(list)
        for rank_results in all_results:
            for file_path, result in rank_results:
                key = (result['dims'], result['problem_type'])
                grouped_results[key].append((file_path, result))
        
        for group_key, group_results in grouped_results.items():
            dims, problem_type = group_key
            generate_collective_stats(group_results, args.output_dir, group_key) 
            plot_field_info(group_results, args.output_dir, group_key)
            plot_case_snapshots(group_results, args.output_dir, group_key)
            # plot_representative_cases(group_results, args.output_dir, group_key)

if __name__ == "__main__":
    main()
