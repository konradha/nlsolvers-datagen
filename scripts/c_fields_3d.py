import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def make_grid(n, L):
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    z = np.linspace(-L, L, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z, x, y, z

def constant(n, L):
    return np.ones((n, n, n),dtype=np.float64)

def periodic_structure(n, L, base_value=1.0, amplitude=0.5, frequency=3):
    X, Y, Z, _, _, _ = make_grid(n, L)
    k = np.pi * frequency / L
    profile = base_value * (1 + amplitude * (
        np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
    ))
    return profile

def piecewise_constant(n, L, num_layers=3, base_value=1.0, contrast_factor=2.0):
    X, Y, Z, _, _, _ = make_grid(n, L)
    profile = np.ones((n, n, n)) * base_value
    layer_width = 2*L / num_layers
    
    for i in range(num_layers):
        if i % 2 == 1:
            x_min = -L + i * layer_width
            x_max = -L + (i + 1) * layer_width
            mask = (X >= x_min) & (X < x_max)
            profile[mask] = base_value * contrast_factor
            
    return profile

def sign_changing_mass(n, L, base_value=1.0, regions='checkerboard', scale=2, sharpness=5.0):
    X, Y, Z, _, _, _ = make_grid(n, L)
    
    if regions == 'checkerboard':
        cell_size = L / scale
        pattern = np.sin(np.pi * X / cell_size) * np.sin(np.pi * Y / cell_size) * np.sin(np.pi * Z / cell_size)
        if sharpness > 0:
            profile = base_value * np.tanh(sharpness * pattern)
        else:
            profile = base_value * np.sign(pattern)
    elif regions == 'half_space':
        profile = base_value * np.tanh(sharpness * X / L)
    else:
        raise NotImplemented
        
    return profile

def layered_materials(n, L, base_value=1.0, num_layers=3, min_amplitude=0.2, max_amplitude=0.8, 
                         min_freq=2, max_freq=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    X, Y, Z, _, _, _ = make_grid(n, L)
    profile = np.ones((n, n, n)) * base_value
    
    for _ in range(num_layers):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        
        direction = nx * X + ny * Y + nz * Z
        
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        freq = np.random.uniform(min_freq, max_freq)
        phase = np.random.uniform(0, 2*np.pi)
        
        profile += amplitude * np.sin(freq * direction + phase)
    
    max_val = np.max(profile)
    min_val = np.min(profile)
    profile = (profile - min_val) / (max_val - min_val)
    
    return base_value * profile

def wave_guiding_structures(n, L, base_value=1.0, num_guides=3, min_width=0.1, max_width=0.5, 
                              guide_amplitude=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    X, Y, Z, x, y, z = make_grid(n, L)
    profile = np.ones((n, n, n)) * base_value
    
    for _ in range(num_guides):
        curve_type = np.random.choice(['line', 'helix', 'spiral'])
        width = np.random.uniform(min_width, max_width)
        
        if curve_type == 'line':
            x0 = np.random.uniform(-L, L)
            y0 = np.random.uniform(-L, L)
            z0 = np.random.uniform(-L, L)
            
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            nx = np.sin(theta) * np.cos(phi)
            ny = np.sin(theta) * np.sin(phi)
            nz = np.cos(theta)
            
            t = np.linspace(-1.5*L, 1.5*L, 100)
            curve_x = x0 + t * nx
            curve_y = y0 + t * ny
            curve_z = z0 + t * nz
            
        elif curve_type == 'helix':
            a = np.random.uniform(0.3*L, 0.8*L)
            b = np.random.uniform(0.3*L, 0.8*L)
            c = np.random.uniform(0.1, 0.5)
            
            t = np.linspace(0, 6*np.pi, 100)
            curve_x = a * np.cos(t)
            curve_y = b * np.sin(t)
            curve_z = c * t
            
        else:  
            a = np.random.uniform(0.1, 0.5)
            b = np.random.uniform(0.5*L, 0.8*L)
            
            t = np.linspace(0, 4*np.pi, 100)
            curve_x = b * t/(4*np.pi) * np.cos(t)
            curve_y = b * t/(4*np.pi) * np.sin(t)
            curve_z = a * t
        
        curve_points = np.column_stack([curve_x, curve_y, curve_z])
        grid_points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        distances = np.zeros(grid_points.shape[0])
        
        batch_size = 10000  
        for i in range(0, grid_points.shape[0], batch_size):
            end_idx = min(i + batch_size, grid_points.shape[0])
            batch = grid_points[i:end_idx]
            distances[i:end_idx] = cdist(batch, curve_points).min(axis=1)
        
        distances = distances.reshape((n, n, n))
        guide_profile = guide_amplitude * np.exp(-(distances**2) / (2 * width**2))
        profile = np.maximum(profile, guide_profile)
    
    return profile

def quasi_periodic_structures(n, L, base_value=1.0, num_waves=6, min_amp=0.1, max_amp=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    X, Y, Z, _, _, _ = make_grid(n, L)
    profile = np.ones((n, n, n)) * base_value
    
    k_vectors = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(num_waves):
        theta = np.arccos(1 - 2 * (i + 0.5) / num_waves)
        phi = np.pi * (1 + np.sqrt(5)) * i
        
        kx = np.sin(theta) * np.cos(phi)
        ky = np.sin(theta) * np.sin(phi)
        kz = np.cos(theta)
        
        scale = golden_ratio ** (i % 3)
        k_vectors.append([kx * scale, ky * scale, kz * scale])
    
    for kx, ky, kz in k_vectors:
        amplitude = np.random.uniform(min_amp, max_amp)
        phase = np.random.uniform(0, 2*np.pi)
        
        k_dot_x = kx * X + ky * Y + kz * Z
        profile += amplitude * np.cos(k_dot_x + phase)
    
    max_val = np.max(profile)
    min_val = np.min(profile)
    profile = (profile - min_val) / (max_val - min_val)
    
    return base_value * profile

def turbulent_like_fields(n, L, base_value=1.0, intensity=0.5, min_scale=2, max_scale=20, 
                             beta=5/3, num_octaves=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X, Y, Z, _, _, _ = make_grid(n, L)
    profile = np.ones((n, n, n)) * base_value
    
    field = np.zeros((n, n, n))
    for octave in range(num_octaves):
        scale = max_scale / (2**octave)
        if scale < min_scale:
            break
            
        noise = np.random.randn(n, n, n)
        smoothed_noise = gaussian_filter(noise, scale)
        
        amplitude = scale**beta
        field += amplitude * smoothed_noise
    
    field -= np.min(field)
    field /= np.max(field)
    
    profile = base_value * np.exp(intensity * (field - 0.5))
    
    return profile

def generate_c_fields(n, L, num_fields=5, field_types=None, base_value=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    if field_types is None:
        field_types = ['layered', 'waveguide', 'quasiperiodic', 'turbulent']
        
    fields = []
    params = []
    
    for i in range(num_fields):
        field_type = np.random.choice(field_types)
        
        if field_type == 'layered':
            num_layers = np.random.randint(2, 6)
            min_amp = np.random.uniform(0.1, 0.3)
            max_amp = np.random.uniform(0.4, 0.8)
            min_freq = np.random.uniform(1, 3)
            max_freq = np.random.uniform(5, 15)
            
            field = layered_materials(n, L, base_value, num_layers, min_amp, max_amp, min_freq, max_freq)
            param = {'type': 'layered', 'num_layers': num_layers, 'min_amp': min_amp, 
                    'max_amp': max_amp, 'min_freq': min_freq, 'max_freq': max_freq}


        elif field_type == 'constant':
            field = constant(n, L)
            param = {'type': 'constant'}

        elif field_type == 'piecewise_constant':
            num_layers = np.random.randint(2, 6)
            contrast_factor = np.random.uniform(1.5, 3.5)
            field = piecewise_constant(n, L, num_layers=num_layers, contrast_factor=contrast_factor)
            param = {'type': 'piecewise_constant', 'num_layers': num_layers, 'contrast_factor': contrast_factor}

        elif field_type == 'sign_changing_mass':
            regions = np.random.choice(['checkerboard', 'half_space'])
            scale = np.random.randint(2, 5)
            sharpness = np.random.uniform(3., 7.)
            field = sign_changing_mass(n, L, regions=regions, scale=scale, sharpness=sharpness) 
            param = {'type': 'sign_changing_mass', 'regions': regions, 'scale': scale, 'sharpness': sharpness} 
 
        elif field_type == 'waveguide':
            num_guides = np.random.randint(1, 5)
            min_width = np.random.uniform(0.05, 0.2)
            max_width = np.random.uniform(0.3, 0.8)
            guide_amp = np.random.uniform(0.5, 1.5)
            
            field = wave_guiding_structures(n, L, base_value, num_guides, min_width, max_width, guide_amp)
            param = {'type': 'waveguide', 'num_guides': num_guides, 'min_width': min_width,
                    'max_width': max_width, 'guide_amp': guide_amp}
            
        elif field_type == 'quasiperiodic':
            num_waves = np.random.randint(4, 10)
            min_amp = np.random.uniform(0.1, 0.3)
            max_amp = np.random.uniform(0.4, 0.8)
            
            field = quasi_periodic_structures(n, L, base_value, num_waves, min_amp, max_amp)
            param = {'type': 'quasiperiodic', 'num_waves': num_waves, 
                    'min_amp': min_amp, 'max_amp': max_amp}
            
        elif field_type == 'turbulent':
            intensity = np.random.uniform(0.3, 0.8)
            min_scale = np.random.uniform(1, 3)
            max_scale = np.random.uniform(10, 30)
            beta = np.random.uniform(1, 3)
            num_octaves = np.random.randint(3, 8)
            
            field = turbulent_like_fields(n, L, base_value, intensity, min_scale, max_scale, beta, num_octaves)
            param = {'type': 'turbulent', 'intensity': intensity, 'min_scale': min_scale,
                    'max_scale': max_scale, 'beta': beta, 'num_octaves': num_octaves}
        else:
            raise NotImplemented
            
        fields.append(field)
        params.append(param)
        
    return fields, params

def visualize_fields_slices(field, slice_indices=None, figsize=(15, 5)):
    n = field.shape[0]
    
    if slice_indices is None:
        slice_indices = [n//4, n//2, 3*n//4]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    im1 = axes[0].imshow(field[slice_indices[0], :, :], cmap='viridis', origin='lower')
    axes[0].set_title(f'X-slice at index {slice_indices[0]}')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(field[:, slice_indices[1], :], cmap='viridis', origin='lower')
    axes[1].set_title(f'Y-slice at index {slice_indices[1]}')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(field[:, :, slice_indices[2]], cmap='viridis', origin='lower')
    axes[2].set_title(f'Z-slice at index {slice_indices[2]}')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig

def compare_fields(fields, params, slice_index=None, figsize=(18, 12)):
    n_fields = len(fields)
    if slice_index is None:
        slice_index = fields[0].shape[0] // 2
        
    fig, axes = plt.subplots(n_fields, 3, figsize=figsize) 
    for i, field in enumerate(fields):
        im1 = axes[i, 0].imshow(field[slice_index, :, :], cmap='viridis', origin='lower')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(field[:, slice_index, :], cmap='viridis', origin='lower')
        plt.colorbar(im2, ax=axes[i, 1])
        
        im3 = axes[i, 2].imshow(field[:, :, slice_index], cmap='viridis', origin='lower')
        plt.colorbar(im3, ax=axes[i, 2])
        
        if params is not None:
            axes[i, 0].set_title(f"Field {i+1}: {params[i]['type']} - X slice")
            axes[i, 1].set_title(f"Y slice")
            axes[i, 2].set_title(f"Z slice")
            
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    for i in range(5):
        for field in ['constant', 'piecewise_constant', 'sign_changing_mass', 'layered', 'waveguide', 'quasiperiodic', 'turbulent']:
            fields, params = generate_multiple_fields(64, 3., num_fields=3, field_types=[field] * 3,)
            compare_fields(fields, params, figsize=(18, 12))
            plt.show()

