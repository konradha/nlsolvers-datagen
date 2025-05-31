import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def make_grid(n, L):
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y, x, y

def constant(n, L):
    return np.ones((n, n), dtype=np.float64)

def periodic_structure(n, L, base_value=1.0, amplitude=0.5, frequency=3):
    X, Y, _, _ = make_grid(n, L)
    k = np.pi * frequency / L
    profile = base_value * (1 + amplitude * (np.sin(k * X) * np.sin(k * Y)))
    return profile

def piecewise_constant(n, L, num_layers=3, base_value=1.0, contrast_factor=2.0):
    X, Y, _, _ = make_grid(n, L)
    profile = np.ones((n, n),dtype=np.float64) * base_value
    layer_width = 2*L / num_layers
    
    for i in range(num_layers):
        if i % 2 == 1:
            x_min = -L + i * layer_width
            x_max = -L + (i + 1) * layer_width
            mask = (X >= x_min) & (X < x_max)
            profile[mask] = base_value * contrast_factor
            
    return profile


def sign_changing_mass(n, L, base_value=1.0, regions='checkerboard', scale=2, sharpness=5.0):
    X, Y, _, _ = make_grid(n, L)
    
    if regions == 'checkerboard':
        cell_size = L / scale
        pattern = np.sin(np.pi * X / cell_size) * np.sin(np.pi * Y / cell_size)
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
                      min_freq=2, max_freq=10):
    X, Y, _, _ = make_grid(n, L)
    profile = np.ones((n, n),dtype=np.float64) * base_value
    
    for _ in range(num_layers):
        theta = np.random.uniform(0, np.pi)
        nx, ny = np.cos(theta), np.sin(theta)
        direction = nx * X + ny * Y
        
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        freq = np.random.uniform(min_freq, max_freq)
        phase = np.random.uniform(0, 2*np.pi)
        
        profile += amplitude * np.sin(freq * direction + phase)
    
    max_val = np.max(profile)
    min_val = np.min(profile)
    profile = (profile - min_val) / (max_val - min_val)
    
    return base_value * profile

def wave_guiding_structures(n, L, base_value=1.0, min_width=0.1, max_width=0.5, 
                           guide_amplitude=0.8):
    num_guides = np.random.randint(3, 12)
        
    X, Y, x, y = make_grid(n, L)
    profile = np.ones((n, n)) * base_value
    
    for _ in range(num_guides):
        if np.random.random() < 0.5:
            curve_type = 'line'
        else:
            curve_type = 'curve'
            
        width = np.random.uniform(min_width, max_width)
        
        if curve_type == 'line':
            x0 = np.random.uniform(-L, L)
            y0 = np.random.uniform(-L, L)
            theta = np.random.uniform(0, 2*np.pi)
            nx, ny = np.cos(theta), np.sin(theta)
            
            t = np.linspace(-1.5*L, 1.5*L, 100)
            curve_x = x0 + t * nx
            curve_y = y0 + t * ny
            
        else:
            a = np.random.uniform(0.5, 2.0)
            b = np.random.uniform(0.5, 2.0)
            phi = np.random.uniform(0, 2*np.pi)
            
            t = np.linspace(0, 2*np.pi, 100)
            curve_x = a * np.cos(t + phi)
            curve_y = b * np.sin(t)
            
        curve_points = np.column_stack([curve_x, curve_y])
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        
        distances = cdist(grid_points, curve_points).min(axis=1).reshape(n, n)
        
        guide_profile = guide_amplitude * np.exp(-(distances**2) / (2 * width**2))
        profile = np.maximum(profile, guide_profile)
    
    return profile

def quasi_periodic_structures(n, L, base_value=1.0, num_waves=5, min_amp=0.1, max_amp=0.5):
    X, Y, _, _ = make_grid(n, L)
    profile = np.ones((n, n),dtype=np.float64) * base_value
    
    k_values = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(num_waves):
        angle = i * np.pi / num_waves
        kx = np.cos(angle)
        ky = np.sin(angle)
        
        scale = golden_ratio ** i
        k_values.append([kx * scale, ky * scale])
    
    for kx, ky in k_values:
        amplitude = np.random.uniform(min_amp, max_amp)
        phase = np.random.uniform(0, 2*np.pi)
        
        k_dot_x = kx * X + ky * Y
        profile += amplitude * np.cos(k_dot_x + phase)
    
    max_val = np.max(profile)
    min_val = np.min(profile)
    profile = (profile - min_val) / (max_val - min_val)
    
    return base_value * profile

def turbulent_like_fields(n, L, base_value=1.0, intensity=0.5, min_scale=2, max_scale=20, 
                          beta=5/3, num_octaves=5): 
    X, Y, x, y = make_grid(n, L)
    profile = np.ones((n, n),dtype=np.float64) * base_value
    
    field = np.zeros((n, n))
    for octave in range(num_octaves):
        scale = max_scale / (2**octave)
        if scale < min_scale:
            break 
        noise = np.random.randn(n, n)
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
        field_types = ['periodic_structure','piecewise_constant','sign_changing_mass',
                'layered', 'waveguide', 'quasiperiodic', 'turbulent']
        
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
            
        elif field_type == 'waveguide':
            min_width = np.random.uniform(0.1, 0.3)
            max_width = np.random.uniform(0.4, 0.8)
            guide_amp = np.random.uniform(0.5, 2.)
            
            field = wave_guiding_structures(n, L, base_value, min_width, max_width, guide_amp)
            param = {'type': 'waveguide',  'min_width': min_width,
                    'max_width': max_width, 'guide_amp': guide_amp} 
        elif field_type == 'quasiperiodic':
            num_waves = np.random.randint(3, 8)
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
        elif field_type == 'periodic_structure':
            amplitude = np.random.uniform(0.2, 0.5)
            frequency = np.random.randint(1, 3)
            field = periodic_structure(n, L, amplitude=amplitude, frequency=frequency)
            param = {'type': 'periodic_structure', 'amplitude': amplitude}

        elif field_type == 'piecewise_constant':
            num_layers = np.random.randint(2, 5)
            contrast_factor = np.random.uniform(1.5, 2.5)
            field = piecewise_constant(n, L, num_layers=num_layers, contrast_factor=contrast_factor)
            param = {'type': 'piecewise_constant', 'num_layers': num_layers, 'contrast_factor': contrast_factor} 

        elif field_type == 'sign_changing_mass':
            regions = ['checkerboard', 'half_space'] 
            regions = np.random.choice(regions)
            scale = np.random.randint(2, 3)
            sharpness = np.random.uniform(3, 6) 
            field = sign_changing_mass(n, L, regions=regions, scale=scale, sharpness=sharpness)
            param = {'type': 'sign_changing_mass', 'regions': regions, 'scale': scale, 'sharpness': sharpness}
        elif field_type == 'constant':
            field = constant(n, L)
            param = {}
        else:
            raise NotImplemented

            
        fields.append(field)
        params.append(param)
        
    return fields, params

def visualize_fields(fields, params=None, figsize=(15, 10)):
    n_fields = len(fields)
    rows = int(np.ceil(n_fields / 3))
    cols = min(n_fields, 3)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, field in enumerate(fields):
        
        if i < len(axes):
            im = axes[i].imshow(field, cmap='viridis', origin='lower')
            plt.colorbar(im, ax=axes[i])
            
            if params is not None:
                title = f"Field {i+1}: {params[i]['type']}"
                axes[i].set_title(title)
            else:
                axes[i].set_title(f"Field {i+1}")
                
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    for field in ['periodic_structure','piecewise_constant','sign_changing_mass',
                'layered', 'waveguide', 'quasiperiodic', 'turbulent']:
        fields, params =generate_c_fields(100, 3., num_fields=9, field_types=[field] * 9)
        visualize_fields(fields, params=params)
        plt.show()

