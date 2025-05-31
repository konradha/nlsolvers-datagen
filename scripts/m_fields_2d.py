import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def make_grid(n, L):
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y, x, y

def constant_mass(n, L, base_value=1.0):
    return np.ones((n, n), dtype=np.float64) * base_value

def piecewise_constant_mass(n, L, m1=1.0, m2=2.0, boundary_type='circle', 
                           boundary_param=0.5, smooth_width=0.05):
    X, Y, _, _ = make_grid(n, L)
    m = np.ones((n, n), dtype=np.float64) * m1
    
    if boundary_type == 'circle':
        r = np.sqrt(X**2 + Y**2)
        boundary = r - boundary_param * L
    elif boundary_type == 'square':
        boundary = np.maximum(np.abs(X), np.abs(Y)) - boundary_param * L
    elif boundary_type == 'horizontal':
        boundary = Y
    elif boundary_type == 'vertical':
        boundary = X
    elif boundary_type == 'diagonal':
        boundary = X + Y
    
    smooth_transition = 0.5 * (1 + np.tanh(boundary / (smooth_width * L)))
    m = m1 + (m2 - m1) * smooth_transition
    
    return m

def gradient_based_mass(n, L, c_field, m0=1.0, gamma=1.0, epsilon=1e-6):
    grad_x = np.zeros_like(c_field)
    grad_y = np.zeros_like(c_field)
    
    grad_x[1:-1, :] = (c_field[2:, :] - c_field[:-2, :]) / 2
    grad_y[:, 1:-1] = (c_field[:, 2:] - c_field[:, :-2]) / 2
    
    grad_sq = grad_x**2 + grad_y**2
    m = m0 * (1 + gamma * grad_sq / (grad_sq + epsilon**2))
    
    return m

def phase_shifted_mass(n, L, c_field, m0=1.0, delta=0.5, shift_fraction=0.05):
    shift_pixels = max(1, int(shift_fraction * n))
    
    shifted_x_plus = np.roll(c_field, shift_pixels, axis=0)
    shifted_x_minus = np.roll(c_field, -shift_pixels, axis=0)
    
    shifted_y_plus = np.roll(c_field, shift_pixels, axis=1)
    shifted_y_minus = np.roll(c_field, -shift_pixels, axis=1)
    
    directional_derivative_x = shifted_x_plus - shifted_x_minus
    directional_derivative_y = shifted_y_plus - shifted_y_minus
    
    derivative_magnitude = np.sqrt(directional_derivative_x**2 + directional_derivative_y**2)
    derivative_normalized = derivative_magnitude / np.max(np.abs(derivative_magnitude))
    
    m = m0 * (1 + delta * derivative_normalized)
    
    return m

def topological_mass(n, L, c_field, m0=1.0, eta=0.8, lambda_param=0.5):
    lap_c = laplace(c_field)
    
    threshold = lambda_param * c_field
    topo_field = np.sign(lap_c - threshold)
    
    smooth_topo = gaussian_filter(topo_field, sigma=1.0)
    normalized_topo = smooth_topo / np.max(np.abs(smooth_topo))
    
    m = m0 * (1 + eta * normalized_topo)
    
    return m

def localized_mass_defects(n, L, m0=1.0, num_defects=10, min_strength=-0.5, max_strength=1.0,
                          min_width=0.05, max_width=0.2):    
    X, Y, _, _ = make_grid(n, L)
    m = np.ones((n, n), dtype=np.float64) * m0
    
    for _ in range(num_defects):
        x0 = np.random.uniform(-L, L)
        y0 = np.random.uniform(-L, L)
        
        strength = np.random.uniform(min_strength, max_strength) * m0
        width = np.random.uniform(min_width, max_width) * L
        
        r_sq = (X - x0)**2 + (Y - y0)**2
        defect = strength * np.exp(-r_sq / (2 * width**2))
        
        m += defect
    
    m = np.maximum(m, 0.1 * m0)
    
    return m

def quasi_periodic_mass(n, L, m0=1.0, num_waves=5, min_amp=0.1, max_amp=0.5):       
    X, Y, _, _ = make_grid(n, L)
    m = np.ones((n, n), dtype=np.float64) * m0
    
    k_values = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(num_waves):
        angle = i * np.pi / num_waves
        kx = np.cos(angle)
        ky = np.sin(angle)
        
        scale = golden_ratio ** i
        k_values.append([kx * scale, ky * scale])
    
    for kx, ky in k_values:
        amplitude = np.random.uniform(min_amp, max_amp) * m0
        phase = np.random.uniform(0, 2*np.pi)
        
        k_dot_x = kx * X + ky * Y
        m += amplitude * np.cos(k_dot_x + phase)
    
    m = np.maximum(m, 0.1 * m0)
    
    return m

def multiscale_mass(n, L, m0=1.0, num_scales=4, min_scale=2, max_scale=16, 
                   min_amp=0.1, max_amp=0.5):     
    m = np.ones((n, n),dtype=np.float64) * m0
    
    scale_factors = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    for scale in scale_factors:
        noise = np.random.randn(n, n)
        smoothed_noise = gaussian_filter(noise, sigma=scale)
        
        amplitude = np.random.uniform(min_amp, max_amp) * m0
        m += amplitude * smoothed_noise / np.max(np.abs(smoothed_noise))
    
    m = np.maximum(m, 0.1 * m0)
    
    return m

def generate_m_fields(n, L, c_field=None, num_fields=8, field_types=None, 
                           m0=1.0): 
    if field_types is None:
        field_types = ['constant', 'piecewise', 'gradient', 'phase', 'topological',
                      'defects', 'quasiperiodic', 'multiscale']
    
    fields = []
    params = []
    
    for i, field_type in enumerate(field_types[:num_fields]):
        if field_type == 'constant':
            field = constant_mass(n, L, m0)
            param = {'type': 'constant', 'm0': m0}
            
        elif field_type == 'piecewise':
            m1 = m0
            m2 = np.random.uniform(1.5, 3.0) * m0
            boundary_types = ['circle', 'square', 'horizontal', 'vertical', 'diagonal']
            boundary_type = np.random.choice(boundary_types)
            boundary_param = np.random.uniform(0.3, 0.7)
            smooth_width = np.random.uniform(0.01, 0.1)
            
            field = piecewise_constant_mass(n, L, m1, m2, boundary_type, boundary_param, smooth_width)
            param = {'type': 'piecewise', 'm1': m1, 'm2': m2, 
                    'boundary_type': boundary_type, 'boundary_param': boundary_param,
                    'smooth_width': smooth_width}
            
        elif field_type == 'gradient' and c_field is not None:
            gamma = np.random.uniform(0.5, 2.0)
            epsilon = np.random.uniform(1e-3, 1e-1)
            
            field = gradient_based_mass(n, L, c_field, m0, gamma, epsilon)
            param = {'type': 'gradient', 'm0': m0, 'gamma': gamma, 'epsilon': epsilon}
            
        elif field_type == 'phase' and c_field is not None:
            delta = np.random.uniform(0.3, 1.0)
            shift_fraction = np.random.uniform(0.02, 0.1)
            
            field = phase_shifted_mass(n, L, c_field, m0, delta, shift_fraction)
            param = {'type': 'phase', 'm0': m0, 'delta': delta, 
                    'shift_fraction': shift_fraction}
            
        elif field_type == 'topological' and c_field is not None:
            eta = np.random.uniform(0.5, 1.0)
            lambda_param = np.random.uniform(0.3, 0.7)
            
            field = topological_mass(n, L, c_field, m0, eta, lambda_param)
            param = {'type': 'topological', 'm0': m0, 'eta': eta, 
                    'lambda_param': lambda_param}
            
        elif field_type == 'defects':
            num_defects = np.random.randint(5, 20)
            min_strength = np.random.uniform(-0.5, -0.1)
            max_strength = np.random.uniform(0.5, 1.0)
            min_width = np.random.uniform(0.03, 0.08)
            max_width = np.random.uniform(0.1, 0.3)
            
            field = localized_mass_defects(n, L, m0, num_defects, min_strength, 
                                         max_strength, min_width, max_width)
            param = {'type': 'defects', 'm0': m0, 'num_defects': num_defects,
                    'min_strength': min_strength, 'max_strength': max_strength,
                    'min_width': min_width, 'max_width': max_width}
            
        elif field_type == 'quasiperiodic':
            num_waves = np.random.randint(3, 8)
            min_amp = np.random.uniform(0.1, 0.3)
            max_amp = np.random.uniform(0.4, 0.8)
            
            field = quasi_periodic_mass(n, L, m0, num_waves, min_amp, max_amp)
            param = {'type': 'quasiperiodic', 'm0': m0, 'num_waves': num_waves,
                    'min_amp': min_amp, 'max_amp': max_amp}
            
        elif field_type == 'multiscale':
            num_scales = np.random.randint(3, 6)
            min_scale = np.random.uniform(1, 3)
            max_scale = np.random.uniform(8, 20)
            min_amp = np.random.uniform(0.1, 0.3)
            max_amp = np.random.uniform(0.4, 0.8)
            
            field = multiscale_mass(n, L, m0, num_scales, min_scale, max_scale, min_amp, max_amp)
            param = {'type': 'multiscale', 'm0': m0, 'num_scales': num_scales,
                    'min_scale': min_scale, 'max_scale': max_scale,
                    'min_amp': min_amp, 'max_amp': max_amp}
            
        else:
            field = constant_mass(n, L, m0)
            param = {'type': 'constant', 'm0': m0}
            
        fields.append(field.astype(np.float64))
        params.append(param)
        
    return fields, params

def visualize_m_fields(fields, params=None, figsize=(15, 10)):
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

if __name__ =='__main__':
    field_types = ['constant', 'piecewise', 'gradient', 'phase', 'topological',
                      'defects', 'quasiperiodic', 'multiscale']
    for field in field_types:
        fields, params = generate_mass_fields(100, 3., c_field=None, num_fields=6, field_types=[field] * 6)
        visualize_mass_fields(fields, params=params)
        plt.show()
   
        

