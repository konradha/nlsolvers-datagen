import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.spatial.distance import cdist
from scipy.fft import fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt

def make_grid(n, L):
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    z = np.linspace(-L, L, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z, x, y, z

def constant_mass(n, L, base_value=1.0):
    return np.ones((n, n, n)) * base_value

def piecewise_constant_mass(n, L, m1=1.0, m2=2.0, boundary_type='sphere',
                           boundary_param=0.5, smooth_width=0.05):
    X, Y, Z, _, _, _ = make_grid(n, L)
    m = np.ones((n, n, n)) * m1

    if boundary_type == 'sphere':
        r = np.sqrt(X**2 + Y**2 + Z**2)
        boundary = r - boundary_param * L
    elif boundary_type == 'cube':
        boundary = np.maximum.reduce([np.abs(X), np.abs(Y), np.abs(Z)]) - boundary_param * L
    elif boundary_type == 'xy_plane':
        boundary = Z
    elif boundary_type == 'xz_plane':
        boundary = Y
    elif boundary_type == 'yz_plane':
        boundary = X

    smooth_transition = 0.5 * (1 + np.tanh(boundary / (smooth_width * L)))
    m = m1 + (m2 - m1) * smooth_transition

    return m

def gradient_based_mass(n, L, c_field, m0=1.0, gamma=1.0, epsilon=1e-6):
    grad_x = np.zeros_like(c_field)
    grad_y = np.zeros_like(c_field)
    grad_z = np.zeros_like(c_field)

    grad_x[1:-1, :, :] = (c_field[2:, :, :] - c_field[:-2, :, :]) / 2
    grad_y[:, 1:-1, :] = (c_field[:, 2:, :] - c_field[:, :-2, :]) / 2
    grad_z[:, :, 1:-1] = (c_field[:, :, 2:] - c_field[:, :, :-2]) / 2

    grad_sq = grad_x**2 + grad_y**2 + grad_z**2
    m = m0 * (1 + gamma * grad_sq / (grad_sq + epsilon**2))

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
                          min_width=0.05, max_width=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X, Y, Z, _, _, _ = make_grid(n, L)
    m = np.ones((n, n, n)) * m0

    for _ in range(num_defects):
        x0 = np.random.uniform(-L, L)
        y0 = np.random.uniform(-L, L)
        z0 = np.random.uniform(-L, L)

        strength = np.random.uniform(min_strength, max_strength) * m0
        width = np.random.uniform(min_width, max_width) * L

        r_sq = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
        defect = strength * np.exp(-r_sq / (2 * width**2))

        m += defect

    m = np.maximum(m, 0.1 * m0)

    return m

def quasi_periodic_mass(n, L, m0=1.0, num_waves=6, min_amp=0.1, max_amp=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X, Y, Z, _, _, _ = make_grid(n, L)
    m = np.ones((n, n, n)) * m0

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
        amplitude = np.random.uniform(min_amp, max_amp) * m0
        phase = np.random.uniform(0, 2*np.pi)

        k_dot_x = kx * X + ky * Y + kz * Z
        m += amplitude * np.cos(k_dot_x + phase)

    m = np.maximum(m, 0.1 * m0)

    return m

def multiscale_mass(n, L, m0=1.0, num_scales=4, min_scale=2, max_scale=16,
                   min_amp=0.1, max_amp=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    m = np.ones((n, n, n)) * m0

    scale_factors = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)

    for scale in scale_factors:
        noise = np.random.randn(n, n, n)
        smoothed_noise = gaussian_filter(noise, sigma=scale)

        amplitude = np.random.uniform(min_amp, max_amp) * m0
        m += amplitude * smoothed_noise / np.max(np.abs(smoothed_noise))

    m = np.maximum(m, 0.1 * m0)

    return m

def generate_m_fields(n, L, c_field=None, num_fields=8, field_types=None,
                           m0=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if field_types is None:
        field_types = ['constant', 'piecewise', 'gradient', 'phase', 'topological',
                      'defects', 'quasiperiodic', 'multiscale']

    fields = []
    params = []

    if c_field is None: c_field = np.ones((n, n, n),dtype=np.float64)

    for i, field_type in enumerate(field_types[:num_fields]):
        if field_type == 'constant':
            field = constant_mass(n, L, m0)
            param = {'type': 'constant', 'm0': m0}

        elif field_type == 'piecewise':
            m1 = m0
            m2 = np.random.uniform(1.5, 3.0) * m0
            boundary_types = ['sphere', 'cube', 'xy_plane', 'xz_plane', 'yz_plane']
            boundary_type = np.random.choice(boundary_types)
            boundary_param = np.random.uniform(0.3, 0.7)
            smooth_width = np.random.uniform(0.01, 0.1)

            field = piecewise_constant_mass(n, L, m1, m2, boundary_type, boundary_param, smooth_width)
            param = {'type': 'piecewise', 'm1': m1, 'm2': m2,
                    'boundary_type': boundary_type, 'boundary_param': boundary_param,
                    'smooth_width': smooth_width}

        elif field_type == 'gradient':
            gamma = np.random.uniform(0.5, 2.0)
            epsilon = np.random.uniform(1e-3, 1e-1)

            field = gradient_based_mass(n, L, c_field, m0, gamma, epsilon)
            param = {'type': 'gradient', 'm0': m0, 'gamma': gamma, 'epsilon': epsilon}


        elif field_type == 'topological':
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
            num_waves = np.random.randint(4, 10)
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

        elif field_type == 'constant':
            field = constant_mass(n, L, m0)
            param = {'type': 'constant', 'm0': m0}
        else:
            raise NotImplemented


        fields.append(field)
        params.append(param)

    return fields, params

def visualize_fields_3d_slices(field, slice_indices=None, figsize=(15, 5)):
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

def compare_mass_fields(fields, params, slice_index=None, figsize=(18, 12)):
    n_fields = len(fields)
    if slice_index is None:
        slice_index = fields[0].shape[0] // 2

    fig, axes = plt.subplots(n_fields, 3, figsize=figsize)
    if n_fields == 1:
        axes = axes.reshape(1, 3)

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

if __name__ =='__main__':
    field_types = ['constant', 'piecewise', 'gradient', 'topological',
                      'defects', 'quasiperiodic', 'multiscale']

    for field in field_types:
        print(field)
        fields, params = generate_mass_fields(64, 3., c_field=None, num_fields=3, field_types=[field] * 3)
        compare_mass_fields(fields, params,)
        plt.show()
