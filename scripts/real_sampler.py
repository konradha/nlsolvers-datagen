import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class RealWaveSampler:
    # phenomena to integrate:
    # - multi-ring
    # - multi-single

    def __init__(self, nx: int, ny: int, L: float):
        self.nx = nx
        self.ny = ny
        self.L = L

        self._setup_grid()

    def _setup_grid(self):
        self.x = np.linspace(-self.L, self.L, self.nx)
        self.y = np.linspace(-self.L, self.L, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.r = np.sqrt(self.X**2 + self.Y**2)
        self.theta = np.arctan2(self.Y, self.X)

        self.dx = 2 * self.L / (self.nx - 1)
        self.dy = 2 * self.L / (self.ny - 1)

        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=2 * self.L / self.nx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=2 * self.L / self.ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2)

    def nonlinear_potential(self, u: np.ndarray,
                            system_type: str = 'sine_gordon') -> np.ndarray:
        if system_type == 'sine_gordon':
            return np.sin(u)
        elif system_type == 'double_sine_gordon':
            lambda_param = 0.3
            return np.sin(u) + lambda_param * np.sin(2 * u)
        elif system_type == 'hyperbolic_sine_gordon':
            return np.sinh(u)
        elif system_type == 'phi4':
            return u**3 - u
        elif system_type == 'klein_gordon':
            return u
        else:
            return np.sin(u)

    def anisotropic_grf(self, length_scale: float = 1.0, anisotropy_ratio: float = 2.0,
                        theta: float = 30.0, power: float = 2.0, amplitude: float = 1.0) -> np.ndarray:
        theta_rad = np.deg2rad(theta)
        ell_x = length_scale * np.sqrt(anisotropy_ratio)
        ell_y = length_scale / np.sqrt(anisotropy_ratio)

        KX_rot = self.KX * np.cos(theta_rad) - self.KY * np.sin(theta_rad)
        KY_rot = self.KX * np.sin(theta_rad) + self.KY * np.cos(theta_rad)

        spectrum = np.exp(-((KX_rot / ell_x)**2 +
                          (KY_rot / ell_y)**2)**(power / 2))
        noise = np.random.randn(self.nx, self.ny) + \
            1j * np.random.randn(self.nx, self.ny)
        field = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectrum)).real

        field = field / np.std(field) * amplitude

        return field

    def wavelet_superposition(self, n_wavelets: int = 20, scale_range: Tuple[float, float] = (0.1, 2.0),
                              kappa: float = 0.5, freq_range: Tuple[float, float] = (0.5, 3.0),
                              amplitude: float = 1.0) -> np.ndarray:
        v0 = np.zeros((self.nx, self.ny))

        for _ in range(n_wavelets):
            scale = scale_range[0] + (scale_range[1] -
                                      scale_range[0]) * np.random.rand()
            theta = 2 * np.pi * np.random.rand()
            x0 = self.L * (np.random.rand() - 0.5)
            y0 = self.L * (np.random.rand() - 0.5)
            k0 = (freq_range[0] + (freq_range[1] - freq_range[0])
                  * np.random.rand()) * (2 * np.pi / (scale * self.L))

            envelope = np.exp(-((self.X - x0)**2 + (self.Y - y0)
                              ** 2) / (2 * (scale * self.L)**2))

            wavelet_type = np.random.rand()
            if wavelet_type < 0.33:
                carrier = np.cos(
                    k0 * ((self.X - x0) * np.cos(theta) + (self.Y - y0) * np.sin(theta)))
            elif wavelet_type < 0.66:
                z = ((self.X - x0) * np.cos(theta) + (self.Y - y0)
                     * np.sin(theta)) / (scale * self.L)
                carrier = -z * np.exp(-z**2 / 2)
            else:
                z = ((self.X - x0) * np.cos(theta) +
                     (self.Y - y0) * np.sin(theta))
                carrier = np.cos(k0 * z) * \
                    np.exp(-(z / (scale * self.L))**2 / 2)

            amp = (1 - kappa) + kappa * np.random.rand()
            v0 += amp * envelope * carrier

        return v0 / np.max(np.abs(v0)) * amplitude

    def kink_solution(self, system_type: str = 'sine_gordon', width: float = 1.0,
                      position: Tuple[float, float] = (0.0, 0.0), orientation: float = 0.0,
                      velocity: Tuple[float, float] = (0.0, 0.0),
                      kink_type: str = 'standard',
                      velocity_type: str = 'fitting') -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = position
        vx, vy = velocity

        X_rot = (self.X - x0) * np.cos(orientation) + \
            (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + \
            (self.Y - y0) * np.cos(orientation)

        if system_type == 'sine_gordon':
            if kink_type == 'anti':
                u = -4 * np.arctan(np.exp(X_rot / width))
                if velocity_type == 'fitting':
                    v = -vx * 4 / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)
            elif kink_type == 'double':
                u = 4 * np.arctan(np.exp(X_rot / width)) + 4 * \
                    np.arctan(np.exp((X_rot - 2 * width) / width))
                if velocity_type == 'fitting':
                    v = vx * 4 / (width * (np.cosh(X_rot / width)**2)) + vx * \
                        4 / (width * (np.cosh((X_rot - 2 * width) / width)**2))
                else:
                    v = np.zeros_like(u)
            else:  # fallback + 'standard'
                u = 4 * np.arctan(np.exp(X_rot / width))
                if velocity_type == 'fitting':
                    v = vx * 4 / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)

        elif system_type == 'phi4':
            if kink_type == 'anti':
                u = -np.tanh(X_rot / width)
                if velocity_type == 'fitting':
                    v = -vx / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)
            elif kink_type == 'double':
                u = np.tanh(X_rot / width) - \
                    np.tanh((X_rot - 4 * width) / width)
                if velocity_type == 'fitting':
                    v = vx / (width * (np.cosh(X_rot / width)**2)) - vx / \
                        (width * (np.cosh((X_rot - 4 * width) / width)**2))
                else:
                    v = np.zeros_like(u)
            else:  # fallback + 'standard'
                u = np.tanh(X_rot / width)
                if velocity_type == 'fitting':
                    v = vx / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)

        elif system_type == 'double_sine_gordon':
            lambda_param = 0.3
            k = 1 / np.sqrt(1 + lambda_param)
            if kink_type == 'anti':
                u = -4 * np.arctan(np.sqrt((1 + lambda_param) / lambda_param) *
                                   np.tanh(np.sqrt(lambda_param) * X_rot / (2 * width)))
                if velocity_type == 'fitting':
                    v = -vx * 4 * np.sqrt((1 + lambda_param) / lambda_param) * np.sqrt(lambda_param) / (2 * width) * (
                        1 - np.tanh(np.sqrt(lambda_param) * X_rot / (2 * width))**2)
                else:
                    v = np.zeros_like(u)
            else:  # fallback + 'standard'
                u = 4 * np.arctan(np.sqrt((1 + lambda_param) / lambda_param) *
                                  np.tanh(np.sqrt(lambda_param) * X_rot / (2 * width)))
                if velocity_type == 'fitting':
                    v = vx * 4 * np.sqrt((1 + lambda_param) / lambda_param) * np.sqrt(lambda_param) / (2 * width) * (
                        1 - np.tanh(np.sqrt(lambda_param) * X_rot / (2 * width))**2)
                else:
                    v = np.zeros_like(u)

        elif system_type == 'hyperbolic_sine_gordon':
            if kink_type == 'anti':
                u = -4 * np.arctan(np.exp(X_rot / width)) + 2 * np.pi
                if velocity_type == 'fitting':
                    v = -vx * 4 / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)
            else:  # fallback + 'standard':
                u = 4 * np.arctan(np.exp(X_rot / width)) - 2 * np.pi
                if velocity_type == 'fitting':
                    v = vx * 4 / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)

        elif system_type == 'klein_gordon':
            if kink_type == 'anti':
                u = -np.tanh(X_rot / width)
                if velocity_type == 'fitting':
                    v = -vx / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)
            else:  # +  'standard':
                u = np.tanh(X_rot / width)
                if velocity_type == 'fitting':
                    v = vx / (width * (np.cosh(X_rot / width)**2))
                else:
                    v = np.zeros_like(u)

        else:
            u = 4 * np.arctan(np.exp(X_rot / width))
            if velocity_type == 'fitting':
                v = vx * 4 / (width * (np.cosh(X_rot / width)**2))
            else:
                v = np.zeros_like(u)

        if velocity_type == 'grf':
            v = self.anisotropic_grf(
                length_scale=width * 2.0,
                amplitude=np.max(
                    np.abs(u)) * 0.2)

        return u, v

    def kink_field(self, system_type: str = 'sine_gordon', winding_x: int = 1, winding_y: int = 0,
                   width_range: Tuple[float, float] = (0.5, 3.0),
                   randomize_positions: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        u0 = np.zeros_like(self.X)

        if winding_x != 0:
            width_x = width_range[0] + \
                (width_range[1] - width_range[0]) * np.random.rand()
            positions_x = []

            if randomize_positions:
                for i in range(abs(winding_x)):
                    positions_x.append(self.L * (2 * np.random.rand() - 1))
            else:
                for i in range(abs(winding_x)):
                    positions_x.append(
                        self.L * (-0.8 + 1.6 * i / (abs(winding_x))))

            sign_x = 1 if winding_x > 0 else -1
            for x0 in positions_x:
                kink_width = width_x * (0.8 + 0.4 * np.random.rand())
                u0 += sign_x * 4 * \
                    np.arctan(np.exp((self.X - x0) / kink_width))

        if winding_y != 0:
            width_y = width_range[0] + \
                (width_range[1] - width_range[0]) * np.random.rand()
            positions_y = []

            if randomize_positions:
                for i in range(abs(winding_y)):
                    positions_y.append(self.L * (2 * np.random.rand() - 1))
            else:
                for i in range(abs(winding_y)):
                    positions_y.append(
                        self.L * (-0.8 + 1.6 * i / (abs(winding_y))))

            sign_y = 1 if winding_y > 0 else -1
            for y0 in positions_y:
                kink_width = width_y * (0.8 + 0.4 * np.random.rand())
                u0 += sign_y * 4 * \
                    np.arctan(np.exp((self.Y - y0) / kink_width))

        v0 = self.anisotropic_grf(
            length_scale=np.mean(width_range) * 2.0,
            amplitude=np.max(
                np.abs(u0)) * 0.1)

        return u0, v0

    def kink_array_field(self, system_type: str = "sine_gordon",
                         num_kinks_x: int = 1, num_kinks_y: int = 1, width_range: Tuple = (0.5, 2.0),
                         jitter: float = 0.3):
        u0 = np.zeros_like(self.X)
        v0 = np.zeros_like(self.X)

        if num_kinks_x > 0:
            width_x = width_range[0] + \
                (width_range[1] - width_range[0]) * np.random.rand()
            spacing_x = 2.0 * self.L / (num_kinks_x + 1)
            for i in range(num_kinks_x):
                x0 = -self.L + (i + 1) * spacing_x
                if jitter > 0:
                    x0 = x0 + jitter * spacing_x * (2 * np.random.rand() - 1)
                sign_x = 1 if np.random.rand() > 0.5 else -1
                kink_width = width_x * (0.8 + 0.4 * np.random.rand())
                u0 += sign_x * 4 * \
                    np.arctan(np.exp((self.X - x0) / kink_width))

        if num_kinks_y > 0:
            width_y = width_range[0] + \
                (width_range[1] - width_range[0]) * np.random.rand()
            spacing_y = 2.0 * self.L / (num_kinks_y + 1)
            for i in range(num_kinks_y):
                y0 = -self.L + (i + 1) * spacing_y
                if jitter > 0:
                    y0 = y0 + jitter * spacing_y * (2 * np.random.rand() - 1)
                sign_y = 1 if np.random.rand() > 0.5 else -1
                kink_width = width_y * (0.8 + 0.4 * np.random.rand())
                u0 += sign_y * 4 * \
                    np.arctan(np.exp((self.Y - y0) / kink_width))

        return u0, v0

    def multi_breather_field(self, system_type: str = 'sine_gordon', num_breathers: int = 3,
                             position_type: str = 'random', amplitude_range: Tuple[float, float] = (0.2, 0.8),
                             width_range: Tuple[float, float] = (0.5, 2.0),
                             frequency_range: Tuple[float, float] = (
                                 0.6, 0.95),
                             time_param: float = 0.0, velocity_type: str = 'fitting') -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)

        positions = []
        if position_type not in ['circle', 'line']:
            for _ in range(num_breathers):
                x0 = self.L * (2 * np.random.rand() - 1)
                y0 = self.L * (2 * np.random.rand() - 1)
                positions.append((x0, y0))
        elif position_type == 'circle':
            radius = 0.6 * self.L * np.random.rand()
            for i in range(num_breathers):
                angle = 2 * np.pi * i / num_breathers
                x0 = radius * np.cos(angle)
                y0 = radius * np.sin(angle)
                positions.append((x0, y0))
        elif position_type == 'line':
            for i in range(num_breathers):
                pos = -self.L + 2 * self.L * i / \
                    (num_breathers - 1 if num_breathers > 1 else 1)
                if np.random.rand() > 0.5:
                    positions.append((pos, 0.0))
                else:
                    positions.append((0.0, pos))

        for x0, y0 in positions:
            width = width_range[0] + (width_range[1] -
                                      width_range[0]) * np.random.rand()
            amplitude = amplitude_range[0] + \
                (amplitude_range[1] - amplitude_range[0]) * np.random.rand()
            frequency = frequency_range[0] + \
                (frequency_range[1] - frequency_range[0]) * np.random.rand()
            phase = 2 * np.pi * np.random.rand()

            breather_type = 'standard' if np.random.rand() > 0.5 else 'radial'
            orientation = 2 * np.pi * np.random.rand()

            u_comp, v_comp = self.breather_solution(
                system_type=system_type,
                amplitude=amplitude,
                frequency=frequency,
                width=width,
                position=(x0, y0),
                phase=phase,
                orientation=orientation,
                breather_type=breather_type,
                time_param=time_param,
                velocity_type=velocity_type
            )

            u += u_comp
            v += v_comp

        return u, v

    def spiral_wave_field(self, num_arms: int = 2, decay_rate: float = 0.5,
                          amplitude: float = 1.0, position: Optional[Tuple[float, float]] = None,
                          phase: float = 0.0, k_factor: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        if position is None:
            x0 = self.L * (2 * np.random.rand() - 1)
            y0 = self.L * (2 * np.random.rand() - 1)
        else:
            x0, y0 = position

        if k_factor is None:
            k = 1.0 + 2.0 * np.random.rand()
        else:
            k = k_factor

        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = np.arctan2(self.Y - y0, self.X - x0)

        spiral_phase = theta + k * r / self.L + phase

        pattern = np.cos(num_arms * spiral_phase)

        decay = np.exp(-decay_rate * r / self.L)

        u = amplitude * pattern * decay
        v = amplitude * 0.1 * self.anisotropic_grf(length_scale=self.L / 5)

        return u, v

    def multi_spiral_state(self, n_spirals: int = 3, amplitude_range: Tuple[float, float] = (0.5, 1.5),
                           num_arms_range: Tuple[int, int] = (1, 4),
                           decay_rate_range: Tuple[float, float] = (0.3, 0.7),
                           position_variance: float = 1.0,
                           interaction_strength: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)

        for i in range(n_spirals):
            amplitude = amplitude_range[0] + \
                (amplitude_range[1] - amplitude_range[0]) * np.random.rand()
            num_arms = np.random.randint(
                num_arms_range[0], num_arms_range[1] + 1)
            decay_rate = decay_rate_range[0] + (
                decay_rate_range[1] - decay_rate_range[0]) * np.random.rand()

            x0 = np.random.normal(0.0, position_variance * self.L / 4)
            y0 = np.random.normal(0.0, position_variance * self.L / 4)
            phase = 2 * np.pi * np.random.rand()
            k_factor = 1.0 + 2.0 * np.random.rand()

            u_comp, v_comp = self.spiral_wave_field(
                num_arms=num_arms,
                decay_rate=decay_rate,
                amplitude=amplitude,
                position=(x0, y0),
                phase=phase,
                k_factor=k_factor
            )

            if i == 0:
                u = u_comp
                v = v_comp
            else:
                u = u + interaction_strength * u_comp
                v = v + interaction_strength * v_comp

        return u, v

    def ring_soliton(self, system_type: str = 'sine_gordon', amplitude: float = 1.0,
                     radius: float = 2.0, width: float = 0.5,
                     position: Tuple[float, float] = (0.0, 0.0), velocity: float = 0.0,
                     ring_type: str = 'expanding', modulation_strength: float = 0.0,
                     modulation_mode: int = 2, time_param: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta_local = np.arctan2(self.Y - y0, self.X - x0)

        if ring_type == 'expanding':
            if system_type == 'sine_gordon':
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / \
                    (width * (np.cosh((radius - r_local) / width)**2))
            elif system_type == 'phi4':
                u = amplitude * np.tanh((radius - r_local) / width)
                v = -velocity * amplitude / \
                    (width * (np.cosh((radius - r_local) / width)**2))
            elif system_type == 'double_sine_gordon':
                lambda_param = 0.3
                u = 4 * np.arctan(np.sqrt((1 + lambda_param) / lambda_param) *
                                  np.tanh(np.sqrt(lambda_param) * (radius - r_local) / (2 * width)))
                v = -velocity * 4 * np.sqrt((1 + lambda_param) / lambda_param) * np.sqrt(lambda_param) / (2 * width) * (
                    1 - np.tanh(np.sqrt(lambda_param) * (radius - r_local) / (2 * width))**2)
            elif system_type == 'hyperbolic_sine_gordon':
                u = 4 * \
                    np.arctan(np.exp((radius - r_local) / width)) - 2 * np.pi
                v = -velocity * 4 / \
                    (width * (np.cosh((radius - r_local) / width)**2))
            elif system_type == 'klein_gordon':
                u = amplitude * np.tanh((radius - r_local) / width)
                v = -velocity * amplitude / \
                    (width * (np.cosh((radius - r_local) / width)**2))
            else:
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / \
                    (width * (np.cosh((radius - r_local) / width)**2))

        elif ring_type == 'kink_antikink':
            inner_radius = radius - width
            outer_radius = radius + width

            if system_type == 'sine_gordon':
                u = 4 * np.arctan(np.exp((inner_radius - r_local) / (width / 2))) - \
                    4 * \
                    np.arctan(np.exp((outer_radius - r_local) / (width / 2)))
                v = -velocity * 4 / (width / 2 * (np.cosh((inner_radius - r_local) / (width / 2))**2)) + \
                    velocity * 4 / \
                    (width / 2 * (np.cosh((outer_radius - r_local) / (width / 2))**2))
            elif system_type == 'phi4':
                u = amplitude * np.tanh((inner_radius - r_local) / (width / 2)) - \
                    amplitude * np.tanh((outer_radius - r_local) / (width / 2))
                v = -velocity * amplitude / (width / 2 * (np.cosh((inner_radius - r_local) / (width / 2))**2)) + \
                    velocity * amplitude / \
                    (width / 2 * (np.cosh((outer_radius - r_local) / (width / 2))**2))
            elif system_type == 'double_sine_gordon':
                lambda_param = 0.3
                u = 4 * np.arctan(np.sqrt((1 + lambda_param) / lambda_param) *
                                  np.tanh(np.sqrt(lambda_param) * (inner_radius - r_local) / (2 * width / 2))) - \
                    4 * np.arctan(np.sqrt((1 + lambda_param) / lambda_param) *
                                  np.tanh(np.sqrt(lambda_param) * (outer_radius - r_local) / (2 * width / 2)))
                v_inner = -velocity * 4 * np.sqrt((1 + lambda_param) / lambda_param) * np.sqrt(lambda_param) / (2 * width / 2) * (
                    1 - np.tanh(np.sqrt(lambda_param) * (inner_radius - r_local) / (2 * width / 2))**2)
                v_outer = velocity * 4 * np.sqrt((1 + lambda_param) / lambda_param) * np.sqrt(lambda_param) / (2 * width / 2) * (
                    1 - np.tanh(np.sqrt(lambda_param) * (outer_radius - r_local) / (2 * width / 2))**2)
                v = v_inner + v_outer
            elif system_type == 'hyperbolic_sine_gordon':
                u = 4 * np.arctan(np.exp((inner_radius - r_local) / (width / 2))) - 4 * \
                    np.arctan(np.exp((outer_radius - r_local) /
                              (width / 2))) - 2 * np.pi
                v = -velocity * 4 / (width / 2 * (np.cosh((inner_radius - r_local) / (width / 2))**2)) + \
                    velocity * 4 / \
                    (width / 2 * (np.cosh((outer_radius - r_local) / (width / 2))**2))
            elif system_type == 'klein_gordon':
                u = amplitude * np.tanh((inner_radius - r_local) / (width / 2)) - \
                    amplitude * np.tanh((outer_radius - r_local) / (width / 2))
                v = -velocity * amplitude / (width / 2 * (np.cosh((inner_radius - r_local) / (width / 2))**2)) + \
                    velocity * amplitude / \
                    (width / 2 * (np.cosh((outer_radius - r_local) / (width / 2))**2))
            else:
                u = 4 * np.arctan(np.exp((inner_radius - r_local) / (width / 2))) - \
                    4 * \
                    np.arctan(np.exp((outer_radius - r_local) / (width / 2)))
                v = -velocity * 4 / (width / 2 * (np.cosh((inner_radius - r_local) / (width / 2))**2)) + \
                    velocity * 4 / \
                    (width / 2 * (np.cosh((outer_radius - r_local) / (width / 2))**2))

        else:
            if system_type == 'sine_gordon':
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / \
                    (width * (np.cosh((radius - r_local) / width)**2))
            elif system_type == 'phi4':
                u = amplitude * np.tanh((radius - r_local) / width)
                v = -velocity * amplitude / \
                    (width * (np.cosh((radius - r_local) / width)**2))
            else:
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / \
                    (width * (np.cosh((radius - r_local) / width)**2))

        if modulation_strength > 0:
            modulation = 1 + modulation_strength * \
                np.cos(modulation_mode * theta_local)
            u *= modulation
            v *= modulation

        return u, v

    def colliding_rings(self, system_type: str = 'sine_gordon', num_rings: int = 2,
                        ring_type: str = 'random', amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        u0 = np.zeros_like(self.X)
        v0 = np.zeros_like(self.X)

        if ring_type not in ['cocentric', 'nested']:
            for _ in range(num_rings):
                x0 = self.L * (2 * np.random.rand() - 1)
                y0 = self.L * (2 * np.random.rand() - 1)
                r0 = 0.1 * self.L + 0.6 * self.L * np.random.rand()
                width = 0.5 + 2.5 * np.random.rand()
                direction = 1 if np.random.rand() > 0.5 else -1

                r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
                if np.random.rand() > 0.5:
                    u0 += 4 * np.arctan(np.exp((r - r0) / width))
                    v0 += direction * np.exp(-(r - r0)**2 / (2 * width**2))
                else:
                    u0 -= 4 * np.arctan(np.exp((r - r0) / width))
                    v0 -= direction * np.exp(-(r - r0)**2 / (2 * width**2))

        elif ring_type == 'concentric':
            x0 = self.L * (2 * np.random.rand() - 1)
            y0 = self.L * (2 * np.random.rand() - 1)

            for i in range(num_rings):
                r0 = (0.2 + 0.6 * i / num_rings) * self.L
                width = 0.5 + 1.5 * np.random.rand()
                direction = 1 if i % 2 == 0 else -1

                r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
                u0 += direction * 4 * np.arctan(np.exp((r - r0) / width))
                v0 += direction * np.exp(-(r - r0)**2 / (2 * width**2))

        elif ring_type == 'nested':
            for i in range(num_rings):
                max_offset = 0.3 * self.L * i / num_rings
                x0 = max_offset * (2 * np.random.rand() - 1)
                y0 = max_offset * (2 * np.random.rand() - 1)
                r0 = (0.2 + 0.5 * (num_rings - i) / num_rings) * self.L
                width = 0.5 + 1.5 * np.random.rand()
                direction = 1 if i % 2 == 0 else -1

                r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
                u0 += direction * 4 * np.arctan(np.exp((r - r0) / width))
                v0 += direction * np.exp(-(r - r0)**2 / (2 * width**2))

        return u0, v0

    def multi_ring_state(self, system_type: str = 'sine_gordon', n_rings: int = 3,
                         radius_range: Tuple[float, float] = (1.0, 5.0),
                         width_range: Tuple[float, float] = (0.3, 0.8),
                         position_variance: float = 0.5, arrangement: str = 'concentric',
                         separation: float = 2.0, interaction_strength: float = 0.7,
                         modulation_strength: float = 0.2,
                         modulation_mode_range: Tuple[int, int] = (1, 4)) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)

        if arrangement not in ['cocentric', 'circular']:
            positions = []
            for _ in range(n_rings):
                x0 = np.random.normal(0.0, position_variance * self.L / 4)
                y0 = np.random.normal(0.0, position_variance * self.L / 4)
                positions.append((x0, y0))
        elif arrangement == 'concentric':
            positions = [(0, 0)] * n_rings

        elif arrangement == 'circular':
            positions = []
            for i in range(n_rings):
                angle = 2 * np.pi * i / n_rings
                x0 = separation * np.cos(angle)
                y0 = separation * np.sin(angle)
                positions.append((x0, y0))

        for i, (x0, y0) in enumerate(positions):
            if arrangement == 'concentric':
                radius = radius_range[0] + (radius_range[1] - radius_range[0]) * i / (
                    n_rings - 1) if n_rings > 1 else radius_range[0]
            else:
                radius = radius_range[0] + \
                    (radius_range[1] - radius_range[0]) * np.random.rand()

            width = width_range[0] + (width_range[1] -
                                      width_range[0]) * np.random.rand()
            velocity = np.random.rand() * 0.4 - 0.2
            ring_type = 'expanding' if np.random.rand() > 0.5 else 'kink_antikink'

            if modulation_strength > 0:
                mod_mode = np.random.randint(
                    modulation_mode_range[0], modulation_mode_range[1] + 1)
            else:
                mod_mode = 0

            u_comp, v_comp = self.ring_soliton(
                system_type=system_type,
                amplitude=1.0,
                radius=radius,
                width=width,
                position=(x0, y0),
                velocity=velocity,
                ring_type=ring_type,
                modulation_strength=modulation_strength,
                modulation_mode=mod_mode
            )

            if i == 0:
                u = u_comp
                v = v_comp
            else:
                u = u + interaction_strength * u_comp
                v = v + interaction_strength * v_comp

        return u, v

    def skyrmion_solution(self, system_type: str = 'sine_gordon', amplitude: float = 1.0,
                          radius: float = 1.0, position: Tuple[float, float] = (0.0, 0.0),
                          charge: int = 1, profile: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta_local = np.arctan2(self.Y - y0, self.X - x0)

        if profile == 'standard':
            phi = 2 * np.arctan(r_local / radius)
        elif profile == 'compact':
            phi = np.pi * (1 - np.exp(-(r_local / radius)**2))
        elif profile == 'exponential':
            phi = np.pi * (1 - np.exp(-r_local / radius))
        else:
            phi = 2 * np.arctan(r_local / radius)

        u = amplitude * np.sin(phi) * np.cos(charge * theta_local)
        v = amplitude * np.sin(phi) * np.sin(charge * theta_local)

        return u, v

    def skyrmion_lattice(self, system_type: str = 'sine_gordon', n_skyrmions: int = 5,
                         radius_range: Tuple[float, float] = (0.5, 1.5), amplitude: float = 1.0,
                         arrangement: str = 'triangular', separation: float = 3.0,
                         charge_distribution: str = 'alternating') -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)

        positions = []
        if arrangement == 'triangular':
            rows = int(np.ceil(np.sqrt(n_skyrmions * 2 / np.sqrt(3))))
            for i in range(rows):
                offset = (i % 2) * 0.5 * separation
                for j in range(int(np.ceil(n_skyrmions / rows))):
                    if len(positions) < n_skyrmions:
                        x = (j - int(np.ceil(n_skyrmions / rows) - 1) /
                             2) * separation + offset
                        y = (i - (rows - 1) / 2) * separation * np.sqrt(3) / 2
                        positions.append((x, y))
        elif arrangement == 'square':
            side = int(np.ceil(np.sqrt(n_skyrmions)))
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_skyrmions:
                        x = (i - (side - 1) / 2) * separation
                        y = (j - (side - 1) / 2) * separation
                        positions.append((x, y))
        else:  # random arrangement
            for _ in range(n_skyrmions):
                x = np.random.rand() * 2 * self.L - self.L
                y = np.random.rand() * 2 * self.L - self.L
                positions.append((x, y))
        charges = []
        if charge_distribution == 'alternating':
            charges = [(-1)**i for i in range(n_skyrmions)]
        elif charge_distribution == 'same':
            charges = [1] * n_skyrmions
        else:  # random charges
            charges = [
                1 if np.random.rand() > 0.5 else -
                1 for _ in range(n_skyrmions)]

        for i, ((x0, y0), charge) in enumerate(zip(positions, charges)):
            radius = radius_range[0] + \
                (radius_range[1] - radius_range[0]) * np.random.rand()
            profile = ['standard', 'compact',
                       'exponential'][np.random.randint(0, 3)]

            u_comp, v_comp = self.skyrmion_solution(
                system_type=system_type,
                amplitude=amplitude,
                radius=radius,
                position=(x0, y0),
                charge=charge,
                profile=profile
            )

            u += u_comp
            v += v_comp

        return u, v

    def skyrmion_like_field(
            self, num_skyrmions: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        phi = np.zeros_like(self.X)

        for _ in range(num_skyrmions):
            x0, y0 = self.L * (2 * np.random.rand(2) - 1)
            lambda_size = 0.2 * self.L + 0.4 * self.L * np.random.rand()
            q = np.random.choice([-1, 1])
            alpha = 2 * np.pi * np.random.rand()
            z = (self.X - x0) + 1j * (self.Y - y0)

            if q > 0:
                w = z / (lambda_size + np.abs(z))
            else:
                w = z.conjugate() / (lambda_size + np.abs(z))

            angle = np.angle(w * np.exp(1j * alpha))

            r = np.abs(z)
            profile = 2 * np.arctan2(lambda_size, r)

            skyrmion_contribution = 2 * profile * angle / np.pi

            cutoff = np.exp(-(r / (0.8 * self.L))**4)
            phi += cutoff * skyrmion_contribution

        return phi, 0.05 * self.anisotropic_grf(length_scale=self.L)

    def q_ball_solution(self, system_type: str = 'sine-gordon', amplitude: float = 1.0,
                        radius: float = 1.0, position: Tuple[float, float] = (0.0, 0.0),
                        phase: float = 0.0, frequency: float = 0.8, charge: int = 1,
                        time_param: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = position
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        omega = frequency * np.sign(charge)  # incorporate m in this term?
        profile = amplitude / np.cosh(r / (radius / np.sqrt(2)))
        time_phase = omega * time_param + phase
        u = profile * np.cos(time_phase)
        v = -profile * omega * np.sin(time_phase)
        return u, v

    def multi_q_ball(self, system_type: str = 'sine-gordon', n_qballs: int = 3,
                     amplitude_range: Tuple = (0.2, 1.0), radius_range: Tuple = (0.5, 2.0),
                     frequency_range: Tuple = (0.4, 0.9), position_variance: float = 0.3,
                     time_param: float = 0.0):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)

        for _ in range(n_qballs):
            amplitude = np.random.uniform(*amplitude_range)
            radius = np.random.uniform(*radius_range)
            frequency = np.random.uniform(*frequency_range)
            phase = 2 * np.pi * np.random.rand()
            charge = 1 if np.random.rand() > 0.5 else -1

            x0 = np.random.normal(0.0, position_variance * self.L / 4)
            y0 = np.random.normal(0.0, position_variance * self.L / 4)

            u_q, v_q = self.q_ball_solution(
                system_type=system_type,
                amplitude=amplitude,
                radius=radius,
                position=(x0, y0),
                phase=phase,
                frequency=frequency,
                charge=charge,
                time_param=time_param
            )

            u += u_q
            v += v_q

        return u, v

    def breather_solution(self, system_type: str = 'sine_gordon', amplitude: float = 0.5,
                          frequency: float = 0.9, width: float = 1.0,
                          position: Tuple[float, float] = (0.0, 0.0), phase: float = 0.0,
                          orientation: float = 0.0, breather_type: str = 'standard',
                          time_param: float = 0.0, velocity_type: str = 'fitting') -> Tuple[np.ndarray, np.ndarray]:
        x0, y0 = position

        X_rot = (self.X - x0) * np.cos(orientation) + \
            (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + \
            (self.Y - y0) * np.cos(orientation)

        if amplitude >= 1.0 and system_type == 'sine_gordon':
            amplitude = 0.999

        if system_type == 'sine_gordon':
            omega = np.sqrt(1 - amplitude**2)

            if breather_type != 'radial':
                xi = X_rot / width
                tau = time_param

                u = 4 * np.arctan(amplitude * np.sin(omega * tau + phase) /
                                  (omega * np.cosh(amplitude * xi)))

                if velocity_type == 'fitting':
                    v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                        omega * np.cosh(amplitude * xi) *
                        (1 + (amplitude**2 / omega**2) *
                         np.sin(omega * tau + phase)**2)
                    )
                else:
                    v = np.zeros_like(u)

            else:
                r_local = np.sqrt(X_rot**2 + Y_rot**2)
                xi = r_local / width
                tau = time_param

                u = 4 * np.arctan(amplitude * np.sin(omega * tau + phase) /
                                  (omega * np.cosh(amplitude * xi)))

                if velocity_type == 'fitting':
                    v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                        omega * np.cosh(amplitude * xi) *
                        (1 + (amplitude**2 / omega**2) *
                         np.sin(omega * tau + phase)**2)
                    )
                else:
                    v = np.zeros_like(u)

        elif system_type == 'phi4':
            xi = X_rot / width
            tau = time_param
            epsilon = amplitude

            if breather_type != 'radial':
                u = amplitude * np.sqrt(2) * np.tanh(xi) / \
                    np.cosh(epsilon * tau)
                if velocity_type == 'fitting':
                    v = amplitude * np.sqrt(2) * epsilon * np.tanh(xi) * \
                        np.sinh(epsilon * tau) / np.cosh(epsilon * tau)**2
                else:
                    v = np.zeros_like(u)

            else:
                r_local = np.sqrt(X_rot**2 + Y_rot**2)
                xi = r_local / width
                u = amplitude * np.sqrt(2) * np.tanh(xi) / \
                    np.cosh(epsilon * tau)
                if velocity_type == 'fitting':
                    v = amplitude * np.sqrt(2) * epsilon * np.tanh(xi) * \
                        np.sinh(epsilon * tau) / np.cosh(epsilon * tau)**2
                else:
                    v = np.zeros_like(u)

        elif system_type == 'double_sine_gordon':
            lambda_param = 0.3
            omega = np.sqrt(1 - amplitude**2)

            # "only" 'standard'
            xi = X_rot / width
            tau = time_param

            u = 4 * np.arctan(amplitude * np.sin(omega * tau + phase) /
                              (omega * np.cosh(amplitude * xi)))

            if velocity_type == 'fitting':
                v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                    omega * np.cosh(amplitude * xi) *
                    (1 + (amplitude**2 / omega**2) *
                     np.sin(omega * tau + phase)**2)
                )
            else:
                v = np.zeros_like(u)

        elif system_type == 'hyperbolic_sine_gordon' or system_type == 'klein_gordon':
            xi = X_rot / width
            tau = time_param

            u = amplitude * np.exp(-(xi**2) / 2) * \
                np.cos(frequency * tau + phase)
            if velocity_type == 'fitting':
                v = -amplitude * frequency * \
                    np.exp(-(xi**2) / 2) * np.sin(frequency * tau + phase)
            else:
                v = np.zeros_like(u)

        else:
            omega = np.sqrt(1 - amplitude**2)
            xi = X_rot / width
            tau = time_param

            u = 4 * np.arctan(amplitude * np.sin(omega * tau + phase) /
                              (omega * np.cosh(amplitude * xi)))

            if velocity_type == 'fitting':
                v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                    omega * np.cosh(amplitude * xi) *
                    (1 + (amplitude**2 / omega**2) *
                     np.sin(omega * tau + phase)**2)
                )
            else:
                v = np.zeros_like(u)

        if velocity_type == 'grf':
            v = self.anisotropic_grf(
                length_scale=width * 2.0,
                amplitude=np.max(
                    np.abs(u)) * 0.2)

        return u, v

    def breather_field(self, system_type: str = 'sine_gordon', num_breathers: int = 1,
                       position_type: str = 'random', time_param: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        u0 = np.zeros_like(self.X)
        v0 = np.zeros_like(self.X)

        positions = []
        if position_type not in ['circle', 'line']:
            for _ in range(num_breathers):
                x0 = self.L * (2 * np.random.rand() - 1)
                y0 = self.L * (2 * np.random.rand() - 1)
                positions.append((x0, y0))
        elif position_type == 'circle':
            radius = 0.6 * self.L * np.random.rand()
            for i in range(num_breathers):
                angle = 2 * np.pi * i / num_breathers
                x0 = radius * np.cos(angle)
                y0 = radius * np.sin(angle)
                positions.append((x0, y0))
        elif position_type == 'line':
            for i in range(num_breathers):
                pos = -self.L + 2 * self.L * i / \
                    (num_breathers - 1 if num_breathers > 1 else 1)
                if np.random.rand() > 0.5:
                    positions.append((pos, 0.0))
                else:
                    positions.append((0.0, pos))

        for x0, y0 in positions:
            width = 0.5 + 2.5 * np.random.rand()
            amplitude = 0.1 + 0.8 * np.random.rand()
            phase = 2 * np.pi * np.random.rand()

            omega = np.sqrt(1.0 - amplitude**2)
            t = time_param

            direction = np.random.rand()
            if direction < 0.33:
                xi = (self.X - x0) / width
                u_comp = 4 * np.arctan(amplitude * np.sin(omega * t + phase) /
                                       (omega * np.cosh(amplitude * xi)))
                v_comp = 4 * amplitude * omega * np.cos(omega * t + phase) / (
                    omega * np.cosh(amplitude * xi) *
                    (1 + (amplitude**2 / omega**2) * np.sin(omega * t + phase)**2)
                )
            elif direction < 0.66:
                yi = (self.Y - y0) / width
                u_comp = 4 * np.arctan(amplitude * np.sin(omega * t + phase) /
                                       (omega * np.cosh(amplitude * yi)))
                v_comp = 4 * amplitude * omega * np.cos(omega * t + phase) / (
                    omega * np.cosh(amplitude * yi) *
                    (1 + (amplitude**2 / omega**2) * np.sin(omega * t + phase)**2)
                )
            else:
                ri = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2) / width
                u_comp = 4 * np.arctan(amplitude * np.sin(omega * t + phase) /
                                       (omega * np.cosh(amplitude * ri)))
                v_comp = 4 * amplitude * omega * np.cos(omega * t + phase) / (
                    omega * np.cosh(amplitude * ri) *
                    (1 + (amplitude**2 / omega**2) * np.sin(omega * t + phase)**2)
                )

            u0 += u_comp
            v0 += v_comp

        return u0, v0

    def multi_q_ball(self, system_type: str = 'phi4', n_qballs: int = 3,
                     amplitude_range: Tuple[float, float] = (0.5, 1.5),
                     radius_range: Tuple[float, float] = (0.5, 2.0),
                     frequency_range: Tuple[float, float] = (0.6, 0.9),
                     position_variance: float = 1.0,
                     interaction_strength: float = 0.7,
                     time_param: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)

        for i in range(n_qballs):
            amplitude = amplitude_range[0] + \
                (amplitude_range[1] - amplitude_range[0]) * np.random.rand()
            radius = radius_range[0] + \
                (radius_range[1] - radius_range[0]) * np.random.rand()
            frequency = frequency_range[0] + \
                (frequency_range[1] - frequency_range[0]) * np.random.rand()
            phase = 2 * np.pi * np.random.rand()
            charge = 1 if np.random.rand() > 0.5 else -1

            x0 = np.random.normal(0.0, position_variance * self.L / 4)
            y0 = np.random.normal(0.0, position_variance * self.L / 4)

            u_q, v_q = self.q_ball_solution(
                system_type=system_type,
                amplitude=amplitude,
                radius=radius,
                position=(x0, y0),
                phase=phase,
                frequency=frequency,
                charge=charge,
                time_param=time_param
            )

            if i == 0:
                u = u_q
                v = v_q
            else:
                u = u + interaction_strength * u_q
                v = v + interaction_strength * v_q

        return u, v

    def soliton_antisoliton_pair(self, system_type: str = 'sine_gordon',
                                 pattern_type: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        if pattern_type == 'auto':
            pattern_type = np.random.choice(
                ['radial', 'linear', 'angular', 'nested'])

        width = 0.8 + 2.2 * np.random.rand()
        x0, y0 = self.L * (2 * np.random.rand(2) - 1)

        if pattern_type == 'radial':
            r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            u0 = 4 * np.arctan(np.exp(r / width)) - 4 * \
                np.arctan(np.exp((r - 0.5 * width) / width))
        elif pattern_type == 'linear':
            theta = np.pi * np.random.rand()
            x_rot = (self.X - x0) * np.cos(theta) + \
                (self.Y - y0) * np.sin(theta)
            u0 = 4 * np.arctan(np.exp(x_rot / width)) - 4 * \
                np.arctan(np.exp(-x_rot / width))
        elif pattern_type == 'angular':
            phi = np.arctan2(self.Y - y0, self.X - x0)
            u0 = 4 * np.arctan(np.exp(np.sin(phi) / width)) - \
                4 * np.arctan(np.exp(-np.sin(phi) / width))
        else:
            r1 = 0.3 * self.L + 0.1 * self.L * np.random.rand()
            r2 = 0.6 * self.L + 0.1 * self.L * np.random.rand()
            r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            u0 = 4 * np.arctan(np.exp((r - r1) / width)) - \
                4 * np.arctan(np.exp((r - r2) / width))

        v0 = self.anisotropic_grf(
            length_scale=width,
            anisotropy_ratio=2.0) * 0.2

        return u0, v0

    def elliptical_soliton(self, system_type: str = 'sine_gordon',
                           complexity: str = 'simple') -> Tuple[np.ndarray, np.ndarray]:
        if complexity != 'complex':
            x0, y0 = (self.L / 2) * (2 * np.random.rand(2) - 1)
            a = (0.1 * self.L + 0.2 * self.L * np.random.rand())
            b = a * (0.2 + 0.8 * np.random.rand())

            theta = np.pi * np.random.rand()
            phase = 2 * np.pi * np.random.rand()

            X_rot = (self.X - x0) * np.cos(theta) + \
                (self.Y - y0) * np.sin(theta)
            Y_rot = -(self.X - x0) * np.sin(theta) + \
                (self.Y - y0) * np.cos(theta)

            r_ellipse = np.sqrt((X_rot / a)**2 + (Y_rot / b)**2)

            omega = (1.0 - 0.3**2)**.5
            u0 = 4 * np.arctan(0.3 * np.sin(phase) /
                               (omega * np.cosh(0.3 * r_ellipse)))

            v0 = 4 * 0.3 * omega * np.cos(phase + omega * 0.0) / (
                omega * np.cosh(0.3 * r_ellipse) *
                (1 + (0.3**2 / omega**2) * np.sin(phase + omega * 0.0)**2)
            )
        else:
            u0 = np.zeros_like(self.X)
            v0 = np.zeros_like(self.X)

            num_features = np.random.randint(2, 5)

            for _ in range(num_features):
                x0, y0 = (self.L / 2) * (2 * np.random.rand(2) - 1)
                a = (0.1 * self.L + 0.2 * self.L * np.random.rand())
                b = a * (0.2 + 0.8 * np.random.rand())

                theta = np.pi * np.random.rand()
                phase = 2 * np.pi * np.random.rand()
                X_rot = (self.X - x0) * np.cos(theta) + \
                    (self.Y - y0) * np.sin(theta)
                Y_rot = -(self.X - x0) * np.sin(theta) + \
                    (self.Y - y0) * np.cos(theta)

                r_ellipse = np.sqrt((X_rot / a)**2 + (Y_rot / b)**2)

                amplitude = 0.2 + 0.3 * np.random.rand()
                omega = (1.0 - amplitude**2)**.5

                u0 += 4 * np.arctan(amplitude * np.sin(phase) /
                                    (omega * np.cosh(amplitude * r_ellipse)))

                v0 += 4 * amplitude * omega * np.cos(phase) / (
                    omega * np.cosh(amplitude * r_ellipse) *
                    (1 + (amplitude**2 / omega**2) * np.sin(phase)**2)
                )

        return u0, v0

    def grf_modulated_soliton_field(self, system_type: str = 'sine_gordon',
                                    grf_length_scale: float = 1.0,
                                    smoothness_scaling: float = 2.0,
                                    anisotropy_ratio: float = 1.0,
                                    anisotropy_angle: float = 0.0,
                                    construction_method: str = 'threshold',
                                    mixture_type: str = 'additive',
                                    velocity_mode: str = 'fitting',
                                    threshold_values: Optional[List[float]] = None,
                                    soliton_types: Optional[List[str]] = None,
                                    level_set_width: float = 0.2,
                                    continuous_range: Optional[Dict[str,
                                                                    Tuple[float, float]]] = None,
                                    random_velocity_scale: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)

        base_grf = self.anisotropic_grf(
            length_scale=grf_length_scale,
            anisotropy_ratio=anisotropy_ratio,
            theta=anisotropy_angle,
            amplitude=1.0
        )

        scaled_grf = smoothness_scaling * base_grf

        if construction_method == 'threshold':
            if threshold_values is None:
                threshold_values = [-1.0, 0.0, 1.0]

            if soliton_types is None:
                soliton_types = ['kink', 'breather', 'antikink']

            for i in range(len(threshold_values) - 1):
                lower_bound = threshold_values[i]
                upper_bound = threshold_values[i + 1]
                soliton_type = soliton_types[i % len(soliton_types)]

                mask = (scaled_grf >= lower_bound) & (scaled_grf < upper_bound)

                if soliton_type == 'kink':
                    width = 0.5 + 1.0 * np.random.rand()
                    soliton_u = 4 * np.arctan(np.exp(scaled_grf[mask] / width))

                    if velocity_mode == 'fitting':
                        soliton_v = 4 / \
                            (width * (np.cosh(scaled_grf[mask] / width)**2))
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(np.sum(mask)) - 1)
                    else:
                        soliton_v = np.zeros(np.sum(mask))

                elif soliton_type == 'antikink':
                    width = 0.5 + 1.0 * np.random.rand()
                    soliton_u = -4 * \
                        np.arctan(np.exp(scaled_grf[mask] / width))

                    if velocity_mode == 'fitting':
                        soliton_v = -4 / \
                            (width * (np.cosh(scaled_grf[mask] / width)**2))
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(np.sum(mask)) - 1)
                    else:
                        soliton_v = np.zeros(np.sum(mask))

                elif soliton_type == 'breather':
                    width = 0.5 + 1.0 * np.random.rand()
                    amplitude = 0.2 + 0.7 * np.random.rand()
                    frequency = np.sqrt(1 - amplitude**2)
                    phase = 2 * np.pi * np.random.rand()

                    soliton_u = 4 * np.arctan(amplitude * np.sin(phase) /
                                              (frequency * np.cosh(amplitude * scaled_grf[mask] / width)))

                    if velocity_mode == 'fitting':
                        soliton_v = 4 * amplitude * frequency * np.cos(phase) / (
                            frequency * np.cosh(amplitude * scaled_grf[mask] / width) *
                            (1 + (amplitude**2 / frequency**2) * np.sin(phase)**2)
                        )
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(np.sum(mask)) - 1)
                    else:
                        soliton_v = np.zeros(np.sum(mask))

                elif soliton_type == 'ring':
                    radius = 0.2 + 0.3 * np.random.rand()
                    width = 0.3 + 0.5 * np.random.rand()

                    r_term = np.abs(scaled_grf[mask]) - radius
                    soliton_u = 4 * np.arctan(np.exp(r_term / width))

                    if velocity_mode == 'fitting':
                        velocity = 0.2 * np.random.rand()
                        soliton_v = -velocity * 4 / \
                            (width * (np.cosh(r_term / width)**2))
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(np.sum(mask)) - 1)
                    else:
                        soliton_v = np.zeros(np.sum(mask))

                u[mask] = soliton_u
                v[mask] = soliton_v

        elif construction_method == 'level_set':
            if threshold_values is None:
                threshold_values = [-1.5, -0.5, 0.5, 1.5]

            if soliton_types is None:
                soliton_types = ['kink', 'breather', 'antikink', 'ring']

            for i, threshold in enumerate(threshold_values):
                soliton_type = soliton_types[i % len(soliton_types)]

                level_set_weight = np.exp(-(scaled_grf - threshold)
                                          ** 2 / (2 * level_set_width**2))

                if soliton_type == 'kink':
                    width = 0.5 + 1.0 * np.random.rand()
                    orientation = np.pi * np.random.rand()
                    x_rot = self.X * np.cos(orientation) + \
                        self.Y * np.sin(orientation)

                    soliton_u = 4 * np.arctan(np.exp(x_rot / width))

                    if velocity_mode == 'fitting':
                        soliton_v = 4 / (width * (np.cosh(x_rot / width)**2))
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(*self.X.shape) - 1)
                    else:
                        soliton_v = np.zeros_like(self.X)

                elif soliton_type == 'antikink':
                    width = 0.5 + 1.0 * np.random.rand()
                    orientation = np.pi * np.random.rand()
                    x_rot = self.X * np.cos(orientation) + \
                        self.Y * np.sin(orientation)

                    soliton_u = -4 * np.arctan(np.exp(x_rot / width))

                    if velocity_mode == 'fitting':
                        soliton_v = -4 / (width * (np.cosh(x_rot / width)**2))
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(*self.X.shape) - 1)
                    else:
                        soliton_v = np.zeros_like(self.X)

                elif soliton_type == 'breather':
                    width = 0.5 + 1.0 * np.random.rand()
                    amplitude = 0.2 + 0.7 * np.random.rand()
                    frequency = np.sqrt(1 - amplitude**2)
                    phase = 2 * np.pi * np.random.rand()

                    r = np.sqrt(self.X**2 + self.Y**2) / width

                    soliton_u = 4 * np.arctan(amplitude * np.sin(phase) /
                                              (frequency * np.cosh(amplitude * r)))

                    if velocity_mode == 'fitting':
                        soliton_v = 4 * amplitude * frequency * np.cos(phase) / (
                            frequency * np.cosh(amplitude * r) *
                            (1 + (amplitude**2 / frequency**2) * np.sin(phase)**2)
                        )
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(*self.X.shape) - 1)
                    else:
                        soliton_v = np.zeros_like(self.X)

                elif soliton_type == 'ring':
                    radius = 1.0 + 1.0 * np.random.rand()
                    width = 0.3 + 0.5 * np.random.rand()

                    r = np.sqrt(self.X**2 + self.Y**2)
                    r_term = r - radius

                    soliton_u = 4 * np.arctan(np.exp(r_term / width))

                    if velocity_mode == 'fitting':
                        velocity = 0.2 * np.random.rand()
                        soliton_v = -velocity * 4 / \
                            (width * (np.cosh(r_term / width)**2))
                    elif velocity_mode == 'random':
                        soliton_v = random_velocity_scale * \
                            (2 * np.random.rand(*self.X.shape) - 1)
                    else:
                        soliton_v = np.zeros_like(self.X)

                if mixture_type == 'additive':
                    u += level_set_weight * soliton_u
                    v += level_set_weight * soliton_v
                elif mixture_type == 'maximum':
                    u = np.maximum(u, level_set_weight * soliton_u)
                    v = np.where(
                        u == level_set_weight *
                        soliton_u,
                        level_set_weight *
                        soliton_v,
                        v)
                else:
                    if i == 0:
                        u = level_set_weight * soliton_u
                        v = level_set_weight * soliton_v
                    else:
                        u = u * (1 - level_set_weight) + \
                            level_set_weight * soliton_u
                        v = v * (1 - level_set_weight) + \
                            level_set_weight * soliton_v

        elif construction_method == 'continuous':
            if continuous_range is None:
                continuous_range = {
                    'amplitude': (0.2, 0.8),
                    'width': (0.5, 2.0),
                    'orientation': (0, np.pi)
                }

            amp_min, amp_max = continuous_range.get('amplitude', (0.2, 0.8))
            width_min, width_max = continuous_range.get('width', (0.5, 2.0))

            normalized_grf = (scaled_grf - np.min(scaled_grf)) / \
                (np.max(scaled_grf) - np.min(scaled_grf))

            amplitude = amp_min + (amp_max - amp_min) * normalized_grf
            width = width_min + (width_max - width_min) * normalized_grf

            if system_type == 'sine_gordon':
                u = 4 * np.arctan(np.exp(scaled_grf / width))

                if velocity_mode == 'fitting':
                    v = 4 / (width * (np.cosh(scaled_grf / width)**2))
                elif velocity_mode == 'random':
                    v = random_velocity_scale * \
                        (2 * np.random.rand(*self.X.shape) - 1)
                else:
                    v = np.zeros_like(self.X)

            elif system_type == 'phi4':
                u = amplitude * np.tanh(scaled_grf / width)

                if velocity_mode == 'fitting':
                    v = amplitude / (width * (np.cosh(scaled_grf / width)**2))
                elif velocity_mode == 'random':
                    v = random_velocity_scale * \
                        (2 * np.random.rand(*self.X.shape) - 1)
                else:
                    v = np.zeros_like(self.X)

            elif system_type == 'double_sine_gordon':
                lambda_param = 0.3
                k = 1 / np.sqrt(1 + lambda_param)

                u = 4 * np.arctan(np.sqrt((1 + lambda_param) / lambda_param) *
                                  np.tanh(np.sqrt(lambda_param) * scaled_grf / (2 * width)))

                if velocity_mode == 'fitting':
                    v = 4 * np.sqrt((1 + lambda_param) / lambda_param) * np.sqrt(lambda_param) / (2 * width) * (
                        1 - np.tanh(np.sqrt(lambda_param) * scaled_grf / (2 * width))**2)
                elif velocity_mode == 'random':
                    v = random_velocity_scale * \
                        (2 * np.random.rand(*self.X.shape) - 1)
                else:
                    v = np.zeros_like(self.X)

            else:
                u = 4 * np.arctan(np.exp(scaled_grf / width))

                if velocity_mode == 'fitting':
                    v = 4 / (width * (np.cosh(scaled_grf / width)**2))
                elif velocity_mode == 'random':
                    v = random_velocity_scale * \
                        (2 * np.random.rand(*self.X.shape) - 1)
                else:
                    v = np.zeros_like(self.X)

        return u, v

    def generate_sample(self, system_type: str = 'sine_gordon', phenomenon_type: str = 'kink_solution',
                        time_param: float = 0.0, velocity_type: str = "fitting",
                        **params) -> Tuple[np.ndarray, np.ndarray]:
        if phenomenon_type == 'kink_solution':
            return self.kink_solution(
                system_type, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'kink_field':
            return self.kink_field(system_type, **params)
        elif phenomenon_type == 'kink_array_field':
            return self.kink_array_field(system_type, **params)
        elif phenomenon_type == 'breather_solution':
            return self.breather_solution(
                system_type, time_param=time_param, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'breather_field':
            return self.breather_field(
                system_type, time_param=time_param, **params)
        elif phenomenon_type == 'multi_breather_field':
            return self.multi_breather_field(
                system_type, time_param=time_param, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'ring_soliton':
            return self.ring_soliton(
                system_type, time_param=time_param, **params)
        elif phenomenon_type == 'colliding_rings':
            return self.colliding_rings(system_type, **params)
        elif phenomenon_type == 'multi_ring_state':
            return self.multi_ring_state(system_type, **params)
        elif phenomenon_type == 'skyrmion_solution':
            return self.skyrmion_solution(system_type, **params)
        elif phenomenon_type == 'skyrmion_lattice':
            return self.skyrmion_lattice(system_type, **params)
        elif phenomenon_type == 'skyrmion_like_field':
            return self.skyrmion_like_field(**params)
        elif phenomenon_type == 'spiral_wave_field':
            return self.spiral_wave_field(**params)
        elif phenomenon_type == 'multi_spiral_state':
            return self.multi_spiral_state(**params)
        elif phenomenon_type == 'q_ball_solution':
            return self.q_ball_solution(
                system_type, time_param=time_param, **params)
        elif phenomenon_type == 'multi_q_ball':
            return self.multi_q_ball(
                system_type, time_param=time_param, **params)
        elif phenomenon_type == 'soliton_antisoliton_pair':
            return self.soliton_antisoliton_pair(system_type, **params)
        elif phenomenon_type == 'elliptical_soliton':
            return self.elliptical_soliton(system_type, **params)
        elif phenomenon_type == 'grf_modulated_soliton_field':
            return self.grf_modulated_soliton_field(system_type, **params)
        else:
            raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")

    def generate_ensemble(self, system_type: str = 'sine_gordon', phenomenon_type: str = 'kink_solution',
                          n_samples: int = 10, parameter_ranges: Optional[Dict[str, Any]] = None,
                          **fixed_params) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples = []

        if parameter_ranges is None:
            parameter_ranges = {}

        for _ in range(n_samples):
            params = fixed_params.copy()

            for param, range_values in parameter_ranges.items():
                if isinstance(range_values, list):
                    params[param] = np.random.choice(range_values)
                elif isinstance(range_values, tuple) and len(range_values) == 2:
                    if isinstance(range_values[0], int) and isinstance(
                            range_values[1], int):
                        params[param] = np.random.randint(
                            range_values[0], range_values[1] + 1)
                    else:
                        params[param] = range_values[0] + \
                            (range_values[1] - range_values[0]) * \
                            np.random.rand()
                else:
                    raise ValueError(
                        f"Invalid parameter range: {range_values}")

            sample = self.generate_sample(
                system_type, phenomenon_type, **params)
            samples.append(sample)

        if n_samples == 1:
            return samples[0]
        else:
            return samples

    def generate_diverse_ensemble(self, system_type: str = 'sine_gordon', phenomenon_type: str = 'kink_solution',
                                  n_samples: int = 10, parameter_ranges: Optional[Dict[str, Any]] = None,
                                  similarity_threshold: float = 0.2, max_attempts: int = 100,
                                  diversity_metric: str = 'l2', **fixed_params) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples = []
        attempts = 0

        if parameter_ranges is None:
            parameter_ranges = {}

        def diversity_distance(s1, s2):
            s1_u, s1_v = s1
            s2_u, s2_v = s2

            if diversity_metric == 'l2':
                s1_norm = np.sqrt(np.sum(s1_u**2 + s1_v**2))
                s2_norm = np.sqrt(np.sum(s2_u**2 + s2_v**2))

                if s1_norm == 0 or s2_norm == 0:
                    return 1.0

                s1_u_norm = s1_u / s1_norm
                s1_v_norm = s1_v / s1_norm
                s2_u_norm = s2_u / s2_norm
                s2_v_norm = s2_v / s2_norm

                u_dist = np.sqrt(np.sum((s1_u_norm - s2_u_norm)**2))
                v_dist = np.sqrt(np.sum((s1_v_norm - s2_v_norm)**2))

                return (u_dist + v_dist) / 2

            elif diversity_metric == 'spectral':
                s1_u_fft = np.fft.fftshift(np.abs(np.fft.fft2(s1_u)))
                s1_v_fft = np.fft.fftshift(np.abs(np.fft.fft2(s1_v)))
                s2_u_fft = np.fft.fftshift(np.abs(np.fft.fft2(s2_u)))
                s2_v_fft = np.fft.fftshift(np.abs(np.fft.fft2(s2_v)))

                s1_u_fft_norm = np.sqrt(np.sum(s1_u_fft**2))
                s1_v_fft_norm = np.sqrt(np.sum(s1_v_fft**2))
                s2_u_fft_norm = np.sqrt(np.sum(s2_u_fft**2))
                s2_v_fft_norm = np.sqrt(np.sum(s2_v_fft**2))

                if s1_u_fft_norm == 0 or s2_u_fft_norm == 0 or s1_v_fft_norm == 0 or s2_v_fft_norm == 0:
                    return 1.0

                u_overlap = np.sum(s1_u_fft * s2_u_fft) / \
                    (s1_u_fft_norm * s2_u_fft_norm)
                v_overlap = np.sum(s1_v_fft * s2_v_fft) / \
                    (s1_v_fft_norm * s2_v_fft_norm)

                return 1.0 - (u_overlap + v_overlap) / 2
            else:
                return 0.0

        while len(samples) < n_samples and attempts < max_attempts:
            params = fixed_params.copy()

            for param, range_values in parameter_ranges.items():
                if isinstance(range_values, list):
                    params[param] = np.random.choice(range_values)
                elif isinstance(range_values, tuple) and len(range_values) == 2:
                    if isinstance(range_values[0], int) and isinstance(
                            range_values[1], int):
                        params[param] = np.random.randint(
                            range_values[0], range_values[1] + 1)
                    else:
                        params[param] = range_values[0] + \
                            (range_values[1] - range_values[0]) * \
                            np.random.rand()
                else:
                    raise ValueError(
                        f"Invalid parameter range: {range_values}")

            sample = self.generate_sample(
                system_type, phenomenon_type, **params)

            if len(samples) == 0:
                samples.append(sample)
            else:
                is_diverse = True
                for existing_sample in samples:
                    dist = diversity_distance(sample, existing_sample)
                    if dist < similarity_threshold:
                        is_diverse = False
                        break

                if is_diverse:
                    samples.append(sample)

            attempts += 1

        return samples

    def generate_initial_condition(self, system_type: str = 'sine_gordon',
                                   phenomenon_type: List[str] = None,
                                   velocity_type: str = 'fitting', **params) -> Tuple[np.ndarray, np.ndarray]:
        if phenomenon_type is None:
            raise Exception
        u0, v0 = self.generate_sample(system_type=system_type,
                                      phenomenon_type=phenomenon_type,
                                      velocity_type=velocity_type,
                                      **params)
        return (u0, v0)


def tsne_on_samples(samples, perplexity=30, n_iter=1000):
    import numpy as np
    from sklearn.manifold import TSNE
    features = []
    for sample in samples:
        feature_vector = sample.flatten()
        features.append(feature_vector)
    features_array = np.array(features)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42)
    embedding = tsne.fit_transform(features_array)
    return embedding

class RealWaveSampler3d:
    def __init__(self, nx: int, ny: int, nz: int, L: float):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.L = L

        self._setup_grid()

    def _setup_grid(self):
        self.x = np.linspace(-self.L, self.L, self.nx)
        self.y = np.linspace(-self.L, self.L, self.ny)
        self.z = np.linspace(-self.L, self.L, self.nz)

        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,)
        self.r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.theta = np.arctan2(self.Y, self.X)

        self.dx = 2 * self.L / (self.nx - 1) # reminder: ghost cells
        self.dy = 2 * self.L / (self.ny - 1)
        self.dy = 2 * self.L / (self.nz - 1)

        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=2 * self.L / self.nx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=2 * self.L / self.ny)
        self.kz = 2 * np.pi * np.fft.fftfreq(self.nz, d=2 * self.L / self.nz)
        
        self.KX, self.KY, self.KZ = np.meshgrid(self.kx, self.ky, self.kz)
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2 + self.KZ**2)

    def nonlinear_potential(self, u: np.ndarray,
                            system_type: str = 'sine_gordon') -> np.ndarray:
        if system_type == 'klein_gordon':
            return u
        else:
            raise NotImplemented

    def anisotropic_grf(self, length_scale: float = 1.0,
                      anisotropy_xy: float = 2.0,
                      anisotropy_xz: float = 2.0,
                      theta_xy: float = 30.0,
                      theta_xz: float = 30.0,
                      theta_yz: float = 30.0,
                      power: float = 2.0,
                      amplitude: float = 1.0) -> np.ndarray:
        theta_xy_rad = np.deg2rad(theta_xy)
        theta_xz_rad = np.deg2rad(theta_xz)
        theta_yz_rad = np.deg2rad(theta_yz)
        ell_x = length_scale * np.sqrt(anisotropy_xy * anisotropy_xz)
        ell_y = length_scale * np.sqrt(1/anisotropy_xy)
        ell_z = length_scale * np.sqrt(1/anisotropy_xz)
        KX_rot = self.KX * np.cos(theta_xy_rad) - self.KY * np.sin(theta_xy_rad)
        KY_rot = self.KX * np.sin(theta_xy_rad) + self.KY * np.cos(theta_xy_rad)
        KZ_rot = self.KZ
        KX_rot_2 = KX_rot * np.cos(theta_xz_rad) - KZ_rot * np.sin(theta_xz_rad)
        KY_rot_2 = KY_rot
        KZ_rot_2 = KX_rot * np.sin(theta_xz_rad) + KZ_rot * np.cos(theta_xz_rad)

        KX_rot_3 = KX_rot_2
        KY_rot_3 = KY_rot_2 * np.cos(theta_yz_rad) - KZ_rot_2 * np.sin(theta_yz_rad)
        KZ_rot_3 = KY_rot_2 * np.sin(theta_yz_rad) + KZ_rot_2 * np.cos(theta_yz_rad)

        spectrum = np.exp(-((KX_rot_3 / ell_x)**2 +
                           (KY_rot_3 / ell_y)**2 +
                           (KZ_rot_3 / ell_z)**2)**(power / 2))

        noise = np.random.randn(self.nx, self.ny, self.nz) + \
                1j * np.random.randn(self.nx, self.ny, self.nz)
        field = np.fft.ifftn(np.fft.fftn(noise) * np.sqrt(spectrum)).real
        field = field / np.std(field) * amplitude
        return field

    def kink_field(self, system_type: str = 'klein_gordon',
                 winding_x: int = 1, winding_y: int = 0, winding_z: int = 0,
                 width_range: Tuple[float, float] = (0.5, 3.0),
                 randomize_positions: bool = True, velocity_type: str = 'zero') -> Tuple[np.ndarray, np.ndarray]:

        u0 = np.zeros_like(self.X)
        if winding_x != 0:
            width_x = width_range[0] + (width_range[1] - width_range[0]) * np.random.rand()
            positions_x = []

            if randomize_positions:
                for i in range(abs(winding_x)):
                    positions_x.append(self.L * (2 * np.random.rand() - 1))
            else:
                for i in range(abs(winding_x)):
                    positions_x.append(self.L * (-0.8 + 1.6 * i / (abs(winding_x))))

            sign_x = 1 if winding_x > 0 else -1
            for x0 in positions_x:
                kink_width = width_x * (0.8 + 0.4 * np.random.rand())
                u0 += sign_x * 4 * np.arctan(np.exp((self.X - x0) / kink_width))

        if winding_y != 0:
            width_y = width_range[0] + (width_range[1] - width_range[0]) * np.random.rand()
            positions_y = []

            if randomize_positions:
                for i in range(abs(winding_y)):
                    positions_y.append(self.L * (2 * np.random.rand() - 1))
            else:
                for i in range(abs(winding_y)):
                    positions_y.append(self.L * (-0.8 + 1.6 * i / (abs(winding_y))))

            sign_y = 1 if winding_y > 0 else -1
            for y0 in positions_y:
                kink_width = width_y * (0.8 + 0.4 * np.random.rand())
                u0 += sign_y * 4 * np.arctan(np.exp((self.Y - y0) / kink_width))

        if winding_z != 0:
            width_z = width_range[0] + (width_range[1] - width_range[0]) * np.random.rand()
            positions_z = []

            if randomize_positions:
                for i in range(abs(winding_z)):
                    positions_z.append(self.L * (2 * np.random.rand() - 1))
            else:
                for i in range(abs(winding_z)):
                    positions_z.append(self.L * (-0.8 + 1.6 * i / (abs(winding_z))))

            sign_z = 1 if winding_z > 0 else -1
            for z0 in positions_z:
                kink_width = width_z * (0.8 + 0.4 * np.random.rand())
                u0 += sign_z * 4 * np.arctan(np.exp((self.Z - z0) / kink_width))

        if velocity_type == 'zero':
            v0 = np.zeros_like(u0)
        else:
            v0 = self.anisotropic_grf(
                length_scale=np.mean(width_range) * 2.0,
                amplitude=np.max(np.abs(u0)) * 0.1)
        return u0, v0

    def q_ball_soliton(self, system_type: str = 'klein_gordon',
            position=None, omega=0.8, amplitude=1.0, w=0.5,
            velocity_type='fitting'):
        X, Y, Z = self.X, self.Y, self.Z
        if position is None:
            xc, yc, zc = .5 * np.random.uniform(-self.L, self.L, 3)
        else:
            xc, yc, zc = position

        R = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - zc)**2)
        profile = amplitude * np.exp(-R**2/(2*w**2))

        u0 = profile * np.cos(omega)
        if velocity_type == 'fitting':
            v0 = -omega * profile * np.sin(omega)
        else:
            v0 = np.zeros_like(u0)
        return u0, v0

    def generate_sample(self, system_type: str = 'klein_gordon', phenomenon_type: str = 'kink_field',
                        time_param: float = 0.0, velocity_type: str = "fitting",
                        **params) -> Tuple[np.ndarray, np.ndarray]:
        if phenomenon_type == 'kink_field':
            return self.kink_field(system_type, **params)
        elif phenomenon_type == 'q_ball_soliton':
            return self.q_ball_soliton(system_type, **params)
        else:
            raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")

    def generate_initial_condition(self, system_type: str = 'klein_gordon',
                                   phenomenon_type: List[str] = None,
                                   velocity_type: str = 'fitting', **params) -> Tuple[np.ndarray, np.ndarray]:
        if phenomenon_type is None:
            raise Exception
        u0, v0 = self.generate_sample(system_type=system_type,
                                      phenomenon_type=phenomenon_type,
                                      velocity_type=velocity_type,
                                      **params)
        
        # watch out for zeros! -- TODO get scaling right
        u0 = u0 / np.max(np.abs(u0) + 1e-10)
        return (u0, v0)


if __name__ == '__main__':
    nx = ny = 128
    L = 3.
    sampler = RealWaveSampler(nx, ny, L)
    params = {
        'n_rings': np.random.randint(2, 6),
        'radius_range': (1.0, min(L / 4, 5.0)),
        'width_range': (0.3, 1.0),
        'arrangement': np.random.choice(['concentric', 'random', 'circular']),
        'interaction_strength': np.random.uniform(0.5, 0.9),
        'modulation_strength': np.random.uniform(0, 0.4),
        'modulation_mode_range': (1, 6)
    }

    n_samples = 100
    ensemble = sampler.generate_diverse_ensemble(system_type='sine_gordon', phenomenon_type='multi_ring_state',
                                                 n_samples=n_samples,
                                                 max_attempts=10 * n_samples, diversity_metric='l2', **params)
