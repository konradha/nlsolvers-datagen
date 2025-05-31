import numpy as np

# TODO look at valid time frames to investigate


def get_parameter_spaces(L):
    system_types = [
        "sine_gordon",
        "double_sine_gordon",
        "hyperbolic_sine_gordon",
        "phi4",
        "klein_gordon"
    ]

    parameter_spaces = {}

    parameter_spaces["kink_solution"] = {
        "system_type": system_types,
        "width": np.linspace(0.3, 3.0, 6).tolist(),
        "position": [
            (x, y) for x in np.linspace(-L * 0.7, L * 0.7, 5)
            for y in np.linspace(-L * 0.7, L * 0.7, 5)
        ],
        "orientation": np.linspace(0, 2 * np.pi, 8).tolist(),
        "velocity": [
            (vx, vy) for vx in np.linspace(-0.4, 0.4, 5)
            for vy in np.linspace(-0.4, 0.4, 5)
        ],
        "kink_type": ["standard", "anti", "double"],
        "velocity_type": ["fitting", "zero", "grf"]
    }

    parameter_spaces["kink_field"] = {
        "system_type": system_types,
        "winding_x": list(range(-4, 5)),
        "winding_y": list(range(-4, 5)),
        "width_range": [
            (min_w, max_w) for min_w in [0.3, 0.5, 0.7]
            for max_w in [1.5, 2.0, 3.0]
        ],
        "randomize_positions": [True, False]
    }

    # using T=40 and appropriate nt (ie. nt=4000) yields a periodicity in time
    # -> breather-like kink!
    parameter_spaces["kink_array_field"] = {
        "system_type": system_types,
        "num_kinks_x": [1, 3, 5],
        "num_kinks_y": [1, 4, 8],
        "width_range": [
            (min_w, max_w) for min_w in [0.3, 0.5, 0.7]
            for max_w in [1.5, 2.0, 3.0]
        ],
        "jitter": [0.1, 0.4, 0.8]
    }

    parameter_spaces["breather_solution"] = {
        "system_type": system_types,
        "amplitude": np.linspace(0.1, 0.95, 9).tolist(),
        "frequency": np.linspace(0.3, 0.95, 7).tolist(),
        "width": np.linspace(0.3, 3.0, 6).tolist(),
        "position": [
            (x, y) for x in np.linspace(-L * 0.7, L * 0.7, 4)
            for y in np.linspace(-L * 0.7, L * 0.7, 4)
        ],
        "phase": np.linspace(0, 2 * np.pi, 8).tolist(),
        "orientation": np.linspace(0, 2 * np.pi, 8).tolist(),
        "breather_type": ["standard", "radial"],
        "time_param": [0.0],  # np.linspace(0, 2.0, 5).tolist(),
        "velocity_type": ["fitting", "zero", "grf"]
    }

    parameter_spaces["breather_field"] = {
        "system_type": system_types,
        "num_breathers": list(range(2, 9)),
        "position_type": ["random", "circle", "line"],
        "time_param": [0.0, .5, 10.]
    }

    parameter_spaces["multi_breather_field"] = {
        "system_type": system_types,
        "num_breathers": list(range(1, 4)),
        "position_type": ["line"],  # ["random", "circle", "line"],
        "amplitude_range": [
            (min_a, max_a) for min_a in [0.1, 0.2, 0.3, 0.4]
            for max_a in [0.6, 0.7, 0.8, 0.9]
        ],
        "width_range": [
            (min_w, max_w) for min_w in [0.3, 0.5, 0.7]
            for max_w in [1., 1.5]
        ],
        "frequency_range": [
            (min_f, max_f) for min_f in [0.3, 0.6, 0.7]
            for max_f in [0.8, 0.9, 0.95]
        ],
        "time_param": [0.0],  # np.linspace(0, 2.0, 5).tolist(),
        "velocity_type": ["fitting", "zero", "grf"]
    }

    parameter_spaces["ring_soliton"] = {
        "system_type": system_types,
        "amplitude": np.linspace(0.5, 2.0, 4).tolist(),
        "radius": np.linspace(0.5, min(L * 0.6, 5.0), 8).tolist(),
        "width": np.linspace(0.2, 1.5, 7).tolist(),
        "position": [
            (x, y) for x in np.linspace(-L * 0.3, L * 0.3, 3)
            for y in np.linspace(-L * 0.3, L * 0.3, 3)
        ],
        "velocity": np.linspace(-0.3, 0.3, 7).tolist(),
        "ring_type": ["expanding", "kink_antikink"],
        "modulation_strength": np.linspace(0, 0.5, 6).tolist(),
        "modulation_mode": list(range(0, 8)),
        "time_param": np.linspace(0, 1.5, 4).tolist()
    }

    parameter_spaces["elliptical_soliton"] = {
        "system_type": system_types,
        "complexity": ["complex", "simple"]
    }

    parameter_spaces["multi_ring_state"] = {
        "system_type": system_types,
        "n_rings": list(range(2, 8)),
        "radius_range": [
            (min_r, max_r) for min_r in [0.5, 1.0, 1.5]
            for max_r in [2.5, 3.5, 4.5]
        ],
        "width_range": [
            (min_w, max_w) for min_w in [0.2, 0.3, 0.4]
            for max_w in [0.6, 0.8, 1.0]
        ],
        "arrangement": ["concentric", "random", "circular"],
        "interaction_strength": np.linspace(0.3, 1.0, 5).tolist(),
        "modulation_strength": np.linspace(0, 0.5, 6).tolist(),
        "modulation_mode_range": [
            (min_m, max_m) for min_m in [1, 2, 3]
            for max_m in [4, 6, 8]
        ]
    }

    parameter_spaces["colliding_rings"] = {
        "system_type": system_types,
        "num_rings": list(range(2, 4)),
        "ring_type": ["cocentric", "nested", "random"],
        "amplitude": [1., 3.],
    }

    parameter_spaces["spiral_wave_field"] = {
        "num_arms": list(range(1, 9)),
        "decay_rate": np.linspace(0.2, 1.0, 5).tolist(),
        "amplitude": np.linspace(0.5, 2.0, 4).tolist(),
        "position": [
            (x, y) for x in np.linspace(-L * 0.5, L * 0.5, 4)
            for y in np.linspace(-L * 0.5, L * 0.5, 4)
        ],
        "phase": np.linspace(0, 2 * np.pi, 8).tolist(),
        "k_factor": np.linspace(0.5, 4.0, 8).tolist()
    }

    parameter_spaces["multi_spiral_state"] = {
        "n_spirals": np.linspace(1, 10, 5).astype(int),
        "amplitude_range": [
            (min_a, max_a) for min_a in [0.1, 0.2, 0.3, 0.4]
            for max_a in [0.6, 0.7, 0.8, 0.9]
        ],
        "num_arms_range": [(1, 3), (3, 12), (1, 8)],
        "decay_rate_range": [
            (min_d, max_d) for min_d in [0.3, 0.6, 0.7]
            for max_d in [0.8, 0.9, 0.95]
        ],
        "position_variance": [.3, 1., 1.5],
        "interaction_strength": [1e-2, .3, .8],
    }

    parameter_spaces["skyrmion_solution"] = {
        "system_type": system_types,
        "amplitude": np.linspace(0.5, 2.0, 4).tolist(),
        "radius": np.linspace(0.3, 2.5, 6).tolist(),
        "position": [
            (x, y) for x in np.linspace(-L * 0.5, L * 0.5, 4)
            for y in np.linspace(-L * 0.5, L * 0.5, 4)
        ],
        "charge": [-2, -1, 1, 2],
        "profile": ["standard", "compact", "exponential"]
    }

    parameter_spaces["skyrmion_lattice"] = {
        "system_type": system_types,
        "n_skyrmions": [4, 7, 9, 12, 16, 25],
        "radius_range": [
            (min_r, max_r) for min_r in [0.3, 0.5, 0.7]
            for max_r in [1.0, 1.5, 2.0]
        ],
        "amplitude": np.linspace(0.5, 2.0, 4).tolist(),
        "arrangement": ["triangular", "square", "random"],
        "separation": np.linspace(1.5, 4.0, 6).tolist(),
        "charge_distribution": ["alternating", "random", "same"]
    }

    parameter_spaces["skyrmion_like_field"] = {
        "num_skyrmions": list(range(2, 9)),
    }

    parameter_spaces["q_ball_solution"] = {
        "system_type": system_types,
        "position": [
            (x, y) for x in np.linspace(-L * 0.5, L * 0.5, 100)
            for y in np.linspace(-L * 0.5, L * 0.5, 100)
        ],
        "phase": [0.0, .5],
        "frequency": [.3, .8],
        "charge": [-1, 1],
    }

    parameter_spaces["multi_q_ball"] = {
        "system_type": system_types,
        "n_qballs": [2, 4, 8],
        "amplitude_range": [(.1, 1.1), (.5, 1.5)],
        "radius_range": [(.5, 2), (.1, 4.)],
    }

    parameter_spaces["grf_modulated_soliton_field"] = {
        "system_type": system_types,
        "grf_length_scale": np.linspace(0.5, 3.0, 6).tolist(),
        "smoothness_scaling": np.linspace(0.5, 5.0, 5).tolist(),
        "anisotropy_ratio": [1.0, 1.5, 2.0, 3.0],
        "anisotropy_angle": np.linspace(0, np.pi, 4).tolist(),
        "construction_method": ["threshold", "level_set", "continuous"],
        "mixture_type": ["additive", "maximum", "blending"],
        "velocity_mode": ["zero", "fitting", "random"],
        "threshold_values": [
            [-1.0, 0.0, 1.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-1.5, -0.5, 0.5, 1.5]
        ],
        "soliton_types": [
            ["kink", "antikink"],
            ["kink", "breather", "antikink"],
            ["kink", "breather", "ring", "antikink"]
        ],
        "level_set_width": [0.1, 0.2, 0.3, 0.5],
        "random_velocity_scale": np.linspace(0.1, 0.5, 5).tolist()
    }

    return parameter_spaces

def get_parameter_spaces_3d(L):
    parameter_spaces = {}
    parameter_spaces["kink_field"] = {
        "system_type": ["klein_gordon"],
        "winding_x": list(range(-4, 5)),
        "winding_y": list(range(-4, 5)),
        "winding_z": list(range(-4, 5)),
        "width_range": [
            (min_w, max_w) for min_w in [0.3, 0.5, 0.7]
            for max_w in [1.5, 2.0, 3.0]
        ],
        "randomize_positions": [True, False],
        "velocity_type": ["zero", "grf"]
    }

    parameter_spaces["q_ball_soliton"] = {
        "omega": [.3, 0.6, 0.8],
        "amplitude": [-.2, .2, .45],
        "w": [.1, .4, .5],
        "velocity_type": ["zero", "fitting"]
    }
    return parameter_spaces 
