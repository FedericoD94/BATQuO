
# Set global parameters
s_eigenvalues = [0, 1]
DEFAULT_PARAMS = {"penalty": 2,
                  "shots": 12800,
                  "num_grid": 4,
                  "seed" : 23, 
                  "initial_length_scale" : 1,
                  "length_scale_bounds" : (0.1, 100),
                  "initial_sigma":1,
                  "constant_bounds":(0.1, 100),
                  "nu" : 2.5,
                  "max_iter_lfbgs": 50000,
                  "normalize_y": False,
                  "gtol": 1e-6,
                  "optimizer_kernel":'fmin_l_bfgs_b', #'fmin_l_bfgs_b' or #monte_carlo', 
                  "diff_evol_func": None, # or 'mc',
                  "n_restart_kernel_optimizer":9,
                  "distance_conv_tol": 0.01,
                  "angle_bounds": [[100, 2000], [100, 2000]]
                  }
                  
Q_DEVICE_PARAMS = {'lattice_spacing': 5, #\mu m
                   'type_of_lattice': 'triangular',
                   'eta': 0.005,
                   'thermal_motion': 85, #nm
                   'doppler_shift': 0.47, #MHz
                   'intensity_fluctuation':0.03, 
                   'laser_waist': 148, #micrometers
                   'rising_time': 50, #ns
                   'epsilon': 0.03, #false positive
                   'epsilon_prime': 0.08, #false negative
                   'temperature': 30, #microKelvin
                   'coherence_time': 5000, #ns
                   'omega_over_2pi': 1, #see notes/info.pdf for this value
                   'delta_over_2pi': -3.1 #see notes/info.pdf for the calculation
}