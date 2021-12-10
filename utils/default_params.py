
# Set global parameters
s_eigenvalues = [0, 1]
DEFAULT_PARAMS = {"penalty": 2,
                  "shots": 128,
                  "num_grid": 20,
                  "seed" : 22, 
                  "initial_length_scale" : 1,
                  "length_scale_bounds" : (0.1, 2),
                  "initial_sigma":1,
                  "constant_bounds":(0.1, 100),
                  "nu" : 1.5,
                  "max_iter_lfbgs": 50000,
                  "optimizer_kernel":'fmin_l_bfgs_b', #'fmin_l_bfgs_b', #monte_carlo', #'fmin_l_bfgs_b',
                  "diff_evol_func": None,
                  "n_restart_kernel_optimizer":9,
                  "distance_conv_tol": 0.01
                  }
