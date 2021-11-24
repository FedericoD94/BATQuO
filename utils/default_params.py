
# Set global parameters
s_eigenvalues = [0, 1]
DEFAULT_PARAMS = {"penalty": 2,
                  "shots": 128,
                  "num_grid": 20,
                  "seed" : 22, 
                  "length_scale_bounds" : (0.01, 100),
                  "nu" : 1.5,
                  "initial_length_scale" : 1,
                  "max_iter_lfbgs": 50000,
                  "optimizer_kernel": 'fmin_l_bfgs_b',
                  "n_restart_kernel_optimizer":9,
                  "distance_conv_tol": 0.01
                  }
