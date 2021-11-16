
# Set global parameters
s_eigenvalues = [0, 1]
DEFAULT_PARAMS = {"penalty": 2,
                  "shots": 1024,
                  "num_grid": 30,
                  "seed" : 22, 
                  "length_scale_bounds" : (0.01, 100),
                  "nu" : 1.5,
                  "initial_length_scale" : 0.11,
                  "max_iter_lfbgs": 50000,
                  "optimizer_kernel": 'fmin_l_bfgs_b'
                  }
