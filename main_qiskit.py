import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.qaoa_qiskit import *
from utils.gaussian_process import *
from utils.create_graphs import create_random_graph
from utils.parameters import parse_command_line
from utils.default_params import *
import json
import time

np.set_printoptions(precision=4, suppress=True)

args = parse_command_line()

seed = args.seed

fraction_warmup = args.fraction_warmup
depth = args.p
trials = args.trials

num_nodes =args.num_nodes
average_connectivity = args.average_connectivity

### PARAMETERS

Nwarmup = int(args.Nbayes * args.fraction_warmup)
Nbayes = args.Nbayes - Nwarmup
method = 'DIFF-EVOL'
param_range = [0.01, np.pi]   # extremes where to search for the values of gamma and beta

global_time = time.time()
results_structure = ['iter ', 'point ', 'energy ', 'fidelity ', 'corr_length ', 'const kernel ',
                    'std energies ', 'average distances ', 'nit ', 'time opt bayes ', 'time qaoa ', 'time opt kernel ', 'time step ']

### CREATE GRAPH
# pos = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [0, 4]])
# pos = np.array([[0, 1], [1, 2], [3, 2], [0, 3], [0, 4], [0,5]])
np.random.seed(seed)
G = create_random_graph(num_nodes, average_connectivity, False)

qaoa = qaoa_qiskit(G)
gs_energy, gs_state, degeneracy = qaoa.calculate_gs_qiskit()

print(gs_energy)
for i_trial in range(trials):
    DEFAULT_PARAMS["seed"] = seed + i_trial
    file_name = 'p={}_punti={}_warmup={}_train={}_trial={}.dat'.format(depth, Nwarmup + Nbayes, Nwarmup, Nbayes, i_trial)
    data = []
    ### CREATE GP AND FIT TRAINING DATA
    #kernel = ConstantKernel(1)*RBF(0.2, length_scale_bounds = (1E-1, 1E2))
    kernel = ConstantKernel(1)*Matern(length_scale=0.11,length_scale_bounds=(0.01, 100), nu=1.5)
    gp = MyGaussianProcessRegressor(kernel=kernel,
                                    optimizer='fmin_l_bfgs_b', #fmin_l_bfgs_bor differential_evolution
                                    #optimizer='differential_evolution', #fmin_l_bfgs_bor
                                    param_range=param_range,
                                    n_restarts_optimizer=10,
                                    alpha=1e-2,
                                    normalize_y=True,
                                    max_iter=50000)

    X_train, y_train = qaoa.generate_random_points(Nwarmup, depth, param_range)
    gp.fit(X_train, y_train)

    data = [[i] + x + [y_train[i],
                        qaoa.fidelity_gs(x),
                        gp.kernel_.get_params()['k2__length_scale'],
                        gp.kernel_.get_params()['k1__constant_value'], 0, 0, 0, 0, 0, 0, 0
                        ] for i,x in enumerate(X_train)]

    ### BAYESIAN OPTIMIZATION
    init_pos = [0.2, 0.2]*depth
    print('Training ...')
    print(X_train)
    for i in range(Nbayes):
        start_time = time.time()
        next_point, n_it, avg_sqr_distances, std_pop_energy = gp.bayesian_opt_step(init_pos, method)
        bayes_time = time.time() - start_time
        y_next_point = qaoa.expected_energy(next_point)
        qaoa_time = time.time() - start_time - bayes_time
        fidelity = qaoa.fidelity_gs(next_point)
        corr_length = gp.kernel_.get_params()['k2__length_scale']
        constant_kernel = gp.kernel_.get_params()['k1__constant_value']
        gp.fit(next_point, y_next_point)
        kernel_time = time.time() - start_time - qaoa_time - bayes_time
        step_time = time.time() - start_time
        new_data = [i+Nwarmup] + next_point + [y_next_point, fidelity, corr_length, constant_kernel,
                                        std_pop_energy, avg_sqr_distances, n_it,
                                        bayes_time, qaoa_time, kernel_time, step_time]
        data.append(new_data)
        print((i+1),' / ',Nbayes)
        print(new_data)
        format = '%.d ' + (len(new_data) - 1)*'%+.8f '
        np.savetxt(file_name, data, fmt=format)

    best_x, best_y, where=gp.get_best_point()

    data.append(data[where])

    np.savetxt(file_name, np.array(data), fmt=format)
    print('Best point: ' , data[where])
    print('time: ', time.time() - global_time)