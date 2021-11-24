import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.qaoa_pulser import *
from utils.gaussian_process import *
import time
import sys

np.set_printoptions(precision = 4, suppress = True)

percentage_warmup = 0.1
Ntot = 1000

### PARAMETERS
#depth = int(sys.argv[1])
depth = 8
Nwarmup = 20
Nbayes = Ntot-Nwarmup
method = 'DIFF-EVOL'
param_range = [100, 3000]   # extremes where to search for the values of gamma and beta
quantum_noise = 0


file_name = 'p={}_punti={}_warmup={}_train={}.dat'.format(depth, Nwarmup + Nbayes, Nwarmup, Nbayes)
data = []
global_time = time.time()
results_structure = ['iter ', 'point ', 'energy ', 'variance', 'fidelity exact', 'fidelity sampled', 'ratio', 'corr_length ', 'const kernel ',
                    'std energies ', 'average distances ', 'nit ', 'time opt bayes ', 'time qaoa ', 'time opt kernel ', 'time step ']

### CREATE GRAPH AND REGISTER 
#pos = np.array([[0., 0.],[-4, -7],[4, -7],[8, 6],[-8, 6]])
pos = np.array([[0., 0.], [0, 10], [10,0], [10,10], [10,20],[20,10]])
               
qaoa = qaoa_pulser(pos, quantum_noise)
gs_en, gs_state, deg = qaoa.calculate_physical_gs()

### INITIAL RANDOM POINTS
X_train = []   #data
y_train = []   #label

### CREATE GP AND FIT TRAINING DATA
kernel =  Matern(length_scale=DEFAULT_PARAMS['initial_length_scale'], length_scale_bounds=DEFAULT_PARAMS['length_scale_bounds'], nu=DEFAULT_PARAMS['nu'])*ConstantKernel(1)
gp = MyGaussianProcessRegressor(kernel=kernel, 
                                optimizer = DEFAULT_PARAMS['optimizer_kernel'],
                                param_range = param_range,
                                n_restarts_optimizer = 9, 
                                normalize_y=True,
                                gtol=1e-06,
                                max_iter=DEFAULT_PARAMS['max_iter_lfbgs'])

X_train, y_train, var_train = qaoa.generate_random_points(Nwarmup, depth, param_range, return_variance = True)
print(X_train)
gp.fit(X_train, y_train)

data = [[i] + x + [y_train[i], 
					var_train[i],
                    qaoa.fidelity_gs_exact(x), 
                    qaoa.fidelity_gs_sampled(x),
                    qaoa.solution_ratio(x),
                    gp.kernel_.get_params()['k1__length_scale'],
                    gp.kernel_.get_params()['k2__constant_value'], 0, 0, 0, 0, 0, 0, 0
                    ] for i, x in enumerate(X_train)]
                    
init_pos = [0.2, 0.2]*depth
format = '%3d ' + 2*depth*'%5d ' + (len(data[0]) - 1 - 2*depth)*'%.4f '
np.savetxt(file_name, data, fmt = format)

print('Training ...')
for i in range(Nbayes):
    print('iteration: {}/{}'.format(i, Nbayes))
    start_time = time.time()
    next_point, n_it, avg_sqr_distances, std_pop_energy = gp.bayesian_opt_step(init_pos, method)
    next_point = [int(i) for i in next_point]
    bayes_time = time.time() - start_time
    y_next_point, variance_next_point = qaoa.expected_energy_and_variance(next_point)
    qaoa_time = time.time() - start_time - bayes_time
    fid_exact = qaoa.fidelity_gs_exact(next_point)
    fid_sampled = qaoa.fidelity_gs_sampled(next_point)
    sol_ratio = qaoa.solution_ratio(next_point)
    corr_length = gp.kernel_.get_params()['k1__length_scale']
    print(corr_length)    
    if np.abs(np.log(np.abs(0.01-corr_length))) > 8:
        if i == 0:
            print(data[-9])
            gp.kernel_.set_params(**{'k1__length_scale': data[-9]})
            corr_length = data[-9]
        else:
            print(data[-9])
            gp.kernel_.set_params(**{'k1__length_scale': new_data[-9]})
            corr_length = new_data[-9]

    constant_kernel = gp.kernel_.get_params()['k2__constant_value']


    gp.fit(next_point, y_next_point)
    kernel_time = time.time() - start_time - qaoa_time - bayes_time
    step_time = time.time() - start_time
    
    new_data = [i+Nwarmup] + next_point + [y_next_point, variance_next_point, fid_exact, fid_sampled, sol_ratio, corr_length, constant_kernel, 
                                    std_pop_energy, avg_sqr_distances, n_it, 
                                    bayes_time, qaoa_time, kernel_time, step_time]     
      
    data.append(new_data)
    print(data)

    np.savetxt(file_name, data, fmt = format)
     
best_x, best_y, where = gp.get_best_point()
data.append(data[where])

np.savetxt(file_name, np.array(data), fmt = format)
print('Best point: ' , data[where])
print('time: ',  time.time() - global_time)