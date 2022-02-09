import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.qaoa_pulser import *
from utils.gaussian_process_mod import *
import time
import random
import datetime

np.set_printoptions(precision = 4, suppress = True)
np.random.seed(DEFAULT_PARAMS['seed'])
random.seed(DEFAULT_PARAMS['seed'])
### TRAIN PARAMETERS
depth = 3
Nwarmup = 20
Ntot = 200
Nbayes = Ntot-Nwarmup
method = 'DIFF-EVOL'
param_range = [100, 3000]   # extremes where to search for the values of gamma and beta
global_time = time.time()
quantum_noise = None #'SPAM', 'doppler', 'dephasing', 'amplitude' or a TUPLE of more than one or None

### CREATE GRAPH AND QAOA INSTANCE
#pos = np.array([[0., 0.],[-4, -7],[4, -7],[8, 6],[-8, 6]])
pos = np.array([[0., 0.], [0, 10], [10,0], [10,10], [10,20],[20,10]])
               
qaoa = qaoa_pulser(depth, param_range, pos, quantum_noise)
gs_en, gs_state, deg = qaoa.calculate_physical_gs()

### CREATE GP 
kernel =  Matern(length_scale=DEFAULT_PARAMS['initial_length_scale'], 
                length_scale_bounds=DEFAULT_PARAMS['length_scale_bounds'], 
                nu=DEFAULT_PARAMS['nu'])*ConstantKernel(DEFAULT_PARAMS['initial_length_scale'], 
                                        constant_value_bounds = DEFAULT_PARAMS['constant_bounds'])
                
gp = MyGaussianProcessRegressor(kernel=kernel, 
                                optimizer = DEFAULT_PARAMS['optimizer_kernel'],
                                param_range = param_range,
                                n_restarts_optimizer = DEFAULT_PARAMS['n_restart_kernel_optimizer'], 
                                normalize_y=False,
                                gtol=1e-06,
                                max_iter=DEFAULT_PARAMS['max_iter_lfbgs'])


### DATA SAVING
file_name = 'p={}_punti={}_warmup={}_train={}'.format(depth, Nwarmup + Nbayes, Nwarmup, Nbayes)
data = []
gamma_names = ['GAMMA' + str(i) + ' ' for i in range(depth)]
beta_names = ['BETA' + str(i) + ' ' for i in range(depth)]
data_names = ['iter'] + gamma_names + beta_names +['energy', 'variance', 'fidelity_exact', 'fidelity_sampled', 'ratio', 'corr_length ', 'const_kernel',
                    'std energies', 'average distances', 'nit', 'time opt bayes', 'time qaoa', 'time opt kernel', 'time step']
data_header = " ".join(["{:>7} ".format(i) for i in data_names]) + '\n'

info_file_name = file_name + '_info.txt'
with open(info_file_name, 'w') as f:
    f.write('BAYESIAN OPTIMIZATION of QAOA \n\n')
    qaoa.print_info_problem(f)
    f.write('QAOA PARAMETERS\n-------------\n')
    qaoa.print_info_qaoa(f)
    f.write('\nGAUSSIAN PROCESS PARAMETERS\n---------------\n')
    gp.print_info(f)
    f.write('\nBAYESIAN OPT PARAMETERS\n------------------\n')
    f.write('Nwarmup points: {} \nNtraining points: {}\n'.format(Nwarmup, Nbayes))
    f.write('FILE.DAT PARAMETERS:\n')
    print(data_names, file = f)

###GENERATE AND FIT TRAINING DATA
X_train, y_train, data_train = qaoa.generate_random_points(Nwarmup)
gp.fit(X_train, y_train)

print(gp.kernel.theta,gp.kernel.get_params()['k1__length_scale'],
                    gp.kernel.get_params()['k2__constant_value'])

### STARTS PLOTTING THE DATA
data_file_name = file_name + '.dat'
data = [[i] + x + [y_train[i]] + data_train[i] +
                    gp.kernel.theta.tolist()+[ 0, 0, 0, 0, 0, 0, 0
                    ] for i, x in enumerate(X_train)]
format = '%3d ' + 2*depth*'%6d ' + (len(data[0]) - 1 - 2*depth)*'%4.4f '
np.savetxt(data_file_name, data, fmt = format)

#### BAYESIAN OPTIMIZATION PROCEDURE
print('Training ...')
for i in range(Nbayes):
    start_time = time.time()
    next_point, n_it, avg_sqr_distances, std_pop_energy = gp.bayesian_opt_step(method)

    next_point = [int(i) for i in next_point]

    bayes_time = time.time() - start_time
    y_next_point, var, fid, fid_exact, sol_ratio, _, _ = qaoa.apply_qaoa(next_point)
    qaoa_time = time.time() - start_time - bayes_time
    corr_length = gp.kernel.get_params()['k1__length_scale']
    constant_kernel = gp.kernel.get_params()['k2__constant_value']
    #if np.abs(np.log(np.abs(0.01-corr_length))) > 8:
     #   if i == 0:
      #      gp.kernel_.set_params(**{'k1__length_scale': data[-1][-9]})
       #     corr_length = data[-1][-9]
       # else:
       #     gp.kernel_.set_params(**{'k1__length_scale': new_data[-9]})
       #     corr_length = new_data[-9]

    gp.fit(next_point, y_next_point)
    kernel_time = time.time() - start_time - qaoa_time - bayes_time
    step_time = time.time() - start_time
    
    new_data = [i+Nwarmup] + next_point + [y_next_point, var, fid, fid_exact, sol_ratio, corr_length, constant_kernel, 
                                    std_pop_energy, avg_sqr_distances, n_it, 
                                    bayes_time, qaoa_time, kernel_time, step_time]     

    data.append(new_data)
    np.savetxt(data_file_name, data, fmt = format)
    print('iteration: {}/{} en: {}, fid: {}'.format(i, Nbayes, y_next_point, fid))

best_x, best_y, where = gp.get_best_point()
data.append(data[where])
np.savetxt(data_file_name, data, fmt = format)
print('Best point: ' , data[where])
print('time: ',  time.time() - global_time)