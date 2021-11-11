import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.qaoa_qiskit import *
from utils.gaussian_process import *
from utils.default_params import *
import json
import time
import sys

'RUNNA IL MAIN VARIANDO NUMERO DI WARMUP POINTS E NUMERO TOTALE DI POINT DA RUNNARE SU CLUSTER IN PARALLELO'

np.set_printoptions(precision = 4, suppress = True)

percentages_warmup = [.1, .2, .3]
Ntots = [100, 150, 200, 250]

for Ntot in Ntots:
	for percentage_warmup in percentages_warmup:
		### PARAMETERS
		#depth = int(sys.argv[1])
		depth = 1
		Nwarmup = int(Ntot*percentage_warmup)
		Nbayes = Ntot - Nwarmup
		backend = 'QISKIT'
		method = 'DIFF-EVOL'
		param_range = [0.1, np.pi]   # extremes where to search for the values of gamma and beta

		file_name = 'qiskit/p={}_punti={}_warmup={}_train={}.dat'.format(depth, Nwarmup + Nbayes, Nwarmup, Nbayes)
		data = []
		global_time = time.time()
		results_structure = ['iter', 'point', 'energy', 'fidelity', 'corr_length', 'const kernel',
							'std energies', 'average distances', 'nit', 'time opt kernel', 'time opt bayes', 'time qaoa', 'time step']

		### CREATE GRAPH 
		#pos = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [0, 4]])
		pos = np.array([[0, 1], [1, 2], [3, 2], [0, 3], [0, 4], [0,5]])

		G = nx.Graph()
		G.add_edges_from(pos)
		qaoa = qaoa_qiskit(G)
		gs_energy, gs_state, degeneracy = qaoa.calculate_gs()

		### CREATE GP AND FIT TRAINING DATA
		#kernel =  ConstantKernel(1)*RBF(0.2, length_scale_bounds = (1E-1, 1E2)) 
		kernel =  ConstantKernel(1)*Matern(length_scale=DEFAULT_PARAMS['initial_length_scale'], length_scale_bounds=DEFAULT_PARAMS['length_scale_bounds'], nu=DEFAULT_PARAMS['nu'])
		gp = MyGaussianProcessRegressor(kernel=kernel, 
									optimizer = DEFAULT_PARAMS['optimizer_kernel'],
									param_range = param_range,
									n_restarts_optimizer = 1, 
									normalize_y = True,
									max_iter=DEFAULT_PARAMS['max_iter_lfbgs'])

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
			#print((i+1),' / ',Nbayes)
			format  = '%.d ' + (len(new_data) - 1)*'%.4f '
			np.savetxt(file_name, data, fmt = format)

		best_x, best_y, where = gp.get_best_point()

		data.append(data[where])

		np.savetxt(file_name, np.array(data), fmt = format)
