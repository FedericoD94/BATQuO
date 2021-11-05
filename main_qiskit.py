import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.qaoa_qiskit import *
from utils.gaussian_process import *
import json
import time

np.set_printoptions(precision = 4, suppress = True)
seed = 22
np.random.seed(seed)
random.seed(seed)

### PARAMETERS
depth = 7
Nwarmup = 10
Nbayes = 50
backend = 'QISKIT'
method = 'DIFF-EVOL'
param_range = [0.1, np.pi]   # extremes where to search for the values of gamma and beta

fidelities = []
delta_Es  = []
corr_lengths = []
average_distance_vectors = []
std_energies = []

### CREATE GRAPH 
pos = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [0, 4]])

#pos = np.array([[0, 1], [1, 2], [3, 2], [0, 3], [0, 4], [0,5]])

G = nx.Graph()
G.add_edges_from(pos)
qaoa = qaoa_qiskit(G)
gs_energy, gs_state, degeneracy = qaoa.calculate_gs_qiskit()


#fixed_params = [2.8701538193522214, 0.9654169506599438, 1.3114765662270935, 1.6981142399479234, 1.6824873990924147, 1.8292098622362, 1.7004006368953093, 1.2588740997641827, 1.4713354779032488, 0.1, 2.2294483412078243, 0.118956282211989, 1.9673648297094237, 3.0904532027356413]

### CREATE GP AND FIT TRAINING DATA
#kernel =  ConstantKernel(1)*RBF(0.2, length_scale_bounds = (1E-1, 1E2)) 
kernel =  ConstantKernel(1)* Matern(length_scale=0.11, length_scale_bounds=(1e-01, 100.0), nu=1.5)
gp = MyGaussianProcessRegressor(kernel=kernel, 
                                seed = seed,
                                param_range = param_range,
                                n_restarts_optimizer=20, 
                                alpha=1e-2,
                                normalize_y=True,
                                max_iter=50000)

X_train, y_train = qaoa.generate_random_points(Nwarmup, depth, param_range)
gp.fit(X_train, y_train)

fidelities = [qaoa.fidelity_gs(i) for i in X_train]
delta_Es = [abs(y_train[i] - gs_energy) for i in range(Nwarmup)]
corr_lengths = [gp.kernel_.get_params()['k2__length_scale']]*Nwarmup
average_distance_vectors = [gp.kernel_.get_params()['k2__length_scale']]*Nwarmup
std_energies = [gp.kernel.get_params()['k2__length_scale']]*Nwarmup
### BAYESIAN OPTIMIZATION
init_pos = [0.2, 0.2]*depth

for i in range(Nbayes):
    start_time = time.time()
    next_point, results, average_norm_distance_vectors, std_population_energy, conv_flag = gp.bayesian_opt_step(init_pos, method)
    mid_time = time.time()
    y_next_point = qaoa.expected_energy(next_point)
    end_time = time.time()
    print('Time gp: {}, time qaoa: {}'.format(mid_time - start_time, end_time - mid_time))
    fidelity = qaoa.fidelity_gs(next_point)
    delta_E = abs(y_next_point - gs_energy)
    corr_length = gp.kernel_.get_params()['k2__length_scale']
    X_train.append(next_point)
    y_train.append(y_next_point)
    fidelities.append(fidelity)
    delta_Es.append(delta_E)
    corr_lengths.append(corr_length)
    average_distance_vectors.append(average_norm_distance_vectors)
    std_energies.append(std_population_energy)
    print(i + Nwarmup, next_point, y_next_point, fidelity, delta_E , corr_length)
    gp.fit(next_point, y_next_point)
    

best_x, best_y, where = gp.get_best_point()
X_train.append(best_x)
y_train.append(best_y)
fidelities.append(fidelities[where])
delta_Es.append(delta_Es[where])
corr_lengths.append(corr_lengths[where])
average_distance_vectors.append(average_distance_vectors[where])
std_energies.append(std_energies[where])
iter = range(Nwarmup + Nbayes)
training_info = np.column_stack(([i for i in iter] + [where], X_train, y_train, fidelities, delta_Es, corr_lengths, average_distance_vectors, std_energies))

np.savetxt('p={}_punti={}_warmup={}_train={}_2.dat'.format(depth, Nwarmup + Nbayes, Nwarmup, Nbayes), training_info)
print('Best point: ' , where,  best_x, best_y, fidelities[where], delta_Es[where], corr_lengths[where])

counts = qaoa.final_sampled_state(best_x)
qaoa.plot_final_state(counts)
#plt.savefig('sampled_state_p{}.png'.format(depth))
end_time = time.time()
print('time: ',  end_time - start_time)