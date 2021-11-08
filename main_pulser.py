import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pulser import Register
from pulser.devices import Chadoq2

from scipy.optimize import minimize
from utils.qaoa_pulser import *
from HMC import *
from gaussian_process import *
import time

seed = 22
np.random.seed(seed)
random.seed(seed)
np.set_printoptions(precision=4)

start_time = time.time()
### TRAIN PARAMETERS
depth = 4
Nwarmup = 10
Nbayes = 50
backend = 'PULSER'
method = 'DIFF-EVOL'
param_range = [100, 2000]   # extremes where to search for the values of gamma and beta

fidelities = []
delta_Es  = []
corr_lengths = []
average_distance_vectors = []
std_energies = []

### CREATE GRAPH AND REGISTER 
pos = np.array([[0., 0.],
                [-4, -7],
                [4, -7],
                [8, 6],
                [-8, 6]]
               )
G, distances = pos_to_graph(pos)

qubits = dict(enumerate(pos))
reg = Register(qubits)
    
### INITIAL RANDOM POINTS
X_train = []   #data
y_train = []   #label

### CREATE GP AND FIT TRAINING DATA
kernel =  ConstantKernel(1)* Matern(length_scale=0.11, length_scale_bounds=(1e-01, 100.0), nu=1.5)
gp = MyGaussianProcessRegressor(kernel=kernel,
                                n_restarts_optimizer=20,
                                param_range = param_range,
                                alpha=1e-2,
                                normalize_y=True,
                                max_iter=50000)

X_train, y_train = generate_random_points(Nwarmup, G, depth, param_range, reg)
gp.fit(X_train, y_train)

sample_points = []
energies = []
fidelities = [0]*Nwarmup
delta_Es = [0]*Nwarmup
average_distance_vectors = [0]*Nwarmup
std_energies = [0]*Nwarmup
corr_lengths = [gp.kernel_.get_params()['k2__length_scale']]*Nwarmup
indexes = ['01011', '00111']
print(' ITER   |   NEXT POINT  |  ENERGY ')
init_pos = [0.2, 0.2]*depth
for i in range(Nbayes):
    next_point, results, average_norm_distance_vectors, std_energy, conv_flag = gp.bayesian_opt_step(init_pos, method)
    
    corr_length = gp.kernel_.get_params()['k2__length_scale']
    next_point = [int(x) for x in next_point]
    X_train.append(next_point)
    y_next_point = apply_qaoa(next_point, reg, G)
    y_train.append(y_next_point)
    C, _= quantum_loop(next_point, r=reg)
    fidelity = (C[indexes[0]] + C[indexes[1]])/1000
    fidelities.append(fidelity)
    print(i, next_point, y_next_point, fidelity)
    gp.fit(next_point, y_next_point)
    sample_points.append(next_point)
    corr_lengths.append(corr_length)
    std_energies.append(std_energy)
    average_distance_vectors.append(average_distance_vectors)
    energies.append(y_next_point)
    delta_Es.append(abs(-3 -y_next_point ))
    
    
best_x, best_y, where = gp.get_best_point()
X_train.append(best_x)
y_train.append(best_y)
fidelities.append(fidelities[where])
delta_Es.append(delta_Es[where])
corr_lengths.append(corr_lengths[where])
average_distance_vectors.append(average_distance_vectors[where])
std_energies.append(std_energies[where])
iter = range(Nwarmup + Nbayes)
training_info = np.column_stack(([i for i in iter] + [where], X_train, y_train, fidelities, delta_Es, corr_lengths, std_energies))
end_time = time.time()
np.savetxt('p={}_training_time_{}.dat'.format(depth, end_time - start_time), training_info)
print('Best point: ' , where,  best_x, best_y, fidelities[where], delta_Es[where])

params = best_x
C, _= quantum_loop(params, r=reg)
plot_distribution(C)