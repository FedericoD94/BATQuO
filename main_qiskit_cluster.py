import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.qaoa_qiskit import *
from utils.gaussian_process import *
import time
import sys

np.set_printoptions(precision = 4, suppress = True)
seed = 22
np.random.seed(seed)
random.seed(seed)

### PARAMETERS
#depth = int(sys.argv[1])
depth = 1
Ntots = [20]
Nwarmups_percentages = [0.1]

for Ntot in Ntots:
    for Nwarmup_percentage in Nwarmups_percentages:
        start_time = time.time()
        Nwarmup = int(Nwarmup_percentage*Ntot)
        Nbayes = Ntot - Nwarmup
        backend = 'QISKIT'
        method = 'DIFF-EVOL'
        param_range = [0.1, np.pi]   # extremes where to search for the values of gamma and beta

        fidelities = []
        delta_Es  = []
        corr_lengths = []
        k1 = []
        k1_constant_kernel = []
        af_samples = []
        af_iters = []
        y_bests = []
        times = []
        average_distance_vectors = []
        std_energies = []

        ### CREATE GRAPH
        pos = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [0, 4]])

        G = nx.Graph()
        G.add_edges_from(pos)
        qaoa = qaoa_qiskit(G)
        gs_energy, gs_state, degeneracy = qaoa.calculate_gs_qiskit()



        ### CREATE GP AND FIT TRAINING DATA
        kernel =  ConstantKernel(1)* Matern(length_scale=0.11, length_scale_bounds=(1e-01, 100.0), nu=1.5)
        gp = MyGaussianProcessRegressor(kernel=kernel,
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
        k1 = [gp.kernel_.get_params()['k1']]*Nwarmup
        k1_constant_kernel = [gp.kernel_.get_params()['k1__constant_value']]*Nwarmup
        af_iters = [0]*Nwarmup
        x_best, y_best, _ = gp.get_best_point()
        y_bests = [0]*(Nwarmup - 1)
        y_bests.append(y_best)
        average_distance_vectors = [0]*Nwarmup
        std_energies = [0]*Nwarmup

        warmup_time = time.time() - start_time
        times = [0]*(Nwarmup -1)
        times.append(warmup_time)
        ### BAYESIAN OPTIMIZATION
        init_pos = [0.2, 0.2]*depth

        for i in range(Nbayes):
            next_point, results, average_norm_distance_vectors, std_population_energy, conv_flag = gp.bayesian_opt_step(init_pos, method)
            y_next_point = qaoa.expected_energy(next_point)
            #gp.kernel_.set_params(k2__length_scale_bounds = (0.1 + i/100 , 1000))

            fidelity = qaoa.fidelity_gs(next_point)
            delta_E = abs(y_next_point - gs_energy)
            corr_length = gp.kernel_.get_params()['k2__length_scale']
            k1_i = gp.kernel_.get_params()['k1']
            k1_constant_kernel_i = gp.kernel_.get_params()['k1__constant_value']
            X_train.append(next_point)
            y_train.append(y_next_point)
            fidelities.append(fidelity)
            delta_Es.append(delta_E)
            corr_lengths.append(corr_length)
            k1.append(k1_i)
            k1_constant_kernel.append(k1_constant_kernel_i)
            average_distance_vectors.append(average_norm_distance_vectors)
            std_energies.append(std_population_energy)
            best_x, best_y, _ = gp.get_best_point()
            y_bests.append(best_y)
            this_time = time.time() - start_time
            times.append(this_time)

            gp.fit(next_point, y_next_point)

        end_time = time.time()

        best_x, best_y, where = gp.get_best_point()
        X_train.append(best_x)
        y_train.append(best_y)
        fidelities.append(fidelities[where])
        delta_Es.append(delta_Es[where])
        corr_lengths.append(corr_lengths[where])
        k1_constant_kernel.append(k1_constant_kernel[where])
        y_bests.append(y_bests[where])
        average_distance_vectors.append(average_distance_vectors[where])
        std_energies.append(std_energies[where])
        times.append(times[-1])
        iter = range(Nwarmup + Nbayes)
        training_info = np.column_stack(([i for i in iter] + [where], X_train, y_train, fidelities, delta_Es, corr_lengths,  k1_constant_kernel, y_bests, average_distance_vectors, std_energies, times))
        np.savetxt('p={}_warmup={}_punti={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes), training_info, fmt = '%.d'+ ((len(training_info[0])-1)*' %.4f'))


