import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

depths = [1,2,3]
Nwarmup = 10
Nbayes = 100
Ntot = Nwarmup + Nbayes

fidelities = np.zeros((len(depths), Ntot))
energies = np.zeros((len(depths), Ntot))
corr_lengths = np.zeros((len(depths), Ntot))
average_distance_vector = np.zeros((len(depths), Ntot))
std_energies = np.zeros((len(depths), Ntot))
energies_best = np.zeros((len(depths), Ntot))
fidelities_best = np.zeros((len(depths), Ntot))
iterations = np.zeros((len(depths), Ntot))
variances = np.zeros((len(depths), Ntot))


energies_best[0] = 0
fidelities_best[0] = 0



for i, depth in enumerate(depths):
    energy_pos = 1 + 2*depth
    fidelity_pos = energy_pos + 4
    variance_pos = energy_pos + 3
    corr_length_pos = energy_pos + 7
    std_energies_pos = energy_pos + 9
    avg_dist_vect_pos = energy_pos + 10
    n_it_pos = energy_pos + 11
    
    if depth ==1:
        divide = -77.8
    else:
        divide = -4
    
    a = pd.read_csv('cici.dat')
    print(a['energy'])
    exit()
    A = np.loadtxt(f'p={depth}_warmup={Nwarmup}_train={Nbayes}.dat')
    energies_best[i, 0] = A[0, energy_pos]/(divide)
    fidelities_best[i, 0] = A[0, fidelity_pos]
    for k, a in enumerate(A[1:Ntot, energy_pos]/(divide)):
    	if a > energies_best[i, k]:
    		energies_best[i, k+1] = a
    	else:
    		energies_best[i, k+1] = energies_best[i, k]
    
    for k, a in enumerate(A[1:Ntot, fidelity_pos]):
    	if a > fidelities_best[i, k]:
    		fidelities_best[i, k+1] = a
    	else:
    		fidelities_best[i, k+1] = fidelities_best[i, k]
    
    energies[i] = A[:Ntot, energy_pos]/(divide)
    fidelities[i] =  A[:Ntot, fidelity_pos]
    variances[i] = A[:Ntot, variance_pos]
    corr_lengths[i] = A[:Ntot, corr_length_pos]
    std_energies[i] = A[:Ntot, std_energies_pos]
    average_distance_vector[i] = A[:Ntot, avg_dist_vect_pos]
    iterations[i] = A[:Ntot, n_it_pos]

lines = ['.-', 'P-', 's-', '^-', 'D-']
markersize = 2
linewidth = 0.8
x = np.arange(0, len(fidelities[0]))
fig, axis = plt.subplots(2,4, figsize = (12, 7))
for i in range(len(depths)):
    axis[0, 0].plot(x, fidelities[i],lines[i],  label = str(depths[i]),markersize = markersize,  linewidth = linewidth)
    axis[0, 0].set_ylabel('Fidelity')
    axis[0, 0].set_xlabel('Steps')
    
    axis[0, 1].plot(x, energies[i], lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[0, 1].set_ylabel('Energy (ratio over ground state)')
    axis[0, 1].set_xlabel('Steps')
    
    axis[0, 2].plot(x, corr_lengths[i], lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[0, 2].set_ylabel('Corr lengths')
    axis[0, 2].set_xlabel('Steps')
    
    axis[0, 3].plot(x, np.log(average_distance_vector[i]), lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[0, 3].set_ylabel('log avg distance vector')
    axis[0, 3].set_xlabel('Steps')
    axis[0, 3].axhline(y = np.log(0.01), c = 'k')
    

    axis[1, 0].plot(x, fidelities_best[i], linewidth = 2, label = str(depths[i]))
    axis[1, 0].set_ylabel('Fidelity best')
    axis[1, 0].set_xlabel('Steps')
    
    axis[1, 1].plot(x, energies_best[i],  linewidth = 2,label = str(depths[i]))
    axis[1, 1].set_ylabel('Energy best (ratio)')
    axis[1, 1].set_xlabel('Steps')
    
    axis[1, 2].plot(x, iterations[i], lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[1, 2].set_ylabel('iterations')
    axis[1, 2].set_xlabel('Steps')
    
    axis[1, 3].plot(x, np.log(variances[i]), lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[1, 3].set_ylabel('log variances')
    axis[1, 3].set_xlabel('Steps')
    

axis[0, 0].legend()
axis[0, 1].legend()
axis[1, 0].legend()
axis[1, 1].legend()
plt.suptitle('Different P, Ntot = {}, warmup = {}%'.format(Ntot, Nwarmup))
plt.tight_layout()
plt.show()

    





