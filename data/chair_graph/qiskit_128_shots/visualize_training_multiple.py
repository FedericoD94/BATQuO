import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

depths = [1,2,3,4]
depths = [5, 6, 7, 8, 9]
Nwarmup_percentage = .1
Ntot = 150

fidelities = np.zeros((len(depths), Ntot))
energies = np.zeros((len(depths), Ntot))
corr_lengths = np.zeros((len(depths), Ntot))
average_distance_vector = np.zeros((len(depths), Ntot))
std_energies = np.zeros((len(depths), Ntot))
energies_best = np.zeros((len(depths), Ntot))
fidelities_best = np.zeros((len(depths), Ntot))
iterations = np.zeros((len(depths), Ntot))

energies_best[0] = 0
fidelities_best[0] = 0

for i, depth in enumerate(depths):
    Nwarmup =int(Nwarmup_percentage * Ntot)
    Nbayes = Ntot - Nwarmup
    A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
    energies_best[i, 0] = A[0, -11]
    fidelities_best[i, 0] = A[0, -10]
    for k, a in enumerate(A[1:-1, -11]):
    	if a < energies_best[i, k]:
    		energies_best[i, k+1] = a
    	else:
    		energies_best[i, k+1] = energies_best[i, k]
    
    for k, a in enumerate(A[1:-1, -10]):
    	if a > fidelities_best[i, k]:
    		fidelities_best[i, k+1] = a
    	else:
    		fidelities_best[i, k+1] = fidelities_best[i, k]
    
    energies[i] = A[:-1, -11]
    fidelities[i] =  A[:-1, -10]
    corr_lengths[i] = A[:-1, -9]
    average_distance_vector[i] = A[:-1, -6]
    std_energies[i] = A[:-1, -7]
    iterations[i] = A[:-1, -5]

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
    axis[0, 1].set_ylabel('Energy')
    axis[0, 1].set_xlabel('Steps')
    
    axis[0, 2].plot(x, corr_lengths[i], lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[0, 2].set_ylabel('Corr lengths')
    axis[0, 2].set_xlabel('Steps')
    
    axis[0, 3].plot(x, np.log(average_distance_vector[i]), lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[0, 3].set_ylabel('avg distance vector')
    axis[0, 3].set_xlabel('Steps')
    axis[0, 3].axhline(y = np.log(0.01), c = 'k')
    

    axis[1, 0].plot(x, fidelities_best[i], linewidth = 2, label = str(depths[i]))
    axis[1, 0].set_ylabel('Fidelity best')
    axis[1, 0].set_xlabel('Steps')
    
    axis[1, 1].plot(x, energies_best[i],  linewidth = 2,label = str(depths[i]))
    axis[1, 1].set_ylabel('Energy best')
    axis[1, 1].set_xlabel('Steps')
    
    axis[1, 2].plot(x, iterations[i], lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[1, 2].set_ylabel('iterations')
    axis[1, 2].set_xlabel('Steps')
    
    axis[1, 3].plot(x, np.log(std_energies[i]), lines[i], label = str(depths[i]), markersize = markersize,  linewidth = linewidth)
    axis[1, 3].set_ylabel('std energies')
    axis[1, 3].set_xlabel('Steps')
    

axis[0, 0].legend()
axis[0, 1].legend()
axis[1, 0].legend()
axis[1, 1].legend()
plt.suptitle('Different P, Ntot = {}, warmup = {}%'.format(Ntot, Nwarmup_percentage))
plt.tight_layout()
plt.show()

    





