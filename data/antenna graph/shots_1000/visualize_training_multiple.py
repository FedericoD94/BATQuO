import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

depths = [1, 2, 3, 4, 5]
Nwarmup_percentage = .1
Ntot = 100

fidelities = np.zeros((len(depths), Ntot))
energies = np.zeros((len(depths), Ntot))
corr_lengths = np.zeros((len(depths), Ntot))
average_distance_vector = np.zeros((len(depths), Ntot))
std_energies = np.zeros((len(depths), Ntot))
energies_best = np.zeros((len(depths), Ntot))
fidelities_best = np.zeros((len(depths), Ntot))

energies_best[0] = 0
fidelities_best[0] = 0

for i, depth in enumerate(depths):
    Nwarmup =int(Nwarmup_percentage * Ntot)
    Nbayes = Ntot - Nwarmup
    A = np.loadtxt('p={}_warmup={}_punti={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
    
    energies[i] = A[:-1, -11]
    for i in range(len(energies[i])):
    	if i < energies_best[i]
    fidelities[i] =  A[:-1, -10]
    corr_lengths[i] = A[:-1, -9]s
    average_distance_vector[i] = A[:-1, -6]
    std_energies[i] = A[:-1, -7]


x = np.arange(0, len(fidelities[0]))
fig, axis = plt.subplots(2,  figsize = (15, 9))
for i in range(len(depths)):
    axis[0].plot(x, fidelities[i], linewidth = 2,label = str(depths[i]))
    axis[0].set_ylabel('Fidelity')
    axis[0].set_xlabel('Steps')
    
    axis[1].plot(x, energies[i],  linewidth = 2,label = str(depths[i]))
    axis[1].set_ylabel('Energy')
    axis[1].set_xlabel('Steps')

    

axis[0].legend()
axis[1].legend()
plt.suptitle('P = {}, Ntot = {} different warmup percentages'.format(depth, Ntot))
plt.tight_layout()
plt.show()

    





