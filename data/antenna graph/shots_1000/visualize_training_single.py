import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

depth = 7
Nwarmup_percentages = [.4]
Ntot = 150

fidelities = np.zeros((len(Nwarmup_percentages), Ntot))
energies = np.zeros((len(Nwarmup_percentages), Ntot))
corr_lengths = np.zeros((len(Nwarmup_percentages), Ntot))
average_distace_vector = np.zeros((len(Nwarmup_percentages), Ntot))
std_energies = np.zeros((len(Nwarmup_percentages), Ntot))


for i, Nwarmup_percentage in enumerate(Nwarmup_percentages):
    Nwarmup =int(Nwarmup_percentage * Ntot)
    Nbayes = Ntot - Nwarmup
    A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
    
    energies[i] = A[:-1, -9]
    fidelities[i] =  A[:-1, -8]
    corr_lengths[i] = A[:-1, -6]
    average_distace_vector = A[:-1, -3]
    std_energies = A[:-1, -2]


x = np.arange(0, len(fidelities[0]))
fig, axis = plt.subplots(2, 2)

for i in range(len(Nwarmup_percentages)):
    axis[0,0].plot(x, fidelities[i], linewidth = 2,label = str(Nwarmup_percentages[i]))
    axis[0,0].set_ylabel('Fidelity')
    axis[0,0].set_xlabel('Steps')
    axis[0,1].plot(x, energies[i],  linewidth = 2,label = str(Nwarmup_percentages[i]))
    axis[0,1].set_ylabel('Energy')
    axis[0,1].set_xlabel('Steps')
    
    axis[1,0].scatter(x, corr_lengths[i], linewidth = 2,label = str(Nwarmup_percentages[i]))
    axis[1,0].set_ylabel('Corr lengths')
    axis[1,0].set_xlabel('Steps')
    
    axis[1,1].scatter(x, average_distance_vector[i],  linewidth = 2, label = 'avg distance vector'))
    axis[1,1].scatter(x, std_energies[i],  linewidth = 2, label = 'avg distance vector'))
    axis[1,1].set_ylabel('diff evol')
    axis[1,1].set_xlabel('Steps')
    

axis[0, 0].legend()
axis[0, 1].legend()
axis[1, 0].legend()
axis[1,1].legend()
plt.suptitle('P = {}, Ntot = {} different warmup percentages'.format(depth, Ntot))
plt.tight_layout()
plt.show()

    





