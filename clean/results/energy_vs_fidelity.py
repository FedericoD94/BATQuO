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
lines = ['.-', 'P-', 's-', '^-', 'D-']

fig  = plt.figure()

for i, depth in enumerate(depths):
    if depth ==1:
        divide = -77.8
    else:
        divide = -4
    A = np.loadtxt(f'p={depth}_warmup={Nwarmup}_train={Nbayes}.dat')

    energy_pos = 1 + 2*depth
    fidelity_pos = energy_pos + 4
    
    energies[i] = A[:Ntot, energy_pos]/(divide)
    fidelities[i] =  A[:Ntot, fidelity_pos]
    
    
    plt.scatter(energies[i], fidelities[i], label = f'p = {depth}')

    
plt.title('Fidelity vs energy')
plt.xlabel('Energy')
plt.ylabel('Fidelity')
plt.legend()
plt.suptitle('Different P, Ntot = {}, warmup = {}%'.format(Ntot, Nwarmup))
plt.tight_layout()
plt.show()

    





