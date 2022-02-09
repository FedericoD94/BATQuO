import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#depths = [1,2,3,4]
depths = [5, 6, 7, 8, 9]
Nwarmup_percentage = .1
Ntots = [100, 150, 200, 250]

fidelities = np.zeros((len(depths), len(Ntots)))
energies = np.zeros((len(depths), len(Ntots)))
corr_lengths = np.zeros((len(depths), len(Ntots)))
average_distance_vector = np.zeros((len(depths), len(Ntots)))


for i, depth in enumerate(depths):
	for k, Ntot in enumerate(Ntots):
		Nwarmup =int(Nwarmup_percentage * Ntot)
		Nbayes = Ntot - Nwarmup
		A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
	
		energies[i, k] = A[-1, -9]
		fidelities[i, k] =  A[-1, -8]

x = np.arange(0, len(Ntots))
x = Ntots
fig, axis = plt.subplots(2, figsize = (8, 6))
for i in range(len(depths)):
    axis[0].plot(x, fidelities[i], 'o-', label = str(depths[i]))
    axis[0].set_ylabel('Fidelity')
    axis[0].set_xlabel('Ntot')
    axis[0].set_xticklabels(Ntots)
    axis[0].set_xticks(Ntots)
    
    axis[1].plot(x, energies[i],  'o-',  label = str(depths[i]))
    axis[1].set_ylabel('Energy')
    axis[1].set_xlabel('Ntot')
    axis[1].set_xticklabels(Ntots)
    axis[1].set_xticks(Ntots)
    
    

axis[0].legend()
axis[1].legend()
plt.suptitle('Different P, Ntot = {}, warmup = {}%'.format(Ntot, Nwarmup_percentage))
plt.tight_layout()
plt.show()

    





