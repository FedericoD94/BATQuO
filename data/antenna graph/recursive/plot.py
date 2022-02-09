import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#depths = [1,2,3,4]
depths = [1, 2, 3, 4,5, 6, 7, 8, 9]
Nwarmup_percentage = .1
Ntot = 100

fidelities = np.zeros(len(depths))
energies = np.zeros(len(depths))


for i, depth in enumerate(depths):
		Nwarmup =int(Nwarmup_percentage * Ntot)
		Nbayes = Ntot - Nwarmup
		A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
	
		energies[i] = A[-1, -11]
		fidelities[i] =  A[-1, -10]

x = depths
fig, axis = plt.subplots(2, figsize = (8, 6))
axis[0].plot(x, fidelities, 'o-', label = str(depths[i]))
axis[0].set_ylabel('Fidelity')
axis[0].set_xlabel('P')
axis[0].set_xticklabels(x)
axis[0].set_xticks(x)

axis[1].plot(x, energies,  'o-',  label = str(depths[i]))
axis[1].set_ylabel('Energy')
axis[1].set_xlabel('P')
axis[1].set_xticklabels(x)
axis[1].set_xticks(x)
    
    

axis[0].legend()
axis[1].legend()
plt.suptitle('Reursive approach p = {}, Ntot = {}, warmup = {}%'.format(depth, Ntot, Nwarmup_percentage))
plt.tight_layout()
plt.show()
