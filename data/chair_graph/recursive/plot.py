import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#depths = [1,2,3,4]
depths = np.arange(1, 19, 1)
Nwarmup= 10
Ntot = 160

fidelities = np.zeros(len(depths))
energies = np.zeros(len(depths))
times = np.zeros(len(depths))


for i, depth in enumerate(depths):
		Nbayes = Ntot - Nwarmup
		A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
	
		energies[i] = A[-1, -11]
		fidelities[i] =  A[-1, -10]
		times[i] = np.sum(A[:-1, -1])/60

x = depths

fig, axis = plt.subplots(3, figsize = (8, 6))
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

axis[2].plot(x, times,  'o-',  label = str(depths[i]))
axis[2].set_ylabel('Times')
axis[2].set_xlabel('P')
axis[2].set_xticklabels(x)
axis[2].set_xticks(x)
    
    

axis[0].legend()
axis[1].legend()
plt.suptitle('Reursive approach p = {}, Ntot = {}, warmup = {}%'.format(depth, Ntot, Nwarmup))
plt.tight_layout()
plt.show()
