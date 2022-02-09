import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

depths = [1,2,3,4, 5]
Nwarmup = 20
Ntots = [200]

fidelities = np.zeros((len(depths), len(Ntots)))
energies = np.zeros((len(depths), len(Ntots)))
corr_lengths = np.zeros((len(depths), len(Ntots)))
average_distance_vector = np.zeros((len(depths), len(Ntots)))
total_time =  np.zeros((len(depths), len(Ntots)))
step_time =  np.zeros((len(depths), len(Ntots)))
save = True

for i, depth in enumerate(depths):
	for k, Ntot in enumerate(Ntots):
		Nbayes = Ntot - Nwarmup
		A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
	
		energies[i] = A[-1, -14]
		fidelities[i] =  A[-1, -12]
		corr_lengths[i] = A[-1, -9]
		total_time[i] = np.sum(A[:-1, -1])/60
		step_time[i] = np.average(A[:-1, -1])/60
		

x = np.arange(0, len(Ntots))
x = depths
fig, axis = plt.subplots(4, figsize = (8, 6))
marker_shape = ['o-', 's-', '^-', 'P-']
for k in range(len(Ntots)):
    axis[0].plot(x, fidelities[:,k],marker_shape[k], label = str(Ntots[k]))
    axis[0].set_ylabel('Fidelity')
    axis[0].set_xlabel('P')
    axis[0].set_xticklabels(depths)
    axis[0].set_xticks(depths)
    
    axis[1].plot(x, energies[:,k],  marker_shape[k],   label = str(Ntots[k]))
    axis[1].set_ylabel('Energy')
    axis[1].set_xlabel('P')
    axis[1].set_xticklabels(depths)
    axis[1].set_xticks(depths)


    axis[2].plot(x, total_time[:,k], marker_shape[k],  label = str(Ntots[k]))
    axis[2].set_ylabel('Time (minutes)')
    axis[2].set_xlabel('P')
    axis[2].set_xticklabels(depths)
    axis[2].set_xticks(depths)
    #axis[2].set_yscale('log')
    
    axis[3].plot(x, step_time[:,k], marker_shape[k],  label = str(Ntots[k]))
    axis[3].set_ylabel('Average Step (minutes)')
    axis[3].set_xlabel('P')
    axis[3].set_xticklabels(depths)
    axis[3].set_xticks(depths)
    #axis[3].set_yscale('log')

axis[0].legend()
axis[1].legend()
axis[2].legend()
axis[3].legend()
plt.suptitle('Different Ntot,depths'.format(Ntot))
plt.tight_layout()
if save:
    plt.savefig('Multiple_best.png')
    
plt.show()



    





