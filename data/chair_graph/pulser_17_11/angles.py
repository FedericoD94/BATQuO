import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


depth = 3
Nwarmup = 20
Ntot =  300

Nbayes = Ntot - Nwarmup
A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))

gammas = A[:Ntot, 1:2*depth+1:2]
betas = A[:Ntot, 2:2*depth+1:2]
        
x = np.arange(0, Ntot)
fig, axis = plt.subplots(2, figsize = (8, 6))
marker_shape = ['o', 's', '^', 'P']
for k in range(depth):
    x = range(len(gammas[:, k]))
    axis[0].plot(x, gammas[:, k],marker_shape[k], label = str(k+1))
    axis[0].set_ylabel('Gamma')
    axis[0].set_xlabel('Step')
    
    x = range(len(betas[:, k]))    
    axis[1].plot(x, betas[:, k],  marker_shape[k],   label = str(k+1))
    axis[1].set_ylabel('Beta')
    axis[1].set_xlabel('Step')




axis[0].legend()
axis[1].legend()
plt.suptitle('gamma beta')
plt.tight_layout()
plt.show()

    





