import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

depths = [1,2,3,4,5,6,7,8,9]
Nwarmup_percentages = [.1,.2,.4,.5]
Ntots = [100, 150, 200, 250]

Nwarmups = np.zeros((len(depths), len(Nwarmup_percentages)))
times = np.zeros((len(depths), len(Ntots), len(Nwarmup_percentages)))
fidelities = np.zeros((len(depths), len(Ntots), len(Nwarmup_percentages)))
delta_energies = np.zeros((len(depths), len(Ntots), len(Nwarmup_percentages)))

for k, depth in enumerate(depths):
    last_time = 0
    for i, Ntot in enumerate(Ntots):
        for j, Nwarmup_percentage in enumerate(Nwarmup_percentages):
            Nwarmup =int(Nwarmup_percentage * Ntot)
            Nbayes = Ntot - Nwarmup
            Nwarmups[k, j] = Nwarmup
            try:
            	A = np.loadtxt('p={}_punti={}_warmup={}_train={}.dat'.format(depth, Ntot, Nwarmup, Nbayes))
            except:
            	continue
            fidelities[k, i, j] = A[-1, -10]
            delta_energies[k, i, j] = A[-1, -11]
            times[k, i, j] = np.sum(A[:-1, -1])/60
            

dfs_time, dfs_energies, dfs_fidelities = [], [], []
with open('Training times.dat', 'wb') as f:
    for i, depth in enumerate(depths):
        df = pd.DataFrame(times[i],
                      index=pd.Index(Ntots, name='p = {}'.format(depth)),
                      columns=Nwarmup_percentages)
        dfs_time.append(df)
        
        df = pd.DataFrame(fidelities[i],
                      index=pd.Index(Ntots, name='p = {}'.format(depth)),
                      columns=Nwarmup_percentages)
        dfs_fidelities.append(df)
                
        df = pd.DataFrame(delta_energies[i],
                      index=pd.Index(Ntots, name='p = {}'.format(depth)),
                      columns=Nwarmup_percentages)
        dfs_energies.append(df)
time_df = pd.concat(dfs_time, keys  = depths)
time_df.to_excel('times.xlsx')

fidelities_df = pd.concat(dfs_fidelities, keys  = depths)
fidelities_df.to_excel('fidelities.xlsx')

energies_df = pd.concat(dfs_energies, keys  = depths)
energies_df.to_excel('energies.xlsx')


