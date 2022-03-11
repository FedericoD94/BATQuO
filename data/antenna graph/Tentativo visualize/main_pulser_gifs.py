import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.qaoa_pulser import *
from utils.gaussian_process import *
import time
from itertools import product

np.set_printoptions(precision = 4, suppress = True)

### TRAIN PARAMETERS
depth = 2
Nwarmup = 10
Nbayes = 50
method = 'DIFF-EVOL'
param_range = [100, 2000]   # extremes where to search for the values of gamma and beta
quantum_noise = False

file_name = 'p={}_punti={}_warmup={}_train={}.dat'.format(depth, Nwarmup + Nbayes, Nwarmup, Nbayes)

data = []
global_time = time.time()
results_structure = ['iter ', 'point ', 'energy ', 'fidelity ', 'corr_length ', 'const kernel ',
                    'std energies ', 'average distances ', 'nit ', 'time opt bayes ', 'time qaoa ', 'time opt kernel ', 'time step ']


### CREATE GRAPH AND REGISTER 
#pos = np.array([[0., 0.],[-4, -7],[4, -7],[8, 6],[-8, 6]])
pos = np.array([[0., 0.], [0, 10], [10,0], [10,10], [10,20],[20,10]])
               
qaoa = qaoa_pulser(pos, quantum_noise)
gs_en, gs_state, deg = qaoa.calculate_physical_gs()

configs = list(product(['0', '1'], repeat = 6))
x = [3000, 100]
C, evol_states = qaoa.get_evolution_states(x)

qaoa.plot_final_state_distribution(C)
print(len(evol_states))
exit()

solution = {'011011': -4}

which_states = evol_states[-2:]
for k in range(len(which_states)):
	res = {}
	for i, state in enumerate(which_states[k]):
		z = "".join(configs[i])
		res[z] = int(np.abs(state)**2*1024)

	C = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
	color_dict = {key: 'g' for key in C}
	for key in solution.keys():
		val = ''.join(str(key[i]) for i in range(len(key)))
		color_dict[val] = 'r'
	fig  = plt.figure(figsize=(12,6))
	plt.xlabel("bitstrings")
	plt.ylabel("counts")
	plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
	plt.xticks(rotation='vertical')
	plt.savefig('images/i_{}.png'.format(k))
	print('ciao')
	plt.close(fig)

		


