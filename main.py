import numpy as np
from scipy.optimize import minimize

from pulser import Register
from utils.quantumloop import quantum_loop, func
from utils.graph import pos_to_graph
from utils.plotdistribution import plot_distribution


#Set parameters
pos = np.array([[-10,-10], [0,-10],[-10,0],[0,0],[10,0],[0,10]]) #graph  with  d = Chadoq2.rydberg_blockade_radius(1)
x0 = np.array([1000,9000])                                       # number of layers and parameter values               
G = pos_to_graph(pos)
qubits = dict(enumerate(pos))
reg = Register(qubits)
#reg.draw()
res = minimize(func, args=(G, reg), x0 =x0,method='Nelder-Mead', tol=1e-5,options = {'maxiter': 15}) #optimization method
count_dict = quantum_loop(res.x, reg)
plot_distribution(count_dict)

