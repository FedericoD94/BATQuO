import numpy as np
from scipy.optimize import minimize

from pulser import Register
from utils.qaoa import *
from utils.graph import *

#Set parameters
x0 = np.array([1000,9000])    
reg, G = chair_graph(a = 10)


#reg.draw()
res = minimize(func, args=(G, reg), x0 =x0,method='Nelder-Mead', tol=1e-5,options = {'maxiter': 15}) #optimization method
count_dict = quantum_loop(res.x, reg)
plot_distribution(count_dict)
print('cia')

