import numpy as np

from pulser import Register
from utils.optimize import scipy_minimize, bayes_opt
from utils.qaoa import quantum_loop, plot_distribution, func
from utils.graph import chair_graph

#Set parameters
depth = 1
x0 = np.array([1000,9000])    
reg, G = chair_graph(a = 10, draw = False)

flag_scipy = 0
flag_bayes = 1

if flag_scipy:
    res = scipy_minimize(G, 
                         func, 
                         reg,
                         x0 =x0,
                         depth =1,
                         method='Nelder-Mead') 
    
    count_dict = quantum_loop(res.x, reg)
    print('Convergence check: ', res.success)
    plot_distribution(count_dict)

if flag_bayes:
    res_bayes = bayes_opt(G, depth, reg, verbose = 1)
    plot_distribution(res_bayes)