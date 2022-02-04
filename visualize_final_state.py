import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.qaoa_pulser import *
from utils.gaussian_process import *
import time
import random
import datetime
import sys


np.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(DEFAULT_PARAMS['seed'])
random.seed(DEFAULT_PARAMS['seed'])
### TRAIN PARAMETERS
depth = 1
Nwarmup = 20
Ntot = 200
Nbayes = Ntot-Nwarmup
method = 'DIFF-EVOL'
angles_bounds = np.array([[100, 3000],[100, 3000]])  # extremes where to search for the values of gamma and beta
global_time = time.time()
quantum_noise = None #'SPAM', 'doppler', 'dephasing', 'amplitude' or a TUPLE of more than one or None



### CREATE GRAPH AND QAOA INSTANCE
#pos = np.array([[0., 0.],[-4, -7],[4, -7],[8, 6],[-8, 6]])
pos = np.array([[0., 0.], [0, 10], [10,0], [10,10], [10,20],[20,10]])
               
qaoa = qaoa_pulser(depth, angles_bounds, pos, quantum_noise)
gs_en, gs_state, deg = qaoa.calculate_physical_gs()

params = [1000,1000]
results = qaoa.apply_qaoa(params, show = True)