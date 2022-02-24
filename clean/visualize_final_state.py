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


### CREATE GRAPH AND QAOA INSTANCE
#pos = np.array([[0., 0.],[-4, -7],[4, -7],[8, 6],[-8, 6]])
               
qaoa = qaoa_pulser(depth, 'chair', quantum_noise = ('SPAM', 'dephasing', 'doppler', 'amplitude'))
gs_en, gs_state, deg = qaoa.calculate_physical_gs()

params = [192, 427]
results = qaoa.apply_qaoa(params, show = True)
print(results[:-1])
plt.title('cv')