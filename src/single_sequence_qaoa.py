import time
import random
import datetime
import sys
import numpy as np
from utils.parameters import parse_command_line
from utils.bayesian_optimization import *
from utils.default_params import *
from utils.qaoa_pulser import *


np.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(threshold=sys.maxsize)


'''
Runs a qaoa sequence of angles for num_repetition times with different seeds
Saves the result of each simulation in a pdf dataframe.
'''
###### TRAIN PARAMETERS ##################

args = parse_command_line()

seed = args.seed
fraction_warmup = args.fraction_warmup
depth = args.p
num_nodes = args.num_nodes
nwarmup = args.nwarmup
nbayes = args.nbayes
average_connectivity = args.average_connectivity
quantum_noise =  args.quantum_noise
type_of_graph = args.type_of_graph
lattice_spacing = args.lattice_spacing
verbose_ = args.verbose
kernel_choice = args.kernel

np.random.seed(seed)
random.seed(seed)

####### CREATE BAYES OPT INSTANCE ########
angles = [100,100,800,504,108,201]
depth = int(len(angles)/2)
num_repetitions = 1
df_results = []

def define_angles_boundaries( depth):
    if depth == 1:
        angle_bounds = [[100, 2000], [100, 2000]]
    if depth == 2:
        angle_bounds = [[100, 1500] for _ in range(depth*2)]
    if depth == 3:
        angle_bounds = [[100, 1000] for _ in range(depth*2)]
    else:
        angle_bounds = [[100, 800] for _ in range(depth*2)]
        
    return np.array(angle_bounds)
        
quantum_noise = 'all'
angles_bounds = define_angles_boundaries(depth)
qaoa = qaoa_pulser(depth = depth, 
                   angles_bounds = angles_bounds,
                    type_of_graph = type_of_graph, 
                    lattice_spacing = lattice_spacing, 
                    seed = seed, 
                    quantum_noise = quantum_noise)
res =  qaoa.apply_qaoa(angles)

df_results.append(res)
    
all_angles = pd.DataFrame.from_dict(df_results)
all_angles.to_pickle(f'output/sequence_100_100_800_504_108_200_noise_result_seed_{seed}')
