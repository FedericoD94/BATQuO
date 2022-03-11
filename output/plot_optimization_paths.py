import numpy as np
import matplotlib.pyplot as plt
#from .src.utils.default_params import DEFAULT_PARAMS

how_many = 5
from_where = 15
p = 4
nbayes = 100
nwarmup = 10
ntot = nbayes + nwarmup

folder_name = f'lfgbs_p_{p}_punti_{ntot}_warmup_{nwarmup}_train_{nbayes}_trial_1_graph_12/'
#folder_name = f'p_3/'
# min_x = DEFAULT_PARAMS['length_scale_bounds'][0]
# max_x = DEFAULT_PARAMS['length_scale_bounds'][1]
# min_y = DEFAULT_PARAMS['constant_bounds'][0]
# max_y = DEFAULT_PARAMS['constant_bounds'][1]
min_x = np.log(0.01)
max_x = np.log(100)
min_y = np.log(0.01)
max_y = np.log(100)

for i in range(from_where, from_where + how_many+1):
    fig = plt.figure()

    A = np.loadtxt(folder_name + "step_{}_kernel_opt.dat".format(i))
    B = np.loadtxt(folder_name + "step_{}_likelihood_grid.dat".format(i))
    cut_off = np.min(A)-100
    B[B<cut_off] = cut_off
    plt.imshow(B, extent = (min_x, max_x, min_y, max_y), origin='lower')
    plt.colorbar()
    plt.xlabel('Const')
    plt.ylabel('corre length')
    path_x = A[:,1]
    path_y = A[:,2]
    plt.plot(path_x, path_y, 'o-')
    max_val  = np.max(B)
    plt.title(f'Step{i}, max: {max_val}')

    max_where = np.argmax(B)
    for j in range(len(A)): 
        plt.annotate(str(j), xy = (path_x[j], path_y[j]))
    plt.show()
    
    
    
