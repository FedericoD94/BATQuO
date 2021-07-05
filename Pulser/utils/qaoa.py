import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2
from utils.graph import get_cost_state

def quantum_loop(param, r):
    seq = Sequence(r, Chadoq2)
    seq.declare_channel('ch0','rydberg_global')
    middle = int(len(param)/2)
    param = np.array(param)*1 #wrapper
    t = param[:middle] #associated to H_c
    tau = param[middle:] #associated to H_0
    p = len(t)
    for i in range(p):
        pulse_1 = Pulse.ConstantPulse(tau[i], 1., 0, 0) # H_M
        pulse_2 = Pulse.ConstantPulse(t[i], 1., 1., 0) # H_M + H_c
        seq.add(pulse_1, 'ch0')
        seq.add(pulse_2, 'ch0')
    seq.measure('ground-rydberg')
    simul = Simulation(seq, sampling_rate=.01)
    results = simul.run()
    count_dict = results.sample_final_state(N_samples=1000) #sample from the state vector
    return count_dict

def plot_distribution(C):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    color_dict = {key: 'g' for key in C}
    indexes = ['011011']  # MIS indexes
    for i in indexes:
        color_dict[i] = 'red'
    plt.figure(figsize=(12,6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
    plt.xticks(rotation='vertical')
    plt.show()



def func(param,*args):
    G = args[0]
    C = quantum_loop(param, r=args[1])
    cost = get_cost_state(C,G)
    return cost

