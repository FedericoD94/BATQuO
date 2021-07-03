import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2

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
    simul = Simulation(seq, sampling_rate=.001)
    results = simul.run()
    count_dict = results.sample_final_state(N_samples=1000) #sample from the state vector
    return count_dict

def get_cost_colouring(z,G,penalty=10):
    """G: the graph (igraph)
       z: a binary colouring
       returns the cost of the colouring z, depending on the adjacency of the graph"""
    cost = 0
    A = G.get_adjacency()
    z = np.array(tuple(z),dtype=int)
    for i in range(len(z)):
        for j in range(i,len(z)):
            cost += A[i][j]*z[i]*z[j]*penalty # if there's an edge between i,j and they are both in |1> state.

    cost -= np.sum(z) #to count for the 0s instead of the 1s
    return cost

def get_cost_state(counter,G):
    cost = 0
    for key in counter.keys():
        cost_col = get_cost_colouring(key,G)
        cost += cost_col * counter[key]
    return cost / sum(counter.values())

def func(param,*args):
    G = args[0]
    C = quantum_loop(param, r=args[1])
    cost = get_cost_state(C,G)
    return cost

