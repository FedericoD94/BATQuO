import numpy as np
import igraph
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2

from scipy.optimize import minimize


def pos_to_graph(pos, d = Chadoq2.rydberg_blockade_radius(1)): #d is the rbr
    g = nx.Graph()
    edges=[]
    for n in range(len(pos)-1):
        for m in range(n+1, len(pos)):
            pwd = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
            if pwd < d:
                edges.append([n,m]) # Below rbr, vertices are connected
    g.add_nodes_from(range(len(pos)))
    g.add_edges_from(edges)
    return g


def quantum_loop(param, r):
    seq = Sequence(r, Chadoq2)
    seq.declare_channel('ch0','rydberg_global')
    middle = int(len(param)/2)
    param = np.array(param)*1 #wrapper
    t = param[:middle] #associated to H_c
    tau = param[middle:] #associated to H_0
    p = len(t)
    for i in range(p):
        ttau = int(tau[i]) - int(tau[i]) % 4
        tt = int(t[i]) - int(t[i]) % 4
        pulse_1 = Pulse.ConstantPulse(ttau, 1., 0, 0) # H_M
        pulse_2 = Pulse.ConstantPulse(tt, 1., 1, 0) # H_M + H_c
        seq.add(pulse_1, 'ch0')
        seq.add(pulse_2, 'ch0')
    seq.measure('ground-rydberg')
    simul = Simulation(seq, sampling_rate=.01)
    results = simul.run()
    count_dict = results.sample_final_state(N_samples=1000) #sample from the state vector
    return count_dict, results.states


def plot_distribution(C):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    color_dict = {key: 'g' for key in C}
    indexes = ['01011', '00111']  # MIS indexes
    for i in indexes:
        color_dict[i] = 'red'
    plt.figure(figsize=(12,6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
    plt.xticks(rotation='vertical')
    plt.show()


def get_cost_colouring(z,G,penalty=10):
    """G: the graph (igraph)
       z: a binary colouring
       returns the cost of the colouring z, depending on the adjacency of the graph"""
    cost = 0
    A = np.array(adjacency_matrix(G).todense())
    z = np.array(tuple(z),dtype=int)
    for i in range(len(z)):
        for j in range(i,len(z)):
            cost += A[i,j]*z[i]*z[j]*penalty # if there's an edge between i,j and they are both in |1> state.

    cost -= np.sum(z) #to count for the 0s instead of the 1s
    return cost


def get_cost(counter,G):
    cost = 0
    for key in counter.keys():
        cost_col = get_cost_colouring(key,G)
        cost += cost_col * counter[key]
    return cost / sum(counter.values())


def apply_qaoa(param, reg, G):
    C, _= quantum_loop(param, r=reg)
    cost = get_cost(C, G)
    return cost
