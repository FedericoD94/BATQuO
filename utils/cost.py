import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2

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
    C = quantum_loop(param, r=reg)
    cost = get_cost(C,G)
    return cost
