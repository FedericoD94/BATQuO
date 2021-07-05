import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2

def pos_to_graph(pos, d = Chadoq2.rydberg_blockade_radius(1) ):
    g=igraph.Graph()
    edges=[]
    for n in range(len(pos)-1):
        for m in range(n+1, len(pos)):
            pwd = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
            if pwd < d:
                edges.append([n,m])                         # Below rbr, vertices are connected
    g.add_vertices(len(pos))
    g.add_edges(edges)
    return g

def chair_graph(a = 10):
    '''
    Cretes the chair graph
    
    a: int, lattice spacing
    '''
    pos = np.array([[-a,-a], [0,-a],[-a,0],[0,0],[a,0],[0,a]]) #graph  with  d = Chadoq2.rydberg_blockade_radius(1)                                   # number of layers and parameter values               
    G = pos_to_graph(pos)
    qubits = dict(enumerate(pos))
    reg = Register(qubits)
    
    return reg, G

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