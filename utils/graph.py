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
