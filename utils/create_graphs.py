import networkx as nx
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

def create_random_graph(num_nodes, average_connectivity, draw=False):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges_graph = np.array(list(combinations(range(num_nodes), 2)))
    s = np.where(np.random.binomial(1, average_connectivity, len(edges_graph)))[0]
    G.add_edges_from(edges_graph[s])

    if draw:
        nx.draw(G, with_labels=True)
        plt.show()
    return G
