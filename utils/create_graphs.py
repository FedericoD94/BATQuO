import networkx as nx
from networkx.readwrite import json_graph
from networkx.generators.random_graphs import random_regular_graph
from itertools import combinations
import json
import numpy as np
import matplotlib.pyplot as plt
from networkx.linalg.graphmatrix import adjacency_matrix


def create_random_graph(num_nodes, average_connectivity, name_plot=False):
    G = nx.Graph()

    G.add_nodes_from(range(num_nodes))
    edges_graph = np.array(list(combinations(range(num_nodes), 2)))
    s = np.where(np.random.binomial(1, average_connectivity, len(edges_graph)))[0]
    G.add_edges_from(edges_graph[s])
    if name_plot:
        graph_name_file = str(draw)
        _plot_graph(G, graph_name_file)

    return G


def create_chair_graph(name_plot=False):
    num_nodes = 6
    G = nx.Graph()
    pos = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [0,5]])
    G.add_edges_from(pos)

    if name_plot:
        graph_name_file = "chair"
        _plot_graph(G, graph_name_file)

    return G


def create_random_regular_graph(num_nodes, degree=3, seed=1, name_plot=False):
    G = random_regular_graph(degree, num_nodes, seed=seed)
    if name_plot:
        graph_name_file = str(name_plot)
        _plot_graph(G, graph_name_file)
    return G


def _plot_graph(G, graph_name_file):
    num_nodes = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos=pos, with_labels=True)
    plt.savefig(graph_name_file + ".pdf")
    A = nx.to_numpy_array(G, nodelist=range(num_nodes), dtype=int)
    np.savetxt(graph_name_file + "_adj_mat.dat", A, fmt="%d")