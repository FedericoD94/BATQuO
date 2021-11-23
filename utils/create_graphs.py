import networkx as nx
from networkx.readwrite import json_graph
from itertools import combinations
import json
import numpy as np
import matplotlib.pyplot as plt

def create_random_graph(num_nodes, average_connectivity, draw=False):
    G = nx.Graph()

    G.add_nodes_from(range(num_nodes))
    edges_graph = np.array(list(combinations(range(num_nodes), 2)))
    s = np.where(np.random.binomial(1, average_connectivity, len(edges_graph)))[0]
    G.add_edges_from(edges_graph[s])
    graph_name_file = draw

    nx.draw(G, with_labels=True)
    plt.savefig(graph_name_file + ".pdf")
    nx.write_edgelist(G, graph_name_file + ".edgelist", data=True)
    nx.write_edgelist(G, graph_name_file + "false.edgelist", data=False)
    mapping = dict([[j,j] for j in range(num_nodes)])
    H =  nx.relabel_nodes(G, mapping)
    data_graph = json_graph.node_link_data(H)
    out_file = open(graph_name_file + ".json", "w")
    json_object = json.dump(data_graph, out_file)
    out_file.close()

    return G


def create_chair_graph(draw=False):
    num_nodes = 6
    G = nx.Graph()
    pos = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [0,5]])
    G.add_edges_from(pos)
    graph_name_file = "chair"

    nx.draw(G, with_labels=True)
    plt.savefig(graph_name_file + ".pdf")
    nx.write_edgelist(G, graph_name_file + ".edgelist", data=True)
    nx.write_edgelist(G, graph_name_file + "false.edgelist", data=False)
    mapping = dict([[j,j] for j in range(num_nodes)])
    H =  nx.relabel_nodes(G, mapping)
    data_graph = json_graph.node_link_data(H)
    out_file = open(graph_name_file + ".json", "w")
    json_object = json.dump(data_graph, out_file)
    out_file.close()

    return G
