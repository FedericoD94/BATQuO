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

    if isinstance(draw, str):
        nx.draw(G, with_labels=True)
        plt.savefig(draw + ".pdf")
        nx.write_edgelist(G, draw + ".edgelist", data=True)
        nx.write_edgelist(G, draw + "false.edgelist", data=False)
        mapping = dict([[j,j] for j in range(num_nodes)])
        H = H = nx.relabel_nodes(G, mapping)
        data_graph = json_graph.node_link_data(H)
        out_file = open(draw + ".json", "w")
        json_object = json.dump(data_graph, out_file)
        out_file.close()

    return G
