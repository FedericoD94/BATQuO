import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2

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
