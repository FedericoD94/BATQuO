import numpy as np
import igraph
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2
from itertools import product

from scipy.optimize import minimize
from qutip import *


omega = 1.
delta = 1.
U = 10

def pos_to_graph(pos, d = Chadoq2.rydberg_blockade_radius(omega)): #d is the rbr
    G = nx.Graph()
    edges=[]
    distances = []
    for n in range(len(pos)-1):
        for m in range(n+1, len(pos)):
            pwd = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
            distances.append(pwd)
            if pwd < d:
                edges.append([n,m]) # Below rbr, vertices are connected
    G.add_nodes_from(range(len(pos)))
    G.add_edges_from(edges)
    return G, np.array(distances)

def create_quantum_loop(param, r, time  = 3000):
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
        pulse_1 = Pulse.ConstantPulse(ttau, omega, 0, 0) # H_M
        pulse_2 = Pulse.ConstantPulse(tt, delta, 1, 0) # H_M + H_c
        seq.add(pulse_1, 'ch0')
        seq.add(pulse_2, 'ch0')
    seq.measure('ground-rydberg')
    simul = Simulation(seq, sampling_rate=.1)
    
    return simul
    
def quantum_loop(param, r):
    simul = create_quantum_loop(param, r)
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
    
def list_operator(N_qubit, op):
    ''''
    returns a a list of tensor products with op on site 0, 1,2 ...
    '''
    op_list = []
    
    for n in range(N_qubit):
        op_list_i = []
        for m in range(N_qubit):
            op_list_i.append(qeye(2))
       
        op_list_i[n] = op
        op_list.append(tensor(op_list_i)) 
    
    return op_list
    
def calculate_physical_gs(G, p = 1, omega = 1, delta = 1, U = 10):
    '''
    returns groundstate and energy 
    '''
    
    N_qubit=len(list(G))
    Omega=omega/2
    delta=delta/2

    ## Defining lists of tensor operators 
    ni = (qeye(2) - sigmaz())/2

    sx_list = list_operator(N_qubit, sigmax())
    sz_list = list_operator(N_qubit, sigmaz())
    ni_list = list_operator(N_qubit, ni)
    
    H=0
    for n in range(N_qubit):
        #H += Omega*sx_list[n]
        H -= delta * sz_list[n]
    for i, edge in enumerate(G.edges):
        H +=  U*sz_list[edge[0]]*sz_list[edge[1]]
    energies, eigenstates = H.eigenstates(sort = 'low')
    print(energies[0], energies[1])
    if energies[0] ==  energies[1]:
        print('DEGENERATE GROUND STATE')
        deg = True
        gs_en = energies[:2]
        gs_state = eigenstates[:2]
    else:
        deg = False
        gs_en = energies[0]
        gs_state = eigenstates[0]
    
    return gs_en, gs_state, deg

def generate_random_points(N, G, depth, extrem_params, reg):
    X = []
    Y = []
    for i in range(N):
        x = [np.random.randint(extrem_params[0],extrem_params[1]),
                np.random.randint(extrem_params[0],extrem_params[1])]*depth
        X.append(x)
        y = apply_qaoa(x, reg, G)
        Y.append(y)
    
    if N == 1:
        X = np.reshape(X, (depth*2,))
    return X, Y

def get_cost_colouring(z,G,penalty=U):
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

def fidelity_groundstate(C, G):
    nodes = G.nodes
    gs_en, gs_state, deg = calculate_physical_gs(G, omega, delta, U)
    gs_state = np.array(gs_state)
    gs_dict = {}
    print(gs_state)
    
    prod_list = product(['0','1'], range(len(nodes)))
    for i, bits in enumerate(prod_list):
        gs_dict[bits] = gs_state[i]
        
    prod = 0
    for i, bits in prod_list:
        prod += np.sqrt(C[bits])*gs_dict[bits]
    
    return prod / sum(counter.values())

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