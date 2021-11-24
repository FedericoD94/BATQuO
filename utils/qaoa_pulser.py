import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2
from pulser.simulation import SimConfig
from itertools import product
from utils.default_params import *
import random
from qutip import *
from scipy.stats import qmc


class qaoa_pulser(object):

    def __init__(self, depth, param_range, pos, state_prep_noise = 0):
        self.omega = 1.
        self.delta = 1.
        self.U = 10
        self.depth = depth
        self.param_range = param_range
        self.G = self.pos_to_graph(pos)
        self.solution = self.classical_solution()
        self.Nqubit = len(self.G)
        self.gs_state = None
        self.gs_en = None
        self.deg = None
        self.qubits_dict = dict(enumerate(pos))
        self.reg = Register(self.qubits_dict)
        self.state_prep_noise = state_prep_noise

    def pulser_info(self):
        '''
        Returns a dictionary of infos on the qaoa to print 
        '''
        info  ={}
        info['depth'] = self.depth
        info['omega'] = self.omega
        info['delta'] = self.delta
        info['U'] = self.U
        info['Nqubits'] = self.Nqubit
        info['graph'] = self.G
        info['classical sol'] = self.solution
        info['quantum noise'] = self.state_prep_noise
        
        return info
        
    def classical_solution(self):
        '''
        Runs through all 2^n possible configurations and estimates the solution
        Returns: dictionary with {[bitstring solution] : energy}
        '''
        results = {}

        string_configurations = list(product(['0','1'], repeat=len(self.G)))

        for string_configuration in  string_configurations:
            single_string = "".join(string_configuration)
            results[single_string] = self.get_cost_string(string_configuration)
        
        d = dict((k, v) for k, v in results.items() if v == np.min(list(results.values())))
        return d
            
    def pos_to_graph(self, pos): 
        '''
        Creates a networkx graph from the relative positions between qubits
        Parameters: positions of qubits in micrometers
        Returns: networkx graph G
        '''
        d = Chadoq2.rydberg_blockade_radius(self.omega)
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
        return G
        
    def list_operator(self, op):
        '''Returns a a list of tensor products with op on site 0, 1, 2 ...
        
        Attributes
        ---------
        op: single qubit operator
        
        Returns
        -------
        op_list: list. Each entry is the tensor product of I on every site except operator op
                       on the position of the entry
        '''
        op_list = []
    
        for n in range(self.Nqubit):
            op_list_i = []
            for m in range(self.Nqubit):
                op_list_i.append(qeye(2))
       
            op_list_i[n] = op
            op_list.append(tensor(op_list_i)) 
    
        return op_list
    
    def calculate_physical_gs(self):
        '''Calculate groundstate state, energy and degeneracy
        
        Returns
        -------
        gs_en: energy of gs
        gs_state: array 
        deg: either 0 if no degeneracy or a number indicating the degeneracy
        '''

        ## Defining lists of tensor operators 
        ni = (qeye(2) - sigmaz())/2

        sx_list = self.list_operator(sigmax())
        sz_list = self.list_operator(sigmaz())
        ni_list = self.list_operator(ni)
    
        H=0
        for n in range(self.Nqubit):
            H += self.omega * sx_list[n]
            H -= self.delta * ni_list[n]
        for i, edge in enumerate(self.G.edges):
            H +=  self.U*ni_list[edge[0]]*ni_list[edge[1]]
        energies, eigenstates = H.eigenstates(sort = 'low')
        _, degeneracies = np.unique(energies, return_counts = True)
        degeneracy = degeneracies[0]
        
        gs_en = energies[0]
        if degeneracy > 1:
            deg = degeneracy
            gs_state = eigenstates[:degeneracy]
        else:
            deg = degeneracy - 1
            gs_state = eigenstates[0]
            gs_state = np.squeeze(gs_state.full())

        self.gs_state = gs_state
        self.gs_en = gs_en
        self.deg = deg
    
        return gs_en, gs_state, deg
        
    def generate_random_points(self, N_points, return_variance = True):
        ''' Generates N_points random points with the latin hypercube method
        
        Attributes:
        N_points: how many points to generate
        return_variance: bool, if to calculate the variance or not
        '''
        X , Y , VAR = [], [], []
        
        hypercube_sampler = qmc.LatinHypercube(d=self.depth*2, seed = DEFAULT_PARAMS['seed'])
        X =  hypercube_sampler.random(N_points)
        l_bounds = np.repeat(self.param_range[0], 2*self.depth)
        u_bounds = np.repeat(self.param_range[1], 2*self.depth)
        X = qmc.scale(X, l_bounds, u_bounds).astype(int)
        X = X.tolist()
        for x in X:
            y, var_y = self.expected_energy_and_variance(x)
            Y.append(y)
            if return_variance:
                VAR.append(var_y)
        
        if return_variance:
            return X, Y, VAR
        else:
            return X, Y

    def create_quantum_circuit(self, params):
        seq = Sequence(self.reg, Chadoq2)
        seq.declare_channel('ch0','rydberg_global')
        p = int(len(params)/2)
        gammas = params[::2]
        betas = params[1::2]
        for i in range(p):
            beta_i = int(betas[i]) - int(betas[i]) % 4
            gamma_i = int(gammas[i]) - int(gammas[i]) % 4
            pulse_1 = Pulse.ConstantPulse(beta_i, self.omega, 0, 0) # H_M
            pulse_2 = Pulse.ConstantPulse(gamma_i, self.delta, 1, 0) # H_M + H_c
            seq.add(pulse_1, 'ch0')
            seq.add(pulse_2, 'ch0')
        seq.measure('ground-rydberg')
        simul = Simulation(seq, sampling_rate=0.1)
    
        return simul
    
    def quantum_loop(self, param):
        sim = self.create_quantum_circuit(param)
        if self.state_prep_noise:
            cfg = SimConfig(noise=('SPAM'), eta = self.state_prep_noise, epsilon = 0, epsilon_prime = 0)
            sim.add_config(cfg)
        results = sim.run()
        count_dict = results.sample_final_state(N_samples=DEFAULT_PARAMS['shots']) #sample from the state vector
        return count_dict, results.states

    def plot_final_state_distribution(self, C):
        C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
        color_dict = {key: 'g' for key in C}
        for key in self.solution.keys():
            val = ''.join(str(key[i]) for i in range(len(key)))
            color_dict[val] = 'r'
        plt.figure(figsize=(10,6))
        plt.xlabel("bitstrings")
        #plt.xticks(size = 15)
        plt.ylabel("counts")
        plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
        plt.xticks(rotation='vertical')
        plt.show()
    
    def plot_landscape(self,
                fixed_params = None,
                num_grid=DEFAULT_PARAMS["num_grid"],
                save = False):
        '''
        Plot energy landscape at p=1 (default) or at p>1 if you give the previous parameters in
        the fixed_params argument
        '''

        lin = np.linspace(self.param_range[0],self.param_range[1], num_grid)
        Q = np.zeros((num_grid, num_grid))
        Q_params = np.zeros((num_grid, num_grid, 2))
        for i, gamma in enumerate(lin):
            for j, beta in enumerate(lin):
                if fixed_params is None:
                    params = [gamma, beta]
                else:
                    params = fixed_params + [gamma, beta]
                Q[j, i] = self.expected_energy(params)
                Q_params[j,i] = np.array([gamma, beta])


        plt.imshow(Q, origin = 'lower', extent = [self.param_range[0],self.param_range[1],self.param_range[0],self.param_range[1]])
        plt.title('Grid Search: [{} x {}]'.format(num_grid, num_grid))
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)

        cb = plt.colorbar()
        plt.xlabel(r'$\gamma$', fontsize=20)
        plt.ylabel(r'$\beta$', fontsize=20)
        cb.ax.tick_params(labelsize=15)
        plt.show()

        if save:
            np.savetxt('../data/raw/graph_Grid_search_{}x{}.dat'.format(num_grid, num_grid), Q)
            np.savetxt('../data/raw/graph_Grid_search_{}x{}_params.dat'.format(num_grid, num_grid), Q)

    def fidelity_gs_sampled(self, x, solution_ratio = False):
        '''
        Fidelity sampled means how many times the solution(s) is measured
        '''
        C = self.get_sampled_state(x)
        fid = 0
        for sol_key in self.solution.keys():
            fid += C[sol_key]
        
        fid = fid/DEFAULT_PARAMS['shots']
        
        if solution_ratio:
            sorted_dict = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
            first_key, second_key =  list(sorted_dict.keys())[:2]
            if (first_key in self.solution.keys()) and (second_key !=0):
                sol_ratio = C[first_key]/C[second_key]
            else:
                sol_ratio = 0
            return fid, sol_ratio
        else:
            return fid
            
    def solution_ratio(self, x):
        sol_ratio = 0
        C = self.get_sampled_state(x)
        sorted_dict = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
        first_key, second_key =  list(sorted_dict.keys())[:2]
        if (first_key in self.solution.keys()) and (second_key !=0):
            sol_ratio = C[first_key]/C[second_key]
            
        return sol_ratio
        
    def fidelity_gs_exact_old(self, param):
        '''
        Return the fidelity of the exact qaoa state (obtained with qutip) and the 
        exact groundstate calculated with the physical hamiltonian of pulser
        '''
        C, evolution_states = self.quantum_loop(param)
        if self.gs_state is None:
            self.calculate_physical_gs()
        fid = fidelity(self.gs_state, evolution_states[-1])
        return fid
        

    def fidelity_gs_exact(self, param):
        '''
        Return the fidelity of the exact qaoa state (obtained with qutip) and the 
        exact groundstate calculated with the physical hamiltonian of pulser
        '''
        C, evolution_states = self.quantum_loop(param)
        final_state = np.squeeze(evolution_states[-1])
        if self.gs_state is None:
            self.calculate_physical_gs()
        if self.deg:
            fidelities = [np.abs(np.dot(final_state, self.gs_state[i]))**2 for i in range(len(self.gs_state))]
            fidelity = np.sum(fidelities)
        else:
            fidelity = np.squeeze(np.abs(np.dot(final_state, self.gs_state))**2)
        return fidelity

    def get_cost_string(self, string):
        'Receives a string of 0 and 1s and gives back its cost to the MIS hamiltonian'
        penalty = self.U
        configuration = np.array(tuple(string),dtype=int)
        
        cost = 0
        cost = -sum(configuration)
        for edge in self.G.edges:
            cost += penalty*(configuration[edge[0]]*configuration[edge[1]])
        
        return cost
            
    def expected_energy_and_variance(self,
             params,
             shots=DEFAULT_PARAMS["shots"]):
        '''
        Applies QAOA circuit and estimates final energy and variance
        '''

        counts = self.get_sampled_state(params)
        extimated_en = 0

        for configuration in counts:
            prob_of_configuration = counts[configuration]/shots
            extimated_en += prob_of_configuration * self.get_cost_string(configuration)

        amplitudes = np.fromiter(counts.values(), dtype=float)
        amplitudes = amplitudes / shots

        return extimated_en, self.sample_variance(extimated_en, counts, shots)
    
    def sample_variance(self, sample_mean, counts, shots):
        estimated_variance = 0
        for configuration in counts:
            hamiltonian_i = self.get_cost_string(configuration) # energy of i-th configuration
            estimated_variance += counts[configuration] * (sample_mean - hamiltonian_i)**2
        
        estimated_variance /= shots - 1 # use unbiased variance estimator

        return estimated_variance
        
    def get_cost_dict(self, counter):
        total_cost = 0
        for key in counter.keys():
            cost = self.get_cost_string(key)
            total_cost += cost * counter[key]
        return total_cost / sum(counter.values())
    
    def get_evolution_states(self, params):
        C, evol= self.quantum_loop(params)
        
        return C, evol
        
    def get_sampled_state(self, params):
        C, _ = self.quantum_loop(params)
        return C
        
    def expected_energy(self, param):
        C, _ = self.quantum_loop(param)
        cost = self.get_cost_dict(C)
        
        return cost
