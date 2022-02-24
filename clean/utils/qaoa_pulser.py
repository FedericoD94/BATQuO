import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2
Chadoq2.change_rydberg_level(60)
from pulser.simulation import SimConfig
from itertools import product
from utils.default_params import *
import random
from qutip import *
from scipy.stats import qmc

np.random.seed(DEFAULT_PARAMS['seed'])
random.seed(DEFAULT_PARAMS['seed'])

class qaoa_pulser(object):

    def __init__(self, depth, type_of_graph, quantum_noise = None):
        self.C_6_over_h = Chadoq2.interaction_coeff
        self.angles_bounds = np.array(DEFAULT_PARAMS['angle_bounds'])
        self.omega = Q_DEVICE_PARAMS['omega_over_2pi'] * 2 * np.pi #see notes/info.pdf for this value
        self.delta = Q_DEVICE_PARAMS['delta_over_2pi'] * 2 * np.pi #see notes/info.pdf for the calculation
        self.U = [] # it is a list because two qubits in rydberg interactiong might be closer than others
        self.depth = depth
        self.G, self.qubits_dict = self.generate_graph(type_of_graph)
        print('graph is\n, ', self.G, self.G.nodes, self.G.edges)
        nx.draw(self.G)
        self.solution, self.solution_energy = self.classical_solution()
        self.Nqubit = len(self.G)
        self.gs_state = None
        self.gs_en = None
        self.deg = None

        self.reg = Register(self.qubits_dict)
        self.quantum_noise = quantum_noise
        if quantum_noise is not None:
            self.noise_config = SimConfig(noise=(self.quantum_noise),
                                          eta = Q_DEVICE_PARAMS['eta'],
                                          epsilon = Q_DEVICE_PARAMS['epsilon'],
                                          epsilon_prime = Q_DEVICE_PARAMS['epsilon_prime'],
                                          temperature = Q_DEVICE_PARAMS['temperature'],
                                          laser_waist = Q_DEVICE_PARAMS['laser_waist'],
                                          )
            self.noise_info = self.noise_config.__str__()
        
        
    def print_info_problem(self,f):
        f.write('Problem: MIS\n')
        f.write('Cost: -\u03A3 Z_i + {} * \u03A3 Z_i Z_j\n'.format(DEFAULT_PARAMS['penalty']))
        f.write('Hamiltonian: \u03A9 \u03A3 X_i - \u03b4 \u03A3 Z_i + U \u03A3 Z_i Z_j\n')
        f.write('Mixing: \u03A9 \u03A3 X_i\n\n')
        f.write('\n')
        
    def print_info_qaoa(self, f):
        '''Prints the info on the passed opened file f'''
        f.write(f'Depth: {self.depth}\n')
        f.write(f'Omega: {self.omega} ')
        f.write(f'delta: {self.delta} ')
        f.write(f'U: {self.U}\n')
        f.write(f'Graph: {self.G.edges}\n')
        f.write(f'Classical sol: {self.solution}\n')
        if self.quantum_noise is not None:
            f.write(f'Noise info: {self.noise_info}')
        f.write('\n')
        
    def generate_graph(self, type_of_graph): 
        '''
        Creates a networkx graph from the relative positions between qubits
        Parameters: positions of qubits in micrometers
        Returns: networkx graph G
        '''
        if type_of_graph == 'chair':
            a = Q_DEVICE_PARAMS['lattice_spacing']
            pos =[[0, 0], [a, 0], [3/2 * a, np.sqrt(3)/2 * a], [3/2 * a, -np.sqrt(3)/2 * a], [2 * a, 0], [3 * a, 0]]
            print(pos)
        else:
            print('type of graph not supported')
            
        d = Chadoq2.rydberg_blockade_radius(self.omega)
        G = nx.Graph()
        edges=[]
        distances = []
        print(d)
        for n in range(len(pos)-1):
            for m in range(n+1, len(pos)):
                pwd = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
                distances.append(pwd)
                if pwd < d:
                    edges.append([n,m]) # Below rbr, vertices are connected
                    self.U.append(self.C_6_over_h/(pwd**6)) #And the interaction is given by C_6/(h*d^6)
        G.add_nodes_from(range(len(pos)))
        G.add_edges_from(edges)
        return G, dict(enumerate(pos))
        
    def classical_solution(self):
        '''
        Runs through all 2^n possible configurations and estimates the solution
        Returns: 
            d: dictionary with {[bitstring solution] : energy}
            en: energy of the (possibly degenerate) solution
        '''
        results = {}

        string_configurations = list(product(['0','1'], repeat=len(self.G)))

        for string_configuration in  string_configurations:
            single_string = "".join(string_configuration)
            results[single_string] = self.get_cost_string(string_configuration)
        
        d = dict((k, v) for k, v in results.items() if v == np.min(list(results.values())))
        en = list(d.values())[0]
        print('classical solution: ', d)
        
        return d, en
        
        
    def get_cost_string(self, string):
        'Receives a string of 0 and 1s and gives back its cost to the MIS hamiltonian'
        penalty = DEFAULT_PARAMS["penalty"]
        configuration = np.array(tuple(string),dtype=int)
        
        cost = 0
        delta = np.max([self.delta, -self.delta]) # to ensure the constant is negative
        cost = -sum(configuration)
        for i,edge in enumerate(self.G.edges):
            cost += penalty*(configuration[edge[0]]*configuration[edge[1]])
        
        return cost
        
    def get_cost_dict(self, counter):
        total_cost = 0
        for key in counter.keys():
            cost = self.get_cost_string(key)
            total_cost += cost * counter[key]
        return total_cost / sum(counter.values())
        
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
    
        H = 0
        for n in range(self.Nqubit):
            delta = np.max([self.delta, -self.delta])  #delta can be negative but for the energy we need -delta + U
            H -= delta * ni_list[n]
            
        for i, edge in enumerate(self.G.edges):
            H +=  self.U[i]*ni_list[edge[0]]*ni_list[edge[1]]
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
        
        print('Groundstate energy: ', gs_en)
        print('Degeneracy: ', deg)
        print('Groundstate: ', gs_state)
        #print('Largest amplitude: ', bin(np.argmax(gs_state)))
        print('delta: ', self.delta, 'omega: ', self.omega, 'U: ', self.U[0])
        
        return gs_en, gs_state, deg
        
    def generate_random_points(self, N_points):
        ''' Generates N_points random points with the latin hypercube method
        
        Attributes:
        N_points: how many points to generate
        return_variance: bool, if to calculate the variance or not
        '''
        X , Y , data_train = [], [], []
        
        hypercube_sampler = qmc.LatinHypercube(d=self.depth*2, seed = DEFAULT_PARAMS['seed'])
        X =  hypercube_sampler.random(N_points)
        l_bounds = np.repeat(self.angles_bounds[:,0], self.depth)
        u_bounds = np.repeat(self.angles_bounds[:,1], self.depth)
        X = qmc.scale(X, l_bounds, u_bounds).astype(int)
        X = X.tolist()
        for x in X:
            y, var_y, fid_sampled, fid_exact, sol_ratio, _ , _ = self.apply_qaoa(x)
            Y.append(y)
            data_train.append([var_y, fid_sampled, fid_exact, sol_ratio])
        
        return X, Y, data_train

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
            pulse_2 = Pulse.ConstantPulse(gamma_i, 0, self.delta, 0) # H_M + H_c
            seq.add(pulse_1, 'ch0')
            seq.add(pulse_2, 'ch0')
        seq.measure('ground-rydberg')
        
        #check if sampling_rate is too small by doing rate*total_duration:
        sampling_rate = 1
        while sampling_rate * sum(params) < 4:
            sampling_rate += 0.1
        simul = Simulation(seq, sampling_rate=sampling_rate)
    
        return simul
    
    def quantum_loop(self, param):
        sim = self.create_quantum_circuit(param)
        if self.quantum_noise is not None:
            sim.add_config(self.noise_config)
        
        results = sim.run()
        #np.random.seed(28)

        count_dict = results.sample_final_state(N_samples=DEFAULT_PARAMS['shots'])
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
    
        lin_gamma = np.linspace(self.angles_bounds[0][0],self.angles_bounds[0][1], num_grid)
        lin_beta = np.linspace(self.angles_bounds[0][1],self.angles_bounds[1][1], num_grid)
        Q = np.zeros((num_grid, num_grid))
        Q_params = np.zeros((num_grid, num_grid, 2))
        for i, gamma in enumerate(lin_gamma):
            for j, beta in enumerate(lin_beta):
                if fixed_params is None:
                    params = [gamma, beta]
                else:
                    params = fixed_params + [gamma, beta]
                a = self.apply_qaoa(params)
                Q[j, i] = a[0]
                Q_params[j,i] = np.array([gamma, beta])


        plt.imshow(Q, origin = 'lower', extent = [item for sublist in self.angles_bounds for item in sublist])
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

    def fidelity_gs_sampled(self, C):
        '''
        Fidelity sampled means how many times the solution(s) is measured
        '''
        fid = 0
        for sol_key in self.solution.keys():
            fid += C[sol_key]
        
        fid = fid/DEFAULT_PARAMS['shots']
        
        return fid
            
    def solution_ratio(self, C):
        sol_ratio = 0
        sorted_dict = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
        if len(sorted_dict)>1 :
            first_key, second_key =  list(sorted_dict.keys())[:2]
            if (first_key in self.solution.keys()):
                sol_ratio = C[first_key]/C[second_key]
        else:
            first_key =  list(sorted_dict.keys())[0]
            if (first_key in self.solution.keys()):
                sol_ratio = -1    #if the only sampled value is the solution the ratio is infinte so we put -1
            
        return sol_ratio

    def fidelity_gs_exact(self, final_state):
        '''
        Return the fidelity of the exact qaoa state (obtained with qutip) and the 
        exact groundstate calculated with the physical hamiltonian of pulser
        '''
        final_state = np.squeeze(final_state)
        if self.gs_state is None:
            self.calculate_physical_gs()
        if self.deg:
            fidelities = [np.abs(np.dot(final_state, self.gs_state[i]))**2 for i in range(len(self.gs_state))]
            fidelity = np.sum(fidelities)
        else:
            fidelity = np.squeeze(np.abs(np.dot(final_state, self.gs_state))**2)
        return fidelity
    
    def expected_variance(self,counts, sample_mean):
        shots=DEFAULT_PARAMS["shots"]
        estimated_variance = 0
        for configuration in counts:
            hamiltonian_i = self.get_cost_string(configuration) # energy of i-th configuration
            estimated_variance += counts[configuration] * (sample_mean - hamiltonian_i)**2
        
        estimated_variance /= shots - 1 # use unbiased variance estimator

        return estimated_variance
        
    def expected_energy(self, C):
        cost = self.get_cost_dict(C)
        
        return cost

    def apply_qaoa(self, params, show = False):
        C, evol= self.quantum_loop(params)
        energy = self.expected_energy(C)
        expected_variance = self.expected_variance(C, energy)
        fidelity_sampled = self.fidelity_gs_sampled(C)
        if self.quantum_noise is None:
            fidelity_exact = self.fidelity_gs_exact(np.flip(evol[-1]))
        else:
            fidelity_exact = 0
        solution_ratio = self.solution_ratio(C)
        
        if show:
            self.plot_final_state_distribution(C)
        return energy, expected_variance, fidelity_sampled, fidelity_exact, solution_ratio, C, evol