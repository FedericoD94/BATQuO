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
import pandas as pd

np.random.seed(DEFAULT_PARAMS['seed'])
random.seed(DEFAULT_PARAMS['seed'])

class qaoa_pulser(object):

    def __init__(self, depth, type_of_graph, lattice_spacing, quantum_noise = None):
        self.C_6_over_h = Chadoq2.interaction_coeff
        self.angles_bounds = np.array(DEFAULT_PARAMS['angle_bounds'])
        self.omega = Q_DEVICE_PARAMS['omega_over_2pi'] * 2 * np.pi #see notes/info.pdf for this value
        self.delta = Q_DEVICE_PARAMS['delta_over_2pi'] * 2 * np.pi #see notes/info.pdf for the calculation
        self.lattice_spacing = lattice_spacing
        self.U = [] # it is a list because two qubits in rydberg interactiong might be closer than others
        self.depth = depth
        self.G, self.qubits_dict = self.generate_graph(type_of_graph)
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
            a = self.lattice_spacing
            pos =[[0, 0], 
                  [a, 0], 
                  [3/2 * a, np.sqrt(3)/2 * a], 
                  [3/2 * a, -np.sqrt(3)/2 * a], 
                  [2 * a, 0], 
                  [3 * a, 0]
                  ]
        else:
            print('type of graph not supported')
            
        rydberg_radius = Chadoq2.rydberg_blockade_radius(self.omega)
        G = nx.Graph()
        edges=[]
        distances = []
        for n in range(len(pos)-1):
            for m in range(n+1, len(pos)):
                pwd = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
                distances.append(pwd)
                if pwd < rydberg_radius:
                    edges.append([n,m]) # Below rbr, vertices are connected
                    self.U.append(self.C_6_over_h/(pwd**6)) #And the interaction is given by C_6/(h*d^6)
        G.add_nodes_from(range(len(pos)))
        G.add_edges_from(edges)
        print('\n###### CREATED GRAPH ######\n')
        print(G.nodes, G.edges)
        print('Rydberg Radius: ', rydberg_radius)
        print('Lattice spacing: ', a)
        
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
        
        #sort the dictionary
        results = dict(sorted(results.items(), key=lambda item: item[1]))
        
        #counts the distribution of energies
        energies, counts = np.unique(list(results.values()), return_counts = True)
        df = pd.DataFrame(np.column_stack((energies, counts)), columns = ['energy', 'counts'])
        print('\n####CLASSICAL SOLUTION######\n')
        print('Lowest energy:', d)
        print('First excited states:', {k: results[k] for k in list(results)[1:5]})
        print('Energy distribution')
        print(df)
        
        
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

        self.gs_state = gs_state
        self.gs_en = gs_en
        self.deg = deg
        self.H = H
        
        print('\n##### QAOA HAMILTONIAN #######')
        print('H = - \u03b4 \u03A3 Z_i + U \u03A3 Z_i Z_j\n')
        print('Mixing: \u03A9 \u03A3 X_i\n')
        print('\u03b4: ', self.delta, '\n\u03A9: ', self.omega, '\nU: ', self.U[0])
        print('Rydberg condition U/\u03A9: ', self.U[0]/self.omega)
        print('Groundstate energy: ', gs_en)
        print('Degeneracy: ', deg)
        
        return gs_en, gs_state, deg, H
        
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
            qaoa_results = self.apply_qaoa(x)
            Y.append(qaoa_results['sampled_energy'])
            data_train.append(qaoa_results)
        
        return X, Y, data_train

    def create_quantum_circuit(self, params):
        seq = Sequence(self.reg, Chadoq2)
        seq.declare_channel('ch0','rydberg_global')
        gammas = params[::2]
        betas = params[1::2]
        
        for i in range(self.depth):
            #Ensures params are multiples of 4 ns
            beta_i = int(betas[i]) - int(betas[i]) % 4
            gamma_i = int(gammas[i]) - int(gammas[i]) % 4
            
            mixing_pulse = Pulse.ConstantPulse(beta_i, self.omega, 0, 0) # H_M
            hamiltonian_pulse = Pulse.ConstantPulse(gamma_i, 0, self.delta, 0) # H_c
            
            seq.add(mixing_pulse, 'ch0')
            seq.add(hamiltonian_pulse, 'ch0')
            
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
        plt.ylabel("counts")
        plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
        plt.xticks(rotation='vertical')
        plt.title('Final sampled state | p = {} | N shots = {}'.format(self.depth, 
                                                                DEFAULT_PARAMS['shots']))
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
        flipped = np.flip(final_state.full())
        flipped = Qobj(flipped, dims = [[2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1]])
        overlap = self.gs_state.overlap(flipped)
        fidelity =  np.abs(overlap)**2

        return fidelity
        
    def calculate_sampled_energy_and_variance(self, sampled_state):
        sampled_energy = self.get_cost_dict(sampled_state)
        
        shots=DEFAULT_PARAMS["shots"]
        estimated_variance = 0
        for configuration in list(sampled_state.keys()):
            hamiltonian_i = self.get_cost_string(configuration) # energy of i-th 
                                                                # configuration

            estimated_variance += sampled_state[configuration] * (sampled_energy - hamiltonian_i)**2
        
        estimated_variance = estimated_variance / (shots - 1) # use unbiased variance estimator
        
        return sampled_energy, estimated_variance

    def calculate_fidelity_sampled(self, C):
        '''
        Fidelity sampled means how many times the solution(s) is measured
        '''
        fid = 0
        for sol_key in self.solution.keys():
            fid += C[sol_key]
        
        fid = fid/DEFAULT_PARAMS['shots']
        
        return fid
        
    def calculate_exact_energy_and_variance(self, final_state):
         
         #Qutip and the results from pulser have opposite endians so we need to flip
         #the array
         flipped = np.flip(final_state.full())
         flipped = Qobj(flipped, dims = [[2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1]])
         
         expected_energy = expect(self.H, flipped)
         expected_variance = variance(self.H, flipped)
         
         return expected_energy, expected_variance
        
    def apply_qaoa(self, params, show = False):
        '''
        Runs qaoa with the specified parameters
        
        Parameters
        ----------
            params: set of angles, needs to be of len 2*depth, can be either int or float
        
        Returns
        -------
            results_dict: Dictionary with all the info of the qaoa final state:
                          sampled_state, sampled_energy, sampled_variance, exact_energy
                          exact_variance, sampled_fidelity, exact_fidelity, solution_ratio
                          states of the evolution
            
        '''
        #Checks the number of parameters
        if len(params) != 2*self.depth:
            print('\nWARNING:\n'
                  f'Running qaoa with a number of params different from '
                  f'2*depth = {2*self.depth}, number of passed parameters is '
                  f'{len(params)}')
        #quantum loop returns a dictionary of N_shots measured states and the evolution
        #of qutip state
        results_dict = {}
        sampled_state, evolution_states= self.quantum_loop(params)
        results_dict['sampled_state'] = sampled_state
        results_dict['evolution_states'] = evolution_states
        
        sampled_energy, sampled_variance = self.calculate_sampled_energy_and_variance(sampled_state)
        results_dict['sampled_energy'] = sampled_energy
        results_dict['sampled_variance'] = sampled_variance
        results_dict['sampled_fidelity'] = self.calculate_fidelity_sampled(sampled_state)
        
        exact_energy, exact_variance = self.calculate_exact_energy_and_variance(evolution_states[-1])
        results_dict['exact_energy'] = exact_energy
        results_dict['exact_variance'] = exact_variance 
        results_dict['solution_ratio'] = self.solution_ratio(sampled_state)

        if self.quantum_noise is None:
            results_dict['exact_fidelity'] = self.fidelity_gs_exact(evolution_states[-1])

        else:
            results_dict['exact_fidelity'] = 0
        
        if show:
            self.plot_final_state_distribution(sampled_state)
            
        return results_dict
                
        