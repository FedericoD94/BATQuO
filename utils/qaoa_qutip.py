import networkx as nx
from collections import Counter, defaultdict, namedtuple
from itertools import product
# import pandas as pd
import random
import numpy as np

# QUANTUM
#from qiskit import Aer, QuantumCircuit, execute
import qutip as qu
from  utils.default_params import *

# VIZ
from matplotlib import pyplot as plt
from utils.default_params import *

class qaoa_qiskit(object):

    def __init__(self, G):
        self.G = G
        self.N = len(G)
        L =  self.N
        self.G_comp = nx.complement(G)
        self.gs_states = None
        self.gs_en = None
        self.deg = None

        dimQ = 2

        self.Id = qu.tensor([qu.qeye(dimQ)] * L)

        temp = [[qu.qeye(dimQ)] * j +
                [qu.sigmax()] +
                [qu.qeye(dimQ)] * (L - j - 1)
                for j in range(L)
                ]

        self.X = [qu.tensor(temp[j]) for j in range(L)]

        temp = [[qu.qeye(dimQ)] * j +
                [qu.sigmay()] +
                [qu.qeye(dimQ)] * (L - j - 1)
                for j in range(L)
                ]
        self.Y = [qu.tensor(temp[j]) for j in range(L)]

        temp = [[qu.qeye(dimQ)] * j +
                [qu.sigmaz()] +
                [qu.qeye(dimQ)] * (L - j - 1)
                for j in range(L)
                ]
        self.Z = [qu.tensor(temp[j]) for j in range(L)]



        self.H_c = self.hamiltonian_cost(penalty=DEFAULT_PARAMS["penalty"])

    def Rx(self, qubit_n, alpha):
        op = np.cos(alpha / 2) * self.Id -1j * np.sin(alpha / 2) * self.X[qubit_n]
        return op


    def Rz(self, qubit_n, alpha):
        op = np.cos(alpha / 2) * self.Id -1j * np.sin(alpha / 2) * self.Z[qubit_n]
        return qu.tensor(temp)


    def Rzz(self, qubit_n, qubit_m, alpha):
        op = (np.cos(alpha / 2) * self.Id
              -1j * np.sin(alpha / 2) * self.Z[qubit_n] * self.Z[qubit_m])
        return op


    def Rxx(self, qubit_n, qubit_m, alpha):
        op = (np.cos(alpha / 2) * self.Id
              -1j * np.sin(alpha / 2) * self.X[qubit_n] * self.X[qubit_m])
        return op


    def s2z(self, configuration):
        return [1 - 2 * s for s in configuration]


    def z2s(self, configuration):
        return [(1-z)/2 for z in configuration]


    def str2list(self, s):
        list_conf = []
        skip = False
        for x in s:
            if skip:
                skip = False
                continue
            if x == "-":
                list_conf.append(-1)
                skip = True
            if x != "-":
                list_conf.append(int(x))
        return list_conf

    def hamiltonian_cost(self, penalty):
        H_0 = [-1*self.Z[i] / 2 for i in range(self.N)]
        H_int = [(Z[i] * Z[j] - Z[i] - Z[j]) / 4 for i, j in  self.G.edges]
        ## Hamiltonian_cost is minimized by qaoa so we need to consider -H_0
        # in order to have a solution labeled by a string of 1s
        H_c = -sum(H_0) + penalty * sum(H_int)
        return H_c


    def evaluate_cost(self, configuration):
        '''
        configuration: strings of 0,1. The solution (minimum of H_c) is labelled by 1
        '''
        qstate_from_configuration = qu.tensor([qu.basis(2, _) for _ in configuration])
        cost = qu.expect(self.H_c, qstate_from_configuration)

        return cost


    def quantum_algorithm(self,
                          params,
                          penalty=DEFAULT_PARAMS["penalty"]):
        '''
        Qiskit implementations of gates:
        Rz: https://qiskit.org/documentation/stubs/qiskit.circuit.library.RZGate.html
        Rzz: https://qiskit.org/documentation/stubs/qiskit.circuit.library.RZZGate.html
        Rx:
        from which we deduce the angles
        '''
        depth = int(len(params)/2)
        gammas = params[::2]
        betas = params[1::2]

        #INIZIALIZE CIRCUIT
        qc = QuantumCircuit(self.N, self.N)
        qc.h(range(self.N))

        #APPLY QAOA no. DEPTH TIMES
        for p in range(depth):
            for edge in self.G.edges():
                qc.rzz(gammas[p] * penalty / 2 , edge[0], edge[1])
                qc.rz(-gammas[p] * penalty / 2, edge[0])
                qc.rz(-gammas[p] * penalty / 2, edge[1])

            for i in self.G.nodes:
                qc.rz(gammas[p], i)

            qc.rx(2*betas[p], range(self.N))

        return qc

    def quantum_measure(self):

         #MEASURE
        meas = QuantumCircuit(self.N,self.N)
        meas.barrier(range(self.N))
        meas.measure(range(self.N), range(self.N))

        return meas

    def run_circuit(self, params, qc,
                    backend_name = 'qasm_simulator',
                    penalty=DEFAULT_PARAMS["penalty"],
                    shots=DEFAULT_PARAMS["shots"] ):

        backend = Aer.get_backend(backend_name)

        #The two seeds are necessary for reproducibility!
        simulate = execute(qc, backend=backend, shots=shots, seed_transpiler = DEFAULT_PARAMS["seed"], seed_simulator = DEFAULT_PARAMS["seed"])
        results = simulate.result()

        return results

    def final_sampled_state(self,
                        params,
                        penalty=DEFAULT_PARAMS["penalty"],
                        shots=DEFAULT_PARAMS["shots"]):

        #MEASURE
        measure = self.quantum_measure()
        qc = self.quantum_algorithm(params)
        #qc.compose(measure, qubits = range(self.N))
        qc.compose(measure, inplace = True)

        results = self.run_circuit(params, qc, 'qasm_simulator', penalty, shots)
        counts = results.get_counts()

        pretty_counts = {k[::-1]:v for k,v in counts.items()}

        return pretty_counts

    def final_exact_state(self,
             params,
             penalty=DEFAULT_PARAMS["penalty"],
             shots=DEFAULT_PARAMS["shots"]):

        qc = self.quantum_algorithm(params)
        results = self.run_circuit(params, qc, "statevector_simulator",  penalty, shots = 1)
        st = results.get_statevector(qc)
        st.reshape((2**self.N,1))

        # Qiskit uses little endian convention (why????) so we need to invert the binary
        #combinations in order in the array, ex: 00111 becomes 11100 while 10101 stays 10101
        ordered_st = np.zeros(len(st), dtype = complex)
        for i, num in enumerate(st):
            string = '{0:b}'.format(i)
            invert_string = string[::-1]
            pos_in_array = int(invert_string, 2)
            ordered_st[pos_in_array] = num
        return ordered_st

    def circuit_unitary(self, params, penalty=DEFAULT_PARAMS["penalty"]):

        qc = self.quantum_algorithm(params)
        results = self.run_circuit(params, qc, 'unitary_simulator', penalty, shots = 1)
        uni = results.get_unitary(qc)

        return uni

    def expected_energy(self,
             params,
             return_state = False,
             shots=DEFAULT_PARAMS["shots"],
             basis="S"):
        '''
        Applies QAOA circuit and estimates final energy
        '''

        counts = self.final_sampled_state(params)
        extimated_en = 0

        if basis=="S":
            pass
        elif basis=="Z":
            counts = {"".join([str(x) for x in s2z([int(x) for x in k])]):v for k,v in counts.items()}
        else:
            raise ValueError('Basis should be specified: it must one of ["S","Z"]')

        for configuration in counts:
            prob_of_configuration = counts[configuration]/shots
            extimated_en += prob_of_configuration * self.evaluate_cost(
                                                                self.str2list(configuration),
                                                                basis=basis)

        amplitudes =  np.fromiter(counts.values(), dtype=float)
        amplitudes = amplitudes / shots

        return extimated_en

    def plot_landscape(self,
                    param_range,
                    fixed_params = None,
                    num_grid=DEFAULT_PARAMS["num_grid"],
                    save = False):
        '''
        Plot energy landscape at p=1 (default) or at p>1 if you give the previous parameters in
        the fixed_params argument
        '''

        lin = np.linspace(param_range[0],param_range[1], num_grid)
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


        plt.imshow(Q, origin = 'lower', extent = [param_range[0],param_range[1],param_range[0],param_range[1]])
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


    def plot_final_state(self, freq_dict):
        ''' Plots the final state given as a dictionary with {binary_strin:counts}'''

        sol = self.classical_solution(basis = 'S')

        sorted_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
        color_dict = {key: 'g' for key in sorted_freq_dict}
        for key in sol.keys():
            val = ''.join(str(key[i]) for i in range(len(key)))
            color_dict[val] = 'r'
        plt.figure(figsize=(12,6))
        plt.xlabel("configuration")
        plt.ylabel("counts")
        plt.xticks(rotation='vertical')
        plt.bar(sorted_freq_dict.keys(), sorted_freq_dict.values(), width=0.5, color = color_dict.values())


    def spectrum_vs_penalty(self, penalty_min=-2,
                            penalty_max=3,
                            penalty_step=0.5,
                            show_plot="show_plot",
                            basis="S"):

        configuration_energies = defaultdict(list)
        penalties=np.arange(penalty_min, penalty_max, penalty_step)

        if basis == "S":
            eigenvalues = s_eigenvalues
        elif basis=="Z":
            eigenvalues = s2z(s_eigenvalues)

        for eigen_configuration in product(eigenvalues, repeat = self.N):
            for penalty in penalties:
                configuration_energies[str(eigen_configuration)].append(evaluate_cost(str2list(eigen_configuration), penalty, basis=basis))

        degeneracies_df = pd.DataFrame(sorted(Counter([v[-1] for k,v in configuration_energies.items()]).items(), key=lambda item: item[0], reverse=False))
        degeneracies_df.columns=["energy","eigenstates"]

        if show_plot:

            for k, v in configuration_energies.items():
                plt.plot(penalties, v, label = k, marker="o")
                plt.xlabel("penalty")
                plt.ylabel("cost")
                plt.legend(title="eigen_configuration",
                           bbox_to_anchor=(1.05, 1),
                           loc='upper left')
            plt.show()

        return degeneracies_df, configuration_energies

    def generate_random_points(self, N_points, depth, extrem_params, fixed_params=None):
        X = []
        Y = []
        np.random.seed(DEFAULT_PARAMS['seed'])
        random.seed(DEFAULT_PARAMS['seed'])

        for i in range(N_points):
            if fixed_params is None:
                x = [random.uniform(extrem_params[0], extrem_params[1]) for _ in range(depth*2)]
            else:
                x = fixed_params + [random.uniform(extrem_params[0], extrem_params[1]) for _ in range(2)]
            X.append(x)
            y = self.expected_energy(x)
            Y.append(y)

        return X, Y

    def list_operator(self, op):
        ''''
        returns a a list of tensor products with op on site 0, 1,2 ...
        '''
        op_list = []

        for qubit in range(self.N):
            op_list_i = []
            for m in range(self.N):
                op_list_i.append(qeye(2))

            op_list_i[qubit] = op
            op_list.append(qu.tensor(op_list_i))

        return op_list

    def calculate_gs_qiskit(self, penalty=DEFAULT_PARAMS["penalty"]):
        '''
        returns groundstate and energy
        '''

        sx_list = self.list_operator(sigmax())
        sz_list = self.list_operator(sigmaz())


        H=0
        for n in range(self.N):
            H +=  0.5 * sz_list[n]
        for i, edge in enumerate(self.G.edges):
            H += penalty/4 * sz_list[edge[0]]*sz_list[edge[1]]
            H -= penalty/4 * sz_list[edge[0]]
            H -= penalty/4 * sz_list[edge[1]]
        energies, eigenstates = H.eigenstates(sort = 'low')

        degeneracy = next((i for i, x in enumerate(np.diff(energies)) if x), 1) + 1
        deg = (degeneracy > 1)
        gs_en = energies[0]
        gs_states = [state_gs.full() for state_gs in eigenstates[:degeneracy]]
        self.gs_states = gs_states
        self.gs_en = gs_states
        self.deg = deg

        return gs_en, gs_states, deg


    def fidelity_gs(self, point):

        #calculate gs if it is not calculated
        if self.gs_states is None:
            self.gs_en, self.gs_states, self.deg = self.calculate_gs_qiskit()

        fin_state_exact = self.final_exact_state(point)
        fidelities= []
        for st_idx, gs_state in enumerate(self.gs_states):
            fidelities.append(np.abs(np.dot(fin_state_exact, gs_state))**2)

        fidelity_tot = np.sum(fidelities)

        return fidelity_tot



