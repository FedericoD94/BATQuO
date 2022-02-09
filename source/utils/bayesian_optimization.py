from .qaoa_pulser import *
from .gaussian_process import *
import numpy as np

class Bayesian_optimization():

    def __init__(self,
                 depth,
                 type_of_graph,
                 quantum_noise,
                 nwarmup,
                 nbayes,
                 kernel_choice,
                 verbose,
                 *args, 
                 **kwargs):
        self.depth = depth
        self.nwarmup = nwarmup
        self.nbayes = nbayes
        
        ### CREATE QAOA
        self.qaoa = qaoa_pulser(depth, type_of_graph, quantum_noise)
        self.qaoa.calculate_physical_gs()
        self.qaoa.classical_solution()
        
        
        ### CREATE GP 
        self.gp = MyGaussianProcessRegressor(depth = depth, kernel_choice = kernel_choice)

        
        
    def print_info(self):
        self.file_name = 'p={}_warmup={}_train={}'.format(self.depth, self.nwarmup, self.nbayes)
        data = []
        gamma_names = ['GAMMA' + str(i) + ' ' for i in range(self.depth)]
        beta_names = ['BETA' + str(i) + ' ' for i in range(self.depth)]
        data_names = ['iter'] + gamma_names + beta_names + ['energy', 
                                                            'variance', 
                                                            'fidelity_exact', 
                                                            'fidelity_sampled', 
                                                            'ratio', 
                                                            'corr_length ', 
                                                            'const_kernel',
                                                            'std energies', 
                                                            'average distances', 
                                                            'nit', 
                                                            'time opt bayes', 
                                                            'time qaoa', 
                                                            'time opt kernel', 
                                                            'time step']
                     
        data_header = " ".join(["{:>7} ".format(i) for i in data_names]) + '\n'


        info_file_name = self.file_name + '_info.txt'
        with open(info_file_name, 'w') as f:
            f.write('BAYESIAN OPTIMIZATION of QAOA \n\n')
            self.qaoa.print_info_problem(f)
            
            f.write('QAOA PARAMETERS')
            f.write('\n-------------\n')
            self.qaoa.print_info_qaoa(f)
            
            f.write('\nGAUSSIAN PROCESS PARAMETERS')
            f.write('\n-------------\n')
            self.gp.print_info(f)
            
            f.write('\nBAYESIAN OPT PARAMETERS')
            f.write('\n-------------\n')
            f.write(f'Nwarmup points: {self.nwarmup} \n')
            f.write(f'Ntraining points: {self.nbayes}\n')
            f.write('FILE.DAT PARAMETERS:\n')
            print(data_names, file = f)
        
    def init_training(self, Nwarmup):
        X_train, y_train, data_train = self.qaoa.generate_random_points(Nwarmup)
        self.gp.fit(X_train, y_train)
        
        
        data_file_name = self.file_name + '.dat'
        kernel_params = np.exp(self.gp.kernel_.theta)
        data = [[i] + x + [y_train[i]] + data_train[i] +
                              [kernel_params[0],
                               kernel_params[1], 0, 0, 0, 0, 0, 0, 0
                              ] for i, x in enumerate(X_train)]
        format = '%3d ' + 2*self.depth*'%6d ' + (len(data[0]) - 1 - 2*self.depth)*'%4.4f '
        np.savetxt(data_file_name, data, fmt = format)
        
    def ciao():
        print('Training ...')
        for i in range(Nbayes):
            start_time = time.time()
            next_point, n_it, avg_sqr_distances, std_pop_energy = gp.bayesian_opt_step()
            next_point = [int(i) for i in next_point]
    
            bayes_time = time.time() - start_time
            y_next_point, var, fid, fid_exact, sol_ratio, _, _ = qaoa.apply_qaoa(next_point)
            qaoa_time = time.time() - start_time - bayes_time
            corr_length = gp.kernel_.get_params()['k1__length_scale']
            constant_kernel = gp.kernel_.get_params()['k2__constant_value']
            #if np.abs(np.log(np.abs(0.01-corr_length))) > 8:
             #   if i == 0:
              #      gp.kernel_.set_params(**{'k1__length_scale': data[-1][-9]})
               #     corr_length = data[-1][-9]
               # else:
               #     gp.kernel_.set_params(**{'k1__length_scale': new_data[-9]})
               #     corr_length = new_data[-9]
            print('iteration: {}/{}  {} en: {}, fid: {}'.format(i, Nbayes, next_point, y_next_point, fid))
            K = gp.covariance_matrix()
            print(K)
            L = np.linalg.cholesky(K)
            gp.fit(next_point, y_next_point)
            #gp.plot_log_marginal_likelihood(show = False, save = True)

    
            kernel_time = time.time() - start_time - qaoa_time - bayes_time
            step_time = time.time() - start_time
    
            new_data = [i+Nwarmup] + next_point + [y_next_point, var, fid, fid_exact, sol_ratio, corr_length, constant_kernel, 
                                            std_pop_energy, avg_sqr_distances, n_it, 
                                            bayes_time, qaoa_time, kernel_time, step_time]     

            data.append(new_data)
            np.savetxt(data_file_name, data, fmt = format)

        best_x, best_y, where = gp.get_best_point()
        data.append(data[where])
        np.savetxt(data_file_name, data, fmt = format)
        print('Best point: ' , data[where])
        print('time: ',  time.time() - global_time)