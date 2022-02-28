from .qaoa_pulser import *
from .gaussian_process import *
import numpy as np
import time
import datetime
from ._differentialevolution import DifferentialEvolutionSolver
import pandas as pd


class Bayesian_optimization():

    def __init__(self,
                 depth,
                 type_of_graph,
                 lattice_spacing,
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
        self.qaoa = qaoa_pulser(depth, type_of_graph, lattice_spacing, quantum_noise)
        self.qaoa.calculate_physical_gs()

        ### CREATE GP 
        self.gp = MyGaussianProcessRegressor(depth = depth, kernel_choice = kernel_choice)
  
        
    def print_info(self):
        self.folder_name = 'results/'
        self.file_name = f'p={self.depth}_warmup={self.nwarmup}_train={self.nbayes}_{datetime.datetime.now().time()}'
        gamma_names = ['GAMMA_' + str(i)  for i in range(self.depth)]
        beta_names = ['BETA_' + str(i) for i in range(self.depth)]
        self.data_names = ['iter'] + gamma_names \
                                   + beta_names \
                                   + ['energy',
                                      'energy_solution',
                                      'energy_ratio', 
                                      'exact_energy',
                                      'sampled_variance', 
                                      'exact_variance',
                                      'fidelity_exact', 
                                      'fidelity_sampled', 
                                      'ratio_solution', 
                                      'corr_length', 
                                      'const_kernel',
                                      'std_energies', 
                                      'average_distances', 
                                      'nit', 
                                      'time_opt_bayes', 
                                      'time_qaoa', 
                                      'time_opt_kernel', 
                                      'time_step']
        self.data_header = " ".join(["{:>7} ".format(i) for i in self.data_names])


        info_file_name = self.folder_name + self.file_name + '_info.txt'
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
            print(self.data_names, file = f)
               
    def init_training(self, Nwarmup):
        X_train, y_train, data_train = self.qaoa.generate_random_points(Nwarmup)
        self.gp.fit(X_train, y_train)
        
        df = pd.DataFrame(np.column_stack((X_train, y_train)))
        print('### TRAIN DATA ###')
        print(df)
        print('\nKernel after training fit')
        print(self.gp.kernel_)
        print('\nStarting K')
        print(self.gp.get_covariance_matrix())
        
        kernel_params = np.exp(self.gp.kernel_.theta)
        self.data_ = []
        for i, x in enumerate(X_train):
            self.data_.append([i +1] 
                               + x  
                               +[y_train[i], 
                                self.qaoa.solution_energy, 
                                y_train[i]/ self.qaoa.solution_energy,
                                data_train[i]['exact_energy'],
                                data_train[i]['sampled_variance'], 
                                data_train[i]['exact_variance'],
                                data_train[i]['sampled_fidelity'],
                                data_train[i]['exact_fidelity'], 
                                data_train[i]['solution_ratio']] 
                               +[kernel_params[0], 
                                 kernel_params[1], 
                                 0, 0, 0, 0, 0, 0, 0])
            
        self.data_file_name = self.file_name + '.dat'
        
               
    def acq_func(self, x):
        
        #check if acq_func is being evaluated on one point (needs reshaping) or many
        if isinstance(x[0], float):
            x = np.reshape(x, (1, -1))
            
        f_x, sigma_x = self.gp.predict(x, return_std=True) 
    
        f_prime = self.gp.y_best #current best value
        
        #Ndtr is a particular routing in scipy that computes the CDF in half the time
        cdf = ndtr((f_prime - f_x)/sigma_x)
        pdf = 1/(sigma_x*np.sqrt(2*np.pi)) * np.exp(-((f_prime -f_x)**2)/(2*sigma_x**2))
        alpha_function = (f_prime - f_x) * cdf + sigma_x * pdf
        
        return alpha_function
        
        
    def acq_func_maximize(self, x):
    
        return (-1)*self.acq_func(x)
        
    def bayesian_opt_step(self, init_pos = None):
        
        samples = []
        acqfunvalues = []

        #callback to save progress data
        def callbackF(Xi, convergence):
            samples.append(Xi.tolist())
            acqfunvalues.append(self.acq_func(Xi, 1)[0])

        repeat = True
        with DifferentialEvolutionSolver(self.acq_func_maximize,
                                         bounds = [(0,1), (0,1)]*self.depth,
                                         callback = None,
                                         maxiter = 100*self.depth,
                                         popsize = 15,
                                         tol = .001,
                                         dist_tol = DEFAULT_PARAMS['distance_conv_tol'],
                                         seed = DEFAULT_PARAMS['seed']
                                         ) as diff_evol:
            results,average_norm_distance_vectors, std_population_energy, conv_flag = diff_evol.solve()
            next_point = results.x
        
            next_point = self.gp.scale_up(next_point)
                
        return next_point, results.nit, average_norm_distance_vectors, std_population_energy

    def check_proposed_point(self, point):
        X_ = self.gp.get_X()
        
        if point in X_:
            return False
        else:
            return True
            
    def run_optimization(self):
        print('Training ...')
        for i in range(self.nbayes):
            start_time = time.time()
            next_point, n_it, avg_sqr_distances, std_pop_energy = self.bayesian_opt_step()
            next_point = [int(i) for i in next_point]
            check_ = self.check_proposed_point(next_point)
            if not check_:
                print('Found the same point twice {next_point} by the optimization')
                print('ending optimization')
                break
            
            bayes_time = time.time() - start_time
            
            qaoa_results = self.qaoa.apply_qaoa(next_point)
            y_next_point = qaoa_results['sampled_energy']
            
            qaoa_time = time.time() - start_time - bayes_time
            
            constant_kernel,corr_length = np.exp(self.gp.kernel_.theta)
            
            print(f'iteration: {i +1}/{self.nbayes}  {next_point}'
                    f' en/ratio: {y_next_point/self.qaoa.solution_energy}'
                    ' en: {}, fid: {}'.format(y_next_point, qaoa_results['sampled_fidelity'])
                    )
            
            self.gp.fit(next_point, y_next_point)
            self.gp.get_log_marginal_likelihood(show = False, save = False)
            kernel_time = time.time() - start_time - qaoa_time - bayes_time
            step_time = time.time() - start_time

            new_data = ([i+self.nwarmup] 
                           + next_point  
                           + [y_next_point, 
                             self.qaoa.solution_energy, 
                             y_next_point/self.qaoa.solution_energy, 
                             qaoa_results['exact_energy'],
                             qaoa_results['sampled_variance'], 
                             qaoa_results['exact_variance'],
                             qaoa_results['sampled_fidelity'],
                             qaoa_results['exact_fidelity'], 
                             qaoa_results['solution_ratio'], 
                             corr_length, 
                             constant_kernel, 
                             std_pop_energy, 
                             avg_sqr_distances, 
                             n_it, 
                             bayes_time, 
                             qaoa_time, 
                             kernel_time, 
                             step_time]  )

            # format_list = ['%+.6f '] * len(new_data)
#             format_list[0] = '% 4d '
#             self.fmt_string = "".join(format_list) 

            self.data_.append(new_data)
            df = pd.DataFrame(data = self.data_, columns = self.data_names)
            df.to_csv(self.folder_name + self.data_file_name, columns = self.data_names, header = self.data_header)
            #np.savetxt(self.folder_name + self.data_file_name, self.data_, fmt = self.fmt_string, header = self.data_header)
            
        best_x, best_y, where = self.gp.get_best_point()
        self.data_.append(self.data_[where])
        df = pd.DataFrame(data = self.data_, columns = self.data_names)
        df.to_csv(self.folder_name + self.data_file_name, columns = self.data_names, header = self.data_header)
        #np.savetxt(self.folder_name + self.data_file_name, self.data_, fmt =  self.fmt_string, header = self.data_header)
        print('Best point: ' , self.data_[where])
