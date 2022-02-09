import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter

from scipy.optimize import minimize
from ._differentialevolution import DifferentialEvolutionSolver
from .HMC import *
from .grad_descent import *
from  utils.default_params import *
# SKLEARN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from itertools import product
from sklearn.utils.optimize import _check_optimize_result
from scipy.stats import norm
from scipy.special import ndtr
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cholesky, cho_solve, solve_triangular

import random
import dill
import warnings
# warnings.filterwarnings("error")
from sklearn.exceptions import ConvergenceWarning


# Allows to change max_iter (see cell below) as well as gtol.
# It can be straightforwardly extended to other parameters
class MyGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, kernel,optimizer, n_restarts_optimizer, param_range, 
                gtol, max_iter, normalize_y):
        '''Initializes gaussian process class
        
        The class also inherits from Sklearn GaussianProcessRegressor
        Attributes for MYgp
        --------
        param_range : range of the angles beta and gamma
        
        gtol: tolerance of convergence for the optimization of the kernel parameters
        
        max_iter: maximum number of iterations for the optimization of the kernel params
        
        Attributes for SKlearn GP:
        ---------
        
        *args, **kwargs: kernel, optimizer_kernel, 
                         n_restarts_optimizer (how many times the kernel opt is performed)
                         normalize_y: standard is yes
        '''
        alpha = 1/np.sqrt(DEFAULT_PARAMS['shots'])
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.alpha = alpha
        self.kernel = kernel
        self.max_iter = max_iter
        self.gtol = gtol
        self.param_range = param_range
        self.X = []
        self.Y = []
        self.x_best = 0
        self.y_best = np.inf
        self.seed = DEFAULT_PARAMS["seed"]
        self.K = []
        
    def get_info(self):
        '''
        Returns a dictionary of infos on the  gp to print 
        '''
        info ={}
        info['param_range'] = self.param_range
        info['acq_fun_optimization_max_iter'] = self.max_iter
        info['seed'] = self.seed
        info['gtol'] = self.gtol
        info['alpha'] = self.alpha
        info['kernel_optimizer'] = self.optimizer
        info['kernel_info'] = self.kernel.get_params()
        info['n_restart_kernel_optimizer'] = self.n_restarts_optimizer
        info['normalize_y'] = self.normalize_y
        
        return info
        
    def print_info(self, f):
        f.write(f'parameters range: {self.param_range}\n')
        f.write(f'acq_fun_optimization_max_iter: {self.max_iter}\n')
        f.write(f'seed: {self.seed}\n')
        f.write(f'tol opt kernel: {self.gtol}\n')
        f.write(f'energy noise alpha 1/sqrt(N): {self.alpha}\n')
        f.write(f'kernel_optimizer: {self.optimizer}\n')
        f.write(f'kernel info: {self.kernel.get_params()}\n')
        f.write(f'n_restart_kernel_optimizer: {self.n_restarts_optimizer}\n')
        f.write(f'normalize_y: {self.normalize_y}\n')
        f.write('\n')
        
    def _constrained_optimization(self,
                                  obj_func,
                                  initial_theta,
                                  bounds):
        '''
        Overrides the super()._constrained_optimization to perform otpimization of the kernel
        parameters by maximizing the log marginal likelihood.
        It is only called by super().fit, so at every fitting of the training points
        or at a new bayesian opt step. Options for the optimization are fmin_l_bfgs_b, 
        differential_evolution or monte_carlo. The latter just averages over M = 30 values
        of the hyperparameters and returns this average.
        
        Do not change the elif option
        '''
        
        def obj_func_no_grad(x):
                return  obj_func(x)[0]
                
        if self.optimizer == "fmin_l_bfgs_b":
#            try:
            opt_res = minimize(obj_func,
                                           initial_theta,
                                           method="L-BFGS-B",
                                           jac=True,
                                           bounds=bounds,
                                           options={'maxiter': self.max_iter,
                                                    'gtol': self.gtol}
                                           )
            _check_optimize_result("lbfgs", opt_res)
#            except ConvergenceWarning:
#                print("BBBBB")
 #               exit()                    
            theta_opt, func_min = opt_res.x, opt_res.fun


        elif self.optimizer == 'differential_evolution':
            diff_evol = DifferentialEvolutionSolver(obj_func_no_grad,
                                            x0 = initial_theta,
                                            bounds = bounds,
                                            popsize = 15,
                                            tol = .001,
                                            dist_tol = 0.01,
                                            seed = self.seed)# as diff_evol:
            results,average_norm_distance_vectors, std_population_energy, conv_flag = diff_evol.solve()
            theta_opt, func_min = results.x, results.fun

        elif self.optimizer == 'monte_carlo':
            M = 30
            hyper_params = self.pick_hyperparameters(M, DEFAULT_PARAMS['length_scale_bounds'], DEFAULT_PARAMS['constant_bounds'])
            theta_opt = np.mean(np.log(hyper_params), axis = 0)
            func_min = obj_func_no_grad(theta_opt)

        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func,
                                                 initial_theta,
                                                 bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

    def fit(self, new_point, y_new_point):
        '''Fits the GP to the new point(s)
        
        Appends the new data to the myGP instance and keeps track of the best X and Y.
        Then uses the inherited fit method which optimizes the kernel (by maximizing the
        log marginal likelihood) with kernel_optimizer for 1 + n_restart_optimier_kernel 
        times and keeps the best value. All points are scaled down to [0,1]*depth.
        
        Attributes
        ---------
        new_point, y_new_point: either list or a single new point with their/its energy
        '''
        new_point = self.scale_down(new_point)

        if isinstance(new_point[0], float): #check if its only one point
            self.X.append(new_point)
            self.Y.append(y_new_point)
            if y_new_point < self.y_best:
                self.y_best = y_new_point
                self.x_best = new_point
        else:
            for i, point in enumerate(new_point):
                self.X.append(point)
                self.Y.append(y_new_point[i])

                if y_new_point[i] < self.y_best:
                    self.y_best = y_new_point[i]
                    self.x_best = point
        
        # Normalize the target values of the dataset
        if self.normalize_y:
            self.y_train_mean = np.mean(self.Y, axis=0)
            self.y_train_std = np.std(self.Y, axis=0)

        else:
            self.y_train_mean = np.zeros(1)
            self.y_train_std = 1
        
        # Remove mean and make unit variance
        self.Y_train = (self.Y - self.y_train_mean) / self.y_train_std
            
        if self.optimizer is not None and self.kernel.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood potentially starting 
            
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel.theta,
                                                      self.kernel.bounds))]
           
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                 if not np.isfinite(self.kernel.bounds).all():
                     raise ValueError(
                         "Multiple optimizer restarts (n_restarts_optimizer>0) "
                         "requires that all bounds are finite.")
                 bounds = self.kernel.bounds
                 for iteration in range(self.n_restarts_optimizer):
                     theta_initial = np.random.uniform(bounds[:, 0], bounds[:, 1])
                     optima.append(self._constrained_optimization(obj_func, theta_initial,
                                                        bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel.theta = optima[np.argmin(lml_values)][0]
            self.kernel._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel.theta)
                
       
        return self

    def predict(self, X, return_std=False, return_cov=False):
        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel(self.X)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.Y_train)  # Line 3
    
        K_trans = self.kernel(X, self.X)
        y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)

        # undo normalisation
        y_mean = self.y_train_std * y_mean + self.y_train_mean

        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T)  # Line 5
            y_cov = self.kernel(X) - K_trans.dot(v)  # Line 6

            # undo normalisation
            y_cov = y_cov * self.y_train_std**2

            return y_mean, y_cov
        elif return_std:
            # cache result of K_inv computation
            if self._K_inv is None:
                # compute inverse K_inv of K based on its Cholesky
                # decomposition L and its inverse L_inv
                L_inv = solve_triangular(self.L_.T,
                                         np.eye(self.L_.shape[0]))
                self._K_inv = L_inv.dot(L_inv.T)

            # Compute variance of predictive distribution
            y_var = self.kernel.diag(X)
            y_var -= np.einsum("ij,ij->i",
                               np.dot(K_trans, self._K_inv), K_trans)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0

            # undo normalisation
            y_var = y_var * self.y_train_std**2

            return y_mean, np.sqrt(y_var)
        else:
            return y_mean
            
    def log_marginal_likelihood(self, theta=None, eval_gradient = False):
        
        if theta is not None:
            self.kernel.theta = theta                
        if eval_gradient:
            K, K_gradient = self.kernel(self.X, eval_gradient=True)
        else:
            K = self.kernel(self.X)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf
        # Support multi-dimensional output of self.y_train_
        Y = np.array(self.Y_train)
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        alpha = cho_solve((L, True), Y)  # Line 3

        # Compute log-likelihood (compare line 7)
        #It is identical to calculate log_marg_like = -0.5*np.dot(self.Y_train, np.dot(np.linalg.inv(K), np.transpose(y_train))) 
                                                    #- 0.5*np.log(np.linalg.det(K)) 
                                                    #-Nwarmup/2*np.log(2*np.pi)
        
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", Y, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimension
        
        

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def my_rescaler(self, x, min_old, max_old, min_new, max_new):
        '''Rescales one or more than one point(s) at a time 
        
        Attributes
        ---------
        x: point(s)
        min_old, max_old: the current param range of x
        min_new, max_new: the next param range of x
        '''
        norm = []
        if isinstance(x[0], float) or isinstance(x[0], int):
            for i in x:
                norm.append(min_new + (max_new - min_new)/(max_old - min_old)*(i - min_old))
        else:
            for i in x:
                norm.append([min_new + (max_new - min_new)/(max_old - min_old)*(j - min_old) for j in i])

        return norm

    def scale_down(self, x):
        'Rescales a(many) point(s) from param range to [0,1]'

        min_old=self.param_range[0]
        max_old=self.param_range[1]
        min_new=0
        max_new=1

        x = self.my_rescaler(x, min_old, max_old, min_new, max_new)

        return x


    def scale_up(self, x):
        'Rescales a(many)point(s) from [0,1] to param range'

        min_old=0
        max_old=1
        min_new=self.param_range[0]
        max_new=self.param_range[1]

        x = self.my_rescaler(x, min_old, max_old, min_new, max_new)

        return x

    def get_best_point(self):
        '''Return the current best point with its energy and position'''
        x_best = self.scale_up(self.x_best)
        where = np.argwhere(self.y_best == np.array(self.Y_train))
        return x_best, self.y_best, where[0,0]

    def acq_func(self, x, *args):
        '''Expected improvement at point x
        
        Arguments
        ---------
        x: point of the prediction
        *args: the sign of the acquisition function (in case you have to minimize -acq fun)
        
        '''
        self = args[0]
        try:
            sign = args[1]
        except:
            sign = 1.0
        if isinstance(x[0], float):
            x = np.reshape(x, (1, -1))
        f_x, sigma_x = self.predict(x, return_std=True) 
        f_prime = self.y_best #current best value
        
        #Ndtr is a particular routing in scipy that computes the CDF in half the time
        cdf = ndtr((f_prime - f_x)/sigma_x)
        pdf = 1/(sigma_x*np.sqrt(2*np.pi)) * np.exp(-((f_prime -f_x)**2)/(2*sigma_x**2))
        alpha_function = (f_prime - f_x) * cdf + sigma_x * pdf
        return sign*alpha_function
        
        
    def pick_hyperparameters(self, N_points, bounds_elle, bounds_sigma):
        ''' Generates array of (N_points,2) random hyperaparameters inside given bounds
        '''
        elle = np.random.uniform(low = bounds_elle[0], high=bounds_elle[1], size=(N_points,))
        sigma = np.random.uniform(low = bounds_sigma[0], high=bounds_sigma[1], size=(N_points,))
        hyper_params = list(zip(elle,sigma))
        
        return hyper_params
        
    def mc_acq_func(self, x, *args):
        ''' Averages the value of the acq_func for different sets of hyperparameters chosen
        (for now) by uniform sampling '''
        
        hyper_params = args[-1]
        acq_func_values = []
        for params in hyper_params:
            self.kernel.set_params(**{'k1__length_scale': params[0]})
            self.kernel.set_params(**{'k2__constant_value': params[1]})
            acq_func_values.append(self.acq_func(x, *args))
            
        return np.mean(acq_func_values)
        
    def bayesian_opt_step(self, method = 'DIFF-EVOL', init_pos = None):
        ''' Performs one step of bayesian optimization
        
        It uses the specified method to maximize the acquisition function
        
        Attributes
        ----------
        method: choice between grid search (not available for depth >1), hamiltonian monte 
                carlo, finite differences gradient, scipy general optimization, diff evolution
                
        Returns (for diff evolution)
        -------
        next_point: where to sample next (rescaled to param range)
        nit: number of iterations of the diff evolution algorithm
        avg_norm_dist_vect: condition of convergence for the positions 
        std_pop_energy: condition of convergence for the energies
        '''
        depth = int(len(self.X[0])/2)

        samples = []
        acqfunvalues = []

        def callbackF(Xi, convergence):
            samples.append(Xi.tolist())
            acqfunvalues.append(self.acq_func(Xi, self, 1)[0])

        if method == 'GRID_SEARCH':
            if depth >1:
                print('PLS No grid search with p > 1')
                raise Error
            X= np.linspace(0, 1, 50, dtype = int)
            Y= np.linspace(0,1, 50, dtype = int)
            X_test = list(product(X, Y))


            alpha = self.acq_func(X_test, self, 1)
            #argmax is a number between 0 and N_test**-1 telling us where is the next point to sample
            argmax = np.argmax(np.round(alpha, 3))
            next_point = X_test[argmax]
            
        if method == 'HMC':
            path_length = .2
            step_size = .02
            epsilon = .001
            epochs = 10
            hmc_samples, traj, success= HMC(func = self.acq_func,
                                            path_len=path_length,
                                            step_size=step_size,
                                            epsilon = epsilon,
                                            epochs=epochs)
            cols = int(len(init_pos)*4)
            rows = int((path_length/step_size+1)*(epochs))
            traj_to_print = np.reshape(traj, (rows, cols))
            np.savetxt('Trajectories.dat',
                        traj_to_print,
                        header='[q_i, q_f],  [p_i, p_f],  self.acq_func(q1), dVdQ_1, dVdq_2',
                        fmt='  %1.3f')

            accepted_samples = hmc_samples[success == True]
            if len(accepted_samples) == 0:
                print('Warning: there where 0 accepted samples. Restarting with new values')
                next_point = 0

            if accepted_samples.shape[0] > 0:
                next_point = np.mean(accepted_samples[-10:], axis = 0)
            else:
                next_point = [0,0]

            if epochs< 100:
                plot_trajectories(self.acq_func, traj, success, save = True)

        if method == 'FD':
            l_rate = 0.001
            if init_pos is None:
                print('You need to pass an initial point if using Finite differences')
                raise Error
            gd_params = gradient_descent(self.acq_func,
                                        l_rate,
                                        init_pos,
                                        max_iter = 400,
                                        method = method,
                                        verbose = 1)
            next_point = gd_params[-1][:2]

        if method == 'SCIPY':
            if init_pos is None:
                print('You need to pass an initial point if using scipy')
                raise Error
            results = minimize(self.acq_func,
                                bounds = [(0,1), (0,1)]*depth,
                                x0 = init_pos,
                                args = (self, -1))
            next_point = results.x

        if method == 'DIFF-EVOL':
            hyper_params = self.pick_hyperparameters(30, DEFAULT_PARAMS['length_scale_bounds'], DEFAULT_PARAMS['constant_bounds'])
            with DifferentialEvolutionSolver(self.acq_func,
                                            bounds = [(0,1), (0,1)]*depth,
                                            callback = None,
                                            maxiter = 100*depth,
                                            popsize = 15,
                                            tol = .001,
                                            dist_tol = DEFAULT_PARAMS['distance_conv_tol'],
                                            seed = self.seed,
                                            args = (self, -1, hyper_params)) as diff_evol:
                results,average_norm_distance_vectors, std_population_energy, conv_flag = diff_evol.solve()
            next_point = results.x
        next_point = self.scale_up(next_point)
        return next_point, results.nit, average_norm_distance_vectors, std_population_energy

    def covariance_matrix(self):
        K = self.kernel_(self.X)
        K[np.diag_indices_from(K)] += self.alpha
        
        return K
        
    def plot_covariance_matrix(self, show = True, save = False):
        K = self.covariance_matrix()
        fig = plt.figure()
        im = plt.imshow(K, origin = 'upper')
        plt.colorbar(im)
        if save:
            plt.savefig('data/cov_matrix_iter={}.png'.format(len(self.X), self.kernel_))
        if show:
             plt.show()

    def plot_posterior_landscape(self, show = True, save = False):
        if len(self.X[0]) > 2:
            raise ValueError(
                        "Non si puo plottare il landscape a p>1"
                    )

        fig = plt.figure()
        num = 100
        x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                x[j, i] = self.predict(np.reshape([i/num,j/num], (1, -1)))
                #NOTARE LO scambio di j, i necessario per fare in modo che gamma sia x e beta sia y!
        im = plt.imshow(x, extent = [0,1,0,1], origin = 'lower')
        samples = np.array(self.X)
        im2 = plt.scatter(samples[:, 0], samples[:,1], marker = '+', c = self.Y_train, cmap = 'Reds')
        plt.title('Landscape at {} sampled points'.format(len(self.X)))
        plt.xlabel('Gamma')
        plt.ylabel('Beta')
        plt.colorbar(im)
        plt.colorbar(im2)
        plt.show()
        

    def plot_acquisition_function(self, show = True, save = False):
        if len(self.X[0]) > 2:
            raise ValueError(
                        "Non si puo plottare l'AF a p>1"
                    )
        fig = plt.figure()
        num = 50
        x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                x[j, i] = self.acq_func([i/num,j/num],self, 1)
        im = plt.imshow(x, extent = [0,1,0,1], origin = 'lower')

        samples = np.array(self.X)
        plt.scatter(samples[:len(self.X), 0], samples[:len(self.X),1], marker = '+', c = 'g')
        plt.scatter(samples[-1, 0], samples[-1,1], marker = '+', c = 'r')
        plt.colorbar(im)
        plt.title('data/ACQ F iter:{} kernel_{}'.format(len(self.X), self.kernel_))

        if save:
            plt.savefig('data/acq_fun_iter={}.png'.format(len(self.X), self.kernel_))
        if show:
            plt.show()
            
    def plot_log_marginal_likelihood(self, show = True, save = False):
        fig = plt.figure()
        num = 50
        x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                x[j, i] = self.log_marginal_likelihood([i/num*2,j/num*2])
        im = plt.imshow(x, extent = [0,2,0,2], origin = 'lower')
        plt.xlabel('Corr length')
        plt.ylabel('Constant')
        plt.colorbar(im)
        plt.title('log_marg_likelihood iter:{} kernel_{}'.format(len(self.X), self.kernel_))

        if save:
            plt.savefig('data/marg_likelihood_iter={}_kernel={}.png'.format(len(self.X), self.kernel_))
        if show:
            plt.show()

