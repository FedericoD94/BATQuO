import numpy as np
import matplotlib.pyplot as plt

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
from sklearn.preprocessing import StandardScaler
import random


# Allows to change max_iter (see cell below) as well as gtol.
# It can be straightforwardly extended to other parameters
class MyGaussianProcessRegressor(GaussianProcessRegressor):
	def __init__(self, *args, optimizer = "fmin_l_bfgs_b", param_range = [0.1, np.pi], max_iter=2e05, gtol=1e-06, **kwargs):
		alpha = 1/np.sqrt(DEFAULT_PARAMS['shots'])
		super().__init__(optimizer = optimizer, alpha = alpha, *args, **kwargs)
		self._max_iter = max_iter
		self._gtol = gtol
		self.param_range = param_range
		self.X = []
		self.Y = []
		self.x_best = 0
		self.y_best = np.inf
		self.seed = DEFAULT_PARAMS["seed"]

	def _constrained_optimization(self,
								  obj_func,
								  initial_theta,
								  bounds):
		'''
		Do not change the elif option
		'''
		
		def obj_func_no_grad(x):
				return 	obj_func(x)[0]
				
		if self.optimizer == "fmin_l_bfgs_b":
			opt_res = minimize(obj_func,
									   initial_theta,
									   method="L-BFGS-B",
									   jac=True,
									   bounds=bounds,
									   options={'maxiter': self._max_iter,
												'gtol': self._gtol}
									   )
			_check_optimize_result("lbfgs", opt_res)
			
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


		elif callable(self.optimizer):
			theta_opt, func_min = self.optimizer(obj_func,
												 initial_theta,
												 bounds=bounds)
		else:
			raise ValueError("Unknown optimizer %s." % self.optimizer)
		return theta_opt, func_min

	def fit(self, new_point, y_new_point):
		new_point = self.scale_down(new_point)

		if isinstance(new_point[0], float):
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

		super().fit(self.X, self.Y)

	def my_rescaler(self, x, min_old, max_old, min_new, max_new):
		'Rescales one or more than one point(s) at a time'
		norm = []
		if isinstance(x[0], float) or isinstance(x[0], int):
			for i in x:
				norm.append(min_new + (max_new - min_new)/(max_old - min_old)*(i - min_old))
		else:
			for i in x:
				norm.append([min_new + (max_new - min_new)/(max_old - min_old)*(j - min_old) for j in i])

		return norm

	def scale_down(self, x):
		'Rescales one or more than one point(s) at a time'

		min_old=self.param_range[0]
		max_old=self.param_range[1]
		min_new=0
		max_new=1

		x = self.my_rescaler(x, min_old, max_old, min_new, max_new)

		return x


	def scale_up(self, x):
		'Rescales one or more than one point(s) at a time'

		min_old=0
		max_old=1
		min_new=self.param_range[0]
		max_new=self.param_range[1]

		x = self.my_rescaler(x, min_old, max_old, min_new, max_new)

		return x

	def get_best_point(self):

		x_best = self.scale_up(self.x_best)
		where = np.argwhere(self.y_best == np.array(self.Y))
		return x_best, self.y_best, where[0,0]

	def acq_func(self, x, *args):
		self = args[0]
		try:
			sign = args[1]
		except:
			sign = 1.0
		if isinstance(x[0], float):
			x = np.reshape(x, (1, -1))
		f_x, sigma_x = self.predict(x, return_std=True)

		K_xx = sigma_x**2
		f_prime = np.min(self.Y)

		cdf = norm.cdf(x = f_prime, loc = f_x , scale = sigma_x)
		pdf = norm.pdf(x = f_prime, loc = f_x , scale = sigma_x)
		alpha_function = (f_prime - f_x) * cdf + sigma_x * pdf

		return sign*alpha_function

	def bayesian_opt_step(self, init_pos, method = 'FD'):
		depth = int(len(init_pos)/2)

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
											initial_position = init_pos,
											path_len=path_length,
											step_size=step_size,
											epsilon = epsilon,
											epochs=epochs)
			cols = int(len(init_pos)*4)
			rows = int((path_length/step_size+1)*(epochs))
			traj_to_print = np.reshape(traj, (rows, cols))
			np.savetxt('Trajectories.dat',
						traj_to_print,
						header='[q_i, q_f],	 [p_i, p_f],  self.acq_func(q1), dVdQ_1, dVdq_2',
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
			gd_params = gradient_descent(self.acq_func,
										l_rate,
										init_pos,
										max_iter = 400,
										method = method,
										verbose = 1)
			next_point = gd_params[-1][:2]

		if method == 'SCIPY':
			results = minimize(self.acq_func,
								bounds = [(0,1), (0,1)]*depth,
								x0 = init_pos,
								args = (self, -1))
			next_point = results.x

		if method == 'DIFF-EVOL':
			with DifferentialEvolutionSolver(self.acq_func,
											bounds = [(0,1), (0,1)]*depth,
											callback = callbackF,
											maxiter = 100,
											popsize = 15,
											tol = .001,
											dist_tol = 0.01,
											seed = self.seed,
											args = (self, -1)) as diff_evol:
				results,average_norm_distance_vectors, std_population_energy, conv_flag = diff_evol.solve()
			next_point = results.x
			success = results.success
		next_point = self.scale_up(next_point)
		return next_point, results.nit, average_norm_distance_vectors, std_population_energy


	def plot_landscape(self, show = True, save = False):
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
		im2 = plt.scatter(samples[:, 0], samples[:,1], marker = '+', c = self.Y, cmap = 'Reds')
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
		plt.title('ACQ F iter:{} kernel_{}'.format(len(self.X), self.kernel_))

		if save:
			plt.savefig('../data/raw/gp_LS_iter_{}_kernel_{}.png'.format(len(self.X), self.kernel_))
		if show:
			plt.show()

