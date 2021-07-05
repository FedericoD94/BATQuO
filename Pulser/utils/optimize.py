import numpy as np
from scipy.optimize import minimize
from utils.qaoa import func, quantum_loop

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from itertools import product
from sklearn.utils.optimize import _check_optimize_result
from scipy.stats import norm
import time

running_params = []
running_state = []

#To be fixed
def callbackF(X, func):
    running_params.append(X)
    running_state.append(func(X,G))


def scipy_minimize(G, func, reg, x0, depth, 
                   method = 'Nelder-Mead',callback = False):
    '''
    Optimization with scipy standard routine
    '''
    
    if callback:
        res = minimize(func, 
                       x0 = np.random.randint(1000,10000, depth*2),
                       args=(G, reg), 
                       callback = callbackF(args = (func, G)),
                       method='Nelder-Mead', 
                       tol=1e-5,
                       options = {'maxiter': 150},
                   )
    else:
        res = minimize(func, 
                   x0 = np.random.randint(1000,10000, depth*2),
                   args = (G, reg),
                   method='Nelder-Mead', 
                   tol=1e-5,
                   options = {'maxiter': 150},
                   )
    print(res)
    
    return res

class MyGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
    
    def _set_alpha(self, alpha):
        self._alpha = alpha
    
def my_rescaler(x, min_old=1000, max_old=10000, min_new=0, max_new=1):
    
    x_sc = min_new + (max_new - min_new)/(max_old - min_old)*(x - min_old)
    
    return x_sc

def bayes_opt(G, depth, reg, verbose = False):
    
    acq_function = 'EI'
    N_train = 10
    N_test = 50 #Number of test elements
    iterations = 100
    gamma_extremes = [500,10000]*depth #extremes where to search for the values of gamma and beta
    beta_extremes = [500,10000]*depth
    
    #create dataset: We start with N random points
    X_train = []   #data
    y_train = []   #label
    
    np.random.seed(3400)
    for i in range(N_train):
        X = [np.random.randint(gamma_extremes[0],gamma_extremes[1]) for _ in range(2*depth)] 
        X_train.append(X)
        Y = func(X, G, reg)
        y_train.append(Y)

    X= np.linspace(gamma_extremes[0], gamma_extremes[1], N_test, dtype = int)
    Y= np.linspace(beta_extremes[0], beta_extremes[1], N_test, dtype = int)
    X_test = list(product(X, Y, repeat = depth))

    X_train = list(my_rescaler(np.array(X_train)))
    X_test = list(my_rescaler(np.array(X_test)))

    #create gaussian process and fit training data
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = MyGaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=9,
                                    alpha=1e-1,
                                    normalize_y = True,
                                    max_iter = 50000)
    gp.fit(X_train, y_train)

    #At each iteration we calculate the best point where to sample from
    sample_points = [0]   #We save every point that was chosen to sample from
    start = time.time()
    times = []

    if verbose:
        print('Starting GP Process...')
        print('Iter  |  Current point    |     Energy | Time \n')
    
    for i in range(iterations):
            # Test GP
            new_mean, new_sigma = gp.predict(X_test, return_std=True)

            #New_mean and new_sigma both are (N_test**2, ) arrays not reshaped yet
            mean_max = np.max(new_mean)
            mean_min = np.min(new_mean)

            #Now calculate acquisitition fn as the cumulative for every point centered around the maximum
            cdf = norm.cdf(x = new_mean, loc =  mean_max, scale = new_sigma)
            pdf = norm.pdf(x = new_mean, loc =  mean_min, scale = new_sigma)

            #The qdf is instead the probability of being lower then the lowest value of the mean (where we wanto to pick the next_value)
            qdf = 1-norm.cdf(x = new_mean, loc =  mean_min, scale = new_sigma)

            if acq_function == 'PI':
                #Next values is calculated as so just because argmax returns a number betwenn 1 and n_test instead of inside the interval
                value = np.argmax(qdf)
                next_point = X_test[value]

            if acq_function == 'EI':
                alpha_function = (new_mean - mean_min - 0.001)*qdf + new_sigma*pdf
                #argmax is a number between 0 and N_test**-1 telling us where is the next point to sample
                argmax = np.argmax(np.round(alpha_function, 3))
                next_point_normalized = X_test[argmax]

            next_point = my_rescaler(np.array(next_point_normalized), min_old=0, max_old=1, min_new=1000, max_new=10000).astype(int) 
            X_train.append(next_point_normalized)
            y_next_point = func(next_point, G, reg)
            y_train.append(y_next_point)

            gp.fit(X_train, y_train)
            sample_points.append(next_point)

            current_time = np.round(time.time() - start, 3)
            
            if verbose:
                print(i, next_point, y_next_point, current_time)
            times.append(current_time)
    
    count_dict = quantum_loop(next_point, r = reg)
    
    return count_dict
    