import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from scipy.optimize import minimize


from utils.gauss_proc import *
from utils.qaoa_pulser import *
from utils.qaoa_qiskit import *

np.random.seed(12)
np.set_printoptions(precision=4, sign="+", suppress=True)

params_kernel = {'sigma': 1, 'ell': 250}
backend = 'QISKIT'

if backend == 'PULSER':
    gamma_range = [500, 2000]   # extremes where to search for the values of gamma and beta
    beta_range = [500, 2000]
if backend == 'QISKIT':
    gamma_range = [0, np.pi]   # extremes where to search for the values of gamma and beta
    beta_range = [0, np.pi]

num_level_p = 8

Nwarmup = 10
Nbayes = 90

pos = np.array([[0., 0.],
                [-4, -7],
                [4, -7],
                [8, 6],
                [-8, 6]]
               )

def plot_acquisition_function(gamma_range, beta_range, gp, y_best):
    num_level_p = 1
    dimX = 11   # number of grid-points for imshow
    gamma_arr = np.linspace(gamma_range[0], gamma_range[1], dimX)
    beta_arr = np.linspace(beta_range[0], beta_range[1], dimX)
    Xs = np.array(list(product(*[gamma_arr, beta_arr] * num_level_p)))
    AFun = []
    for xs in Xs:
        AFun.append(gp.acq_function_optimize(xs, y_best))
    AFun = np.array(AFun)

    fig = plt.figure()
    plt.imshow(AFun.reshape((dimX, dimX)).T,
               extent=[min(gamma_arr), max(gamma_arr),
                       min(beta_arr), max(beta_arr)],
               origin="lower")

    plt.xlabel("gamma")
    plt.ylabel("beta")

    plt.colorbar()
    plt.show()

def generate_random_points():
    if backend == 'PULSER':
        gamma_random = np.random.randint(gamma_range[0],
                                         gamma_range[1],
                                         size=num_level_p
                                         )
        beta_random = np.random.randint(beta_range[0],
                                        beta_range[1],
                                        size=num_level_p)

        X = np.array([gamma_random, beta_random]).T.ravel()

        Y = apply_qaoa(X, reg, G)
    if backend == 'QISKIT':
        gamma_random = np.random.random(size=num_level_p)*(gamma_range[1]-gamma_range[0]) + gamma_range[0]
        beta_random = np.random.random(size=num_level_p)*(gamma_range[1]-gamma_range[0]) + gamma_range[0]

        X = np.array([gamma_random, beta_random]).T.ravel()
    return X

# Grafo di PULSER originale
# pos = np.array([[0,0], [0,10],[0,-10],[10,0],[-10,0]])
G = pos_to_graph(pos)
if backend == 'PULSER':
    qubits = dict(enumerate(pos))
    reg = Register(qubits)




# warm up stage
y_best = np.inf
y_train = []
X_train = []
for i in range(Nwarmup):
    X = generate_random_points()
    if backend == 'PULSER':
        Y = apply_qaoa(X, reg, G)
    if backend == 'QISKIT':
        Y = QAOA(G, X)

    if Y <= y_best:
        x_best = X
        y_best = Y
        print(f"{X}, {Y:+.4f}, accepted")
    else:
        print(f"{X}, {Y:+.4f}, not accepted")

    X_train.append(X)
    y_train.append(Y)

    print("X=", X, X.shape)

print(y_best)

# end - warm up stage

# Bayesian optimization steps
j_bayes = 0
j_step = 0
while j_bayes < Nbayes:
    j_step += 1
    gp = gaussian_process(X_train, y_train, params=params_kernel)

    X = generate_random_points()

    x0 = X

    res = minimize(gp.acq_function_optimize,
                   x0,
                   args=(y_best),
                   method='L-BFGS-B',
                   jac=gp.der_acq_function_optimize,
                   # options={'disp': True},
                   tol=1e-9)
#    print(res)

    if backend == 'PULSER':
        X = res.x.astype(int)
    if backend == 'QISKIT':
        X = res.x
    gammas = X[0::2]
    betas = X[1::2]
    # check that the values found for gamma and beta are bigger than 8ns
    if backend == 'PULSER':
        cond_gamma = (gammas <= 8)
        cond_beta = (betas <= 8)
        if (gammas <= 8).all() or (betas <= 8).all():
            print(f"{j_step}, {X}, out of range")
            # continue the optimization by throwing away these values
            continue

    j_bayes += 1
    if backend == 'PULSER':
        Y = apply_qaoa(X, reg, G)
    if backend == 'QISKIT':
        Y = QAOA(G, X)

    if Y <= y_best:
        x_best = X
        y_best = Y
        print(f"{j_bayes}, {j_step}, {X}, {Y:+.4f}, accepted")
    else:
        print(f"{j_bayes}, {j_step}, {X}, {Y:+.4f}, not accepted")

    # if num_level_p == 1:
    #    plot_acquisition_function(gamma_range, beta_range, gp, y_best)

    X_train.append(X)
    y_train.append(Y)

print()
print(x_best, y_best, f"after {Nwarmup} warmup steps and {j_bayes} bayesian optimization steps")

# measure with x_best
# count_dict,_ = quantum_loop(res.x, reg)
# plot_distribution(count_dict)