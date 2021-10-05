import numpy as np

import matplotlib.pyplot as plt
from itertools import product
from utils.gauss_proc import *

np.random.seed(6)
np.set_printoptions(precision=4, sign="+")

params_kernel = {'sigma': 1, 'ell': 3.25}


# def func(x):
#     return np.sin(np.pi * x / 2)**2 + 1

D = 2

dimX = 5

number_of_functions = 6
x = np.linspace(-4, 4, dimX)
Xs = np.array(list(product(*[x for _ in range(D)])))

Sigma0 = Kmatrix_gaussian(Xs, Xs)

if D == 3:
    X = np.array([[4, 4, 4], [0, 0, 0]])
    f = [2, 4]
elif D == 2:
    X = np.array([[-2, 0], [0, 0], [2, 0]])
    f = [2, 4, -3]
elif D == 1:
    X = np.array([0, 4])
    f = [2, 4]


Xs =  [-4,  4]

print(np.array(X), np.array(Xs))

A = np.array(X)
B = np.array(Xs)
w = (A - B).reshape(A.shape)
print(w)
print("w.shape", w.shape)
print("np.array(X).shape", np.array(X).shape)

gp = gaussian_process(X, f, params=params_kernel)

print("KXX")
print(np.array(Kmatrix_gaussian(X, X, params_kernel)))


print("KXXs")
print(Kmatrix_gaussian(X, [Xs], params_kernel))
print("derivative KXXs")
print(w * Kmatrix_gaussian(X, [Xs], params_kernel) / 3.25**2)
print(der_Kmatrix(X, [Xs], params_kernel))
print("----")


print("Kmatrix_gaussian(X, X, params_kernel)")
print(Kmatrix_gaussian(X, X, params_kernel))


print("gp.der_mean([Xs])")
print(gp.der_mean([Xs]))
print("bef")
a = gp.der_variance([Xs])
print(a)
print("aff")

y_best=0.1
AFun = gp.acq_function(Xs, y_best)
dAfun = gp.der_acq_function_optimize(Xs, y_best)

print("AFun=", AFun)
print("der_AFun=", dAfun)



exit()
new_ys = np.random.multivariate_normal(
    mean=new_mean, cov=new_sigma,
    size=number_of_functions)
print(Xs)
print(new_ys)


# warm up stage
Nwarmup = 2
y_best = np.inf
yi_array = []
xi_array = []
for i in range(Nwarmup):
    x_i = 2 * np.random.random() - 1
    y_i = f(x_i)

    if y_i <= y_best:
        x_best = x_i
        y_best = y_i
        print(f"{x_i:+.4f}, {y_i:+.4f}, accepted")
    else:
        print(f"{x_i:+.4f}, {y_i:+.4f}")
    xi_array.append(x_i)
    yi_array.append(y_i)
# end - warm up stage

X = xi_array
fv = yi_array


plt.plot(Xs, AFun)
plt.show()