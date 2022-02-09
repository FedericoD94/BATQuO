import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('not_converg_diff_ev.txt')

print(A)


plt.plot(np.log(A[:, 0]))
plt.axhline(y = np.log(1e-2),  c = 'r')
plt.xlabel('Steps')
plt.ylabel('log avg distances^2')
plt.show()