import numpy as np
import matplotlib.pyplot as plt

import pdb

def inner_prod(a, b):
    # ans = np.sum(np.multiply(a,b))
    # ans = np.tensordot(a.reshape((len(a), 1)), b.reshape((len(b), 1)))
    ans = np.tensordot(a, b)
    return ans

def mod_cholesky(A):
    L = np.linalg.cholesky(A)
    D_diag = np.diag(L).copy()
    rows, cols = A.shape
    for i in range(cols):
        L[:,i] /= D_diag[i]
    D = np.diag(D_diag**2)

    return L, D

def diag_scaling(A, b):
    D_inv = np.diag(1.0/np.diag(A))
    A = np.dot(D_inv, A)
    b = np.dot(D_inv, b)
    # pdb.set_trace()
    return A, b

x = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
A = np.array([[100006.000, 100009.000, 100002.000, 100005.000, 100004.000, 100002.000, 100001.000, 100005.000],
              [100001.000,100001.000,100000.000,100003.000,100000.000,100002.000,100002.000,100009.000],
              [100007.000,100003.000,100001.000,100009.000,100003.000,100008.000,100000.000,100005.000],
              [100005.000,100007.000,100009.000,100004.000,100005.000,100008.000,100000.000,100005.000],
              [100003.000,100004.000,100000.000,100009.000,100000.000,100005.000,100002.000,100007.000],
              [100004.000,100002.000,100002.000,100006.000,100009.000,100003.000,100004.000,100006.000],
              [100008.000,100001.000,100008.000,100006.000,100009.000,100002.000,100007.000,100004.000],
              [100009.000,100005.000,100000.000,100006.000,100009.000,100001.000,100005.000,100008.000]])
b = np.array([[800034.000], [800018.000], [800036.000], [800043.000], [800030.000], [800036.000], [800045.000], [800043.000]])

# A -= 100000.0
# b -= 800000.0

P_inv = np.dot(A[0:2,:], A[0:2,:].T)
L, D = mod_cholesky(P_inv)

D_inv = np.diag(1.0/np.diag(D))
L_inv = np.linalg.inv(L)
P = np.dot(L_inv.T, np.dot(D_inv, L_inv))
x = np.dot(A[0:2,:].T, np.dot(P, b[0:2]))

A, b = diag_scaling(A, b)

m = np.dot(A.T, np.dot(A, x) - b)
mat0 = np.dot(A.T, np.dot(A, x) - b)
mat1 = np.dot(A.T, np.dot(A, m))
t = -(inner_prod(m, mat0))/(inner_prod(m, mat1))
# x = x + t*m
x = x + 1e-6
# pdb.set_trace()

error_list = []
for i in range(100):
    # print(i)
    mat0 = np.dot(A.T, np.dot(A, np.dot(A.T, (np.dot(A, x)-b))))
    mat1 = np.dot(A.T, np.dot(A, m))
    alpha = - (inner_prod(m, mat0))/(inner_prod(m, mat1))

    m = (np.dot(A.T, np.dot(A, x) - b) + alpha*m)
    mat0 = np.dot(A.T, np.dot(A, x) - b)
    mat1 = np.dot(A.T, np.dot(A, m))
    t = -(inner_prod(m, mat0))/(inner_prod(m, mat1))
    x = x + t*m

    # pdb.set_trace()
    error_list.append(np.sum(abs(1 - x)))

plt.plot(error_list)
plt.show()
pdb.set_trace()
