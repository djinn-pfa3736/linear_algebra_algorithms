import numpy as np

import matplotlib.pyplot as plt

import pdb

def mod_cholesky(A):
    L = np.linalg.cholesky(A)
    D_diag = np.diag(L).copy()
    rows, cols = A.shape
    for i in range(cols):
        L[:,i] /= D_diag[i]
    D = np.diag(D_diag**2)

    return L, D

x = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
A = np.array([[100006.000, 100009.000, 100002.000, 100005.000, 100004.000, 100002.000, 100001.000, 100005.000],
              [100001.000,100001.000,100000.000,100003.000,100000.000,100002.000,100002.000,100009.000],
              [100007.000,100003.000,100001.000,100009.000,100003.000,100008.000,100000.000,100005.000],
              [100005.000,100007.000,100009.000,100004.000,100005.000,100008.000,100000.000,100005.000],
              [100003.000,100004.000,100000.000,100009.000,100000.000,100005.000,100002.000,100007.000],
              [100004.000,100002.000,100002.000,100006.000,100009.000,100003.000,100004.000,100006.000],
              [100008.000,100001.000,100008.000,100006.000,100009.000,100002.000,100007.000,100004.000],
              [100009.000,100005.000,100000.000,100006.000,100009.000,100001.000,100005.000,100008.000]])
b = np.array([[800034.000], [800018.000], [800036.000], [800043.000], [800030.000], [800036.000], [800045.000],[800043.000]])

# A -= 100000.0
# b -= 800000.0

rows, cols = A.shape

P_inv = np.dot(A[0:2,:], A[0:2,:].T)
L, D = mod_cholesky(P_inv)

D_inv = np.diag(1.0/np.diag(D))
L_inv = np.linalg.inv(L)
# pdb.set_trace()

d_00 = P_inv[0, 0]
l_10 = P_inv[1, 0]*(1/d_00)
d_11 = P_inv[1, 1] - P_inv[1, 0]*l_10
l_00 = l_11 = 1
l_01 = d_01 = d_10 = 0

L_inv_calc = [[l_00, l_01], [l_10, l_11]]
D_inv_calc = [[1.0/d_00, d_01], [d_10, 1.0/d_11]]

P = np.dot(L_inv.T, np.dot(D_inv, L_inv))
x = np.dot(A[0:2,:].T, np.dot(P, b[0:2,0]))

for i in range(1, rows-1):
    c_next = np.dot(A[0:(i+1),:], A[(i+1),:].T)
    l_next = np.dot(D_inv, np.dot(L_inv, c_next))

    d_next = np.dot(A[(i+1),:], A[(i+1),:].T) - np.dot(l_next.T, np.dot(D, l_next))
    g_next_T = np.dot(l_next.T, L_inv)
    # g_next_T = g_next_T.reshape((1, len(g_next_T)))
    # term0 = np.dot(A[0:(i+1),:].T, g_next_T.T) - A[(i+1),:].reshape((8, 1))
    term0 = np.dot(A[0:(i+1),:].T, g_next_T.T) - A[(i+1),:].T
    term1 = np.dot(g_next_T, b[0:(i+1),0]) - b[(i+1),0]
    x += (1.0/d_next)*term0*term1
    # pdb.set_trace()
    # print(x)

    row, col = L_inv.shape
    L_tmp = np.hstack([L_inv.copy(), np.zeros((row, 1))])
    tmp = (-1*g_next_T).tolist()
    tmp.append(1.0)
    g_tmp = np.array(tmp)

    L_inv = np.vstack([L_tmp, g_tmp])
    D_inv = np.diag(np.hstack([np.diag(D_inv).copy(), 1.0/d_next]))
    D = np.diag(1.0/np.diag(D_inv)).copy()
    # pdb.set_trace()

pdb.set_trace()
