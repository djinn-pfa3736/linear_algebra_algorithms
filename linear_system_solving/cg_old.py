import numpy as np
import matplotlib.pyplot as plt
import pdb
def inner_prod(a, b):
    ans = np.sum(np.multiply(a,b))
    return ans

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
alpha = 0

A -= 100000
b -= 800000

m = np.dot(A.T, np.dot(A, x)-b)
mat0 = np.dot(A.T, np.dot(A, x)-b)
mat1 = np.dot(A.T, np.dot(A, m))
t = -(inner_prod(m, mat0))/(inner_prod(m, mat1))
x = x + t*m
error_list = []
for i in range(1000):
    mat0 = np.dot(A.T, np.dot(A, np.dot(A.T, np.dot(A, x)-b)))
    mat1 = np.dot(A.T, np.dot(A, m))
    alpha = - (inner_prod(m, mat0))/(inner_prod(m, mat1))
    m = np.dot(A.T, np.dot(A, x)-b) + alpha*m
    mat0 = np.dot(A.T, np.dot(A, x)-b)
    mat1 = np.dot(A.T, np.dot(A, m))
    t = -(inner_prod(m, mat0))/(inner_prod(m, mat1))
    x = x + t*m
    # pdb.set_trace()
    error_list.append(np.sum(abs(1 - x)))
plt.plot(error_list)
plt.show()
pdb.set_trace()
