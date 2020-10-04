import numpy as np

import gauss_jordan

def least_square(x_vec, y_vec):

    x_sum = np.sum(x_vec)
    y_sum = np.sum(y_vec)
    xy_sum = np.sum(x_vec*y_vec)
    x2y_sum = np.sum(x_vec**2*y_vec)
    x2_sum = np.sum(x_vec**2)
    x3_sum = np.sum(x_vec**3)
    x4_sum = np.sum(x_vec**4)
    n = len(x_vec)

    A = np.array([[x4_sum, x3_sum, x2_sum],
                 [x3_sum, x2_sum, x_sum],
                 [x2_sum, x_sum, n]])
    b = np.array([x2y_sum, xy_sum, y_sum])

    coef = gauss_jordan.gauss_jordan(A, b)

    return coef

if __name__ == '__main__':
    x_vec = np.array([0, 0.3, 0.8, 1.1, 1.6, 2.3])
    y_vec = np.array([0.6, 0.67, 1.01, 1.35, 1.47, 1.25])

    coef = least_square(x_vec, y_vec)
    print(coef)
