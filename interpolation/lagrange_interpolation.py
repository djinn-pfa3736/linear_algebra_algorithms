import numpy as np

def lagrange_interpolation(x_vec, y_vec, x):

    n = len(x_vec)
    L_x = 0
    for j in range(n):
        l_j = 1
        for m in range(n):
            if j != m:
                l_j *= (x - x_vec[m])/(x_vec[j] - x_vec[m])
        L_x += l_j*y_vec[j]

    return L_x

if __name__ == '__main__':

    x_vec = np.array([1, 2, 4], dtype=np.float32)
    y_vec = 2*x_vec**2 + 3*x_vec + 4

    L_x = lagrange_interpolation(x_vec, y_vec, 3)
    print(L_x)
