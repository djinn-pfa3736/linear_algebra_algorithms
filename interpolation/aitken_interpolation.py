import numpy as np

def aitken_interpolation(x_vec, y_vec, x):

    n = len(x_vec)
    P = np.array([[0.0]*n for _ in range(n)])
    for i in range(n):
        P[i,0] = y_vec[i]
        for k in range(i):
            P[i,k+1] = ((x_vec[i] - x)*P[i-1,k] - (x_vec[i-k-1] - x)*P[i,k])/(x_vec[i] - x_vec[i-k-1])
    return P[n-1,n-1]

if __name__ == '__main__':

    x_vec = np.array([1, 2, 4], dtype=np.float32)
    y_vec = 2*x_vec**2 + 3*x_vec + 4

    L_x = aitken_interpolation(x_vec, y_vec, 3)
    print(L_x)
