import numpy as np
import pprint

def cholesky_decomposition(A):
    L = np.zeros_like(A)
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                tmp = 0
                for k in range(j):
                    tmp += L[j,k]**2
                L[j,j] = np.sqrt(A[j,j] - tmp)
            elif j < i:
                tmp = 0
                for k in range(j):
                    tmp += L[i,k]*L[j,k]
                L[i,j] = (1/L[j,j])*(A[i,j] - tmp)

    return L

if __name__ == '__main__':

    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]], dtype=np.float32)

    L = cholesky_decomposition(A)
    pprint.pprint(L)
