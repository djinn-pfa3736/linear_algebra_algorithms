import numpy as np
import pprint

def modified_cholesky(A):
    L = np.zeros_like(A)
    D = np.zeros_like(A)

    D[0,0] = A[0,0]
    for i in range(len(A)):
        L[i,i] = 1
        for j in range(i):
            tmp = 0
            for k in range(j):
                tmp += L[i,k]*D[k,k]*L[j,k]
            L[i,j] = (A[i,j] - tmp)/D[j,j]

        tmp = 0
        for k in range(i):
            tmp += L[i,k]**2*D[k,k]
        D[i,i] = A[i, i] - tmp

    return L, D

if __name__ == '__main__':

    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]], dtype=np.float32)

    L, D = modified_cholesky(A)
    pprint.pprint(L)
    pprint.pprint(D)
    pprint.pprint(np.dot(L, np.dot(D, L.T)))
