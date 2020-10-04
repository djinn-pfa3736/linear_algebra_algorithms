import math
import numpy as np

import pdb

def gauss_jordan(A, b):
    for i in range(len(A)):
        if A[i, i] == 0:
            valid_flag = False
            for j in range(i+1, len(A)):
                if A[j, i] != 0:
                    tmp = A[i,:]
                    A[i,:] = A[j,:]
                    A[j,:] = tmp
                    valid_flag = True
            if not valid_flag:
                print("There is no valid pivot...")
                return -1
        else:
            a_ii = A[i, i]
            A[i,:] /= a_ii
            b[i] /= a_ii
            for j in range(len(A)):
                if i != j:
                    a_ji = A[j,i]
                    A[j,:] -= a_ji*A[i,:]
                    b[j] -= a_ji*b[i]

    return b

if __name__ == '__main__':

    A = np.array([[1, 2, 3],
                  [2, 2, 3],
                  [2, 2, 1]], dtype=np.float32)
    b = np.array([2, 1, -1], dtype=np.float32)

    x = gauss_jordan(A, b)
    print(x)
