import numpy as np
import pprint

"""
A = np.array([[1, 2, 3],
              [2, 2, 3],
              [2, 2, 1]], dtype=np.float32)
b = np.array([2, 1, -1], dtype=np.float32)
"""

A = np.array([[3, 2, 1],
              [1, 4, 1],
              [2, 2, 5]], dtype=np.float32)
b = np.array([10, 12, 21], dtype=np.float32)

x = np.zeros_like(b)
# x = np.array([1, 0, 1], dtype=np.float32)

max_iter = 100

count = 0
while count < max_iter:
    for i in range(len(A)):
        residual = b[i]
        for j in range(len(A)):
            if j != i:
                residual -= A[i,j]*x[j]
        x[i] = residual/A[i,i]

    count += 1

pprint.pprint(x)

# 08027285364
