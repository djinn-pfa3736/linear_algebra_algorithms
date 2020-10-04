import numpy as np

def trapezoidal_rule(f, a, b, n):
    h = (b - a)/n

    S = 0

    for i in range(0, n, 2):
        x = a + i*h
        Si = h/3*(f(x) + 4*f(x + h) + f(x + 2*h))
        S += Si

    return S

if __name__ == '__main__':

    f = lambda x: 3*x**2 + 4
    a = 1
    b = 4
    n_vec = [2, 4, 8, 16, 32, 64, 128, 256]

    for n in n_vec:
        S = trapezoidal_rule(f, a, b, n)
        print(S)
