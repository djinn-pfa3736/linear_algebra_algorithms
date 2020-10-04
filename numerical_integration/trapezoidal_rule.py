import numpy as np

def trapezoidal_rule(f, a, b, n):
    h = (b - a)/n

    S = 0

    for i in range(n):
        x = a + h*i
        Si = h/2*(f(x) + f(x + h))
        S += Si

    return S

if __name__ == '__main__':

    f = lambda x: 3*x**2 + 4
    a = 1
    b = 4
    n_vec = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for n in n_vec:
        S = trapezoidal_rule(f, a, b, n)
        print(S)
