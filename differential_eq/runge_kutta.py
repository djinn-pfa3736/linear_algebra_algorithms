import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(f, x0, y0, h, m):
    x_prev = x0
    y_prev = y0

    y_vec = []
    for i in range(m):
        k1 = h*f(x_prev, y_prev)
        k2 = h*f(x_prev + h/2, y_prev + k1/2)
        k3 = h*f(x_prev + h/2, y_prev + k2/2)
        k4 = h*f(x_prev + h, y_prev + k3)

        y_next = y_prev + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        y_vec.append(y_next)

        y_prev = y_next
        x_prev += h

    y_vec = np.array(y_vec)
    return y_vec

if __name__ == '__main__':
    f = lambda x, y: x + y

    x0 = 0
    y0 = 0

    h = 0.001
    m = 100

    y_vec = runge_kutta(f, x0, y0, h, m)
    plt.plot(y_vec)
    plt.show()

    x_vec = np.arange(x0, h*m, h)
    true_y_vec = np.exp(x_vec) - x_vec - 1
    plt.plot(np.abs(true_y_vec - y_vec))
    plt.show()
