import numpy as np
import matplotlib.pyplot as plt
import runge_kutta

def adams_moulton(f, y0, a, b, h, eps, max_iter):

    tmp = runge_kutta.runge_kutta(f, a, y0, h, 4)
    f_vec = [tmp[0], tmp[1], tmp[2], tmp[3]]
    m = int((b - a)/h)

    y_prev = y0
    x = x0
    for i in range(3, m-1):

        y_pred = y_prev + h/24*(55*f_vec[i] - 59*f_vec[i-1] + 37*f_vec[i-2] - 9*f_vec[i-3])
        y_corr = y_prev + h/24*(9*f(x, y_pred) + 19*f_vec[i] - 5*f_vec[i-1] + f_vec[i-2])

        err = np.abs(y_pred - y_corr)

        iter = 0
        while (eps < err) and (iter <= max_iter):
            y_corr_next = y_prev + h/24*(9*f(x + h, y_corr) + 19*f_vec[i] - 5*f_vec[i-1] + f_vec[i-2])

            err = np.abs(y_corr_next - y_corr)
            y_corr = y_corr_next
            iter += 1

        y_prev = y_corr
        x += h
        f_vec.append(f(x, y_corr))

        if max_iter == iter:
            print("No convergence...")
            return -1

    return f_vec

if __name__ == '__main__':
        f = lambda x, y: x + y

        x0 = 0
        y0 = 0

        y0 = 0
        a = 0
        b = 4
        h = 0.001
        eps = 1e-8
        max_iter = 10

        y_vec = adams_moulton(f, y0, a, b, h, eps, max_iter)
        plt.plot(y_vec)
        plt.show()

        x_vec = np.arange(a, b, h)
        true_y_vec = np.exp(x_vec) - x_vec - 1
        plt.plot(np.abs(true_y_vec - y_vec))
        plt.show()
