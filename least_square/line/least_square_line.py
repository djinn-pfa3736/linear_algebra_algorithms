import numpy as np

def least_square(x_vec, y_vec):

    x_sum = np.sum(x_vec)
    y_sum = np.sum(y_vec)
    xy_sum = np.sum(x_vec * y_vec)
    x2_sum = np.sum(x_vec**2)

    a = (len(x_vec)*xy_sum - x_sum*y_sum)/(len(x_vec)*x2_sum - x_sum**2)
    b = (x2_sum*y_sum - x_sum*xy_sum)/(len(x_vec)*x2_sum - x_sum**2)

    return a, b


if __name__ == '__main__':
    x_vec = np.arange(0, 1.4, 0.2)
    y_vec = np.array([1.0, 1.9, 3.2, 4.3, 4.8, 6.1, 7.2])

    a, b = least_square(x_vec, y_vec)
    print(a, b)
