import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares, minimize


def func_a(x, v, k):
    return (1 / v) + (k / (v * x))


def func_b(x, v, k):
    return v / (1 + (k / x))


def func_MM(x, v, k):
    return (x * v) / (x + k)


def func_err_c(p, x, y):
    v, k = p
    err = (y - v * x / (k + x)) ** 2
    return err


def func_err_d(p, x, y):
    v, k = p
    err = 0
    for i in range(len(x)):
        err = err + (y[i] - v * x[i] / (k + x[i]))**2
    return err


if __name__ == '__main__':

    # problem 1 (a)
    # Density x Eaten y
    Density = np.array([4, 10, 30, 90, 173, 256])
    Eaten = np.array([2.5, 9.5, 12.5, 19.5, 21.5, 19.0])

    x = np.linspace(4, 256, 300)
    y = 1 / Eaten
    popt, _ = curve_fit(func_a, Density, y)
    plt.plot(x, func_a(x, *popt), 'r-',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

    plt.scatter(Density, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Lineweaver-Burke transform")
    plt.legend()
    plt.savefig("4-1a.png")
    # plt.show()
    plt.close()

    # problem 1 (b)
    x = np.linspace(4, 256, 300)
    popt, _ = curve_fit(func_a, Density, Eaten)
    plt.plot(x, func_a(x, *popt), 'r-',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.scatter(Density, Eaten)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Eadie-Hofstee transform")
    plt.legend()
    plt.savefig("4-1b.png")
    # plt.show()
    plt.close()

    # problem 1 (c)
    init_point = np.array([0, 0])
    result = least_squares(fun=func_err_c, x0=init_point, args=(Density, Eaten), method='lm')
    plt.plot(Density, func_MM(Density, *result.x), 'r-',
             label='fit: a=%5.3f, b=%5.3f' % tuple(result.x))
    plt.scatter(Density, Eaten)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Levenberg-Marquardt")
    plt.legend()
    plt.savefig("4-1c.png")
    plt.show()

    # problem 1 (d)

    init_point = np.array([0, 0])
    result = minimize(fun=func_err_d, x0=init_point, args=(Density, Eaten), method='Nelder-Mead')
    plt.plot(Density, func_MM(Density, *result.x), 'r-',
             label='fit: a=%5.3f, b=%5.3f' % tuple(result.x))
    plt.scatter(Density, Eaten)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Nelder-Mead simplex")
    plt.legend()
    plt.savefig("4-1d.png")
    # plt.show()
