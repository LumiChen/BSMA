import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def B(t, y):
    X1, X2, X3 = y

    r1 = 1
    r2 = 0.1
    r3 = 0.3
    v = 1
    n = 1
    F1 = 0.5
    F2 = 0.8

    dX1 = r1 * X1 * (1 - F1 - X1 - (v * X2) - (n * X3))
    dX2 = r2 * X2 * (1 - F2 - (X2 / X1))
    dX3 = r3 * X3 * (1 - (X3 / X1))

    return dX1, dX2, dX3


if __name__ == '__main__':

    t0 = 0
    tf = 25
    initial = 0.45, 0.09, 0.45
    t_eval = np.arange(t0, tf, 0.01)
    sol = solve_ivp(B, (t0, tf), initial, t_eval=t_eval, method='RK45')
    T = sol.t
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]

    plt.figure(figsize=(5, 5))
    plt.plot(T, x, label='krill')
    plt.plot(T, y, label='whales', ls='--')
    plt.plot(T, z, label='seals', ls='-.')
    plt.legend()
    plt.show()
    plt.close()



