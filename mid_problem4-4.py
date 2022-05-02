import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from scipy.integrate import solve_ivp


def B(t, y):
    X1, X2 = y

    v = 1
    r1 = 1
    F2 = 0.8
    F1 = 0.5
    r2 = 0.04

    dX1 = r1 * X1 * (1 - F1 - X1 - (v * X2))
    dX2 = r2 * X2 * (1 - F2 - (X2 / X1))

    return dX1, dX2


if __name__ == '__main__':

    t0 = 0
    tf = 25
    initial = 0.65, 0.32
    t_eval = np.arange(t0, tf, 0.01)
    sol = solve_ivp(B, (t0, tf), initial, t_eval=t_eval, method='RK45')
    T = sol.t
    x = sol.y[0]
    y = sol.y[1]

    plt.figure(figsize=(5, 5))
    plt.plot(T, x, label='krill')
    plt.plot(T, y, label='whales', ls='-')
    plt.xlabel("Time, t")
    plt.ylabel("Krill and Whale population")
    plt.legend()
    plt.show()
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(x, y)
    plt.title('krill v.s. whale population')
    plt.xlim(0, 0.7)
    plt.ylim(0, 0.4)
    plt.xlabel("Krill population")
    plt.ylabel("Whale population")
    plt.legend()
    plt.plot()
    plt.show()


