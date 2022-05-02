import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from scipy.integrate import solve_ivp


def deriv(t, y):
    N1, N2 = y
    r = 20
    K = 2
    a = 5
    b = 3
    s = 2

    dN1 = r * N1 * (1 - N1 / K) - a * N1 * N2
    dN2 = s * N2 * (1 - N2 / (b * N1))

    return dN1, dN2


if __name__ == "__main__":

    t0 = 0
    tf = 3
    xi = 0.1
    yi = 0.8
    t_eval = np.arange(t0, tf, 0.01)
    sol = solve_ivp(deriv, (t0, tf), (xi, yi), t_eval=t_eval, method='RK45')
    T = sol.t
    x = sol.y[0]
    y = sol.y[1]

    x0 = 50
    it = 0
    # print(x)
    for i in x:
        if abs(x0 - i) < 0.000005:
            break
        it += 1
        x0 = i
    print(x[it])
    print(y[it])
    print(it)
    t = np.arange(0, it, 1)
    plt.figure(figsize=(5, 5))
    plt.plot(t, x[0: it], label='krill')
    plt.plot(t, y[0: it], label='whal', ls='-')
    plt.title("initial x = {} and initial y = {}".format(xi, yi))
    plt.legend()
    plt.xlabel("Time, t")
    plt.ylabel("Krill and Whale population")
    plt.show()


