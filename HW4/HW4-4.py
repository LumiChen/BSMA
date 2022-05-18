import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit


def P4(t, y):
    X, Y = y

    p = 1.85
    K = 898
    w = 25.5
    o = 284.1
    delta = 12.40
    r = 2.07

    dx = p * (1 - (X / K)) * X - ((w * Y * X) / (o + X))
    dy = delta * Y * (X / (o + X)) - (r * Y)

    return dx, dy


def U(x, y):
    n = len(x)
    upper = 0
    bottom_left = 0
    bottom_right = 0
    for i in range(n):
        upper += (x[i] - y[i])**2
        bottom_left += x[i]**2
        bottom_right += y[i]**2

    temp = np.sqrt(upper / n) / (np.sqrt(bottom_left /n) + np.sqrt(bottom_right / n))
    return temp


def ononeregression(x, m, b):
    return m * x + b


if __name__ == '__main__':

    Dnasutum_18 = pd.read_table("Luckinbill18_Dnasutum.dat", sep="\s+", usecols=[0, 1])
    Paurelia_18 = pd.read_table("Luckinbill18_Paurelia.dat", sep="\s+", usecols=[0, 1])
    Dnasutum_33 = pd.read_table("Luckinbill33_Dnasutum.dat", sep="\s+", usecols=[0, 1])
    Paurelia_33 = pd.read_table("Luckinbill33_Paurelia.dat", sep="\s+", usecols=[0, 1])

    t0 = 0
    tf = 18.2
    initial = 15.0, 5.833
    t_eval = np.arange(t0, tf, 0.05)
    sol = solve_ivp(P4, (t0, tf), initial, t_eval=t_eval, method='RK45')
    T = sol.t
    x = sol.y[0]
    y = sol.y[1]
    for i in range(len(T)):
        T[i] = round(T[i], 2)

    x_m = []  # model
    x_r = []  # real
    x_y = []  # t
    y_m = []  # model
    y_r = []  # real
    y_y = []  # t

    # prey
    for j in range(len(Paurelia_18['Luckinbill,'])):
        for i in range(len(T)):
            if T[i] == Paurelia_18['Luckinbill,'].loc[j]:
                x_m.append(x[i])
                x_r.append(Paurelia_18['P.'].loc[j])
                x_y.append(T[i])

    # predator
    for i in range(len(T)):
        for j in range(len(Dnasutum_18['Day'])):
            if T[i] == Dnasutum_18['Day'].loc[j]:
                y_m.append(y[i])
                y_r.append(Dnasutum_18['#/ml'].loc[j])
                y_y.append(T[i])

    print(len(Paurelia_18['Luckinbill,']))
    print(len(x_m))
    print("")
    print(len(Dnasutum_18['Day']))
    print(len(y_m))

    print(ttest_rel(x_m, x_r, alternative='two-sided'))
    print(ttest_rel(y_m, y_r, alternative='two-sided'))

    print(U(x_m, x_r))
    print(U(y_m, y_r))

    plt.figure(figsize=(10, 5))
    plt.plot(T, x, '-', label='prey density (model)')
    plt.plot(Paurelia_18['Luckinbill,'], Paurelia_18['P.'], ls='-.', label='prey density (real)')
    plt.ylim(-50, 600)
    plt.xlim(0, 18.5)
    plt.scatter(x_y, x_m, label='model')
    plt.scatter(x_y, x_r, label='real')
    plt.title("Luckinbill18_Paurelia")
    plt.legend()
    plt.savefig("4-4dx.png", dpi=300)
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(T, y, '-', label='predator density (model)')
    plt.plot(Dnasutum_18['Day'], Dnasutum_18['#/ml'], ls='-.', label='predator density (real))')
    plt.ylim(-50, 365)
    plt.xlim(0, 18.5)
    plt.scatter(y_y, y_m, label='model')
    plt.scatter(y_y, y_r, label='real')
    plt.title("Luckinbill18_Dnasutum")
    plt.legend()
    plt.savefig("4-4dy.png", dpi=300)
    # plt.show()
    plt.close()

    x = Paurelia_18['P.']
    y = Dnasutum_18['#/ml']

    x_fit_reg = np.linspace(-1.5, 600, 500)
    popt_oneone_prey, _ = curve_fit(ononeregression, xdata=x_r, ydata=x_m)
    plt.figure(figsize=(5, 5))
    plt.plot(x_fit_reg, ononeregression(x_fit_reg, *popt_oneone_prey), ls='-',
             label='Prey: m=%5.5f b=%5.5f' % tuple(popt_oneone_prey))
    plt.legend()
    plt.scatter(x_r, x_m)
    plt.ylim(-5, 600)
    plt.xlim(-5, 600)
    plt.plot(x_fit_reg, x_fit_reg, '-.')
    plt.savefig("4-4dx11.png", dpi=300)
    plt.close()

    popt_oneone_predator, _ = curve_fit(ononeregression, xdata=y_r, ydata=y_m)
    plt.figure(figsize=(5, 5))
    plt.plot(x_fit_reg, ononeregression(x_fit_reg, *popt_oneone_predator), ls='-',
             label='Predator: m=%5.5f b=%5.5f' % tuple(popt_oneone_predator))
    plt.legend()
    plt.scatter(y_r, y_m)
    plt.plot(x_fit_reg, x_fit_reg, '-.')
    plt.savefig("4-4dy11.png", dpi=300)
