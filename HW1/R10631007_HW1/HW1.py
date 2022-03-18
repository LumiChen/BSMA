import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.optimize import curve_fit


def Estimate(x, y, inter=False):

    if inter:
        model = LinearRegression()
    else:
        model = LinearRegression(fit_intercept=False)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    model.fit(x, y)

    y_hat = model.predict(x)

    if inter:
        print("{}*R + {}".format(round(float(model.coef_[0]), 5), round(float(model.intercept_), 5)))
    else:
        print("{}*R".format(round(float(model.coef_[0]), 5)))

    # print("R2 : {}".format(round(metrics.r2_score(y, y_hat), 5)))
    # print("MSE : {}".format(round(metrics.mean_squared_error(y, y_hat), 5)))
    plt.plot(x, y, 's', label='original values')
    plt.plot(x, y_hat, 'r', label='polyfit values')

    return plt


def Simulate(Rt, i):
    Rtt = Rt
    Rt = Rt + (6.19 - (0.0113 * Rt)) - (0.0066 * Rt)
    i += 1

    if abs(Rtt-Rt) < 0.001:  # for find the dynamic equilibrium
        print("Number of iteration times: {}".format(i))
        print("Rt : {}".format(str(Rt)))
        return i

    return Simulate(Rt, i)


def Simulate_re(Rt, i):
    Rtt = Rt
    Rt = Rt + (6.19 - (0.0113 * Rt)) - (0.0066 * Rt) + (-0.006892 * Rt)
    i += 1

    if abs(Rtt-Rt) < 0.001:  # for find the dynamic equilibrium
        print("Number of iteration times: {}".format(i))
        print("Rt : {}".format(str(Rt)))
        return i

    return Simulate_re(Rt, i)


def func_e_I(x, a, b, c):  # negative exponential Immigration
    return a * (-(np.exp(b * x))) + c


def func_e_E(x, a, b):  # negative exponential Extinction
    return a * (-(np.exp((b * x))) + 1)


def func_q_I(x, a, b, c):  # quadratic Immigration
    return a * (x**2) + b * x + c


def func_q_E(x, a, b):  # quadratic Extinction
    return a * (x**2) + b * x


def Alternative(x, y, func, coef, dataset):

    if coef == 1:
        popt, pcov = curve_fit(func, x, y, bounds=([0, 0, 2], [0.1, 0.1, 10]))
    elif coef == 2:
        popt, pcov = curve_fit(func, x, y, bounds=([-10, 0], [0, 0.1]))
    elif coef == 3:
        popt, pcov = curve_fit(func, x, y, bounds=([-np.inf, 0, 0], [0, 1, np.inf]))
    else:
        popt, pcov = curve_fit(func, x, y, bounds=([0, 0], [np.inf, 1]))

    if coef == 1 or coef == 3:
        print("------------------")
        print("a = {}, b = {}, c = {}".format(float(popt[0]), float(popt[1]), float(popt[2])))
        print("------------------")

    else:
        print("------------------")
        print("a = {}, b = {}".format(float(popt[0]), float(popt[1])))
        print("------------------")

    X = np.linspace(np.min(R), np.max(R), 50)

    if dataset == 'E':
        plt.plot(R, E, 'bo', label='data')

    else:
        plt.plot(R, I, 'bo', label='data')

    plt.plot(X, func(X, *popt), 'r-')

    return plt


def Exponential_Simulate(Rt, i):

    Rtt = Rt
    Rt = Rt + ((0.00109 * (-np.exp(Rt * 0.033258))) + 5.5962) - (-0.1733 * (-(np.exp(Rt * 0.01041))) + 1)
    i += 1

    if abs(Rtt - Rt) < 0.001:
        print("Number of iteration times: {}".format(i))
        print("Rt : {}".format(str(Rt)))
        return i

    return Exponential_Simulate(Rt, i)


def Quadratic_Simulate(Rt, i):

    Rtt = Rt
    Rt = Rt + ((-4.5417 * pow(10, -5)) * (Rt**2) + ((1.774 * pow(10, -11)) * Rt) + 5.843) - \
        ((3.2417 * pow(10, -5) * (Rt**2)) + (5.9761*pow(10, -12)) * Rt)
    i += 1

    if abs(Rtt - Rt) < 0.001:
        print("Number of iteration times: {}".format(i))
        print("Rt : {}".format(str(Rt)))
        return i

    return Quadratic_Simulate(Rt, i)


def Simulat_graph(Rt, i):
    y = []
    I = []
    E = []

    for x in range(i):
        Rt = Rt + (6.19 - (0.0113 * Rt)) - (0.0066 * Rt) + (-0.006892 * Rt)
        y.append(Rt)

    x = np.arange(1, i + 1)
    y = np.array(y)

    plt.plot(x, y, 'r-', label='Number')
    plt.xlabel('Iteration (n)')
    plt.ylabel('Number')

    return plt


def Graph(Rt, i, eq):

    y = []
    I = []
    E = []

    for x in range(i):
        if eq == '1':
            Rt = Rt + (6.19 - (0.0113 * Rt)) - (0.0066 * Rt)
            It = (6.19 - (0.0113 * Rt))
            Et = (0.0066 * Rt)
        elif eq == '2':
            Rt = Rt + ((0.00109 * (-np.exp(Rt * 0.033258))) + 5.5962) - (-0.1733 * (-(np.exp(Rt * 0.01041))) + 1)
            It = ((0.00109 * (-np.exp(Rt * 0.033258))) + 5.5962)
            Et = (-0.1733 * (-(np.exp(Rt * 0.01041))) + 1)

        else:
            Rt = Rt + ((-4.5417 * pow(10, -5)) * (Rt ** 2) + ((1.774 * pow(10, -11)) * Rt) + 5.843) - \
                 ((3.2417 * pow(10, -5) * (Rt ** 2)) + (5.9761 * pow(10, -12)) * Rt)
            It = ((-4.5417 * pow(10, -5)) * (Rt ** 2) + ((1.774 * pow(10, -11)) * Rt) + 5.843)
            Et = ((3.2417 * pow(10, -5) * (Rt ** 2)) + (5.9761 * pow(10, -12)) * Rt)
        y.append(Rt)
        I.append(It)
        E.append(Et)

    x = np.arange(1, i + 1)
    y = np.array(y)

    plt.subplot(3, 1, 1)
    plt.plot(x, y, 'r-', label='number')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x, I, 'r-', label='Immigration')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x, E, 'r-', label='Extinction')
    plt.legend()

    return plt


if __name__ == '__main__':

    # loading the dataset
    R = [0, 36, 80, 155, 210, 240]  # the number of species
    I = [8.0, 3.0, 5.0, 6.5, 4.0, 2.5]  # immigration rate
    E = [0.0, 0.05, 0.10, 0.50, 1.75, 1.75]  # extinction rate

    R = np.array(R)
    I = np.array(I)
    E = np.array(E)
    #
    # problem 1 (a)
    #
    Immigration_plt = Estimate(R, I, inter=True)
    Immigration_plt.xlabel('Species numbers')
    Immigration_plt.ylabel('Immigration rate')
    # Immigration_plt.show()
    Immigration_plt.close()

    Extinction_plt = Estimate(R, E, inter=False)
    Extinction_plt.xlabel('Species numbers')
    Extinction_plt.ylabel('Extinction rate')
    # Extinction_plt.show()
    Extinction_plt.close()

    #
    # problem 1 (e)
    #
    Simulate_re(500, 0)
    Simulate_re(0, 0)

    From500_plt = Simulat_graph(500, Simulate_re(500, 0))
    From500_plt.savefig("from500.png")
    From500_plt.close()

    From0_plt = Simulat_graph(0, Simulate_re(0, 0))
    From0_plt.savefig("from0.png")
    From0_plt.close()

    #
    # problem 2 (a)
    #
    e_I_plt = Alternative(R, I, func=func_e_I, coef=1, dataset='I')  # exponential Immigration function
    e_I_plt.xlabel('Species numbers')
    e_I_plt.ylabel('Immigration rate')
    e_I_plt.title("Exponential Immigration")
    # e_I_plt.savefig("Exponential Immigration.png")
    # e_I_plt.show()
    e_I_plt.close()

    e_E_plt = Alternative(R, E, func=func_e_E, coef=2, dataset='E')
    e_E_plt.xlabel('Species numbers')
    e_E_plt.ylabel('Extinction rate')
    e_E_plt.title("Exponential Extinction")
    # e_E_plt.savefig("Exponential Extinction.png")
    # e_E_plt.show()
    e_E_plt.close()

    q_I_plt = Alternative(R, I, func=func_q_I, coef=3, dataset='I')  # quadratic Immigration function
    q_I_plt.xlabel('Species numbers')
    q_I_plt.ylabel('Immigration rate')
    q_I_plt.title("Quadratic Immigration")
    # q_I_plt.savefig("Quadratic Immigration.png")
    # q_I_plt.show()
    q_I_plt.close()

    q_E_plt = Alternative(R, E, func=func_q_E, coef=4, dataset='E')
    q_E_plt.xlabel('Species numbers')
    q_E_plt.ylabel('Extinction rate')
    q_E_plt.title("Quadratic Extinction")
    # q_E_plt.savefig("Quadratic Extinction.png")
    # q_E_plt.show()
    q_E_plt.close()

    # plot the rate and number
    Ori_plt = Graph(0, Simulate(0, 0), eq=1)
    Ori_plt.suptitle("Original model")
    Ori_plt.xlabel("Iteration (n)")
    # Ori_plt.savefig("Original model.png")
    # Ori_plt.show()
    Ori_plt.close()

    Ex_plt = Graph(0, Exponential_Simulate(0, 0), eq=2)
    Ex_plt.suptitle("Exponential model")
    Ex_plt.xlabel("Iteration (n)")
    # Ex_plt.savefig("Exponential model.png")
    # Ex_plt.show()
    Ex_plt.close()

    Qua_plt = Graph(0, Quadratic_Simulate(0, 0), eq=3)
    Qua_plt.suptitle("Quadratic model")
    Qua_plt.xlabel("Iteration (n)")
    # Qua_plt.savefig("Quadratic model.png")
    # Qua_plt.show()
    Qua_plt.close()

