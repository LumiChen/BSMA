import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from sklearn.metrics import mean_squared_error, r2_score


def model1(x, a1):
    return a1 * x


def model2(x, a0, a1):
    return a0 + a1 * x


def model3(x, a0, a1, a2):
    return a0 + a1 * x + a2 * (x**2)


def model4(x, a3, a4):
    return a3 * np.exp(a4 * x)


def ononeregression(x, m, b):
    return m * x + b


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


if __name__ == '__main__':

    x = np.array([0, 1, 2, 3])
    y = np.array([-1.290, 5.318, 7.049, 19.886])

    x_fit = np.linspace(0, 3, 50)
    # fitting
    plt.figure(figsize=(5, 5))

    popt_model1, _ = curve_fit(model1, x, y)
    plt.plot(x_fit, model1(x_fit, *popt_model1), '-',
             label='Model1: a1=%5.3f' % tuple(popt_model1))

    popt_model2, _ = curve_fit(model2, x, y)
    plt.plot(x_fit, model2(x_fit, *popt_model2), ls=(0, (5, 10)),
             label='Model2: a0=%5.3f a1=%5.3f' % tuple(popt_model2))

    popt_model3, _ = curve_fit(model3, x, y)
    plt.plot(x_fit, model3(x_fit, *popt_model3), ls='dashdot',
             label='Model3: a0=%5.3f a1=%5.3f a2=%5.3f' % tuple(popt_model3))

    popt_model4, _ = curve_fit(model4, x, y)
    plt.plot(x_fit, model4(x_fit, *popt_model4), ls=(0, (3, 5, 1, 5, 1, 5)),
             label='Model4: a3=%5.3f a4=%5.3f' % tuple(popt_model4))

    plt.scatter(x, y)
    plt.legend()
    plt.savefig("4-3_fitting.png", dpi=300)
    # plt.show()
    plt.close()

    # 1:1 regression
    y_model1 = model1(x, *popt_model1)
    y_model2 = model2(x, *popt_model2)
    y_model3 = model3(x, *popt_model3)
    y_model4 = model4(x, *popt_model4)

    x_fit_reg = np.linspace(-1.5, 20, 50)

    # model1
    plt.figure(figsize=(5, 5))
    popt_oneone_model1, _ = curve_fit(ononeregression, xdata=y, ydata=y_model1)
    plt.plot(x_fit_reg, ononeregression(x_fit_reg, *popt_oneone_model1), '-',
             label='Model1: m=%5.5f b=%5.5f' % tuple(popt_oneone_model1))
    print(popt_oneone_model1)
    # print(r2_score(ononeregression(y_model1, *popt_oneone_model1), y))
    plt.scatter(y, y_model1)
    plt.plot(x_fit_reg, x_fit_reg, '-.', label='45 degree')
    plt.title("model1 R^2: {}".format(round(r2_score(ononeregression(y_model1, *popt_oneone_model1), y), 2)))
    plt.xlabel("Model")
    plt.ylabel("Date")
    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.legend()
    plt.savefig("4-3_model1.png", dpi=300)
    # plt.show()
    plt.close()

    # model2
    plt.figure(figsize=(5, 5))
    popt_oneone_model2, _ = curve_fit(ononeregression, xdata=y, ydata=y_model2)
    plt.plot(x_fit_reg, ononeregression(x_fit_reg, *popt_oneone_model2), ls='-',
             label='Model2: m=%5.5f b=%5.5f' % tuple(popt_oneone_model2))
    print(popt_oneone_model2)
    # print(r2_score(ononeregression(y_model2, *popt_oneone_model2), y))
    plt.scatter(y, y_model2)
    plt.plot(x_fit_reg, x_fit_reg, '-.', label='45 degree')
    plt.title("model2 R^2: {}".format(round(r2_score(ononeregression(y_model2, *popt_oneone_model2), y), 2)))
    plt.xlabel("Model")
    plt.ylabel("Date")
    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.legend()
    plt.savefig("4-3_model2.png", dpi=300)
    # plt.show()
    plt.close()

    # model3
    plt.figure(figsize=(5, 5))
    popt_oneone_model3, _ = curve_fit(ononeregression, xdata=y, ydata=y_model3)
    plt.plot(x_fit_reg, ononeregression(x_fit_reg, *popt_oneone_model3), ls='-',
             label='Model3: m=%5.5f b=%5.5f' % tuple(popt_oneone_model3))
    print(popt_oneone_model3)
    # print(r2_score(ononeregression(y_model3, *popt_oneone_model3), y))
    plt.scatter(y, y_model3)
    plt.plot(x_fit_reg, x_fit_reg, '-.', label='45 degree')
    plt.title("model3 R^2: {}".format(round(r2_score(ononeregression(y_model3, *popt_oneone_model3), y), 2)))
    plt.xlabel("Model")
    plt.ylabel("Date")
    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.legend()
    plt.savefig("4-3_model3.png", dpi=300)
    # plt.show()
    plt.close()

    # model4
    plt.figure(figsize=(5, 5))
    popt_oneone_model4, _ = curve_fit(ononeregression, xdata=y, ydata=y_model4)
    plt.plot(x_fit_reg, ononeregression(x_fit_reg, *popt_oneone_model4), ls='-',
             label='Model4: m=%5.5f b=%5.5f' % tuple(popt_oneone_model4))
    print(popt_oneone_model4)
    # print(r2_score(ononeregression(y_model4, *popt_oneone_model4), y))
    plt.scatter(y, y_model4)
    plt.plot(x_fit_reg, x_fit_reg, '-.', label='45 degree')
    plt.title("model4 R^2: {}".format(round(r2_score(ononeregression(y_model4, *popt_oneone_model4), y)), 2))
    plt.xlabel("Model")
    plt.ylabel("Date")
    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.legend()
    plt.savefig("4-3_model4.png", dpi=300)
    # plt.show()

    # paired t-test H0: delta D =0
    # p-value = 3.182 at alpah = 0.05
    # print(ttest_rel(y, y_model1, alternative='two-sided'))
    # print(ttest_rel(y, y_model2, alternative='two-sided'))
    # print(ttest_rel(y, y_model3, alternative='two-sided'))
    # print(ttest_rel(y, y_model4, alternative='two-sided'))
    #
    # print(U(y, y_model1))
    # print(U(y, y_model2))
    # print(U(y, y_model3))
    # print(U(y, y_model4))
    #
    # print(y_model1)
    # print(y_model2)
    # print(y_model3)
    # print(y_model4)
