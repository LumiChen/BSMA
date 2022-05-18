import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from scipy.optimize import curve_fit


def func_a(t, M0, K, r):
    return (M0 * K) / (M0 + (K - M0) * np.exp(-r * t))


def AGR_a(t, M0, K, r):
    return (r * M0 * K * np.exp(-r * t) * (K-M0)) / (M0 + np.exp(-r * t) * (K-M0))**2


def RGR_a(t, M0, K, r):
    return (r * np.exp((-r * t)) * (K - M0)) / (M0 + np.exp(-r *t) * (K - M0))


def func_b(t, M0, K, r):
    return K * (M0 / K)**np.exp(-r * t)


def AGR_b(t, M0, K, r):
    return r * K * np.exp(-r * t) * (M0 / K) ** np.exp(-r * t) * np.log(K / M0)


def RGR_b(t, M0, K, r):
    return r * np.exp(-r * t) * np.log(K / M0)


if __name__ == "__main__":

    # initial
    data = pandas.read_excel("HW4-PROBLEM-2.xlsx")
    x = data['Time (Day)']
    y = data['16-Hr Lighting']
    y2 = data['24-Hr Lighting']

    # 16-HR
    popt, _ = curve_fit(func_a, x, y)
    plt.plot(x, func_a(x, *popt), '-',
             label='Logistic: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt))

    popt2, _ = curve_fit(func_b, x, y)
    plt.plot(x, func_b(x, *popt2), '-',
             label='Gompertz: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt2))

    plt.scatter(x, y, c='peru')
    plt.xlabel('Time (Day)')
    plt.ylabel('Projected Leaf　Area (cm^2)')
    plt.legend()
    plt.savefig("16HR.png")
    plt.show()
    plt.close()

    AGR_x = np.linspace(0, 18, 1000)
    AGR_y = AGR_a(AGR_x, *popt)
    plt.plot(AGR_x, AGR_y, '-',
             label='Logistic: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt))

    AGR_y = AGR_b(AGR_x, *popt2)
    plt.plot(AGR_x, AGR_y, '-',
             label='Gompertz: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt2))

    plt.xlabel("Time (Day)")
    plt.ylabel("Absolute growth rate")
    plt.title("16Hr Absolute growth rate")
    plt.savefig("16HR_AGR.png")
    plt.legend()
    plt.show()
    plt.close()

    RGR_x = np.linspace(0, 18, 1000)
    RGR_y = RGR_a(RGR_x, *popt)
    plt.plot(RGR_x, RGR_y, '-',
             label='Logistic: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt))

    RGR_y = RGR_b(RGR_x, *popt2)
    plt.plot(RGR_x, RGR_y, '-',
             label='Gompertz: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt2))

    plt.xlabel("Time (Day)")
    plt.ylabel("Related growth rate")
    plt.title("16Hr Related growth rate")
    plt.savefig("16HR_RGR.png")
    plt.legend()
    plt.show()
    plt.close()

    # 24-HR
    popt, _ = curve_fit(func_a, x, y2)
    plt.plot(x, func_a(x, *popt), '-',
             label='Logistic: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt))

    popt2, _ = curve_fit(func_b, x, y2)
    plt.plot(x, func_b(x, *popt2), '-',
             label='Gompertz: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt2))

    plt.scatter(x, y2, c='peru')
    plt.xlabel('Time (Day)')
    plt.ylabel('Projected Leaf　Area (cm^2)')
    plt.legend()
    plt.savefig("24HR.png")
    plt.show()
    plt.close()

    AGR_x = np.linspace(0, 18, 1000)
    AGR_y = AGR_a(AGR_x, *popt)
    plt.plot(AGR_x, AGR_y, '-',
             label='Logistic: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt))

    AGR_y = AGR_b(AGR_x, *popt2)
    plt.plot(AGR_x, AGR_y, '-',
             label='Gompertz: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt2))

    plt.xlabel("Time (Day)")
    plt.ylabel("Absolute growth rate")
    plt.title("24Hr Absolute growth rate")
    plt.savefig("24HR_AGR.png")
    plt.legend()
    plt.show()
    plt.close()

    RGR_x = np.linspace(0, 18, 1000)
    RGR_y = RGR_a(RGR_x, *popt)
    plt.plot(RGR_x, RGR_y, '-',
             label='Logistic: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt))

    RGR_y = RGR_b(RGR_x, *popt2)
    plt.plot(RGR_x, RGR_y, '-',
             label='fit: M0=%5.3f, K=%5.3f, r=%5.3f' % tuple(popt2))

    plt.xlabel("Time (Day)")
    plt.ylabel("Related growth rate")
    plt.title("24Hr Related growth rate")
    plt.savefig("24HR_RGR.png")
    plt.legend()
    plt.show()
    plt.close()







