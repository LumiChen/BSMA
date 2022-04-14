import numpy as np
import matplotlib.pyplot as plt
import timeit


def f(y, t):
    u, v = y
    return np.asarray([998*u+1998*v, -999*u-1999*v])


def h(y, t):
    dh = -1 * np.sin(np.pi * t / 60) / (16 * (5 * y - y ** 2) ** 0.5)
    return dh


def RK_4(f, uv, Dt, FT):
    total_step = int(round(FT/Dt))  # how many steps
    time_range = np.linspace(0, Dt * total_step, total_step + 1)

    y0 = np.copy(uv)
    fsol = [y0]
    for t in time_range[:-1]:
        y1 = f(y0, t)*Dt
        y2 = f(y0 + (y1 / 2), (t + Dt / 2))*Dt
        y3 = f(y0 + (y2 / 2), (t + Dt / 2))*Dt
        y4 = f(y0 + y3, t + Dt)*Dt
        y0 = y0 + ((y1 + 2 * (y2 + y3) + y4) / 6)
        fsol.append(y0)
    return fsol, time_range


def Euler(f, h, Dt, FT):
    total_step = int(round(FT/Dt))  # how many steps
    time_range = np.linspace(0, Dt * total_step, total_step + 1)
    fsol = [h]
    y0 = np.copy(h)
    # Euler function
    for t in time_range[:-1]:
        y0 = y0 + Dt * f(y0, t)
        fsol.append(y0)
    return fsol, time_range


if __name__ == '__main__':
    t0, tf = 0, 10
    # P3
    y0 = 1, 0
    t = np.linspace(0, 10, 100)
    Dt = 0.0024
    u, t = RK_4(f, y0, Dt, tf)
    u1 = [a[0] for a in u]
    u2 = [a[1] for a in u]
    t = [i for i in range(len(t))]

    # plot
    plt.plot(t, u1, label='[u]')
    plt.plot(t, u2, label='[v]')
    plt.xlabel('steps')
    plt.ylabel('value')
    plt.title('RK4, dt=' + str(Dt))
    plt.legend()
    # plt.savefig("RK4_" + str(Dt) + '.png')
    plt.show()

    # p4

    Dt = 5
    h0 = [2]
    t0, tf = 0, 2400

    u_RK4, t = RK_4(h, h0, Dt, tf)
    u_Euler, t = Euler(h, h0, Dt, tf)

    u1 = [a[0] for a in u_RK4]
    u2 = [a[0] for a in u_Euler]
    Er = [100*abs(a - b) for a, b in zip(u1, u2)]

    fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1)
    ax1.plot(t, u1, label='RK4')
    ax1.set(xlabel='steps',
            ylabel='value',
            title='RK-4, dt=' + str(Dt))
    ax1.legend(loc='upper left')

    ax2.plot(t, u2, label='Euler')
    ax2.set(xlabel='steps',
            ylabel='value',
            title='Euler, dt=' + str(Dt))
    ax2.legend(loc='upper left')

    ax3.plot(t, u1, label='RK4')
    ax3.plot(t, u2, label='Euler')
    ax3.set(xlabel='steps',
            ylabel='value',
            title='RK4 vs Euler, dt=' + str(Dt))

    ax4.plot(t, Er, label='Error')
    ax4.set(xlabel='steps',
            ylabel='Error (%)')
    plt.legend()
    # plt.savefig("p4_" + str(Dt) + '.png', dpi=150, figsize=(30, 12))
    plt.show()
