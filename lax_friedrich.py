import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use("ggplot")

N = 500
M = 100
x_max = 1.
dx = x_max / (M+1)
T = 1.
dt = T / (N + 1)
r = 0.5 * dt / dx
g = 9.81

def h_0(x): return 1. + np.maximum( 0., 1. * np.sin(np.pi / x_max * x) - 0.99 * 1. )
def q_0(x): return h_0(x) * 0.

xvalues = np.array([ i * dx for i in range(M + 2) ])
tvalues = np.array([ i * dt for i in range(N + 2) ])

w = np.array( [ np.zeros(2 * M + 4) ] * (N + 2) )

w[0,::2] = h_0(xvalues)
w[0,1::2] = q_0(xvalues)
w[0,1] = 0.
w[0,-1] = 0.

def w_next(w,n):
    for m in range(2, 2 * M+2,2):
        w[n + 1, m : m + 2] = np.array( [ 0.5 * (w[n, m - 2] + w[n, m + 2]), 0.5 * (w[n, m - 1] + w[n, m + 3]) ] )\
                              - r * ( f(w, n, m + 2) - f(w, n, m - 2) )
    w[n + 1, 0] = 2 * w[n + 1, 2] - w[n + 1, 4]
    w[n + 1, -2] = 2 * w[n + 1, -4] - w[n + 1, -6]

def f(w,n,m):
    return np.array( [w[n, m + 1], w[n, m + 1]**2 / w[n, m] + 0.5 * g * w[n, m ]])


def main():

    for n in range(N+1):
        w_next(w,n)

    y1 = w[:, 0::2]
    fig, ax = plt.subplots()

    line, = ax.plot(xvalues, y1[0])

    def animate(i):
        line.set_ydata(y1[i])
        return line,

    def init():
        line.set_ydata(np.ma.array(xvalues, mask=True))
        return line,

    ax.ani = animation.FuncAnimation(fig, animate, np.arange(1, N - 1), init_func=init,
                                     interval=25, blit=True)

    plt.show()

    return 0

if __name__ == "__main__":
    main()
