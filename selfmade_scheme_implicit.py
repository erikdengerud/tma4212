import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use("ggplot")

#This method uses BC
#  h(0,t_n+1))=h(x_1,t_n))
#  h(x_m+1,t_n+1))=h(x_m,t_n))


N = 500
M = 200
dx = 10 / (M+1)
T = 10
dt = T / (N + 1)
g = 9.81

xvalues = np.array([i * dx for i in range(M+2)])
tvalues = np.array([i * dt for i in range(N+2)])

h = np.array([[1.] * (M + 2)] * (N+2))
u = np.array([[0.] * (M + 2)] * (N+2))

#h[0,0:M+2] = 1+2/5*np.exp(-5*(xvalues-dx*(M+1)/2)**2)
h[0,0:M+2] = 1+1/2*np.sin(xvalues/(2*np.pi))


def main():
    A = np.array([[0.] * (2*M)] * 2*M)
    b = np.array([0.] * (2*M))
    for n in range(N+1):
        for i in range(1,M-1):
            A[(2*(i-1)):2*(i-1)+6,i] = np.array([dt*h[n][i],dt*u[n][i]*h[n][i],0,2*dx*h[n][i+1],-dt*h[n][i+2],-dt*u[n][i+2]*h[n][i+2]])
        for i in range(M+1,2*M-1):
            A[(2*(i-M-1)):2*(i-M-1)+6,i] = np.array([dt*u[n][i-M],dt*g*h[n][i-M],2*dx,0,-dt*u[n][i+2-M],-dt*g*h[n][i+2-M]])
        A[1:4,0]                = np.array([2*dx*h[n][1],-dt*h[n][2],-dt*u[n][2]*h[n][2]])
        A[2*M-4:2*M,M-1]        = np.array([dt*h[n][M-1],dt*u[n][M-1]*h[n][M-1],0,2*dx*h[n][M]])
        A[0:4, M]               = np.array([2*dx,0,-dt*u[n][2],-dt*g*h[n][2]])
        A[2 * M - 4:2 * M, 2*M -1] = np.array([dt*u[n][M-1],dt*g*h[n][M-1],2*dx,0])

        for i in range(0,2*M,2):
            b[i]    =  2*dx*h[n][i//2+1]
            b[i+1]  =  2*dx*h[n][i//2+1]*u[n][i//2+1]
        b[0]    += dt*u[n][1]*h[n][1]
        b[1]    += dt*g*h[n][1]*h[n][1]
        b[-2]   += -dt*u[n][-2]*h[n][M]
        b[-1]   += -dt*g*h[n][-2]*h[n][M]


        x = np.linalg.solve(A,b)
        h[n+1][1:M+1] = x[M:2*M]
        h[n+1][0]     = h[n][1]
        h[n+1][M+1]   = h[n][M]
        u[n+1][1:M+1] = x[0:M]

    fig, ax = plt.subplots()

    line, = ax.plot(xvalues, h[0])

    def animate(i):
        line.set_ydata(h[i])
        return line,

    def init():
        line.set_ydata(np.ma.array(xvalues, mask=True))
        return line,

    ax.ani = animation.FuncAnimation(fig, animate, np.arange(1, N+2), init_func=init,
                                     interval=25, blit=True)
    plt.ylim(0, 3)
    plt.show()

    return 0


if __name__ == "__main__":
    main()




