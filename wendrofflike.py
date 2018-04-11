import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use("ggplot")

N = 2000
M = 100
dx = 10 / (M+1)
T = 10
dt = T / (N + 1)
g = 10

xvalues = np.array([i * dx for i in range(M+2)])
tvalues = np.array([i * dt for i in range(N+2)])

h = np.array([[1.] * (M + 2)] * (N+2))
u = np.array([[0.] * (M + 2)] * (N+2))

h[0,0:M+2] = 1+2/5*np.exp(-5*(xvalues-dx*(M+1)/2)**2)


def ddx(y,n, m):
    return (y[n][m+1] - y[n][m - 1]) / (2*dx)


def main():

	for n in range(N+1):
		for m in range(1,M+1):
			h[n+1][m] = max(0, -(ddx(h,n,m)*u[n][m]+h[n][m]*ddx(u,n,m))*dt+h[n][m])
			if h[n+1][m] == 0:
				u[n+1][m] = 0
			else:
				u[n+1][m] = (-u[n][m]*(h[n+1][m]-h[n][m])/dt-2*h[n][m]*u[n][m]*ddx(u,n,m)-u[n][m]**2*ddx(h,n,m)-g*h[n][m]*ddx(h,n,m))*dt/h[n][m]+u[n][m]
    

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




