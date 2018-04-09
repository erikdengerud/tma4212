import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use("ggplot")

N = 500
M = 100
x_max = 1.
h = x_max / M
T = 1.
k = T / (N - 1)
r = k / (2 * h)

def h_0(x): return 1. + np.maximum( 0., 1. * np.sin(np.pi / x_max * x) - 0.99 * 1. )
def q_0(x): return h_0(x) * 0. #np.sin(x) * 1e-2

xvalues = np.array([ i * h for i in range(M + 1) ])
tvalues = np.array([ i * k for i in range(N) ])

w = np.array( [ np.zeros(2 * M + 2) ] * N )

w[0,::2] = h_0(xvalues)
w[0,1::2] = q_0(xvalues)
w[0,1] = 0.
w[0,-1] = 0.

def A(x):
	return np.array( [ [ 0., 1. ], [ -(x[1] / x[0])**2 + 9.81 * x[0], 2. * x[1] / x[0] ] ] )
res = np.array( [ np.zeros(2 * (M + 1)) ] * 2 * (M + 1) )
def B(n):
	res[0:2,0:2] = 2 * A(w[n,0:2])
	for i in range(1, M):
		res[2 * i:2 * i + 2,2 * i:2 * i + 2] = A(w[n,2 * i:2 * i + 2])
	res[2 * M:2 * M + 2,2 * M:2 * M + 2] = 2 * A(w[n,-2:])
	return res
D = np.array( [ np.zeros(2 * (M + 1)) ] * 2 * (M + 1) )
D[0,0] = -1.
D[1,1] = -1.
D[-1,-1] = 1.
D[-2,-2] = 1.
D[2:,:-2] -= np.identity(2 * M)
D[:-2,2:] += np.identity(2 * M)

def main():
    
    for i in range(0, N - 1):
        C = np.identity(2 * M + 2) + r * B(i) @ D
        # for the velocity in the edges to be constaint we change
        # the matrix a bit
        C[1,:] = 0.
        C[-1,:] = 0.
        C[1,1] = 1.
        C[-1,-1] = 1.
        
        w[i + 1] = np.linalg.solve(C, w[i])
        
    
    y1 = w[:,0::2]
    fig, ax = plt.subplots()
    
    line, = ax.plot(xvalues, y1[0])
    def animate(i):
        line.set_ydata(y1[i])
        return line,
    def init():
        line.set_ydata(np.ma.array(xvalues, mask=True))
        return line,
    
    ax.ani = animation.FuncAnimation(fig, animate, np.arange(1, N - 1), init_func = init,
                                  interval = 25, blit=True)
    
    plt.show()
    

if __name__ == "__main__":
    main()




