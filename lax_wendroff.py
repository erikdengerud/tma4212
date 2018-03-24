import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use("ggplot")

N = 300
M = 50
h = 1 / M
T = 1
k = T / (N - 1)
r = k / (2 * h)

def y1_0(x): return np.ones(len(x)) #np.sin(np.pi * x)
def y2_0(x): return y1_0(x) * 0.
def b_0(t): return 1. + 0.01 * np.sin(20 * tvalues)
def b_1(t): return 1. + 0.01 * np.sin(10 * tvalues)

xvalues = np.array([ i * h for i in range(M + 1) ])
tvalues = np.array([ i * k for i in range(N) ])

y1 = np.array( [ [ 0. ] * (M + 1) ] * N )
y2 = np.array( [ [ 0. ] * (M + 1) ] * N )

y1[0,0:(M + 1)] = y1_0(xvalues)
y2[0,0:(M + 1)] = y2_0(xvalues)

y1[0:N,0] = b_0(tvalues)
#y1[0:N,M] = b_1(tvalues)

y2[0:N,0] = 0.
y2[0:N,M] = 0.

def f(n, m):
    if np.abs(y1[n,m]) < 1e-6:
        return np.array( [ 0., 0. ] )
    return np.array( [ y2[n,m], y2[n,m]**2 / y1[n,m] + 0.5 * 9.81 * y1[n,m]**2 ] )

def J(n, m):
    if np.abs(y1[n,m]) < 1e-6:
        return np.array([[0., 1.],
                         [0., 0.]])
    return np.array([[ 0., 1. ],
                     [ -y2[n,m]**2 / y1[n,m]**2 + 9.81 * y1[n,m], 2 * y2[n,m] / y1[n,m] ]])
def meanJ(n, m):
    return 0.5 * (J(n, m) + J(n, m + 1))

def main():
    
    nfixed = N - 1
    assert(nfixed < N)
    
    # explicit methods
    for i in range(nfixed):
        for j in range(1, M):
            #delta_y = -r * (f(i, j + 1) - f(i, j - 1))
            delta_y = -r * (f(i, j + 1) - f(i, j - 1)) + 2 * r**2 * ( meanJ(i, j) @ (f(i, j + 1) - f(i, j)) - 
                            meanJ(i, j - 1) @ (f(i, j) - f(i, j - 1)) )
            
            y1[i + 1,j] = y1[i,j] + delta_y[0]
            y2[i + 1,j] = y2[i,j] + delta_y[1]
        #y1[i + 1,0] = 2 * y1[i + 1,1] - y1[i + 1,2]
        y1[i + 1,M] = 2 * y1[i + 1,M - 1] - y1[i + 1,M - 2]
        y2[i + 1,0] = 2 * y2[i + 1,1] - y2[i + 1,2]
        #y2[i + 1,M] = 2 * y2[i + 1,M - 1] - y2[i + 1,M - 2]
    
    
    
    fig, ax = plt.subplots()
    
    line, = ax.plot(xvalues, y1[0])
    def animate(i):
        line.set_ydata(y1[i])
        return line,
    def init():
        line.set_ydata(np.ma.array(xvalues, mask=True))
        return line,
    
    ax.ani = animation.FuncAnimation(fig, animate, np.arange(1, nfixed), init_func = init,
                                  interval=25, blit=True)
    
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()




