import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use("ggplot")

N = 2000
M = 100
h = 10 / (M +1)
T = 10
k = T / (N + 1)
p = k / (2 * h)
g = 9.81
H = 1

def y1_0(x): return np.ones(len(x))*np.sin(np.pi * x) #sin to make initial wave
def y2_0(x): return y1_0(x) * 0.
def b_0(t): return 1. + 0.01 * np.sin(20 * tvalues)
def b_1(t): return 1. + 0.01 * np.sin(10 * tvalues)

xvalues = np.array([ i * h for i in range(M + 1) ])
tvalues = np.array([ i * k for i in range(N) ])

#making grids
h_grid = np.array( [ [ 0. ] * (M + 1) ] * N )
u_grid = np.array( [ [ 0. ] * (M + 1) ] * N )

#setting initial cond.
#h_grid[0,0:(M + 1)] = 1+y1_0(xvalues)
h_grid[0,0:M+2] = 1+2/5*np.exp(-5*(xvalues-h*(M+1)/2)**2)

u_grid[0,0:(M + 1)] = y2_0(xvalues)
print (h_grid)

#to make waves from the side
#h_grid[0:N,0] = b_0(tvalues)
#h_grid[0:N,M] = b_1(tvalues)

#??
#u_grid[0:N,0] = 0.
#u_grid[0:N,M] = 0.




def main():

	nfixed = N-1
	assert(nfixed < N)

	#method
	for i in range(nfixed):
		#can be done in one loop, but it is easier to read if we use two
		for j in range(M):#u
			du = p * g * (h_grid[i, j+1] - h_grid[i, j-1])
			u_grid[i + 1,j] = u_grid[i,j] - du


		for j in range(M):#h
			dh = p * H * (u_grid[i+1, j+1] - u_grid[i+1, j-1])
			h_grid[i + 1,j] = h_grid[i,j] - dh


		h_grid[i + 1,M] = 2 * h_grid[i + 1,M - 1] - h_grid[i + 1,M - 2]
		u_grid[i + 1,0] = 2 * u_grid[i + 1,1] - u_grid[i + 1,2]

	print(u_grid)
	print(h_grid)
	'''
	ones = 0
	for n in range(300):
		for m in range(50):
			if h_grid[n, m] == 1:
				ones+=1
	print(ones, ones/(300*50))
	'''
	#h_grid2 = h_grid[::2]
	#xvalues2 = xvalues[::2]

	fig, ax = plt.subplots()
	#j = int(i/2)
	line, = ax.plot(xvalues, h_grid[i])
	
	def animate(i):
		line.set_ydata(h_grid[i])
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