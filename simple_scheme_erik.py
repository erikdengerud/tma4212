import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use("ggplot")

#Values
m = 5000		#space
n = 12	#time
h = 1 / m
k =	1 / (n - 1)
p = k / h #could inser 2*h
g = 9.81

#grid values
xvalues = np.array([ i * h for i in range(m + 1) ])
tvalues = np.array([ i * k for i in range(n) ])

#grid 0. (m+1)xn
h_grid = np.array([ [ 0. ] * (m + 1) ] * n)
u_grid = np.array([ [ 0. ] * (m + 1) ] * n)

#initial conditions:
h_grid[0,0:(m + 1)] = np.ones(len(xvalues))*np.sin(np.pi*xvalues) #sin to make inital disturbance

#continnuous wave from the left:
#h_grid[0:n,0] = 1. + 0.01 * np.sin(20 * tvalues)

#if something doesnt work try this
u_grid[0:n,0] = 0.
u_grid[0:n,m] = 0.

def main():

	#time
	for i in range(n-1): # all timesteps except the forst one as this is the initial condition

		#space
		for j in range(1,m): #not the endpoints as these are defined? boundaries?
			
			#the change from last point
			#h first as this is used in u
			dh = - p / 2 * h_grid[i,j] * (u_grid[i, j+1] - u_grid[i, j-1]) - p / 2 * u_grid[i,j] * (h_grid[i, j+1] - h_grid[i, j-1])

			h_grid[i+1,j] = h_grid[i,j] + dh
			
			#the new point
			du = - u_grid[i,j] / h_grid[i,j] * (h_grid[i+1, j] - h_grid[i,j]) \
					- p * u_grid[i,j] / h_grid[i,j] * (u_grid[i, j+1] - u_grid[i, j-1]) \
					- p / 2 * u_grid[i,j]**2 / h_grid[i,j] * (h_grid[i, j+1] - h_grid[i, j-1]) \
					- p * g / 2 * (h_grid[i, j+1] - h_grid[i, j-1])


			u_grid[i+1,j] = u_grid[i,j] + du

	print(h_grid[10])
	#explodes with time
	h_grid_plot = h_grid[:10,]
	print(h_grid_plot)

	fig, ax = plt.subplots()
    
	line, = ax.plot(xvalues,h_grid_plot[0])

	def animate(i):
		line.set_ydata(h_grid_plot[i])
		return line,
	def init():
		line.set_ydata(np.ma.array(xvalues, mask=True))
		return line,
    
	ax.ani = animation.FuncAnimation(fig, animate, np.arange(1, 10), init_func = init,
                                  interval=25, blit=True)
	plt.show()
	

	return 0

main()