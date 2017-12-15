import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from gradient_descent import *

def quad(x):
    # a 3d parabola
    return x[0]**2 + x[1]**2

def double_dip(x):
    x, y = x[0], x[1]
    return x*math.exp(-(x**2 + y**2))

def cool_function(x):
    # a 3d sine wave
    x, y = x[0],x[1]
    return (math.sin(x)**2 + math.cos(y)**2)/(5 + x**2 + y**2)

f = double_dip
op = gradientDescentOp(double_dip, [1,0], 0.5, 0.9)
def update(frame):
    op.update()
    x = op.x
    return [plt.plot([x[0]], [x[1]], [op.current()], marker='o', markersize=7, color="red")]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# change these depending on function
x = y = np.arange(-5.0, 5.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([f([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

line_ani = animation.FuncAnimation(fig, update, 25, interval=500, blit=False)

plt.show()
