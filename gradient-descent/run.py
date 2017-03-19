import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def fun(x, y):
    return (x**2) / (2*y**2)

def fun_der_x(x,y):
    return (x/y**2)

def fun_der_y(x,y):
    return (-3 * x**2/y**3)

# Initial values - Start the solution search randomly
random = np.random.randint(-100,100,2)
x, y = random[0], random[1]
f = fun(x, y)

# Figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f")

# Parameters
step_size = 0.001
n_iter = 1000

# Do gradient descent
for i in range(n_iter):
    der_x = fun_der_x(x,y)
    der_y = fun_der_y(x,y)
    x_prev, y_prev, f_prev = x,y,f
    x -= step_size * der_x
    y -= step_size * der_y
    f = fun(x, y)
    ax.scatter(x_prev, y_prev, f_prev, color="blue", s=2)

ax.scatter(x, y, f, color="red", s=100)
print("Minimum value of f :" + str(f) + " found at ("+str(x)+","+str(y)+")")
plt.show()
