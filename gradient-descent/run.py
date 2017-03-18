import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def fun(x, y):
    return x**2 + y**2

def fun_der_x(x):
    return 2 * x

def fun_der_y(y):
    return 2 * y

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
step_size = 0.07
n_iter = 100

# Do gradient descent
for i in range(n_iter):
    der_x = fun_der_x(x)
    der_y = fun_der_y(y)
    x_prev, y_prev, f_prev = x,y,f
    x -= step_size * der_x
    y -= step_size * der_y
    f = fun(x, y)
    ax.scatter(x_prev, y_prev, f_prev)
plt.show()
