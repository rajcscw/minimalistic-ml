import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal

def fun(W, X):
    return np.dot(W, X)

# Data
N = 100
D = 2
X = np.random.uniform(0,100,N*D).reshape((D, N))
Y = np.dot(np.array([2, 3]), X).reshape(1, N)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0, :], X[1,:],Y, "blue")

# Initialize weight W
W = np.random.uniform(-100,100, D)

# Parameters
n_iter = 500
step_size = 9e-6

# Fit a model using stochastic gradient descent
for i in range(n_iter):
    for j in range(N):
        # x and y
        x = X[:,j]
        y = Y[:,j]

        # f
        f = fun(W,x)

        # Compute gradients
        w_grad = 2 * (f - y) * x.T

        # Update weights using stochastic gradient descent
        W -= step_size * w_grad

    # Print squared loss every 100 iterations
    if i % 10 == 0:
        loss = np.sum((fun(W,X) - Y)**2)
        print("Loss: "+str(loss))

# Plot the plane
print("W: "+ np.array_repr(W,  precision=6))
xx, yy = np.meshgrid(X[0, :],X[1,:])
Y_pred = np.dot(np.array(W[0]), xx) + np.dot(np.array(W[1]), yy)
ax.plot_surface(xx,yy,Y_pred)
plt.show()
