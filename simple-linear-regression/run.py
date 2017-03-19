import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def fun(w0, w1, x):
    return w1 * x + w0

# Generate sample points
X = 1.0 * np.random.randint(0, 100, 100)
Y = 2 * X + 10
noise = 2.0*np.random.randn(Y.shape[0])
Y += noise

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X,Y)
plt.savefig("points.svg")

# Initialize weight w0, w1
w0 = np.random.randint(0,100)
w1 = np.random.randint(0,100)

# Parameters
n_iter = 1000
step_size = 9e-5
n = X.shape[0]

# Fit a model using stochastic gradient descent
for i in range(n_iter):
    for j in range(n):
        # x and y
        x = X[j]
        y = Y[j]

        # f
        f = fun(w0, w1, x)

        # Compute gradients
        w0_grad = 2 * (f - y)
        w1_grad = 2 * (f - y) * x

        # Update weights using stochastic gradient descent
        w0 += -step_size * w0_grad
        w1 += -step_size * w1_grad

    # Print squared loss every 100 iterations
    if i % 100 == 0:
        loss = np.sum((fun(w0, w1, X) - Y)**2)
        print("Loss: "+str(loss))

# Plot the predicted line
Y_pred = fun(w0, w1, X)
ax.plot(X,Y_pred)
plt.show()
print("w1="+str(w1)+", w0="+str(w0))