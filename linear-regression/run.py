import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def fun(w0, w1, x):
    return w1 * x + w0

# Generate sample points
X = 1.0 * np.random.randint(0, 100, 100)
Y = 2 * X + 10
noise = 5.0*np.random.randn(Y.shape[0])
Y += noise

# Weight matrices
scaling = 1
w0 = np.random.uniform(-scaling, +scaling)
w1 = np.random.uniform(-scaling, +scaling)

# Fit a model using stochastic gradient descent
n_iter = 100
step_size = 1e-7
n = X.shape[0]
for i in range(n_iter):
    w0_grad = 0.0
    w1_grad = 0.0
    for j in range(n):

        # x and y
        x = X[j]
        y = Y[j]

        # Loss
        f = fun(w0, w1, x)
        l = f - y

        # Compute gradients
        w0_grad += 2 * l
        w1_grad += 2 * l * x

        # Update weights using LMS (stochastic gradient descent)
        w0 += -step_size * w0_grad
        w1 += -step_size * w1_grad

    # Print squared loss at each iteration
    loss = np.sum((fun(w0, w1, X) - Y)**2)
    print("Loss: "+str(loss))

# Compute predicted line
print(w1,w0)
Y_pred = fun(w0, w1, X)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X,Y)
ax.plot(X,Y_pred)
plt.show()