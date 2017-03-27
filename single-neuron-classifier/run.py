from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fun(W, b, X):
    return np.tanh(np.dot(W, X)+b)

def generate_points(N,D):
    X, Y = datasets.make_moons(n_samples=N, noise=0.3, random_state=0)
    X = X.T
    Y[Y==0] = -1
    return X,Y

def predict(W,b, X):
    pred = fun(W, b, X)
    pred[pred <= 0] = -1
    pred[pred > 0] = 1
    return pred

# Generate data
N = 1000
D = 2
np.random.seed(0)
X,Y = generate_points(N,D)

# Initialize weight W
W = np.random.rand(D)
b = np.random.rand(1)

# Parameters
n_iter = 500
step_size = 0.01

# Train the neuron
for i in range(n_iter):
    for j in range(N):
        # x and y
        x = X[:,j]
        y = Y[j]

        # Do a forward pass
        f = fun(W, b, x)

        # Compute gradients
        w_grad = (f - y) * (1-f**2) * x.T
        b_grad = (f-y) * (1-f**2)

        # Do a backward pass (updates gradient)
        W -= step_size * w_grad
        b -= step_size * b_grad

    # Print squared loss every 10 iterations
    if i % 10 == 0:
        Y_pred = predict(W, b, X)
        incorrect = np.sum(Y != Y_pred)
        print("Loss: "+str(incorrect/N * 100)+" %")

# Plot the decision boundary
fig = plt.figure()
ax = fig.add_subplot(111)
h = .02
x_min, x_max = X[0, :].min() - 1, X[0,:].max() + 1
y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = predict(W, b, np.c_[xx.ravel(),yy.ravel()].T)
Z = Z.reshape(xx.shape)
cm = plt.cm.YlGn
ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
col = np.where(Y==-1,'y','g')
ax.scatter(X[0, :],X[1,:],c=col)
plt.show()

# Print accuracy
Y_pred = predict(W, b, X)
correct = np.sum(Y == Y_pred)
print("The accuracy is: " + str(correct/N * 100)+" %")