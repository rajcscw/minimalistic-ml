from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fun(W, X):
    return np.tanh(np.dot(W, X))

def generate_points(N,D):
    X, Y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, class_sep=1.0)
    X = X.T
    Y[Y==0] = -1
    return X,Y

def predict(W,X):
    pred = fun(W,X)
    pred[pred <= 0] = -1
    pred[pred > 0] = 1
    return pred

# Generate data
N = 1000
D = 2
np.random.seed(20)
X,Y = generate_points(N,D)

# Initialize weight W
W = np.random.uniform(-5,5, D)

# Parameters
n_iter = 500
step_size = 5e-5

# Train the neuron
for i in range(n_iter):
    for j in range(N):
        # x and y
        x = X[:,j]
        y = Y[j]

        # Do a forward pass
        f = fun(W,x)

        # Compute gradients
        w_grad = (f - y) * (1-f**2) * x.T

        # Do a backward pass (updates gradient)
        W -= step_size * w_grad

    # Print squared loss every 10 iterations
    if i % 10 == 0:
        Y_pred = predict(W,X)
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
Z = predict(W, np.c_[xx.ravel(),yy.ravel()].T)
Z = Z.reshape(xx.shape)
cm = plt.cm.YlGn
ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
col = np.where(Y==-1,'y','g')
ax.scatter(X[0, :],X[1,:],c=col)
plt.show()

# Print accuracy
Y_pred = fun(W,X)
Y_pred[Y_pred <= 0] = -1
Y_pred[Y_pred > 0] = 1
correct = np.sum(Y == Y_pred)
print("The accuracy is: " + str(correct/N * 100)+" %")