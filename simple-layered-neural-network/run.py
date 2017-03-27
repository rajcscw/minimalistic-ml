from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fun(W, X):
    return np.tanh(np.dot(W, X))

def generate_points(N,D):
    X, Y = datasets.make_classification(n_samples=N, n_features=D, n_informative=2, n_redundant=0, class_sep=1.0)
    X = X.T
    Y[Y==0] = -1
    return X,Y

def predict(W_ih, W_ho,X):
    hidden_layer = fun(W_ih, X)
    output_layer = fun(W_ho, hidden_layer)
    output_layer[output_layer <= 0] = -1
    output_layer[output_layer > 0] = 1
    return output_layer

# Generate data
N = 1000
D = 2
np.random.seed(10)
X,Y = generate_points(N,D)
X = np.vstack((X,np.ones((1,N))))
D += 1

# Initialize weight W
hid = 3
W_ih = np.random.rand(hid+1,3)
W_ho = np.random.rand(1,hid+1)

# Parameters
n_iter = 500
step_size = 9e-5

# Train the neuron
for i in range(n_iter):
    for j in range(N):
        # x and y
        x = X[:,j].reshape(D,1)
        y = Y[j]

        # Do a forward pass
        hidden_layer = fun(W_ih, x)
        output_layer = fun(W_ho, hidden_layer)

        # Do a backward pass
        loss_gradient = (output_layer - y)
        delta_output = loss_gradient * (1-output_layer**2)
        delta_hidden = np.dot(W_ho.transpose(), delta_output) * (1-hidden_layer**2)

        W_ho -= step_size * np.dot(delta_output, hidden_layer.transpose())
        W_ih -= step_size * np.dot(delta_hidden, x.transpose())

    # Print squared loss every 10 iterations
    if i % 10 == 0:
        Y_pred = predict(W_ih, W_ho, X)
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

input = np.c_[xx.ravel(),yy.ravel()].T
input = np.vstack((input,np.ones((1,input.shape[1]))))
Z = predict(W_ih, W_ho, input)
Z = Z.reshape(xx.shape)
cm = plt.cm.YlGn
ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
col = np.where(Y==-1,'y','g')
ax.scatter(X[0, :],X[1,:],c=col)
plt.show()

# Print accuracy
Y_pred = predict(W_ih, W_ho, X)
correct = np.sum(Y == Y_pred)
print("The accuracy is: " + str(correct/N * 100)+" %")