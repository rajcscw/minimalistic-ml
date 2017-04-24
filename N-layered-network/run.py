from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fun(W, b, X):
    return np.tanh(np.dot(W, X) + b)

def generate_points(N,D):
    X, Y = datasets.make_moons(n_samples=N, noise=0.3, random_state=0)
    X = X.T
    Y[Y==0] = -1
    return X,Y

def predict(W, b, X):
    activations = []
    activations.append(X)
    for h in range(1, n_hid+2):
        activation = fun(W[h-1], b[h-1], activations[h-1])
        activations.append(activation)
    output_activation = activations[n_hid+1]
    output_activation[output_activation <= 0] = -1
    output_activation[output_activation > 0] = 1
    return output_activation

# Generate data
N = 1000
D = 2

np.random.seed(0)
X,Y = generate_points(N,D)

# Initialize weights W and biases b for all hidden layers and output layer
n_input = D
n_output = 1
hidden =[10, 5, 5]
n_hid = len(hidden)
W = []
b = []
prev_dim = n_input
for i in range(0,n_hid):
    W.append(np.random.rand(hidden[i], prev_dim))
    b.append(np.random.rand(hidden[i], 1))
    prev_dim = hidden[i]
W.append(np.random.rand(n_output, prev_dim))
b.append(np.random.rand(n_output,1))

# Parameters
n_iter = 2000
step_size = 1e-2

# Train the neuron
for i in range(n_iter):
    for j in range(N):
        # x and y
        x = X[:,j].reshape(D,1)
        y = Y[j]

        # Do a forward pass
        activations = []
        activations.append(x)
        for h in range(1, n_hid+2):
            activation = fun(W[h-1], b[h-1], activations[h-1])
            activations.append(activation)
        output_activation = activations[n_hid+1]

        # Do a backward pass
        loss_gradient = (output_activation - y)
        delta_output = loss_gradient * (1-output_activation**2)
        prev_delta = delta_output
        delta_hidden = {}
        for h in range(n_hid, 0, -1):
            delta = np.dot(W[h].transpose(), prev_delta) * (1-activations[h]**2)

            # Now that we have used the previous layer's weight, we can safely update
            # Update the weights and bias for the previous layer
            W[h] -= step_size * np.dot(prev_delta, activations[h].transpose())
            b[h]-= step_size * prev_delta

            prev_delta = delta

        # Update the weights to input-to-1st hidden layer
        W[0] -= step_size * np.dot(prev_delta, activations[0].transpose())
        b[0] -= step_size * prev_delta

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

input = np.c_[xx.ravel(),yy.ravel()].T
Z = predict(W, b, input)
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