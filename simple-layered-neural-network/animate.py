from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def fun(W, b, X):
    return np.tanh(np.dot(W, X) + b)

def generate_points(N,D):
    X, Y = datasets.make_moons(n_samples=N, noise=0.3, random_state=0)
    X = X.T
    Y[Y==0] = -1
    return X,Y

def predict(W_ih, W_ho,bi_h, bi_o, X):
    hidden_layer = fun(W_ih, bi_h, X)
    output_layer = fun(W_ho, bi_o, hidden_layer)
    output_layer[output_layer <= 0] = -1
    output_layer[output_layer > 0] = 1
    return output_layer

def update(frame):
    global W_ih, bi_h, W_ho, bi_o, X, Y, N, D, step_size, next, xx, yy, Z, contourplot, scatterplot

    # Train
    # x and y
    x = X[:,next].reshape(D,1)
    y = Y[next]

    # Do a forward pass
    hidden_layer = fun(W_ih, bi_h, x)
    output_layer = fun(W_ho, bi_o, hidden_layer)

    # Do a backward pass
    loss_gradient = (output_layer - y)
    delta_output = loss_gradient * (1-output_layer**2)
    delta_hidden = np.dot(W_ho.transpose(), delta_output) * (1-hidden_layer**2)

    W_ho -= step_size * np.dot(delta_output, hidden_layer.transpose())
    bi_o -= step_size * delta_output
    W_ih -= step_size * np.dot(delta_hidden, x.transpose())
    bi_h -= step_size * delta_hidden

    next = next + 1
    if next == N-1:
        next = 0

    # Update every 200th update
    # Set the plot data

    if next % N == 0:
        print("Updating...")
        input = np.c_[xx.ravel(),yy.ravel()].T
        Z = predict(W_ih, W_ho, bi_h, bi_o, input)
        Z = Z.reshape(xx.shape)

        for c in contourplot.collections:
            c.remove()

        contourplot = ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
        scatterplot = ax.scatter(X[0, :],X[1,:],c=col)

    # return plots
    return scatterplot, contourplot

# Generate data
N = 100
D = 2
np.random.seed(0)
X,Y = generate_points(N,D)

# Initialize weight W and b
hid = 3
W_ih = np.random.rand(hid,2)
bi_h = np.random.rand(hid, 1)
W_ho = np.random.rand(1,hid)
bi_o = np.random.rand(1,1)

# Parameters
n_iter = 10
step_size = 0.01
next = 0

# Plot the decision boundary
fig = plt.figure()
ax = fig.add_subplot(111)
col = np.where(Y==-1,'y','g')
scatterplot = ax.scatter(X[0, :],X[1,:],c=col)

h = .02
x_min, x_max = X[0, :].min() - 1, X[0,:].max() + 1
y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

input = np.c_[xx.ravel(),yy.ravel()].T
Z = predict(W_ih, W_ho, bi_h, bi_o, input)
Z = Z.reshape(xx.shape)
cm = plt.cm.YlGn
contourplot = ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)


# Create the animation
ani = FuncAnimation(fig, update, interval=1, frames=500, repeat=False)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
plt.show()
ani.save('animation.mp4', writer=writer)
