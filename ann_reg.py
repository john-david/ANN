
# using ANN for regression

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate and plot the dataset
N = 500
# x-values between [-2, 2]
X = np.random.random((N, 2))  * 4 - 2
# create a saddle y = x*x
Y = X[:, 0]*X[:, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:, 1], Y)
plt.show()

# make neural network and train 
D = 2
M = 100 # number of hidden units

# hidden layer (layer 1) 
W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)

# output layer (layer 2)
V = np.random.randn(M) / np.sqrt(M)
c = 0

def forward(X):
    Z = X.dot(W) + b
    # relu
    Z = Z * (Z > 0)

    Yhat = Z.dot(V) + c
    return Z, Yhat

def derivative_V(Z, Y, Yhat):
    return (Y - Yhat).dot(Z)

def derivative_c(Y, Yhat):
    return (Y - Yhat).sum()

def derivative_W(X, Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0) # relu activation
    return X.T.dot(dZ)

def derivative_b(Z, Y, Yhat, V):
    dZ = np.outer(Y - Yhat, V) * (Z > 0) # relu activation
    return dZ.sum(axis = 0)

def update(X, Z, Y, Yhat, W, b, V, c, learning_rate = 1e-4):
    gV = derivative_V(Z, Y, Yhat)
    gc = derivative_c(Y, Yhat)
    gW = derivative_W(X, Z, Y, Yhat, V)
    gb = derivative_b(Z, Y, Yhat, V)

    V += learning_rate * gV
    c += learning_rate * gc
    W += learning_rate * gW
    b += learning_rate * gb

    return W, b, V, c

def get_cost(Y, Yhat):
    # mean squared error
    return ((Y - Yhat)**2).mean()

# training loop

costs = []

for i in range(200):
    Z, Yhat = forward(X)
    W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)
    cost = get_cost(Y, Yhat)
    costs.append(cost)
    if i % 25 == 0:
        print(cost)

plt.plot(costs)
plt.show()

# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], Y)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, Yhat = forward(Xgrid)
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat, linewidth = 0.2, antialiased = True)
plt.show()

# plot the magnitude of residuals
Ygrid = Xgrid[:, 0] * Xgrid[:, 1]
R = np.abs(Ygrid - Yhat)

plt.scatter(Xgrid[:, 0], Xgrid[:, 1], c = R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], R, linewidth = 0.2, antialiased = True)
plt.show()


