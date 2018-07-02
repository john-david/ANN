
# using a neural network on XOR and the donut problems

import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    # this is a binary classification problem, using two sigmoids
    Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))
    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z

def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)

def derivative_w2(Z, T, Y):
    # Z is NxM mat
    return Z.T.dot(T - Y)

def derivative_b2(T, Y):
    return (T - Y).sum()

def derivative_w1(X, Z, T, Y, W2):
    dZ = np.outer(T - Y, W2) * (1 - Z * Z)
    return dZ.sum(axis = 0)

def derivative_b1(Z, T, Y, W2):
    dZ = np.outer(T - Y, W2) * (1 - Z * Z)
    return dZ.sum(axis = 0)

def cost(T, Y):
    # binary cross entropy cost function
    total = 0
    for n in range(len(T)):
        if T[n] == 1:
            total =+ np.log(Y[n])
        else:
            total += np.log(1 - Y[n])
    return total

# XOR problem
def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4)
    b2 = np.random.randn(1)
    # array to track log likelihood
    LL = []
    learning_rate = 0.0005
    regularization = 0.
    last_error_rate = None
    
    # gradient descent
    for i in range(100000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        err = np.abs(prediction - Y).mean()
        if err != last_error_rate:
            last_error_rate = err
            print("error rate: ", err)
            print("true: ", Y)
            print("pred: ", prediction)
        # exit early if the log likelihood increases
        if LL and ll < LL[-1]:
            print("early exit")
            break
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 10000 == 0:
            print(ll)

    print("final classification rate: ", 1 - np.abs(prediction - Y).mean() )
    
    plt.title("XOR problem classification rate")
    plt.plot(LL)
    plt.show()


# donut problem
def test_donut():
    # create donut data set
    N = 1000
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer ])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    # log likelihood array
    LL = []
    learning_rate = 0.00005
    regularization = 0.
    last_error_rate = None
    for i in range(160000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        err = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 1000 == 0:
            print("ll: ", ll, " classification rate:", 1 - err)

    plt.title("donut classification rate")
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    # test_xor()
    test_donut()





