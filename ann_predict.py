

# make predictions from ecommerce.csv file using ann
# going in the forward direction using randomly initialized weights
# (no training)

import numpy as np
from process import get_data

X, Y = get_data()

M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis = 1, keepdims = True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y_given_X, axis = 1)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("score: ", classification_rate(Y, predictions))


