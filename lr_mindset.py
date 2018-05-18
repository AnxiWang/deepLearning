import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px1 = train_set_x_orig.shape[1]
num_px2 = train_set_x_orig.shape[2]

train_set_x_flatten = train_set_x_orig.reshape(-1, num_px1 * num_px2 * 3).T
test_set_x_flatten = test_set_x_orig.reshape(-1, num_px1 * num_px2 * 3).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    s = 1 /(1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w, b = np.zeros((dim, 1)), 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 -Y) * np.log(1 - A))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {
        "dw":dw,
        "db":db
    }
    return grads, cost

def optimize(w, b, X, Y, num_interations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_interations):
         grads, cost = propagate(w, b, X, Y)
         dw = grads["dw"]
         db = grads["db"]
         w = w - learning_rate * dw
         b = b - learning_rate * db
         if i % 100 == 0:
             costs.append(cost)
         if print_cost and i % 100 == 0:
            print("cost after interation %i: %f" %(i, cost))
    params = {
        "w":w,
        "b":b
    }
    grads = {
        "dw":dw,
        "db":db
    }
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_predict = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_predict[0, i] = 0
        else:
            Y_predict[0, i] = 1
    assert (Y_predict.shape == (1, m))
    return Y_predict

def model(X_train, Y_train, X_test, Y_test, num_interations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_interations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_predict_train = predict(w, b, X_train)
    Y_predict_test = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100- np.mean(np.abs(Y_predict_test- Y_test)) * 100))

    d = {
        "costs":costs,
        "Y_prediction_test":Y_predict_test,
        "Y_prediction_train":Y_predict_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_interations":num_interations
    }
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_interations=2000, learning_rate=0.005, print_cost=True)







