import numpy as np

def load_planar_datasets():
    m = 400
    N = int(m/2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4
    for j in range(2):
        ix = range(N * j, N *(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.rand(N) * 0.2
        r = a*np.sin(4*t) + np.random.rand(N) * 0.2
        X[ix] = np.c_[r*np.sin(t), r* np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T

    return X, Y

def sigmoid(x):
    z = 1/ (1 + np.exp(-x))
    return z