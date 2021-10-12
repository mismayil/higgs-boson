import numpy as np

def compute_error(y, tx, w):
    N = len(y)
    y = y.reshape((-1, 1))
    X = tx.reshape((N, -1))
    w = w.reshape((-1, 1))
    e = y - np.dot(X, w)
    return e


def compute_loss(y, tx, w):
    N = len(y)
    e = compute_error(y, tx, w)
    return (1/(2*N)) * np.sum(e**2)


def least_squares(y, tx):
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_loss(y, tx, w)
    return w, loss