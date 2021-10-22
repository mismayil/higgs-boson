import numpy as np
from proj1_helpers import predict_labels
import matplotlib.pyplot as plt

def compute_error(y, tx, w):
    N = len(y)
    y = y.reshape((-1, 1))
    X = tx.reshape((N, -1))
    w = w.reshape((-1, 1))
    e = y - np.dot(X, w)
    return e


def compute_mse(y, tx, w):
    N = len(y)
    e = compute_error(y, tx, w)
    return (1/(2*N)) * np.sum(e**2)


def compute_accuracy(y_true, y_pred):
    y_true = y_true.reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    N = len(y_true)
    return ((y_true == y_pred).sum() / N) * 100


def compute_f1(y_true, y_pred, pos_val=1, neg_val=-1):
    y_true = y_true.reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    tp = ((y_pred == pos_val) & (y_pred == pos_val)).sum()
    fp = ((y_pred == pos_val) & (y_true == neg_val)).sum()
    fn = ((y_pred == neg_val) & (y_true == pos_val)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def least_squares(y, tx):
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    N, D = tx.shape
    w = np.linalg.solve(np.dot(tx.T, tx) + 2 * lambda_ * N * np.eye(D), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    return loss, w


def build_poly(x, degree):
    return np.apply_along_axis(lambda r: np.array([r**d for d in range(degree+1)]), 0, x).T


def ridge_regression_cv(y, x, k_indices, k, lambda_, degree):
    test_indices = k_indices[k]
    train_indices = list(set(range(len(y))) - set(k_indices[k]))
    train_x, train_y = x[train_indices], y[train_indices]
    test_x, test_y = x[test_indices], y[test_indices]

    # train_x_poly = build_poly(train_x, degree)
    # test_x_poly = build_poly(test_x, degree)

    loss_tr, weights = ridge_regression(train_y, train_x, lambda_)

    loss_te = compute_mse(test_y, test_x, weights)
    y_pred = predict_labels(weights, test_x)
    accuracy = compute_accuracy(test_y, y_pred)
    f1 = compute_f1(test_y, y_pred)
    return weights, loss_tr, loss_te, accuracy, f1


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def ridge_regression_kfold(y, tx, lambdas=np.logspace(-4, 0, 30), k_fold=4, degree=7, seed=1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_lambda = None
    best_weights = None
    best_accuracy = None
    best_f1 = None
    min_loss = None

    for lambda_ in lambdas:
        rmse_losses = []
        accuracies = []
        f1s = []
        for k in range(len(k_indices)):
            weights, loss_tr, loss_te, accuracy, f1 = ridge_regression_cv(y, tx, k_indices, k, lambda_, degree)
            rmse_losses.append(np.sqrt(2 * loss_te))
            accuracies.append(accuracy)
            f1s.append(f1)
        loss = np.array(rmse_losses).mean()
        accuracy = np.array(accuracies).mean()
        f1 = np.array(f1s).mean()

        if min_loss is None or loss < min_loss:
            best_lambda = lambda_
            best_weights = weights
            best_accuracy = accuracy
            best_f1 = f1
            min_loss = loss

    return best_weights, best_lambda, min_loss, best_accuracy, best_f1

def standardize_data(x):
    return (x-np.mean(x, axis=0)) / np.std(x, axis=0)


def logistic_regression(y, tx, initial_w = None, max_iters = 1000, gamma = 0.1, plot=True):
    losses = []
    accuracies = []
    max_accuracy = 0
    best_epoch = None
    loss = None
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))
    
    for i in range(max_iters):
        y_pred = predict(w, tx)
        loss = compute_binary_loss(y, tx, w)
        losses.append(loss)

        accuracy = compute_accuracy(y, y_pred)
        accuracies.append(accuracy)

        if accuracies[i] > max_accuracy:
            max_accuracy = accuracies[i]
            best_epoch = i

        if i % 10 == 0:
            print("Iteration {}/{}".format(i, max_iters))
            print("Accuracy = {}%".format(accuracies[i]))
            print("Loss = {}".format(loss))

        w -= gamma * logistic_gradient(tx, y, w)
        losses.append(loss)
        
    if plot:
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(list(range(max_iters)), accuracies)
        axs[1].plot(list(range(max_iters)), losses)
    
    print("Best Accuracy : {}% reached at epoch {}".format(max_accuracy, best_epoch))
    
    return w, losses, accuracies


def logistic_gradient(x, y, w):
    return (1 / len(y)) * x.T @ (sigmoid(x @ w) - y)


def sigmoid(x):
    x = np.clip(x, -20, 20)
    return np.exp(x) / (1 + np.exp(x))


def predict(w, x):
    y_pred = sigmoid(x @ w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def compute_binary_loss(y, x, w):
    probs = sigmoid(x @ w)
    return -(1 / len(y)) * np.sum(y * np.log(probs) + (1-y) * np.log(1-probs))


def encode_binary(y):
    y_copy = y.copy()
    y_copy[np.where(y_copy == -1)] = 0
    return y_copy


def add_bias(x):
    return np.c_[np.ones(x.shape[0]), x]