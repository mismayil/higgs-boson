import numpy as np
from proj1_helpers import predict_labels
import matplotlib.pyplot as plt
from itertools import product

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


def compute_f1(y_true, y_pred, pos_val=1, neg_val=0):
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
    return w, loss


def least_squares_gradient(y, tx, w):
    return tx.T @ ((tx @ w) - y)


def build_poly(X, degree, cont_features=None):
    X_poly = []
    for x in X:
        x_poly = []
        for d in range(1, degree+1):
            x = x if cont_features is None else x[cont_features]
            x_poly.append(x**d)
        X_poly.append(np.hstack(x_poly))
    return np.array(X_poly)


def least_squares_GD(y, tx, initial_w = None, max_iters = 100, gamma = 0.1, batch_size = None, num_batches = None, verbose = False):
    losses = []
    accuracies = []
    max_accuracy = 0
    best_epoch = None
    loss = None
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for tx_sample, y_sample in batch_iter(tx, y, batch_size=batch_size, num_batches=num_batches):
            y_pred = predict_labels(w, tx_sample)
            loss = compute_mse(y_sample, tx_sample, w)
            accuracy = compute_accuracy(y_sample, y_pred)
            w -= gamma * least_squares_gradient(tx, y, w)
        
        losses.append(loss)
        accuracies.append(accuracy)

        if accuracies[i] > max_accuracy:
            max_accuracy = accuracies[i]
            best_epoch = i

        if verbose and i % 10 == 0:
            print("Iteration {}/{}".format(i, max_iters))
            print("Accuracy = {}%".format(accuracies[i]))
            print("Loss = {}".format(loss))
            print('\n')
        
    if verbose:
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(list(range(max_iters)), accuracies)
        axs[1].plot(list(range(max_iters)), losses)
    
    print("Best Accuracy : {}% reached at epoch {}".format(max_accuracy, best_epoch))
    
    return w, loss

def least_squares_GD(y, tx, initial_w = None, max_iters = 100, gamma = 0.1, verbose = False):
    return least_squares_GD(y, tx, initial_w=initial_w, max_iters=max_iters, gamma=gamma, verbose=verbose, batch_size=1)


def kfold_cv_iter(y, tx, k = 5, seed = 1):
    num_row = y.shape[0]
    fold_size = int(num_row / k)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = list(set(range(num_row)) - set(test_indices))
        yield tx[train_indices], y[train_indices], tx[test_indices], y[test_indices]


def ridge_regression_cv(y, tx, lambdas=np.logspace(-4, 0, 30), k_fold=4, degree=7, seed=1):
    best_lambda = None
    best_weights = None
    best_accuracy = None
    best_f1 = None
    min_loss = None

    for lambda_ in lambdas:
        rmse_losses = []
        accuracies = []
        f1s = []
        for train_x, train_y, test_x, test_y in kfold_cv_iter(y, tx, k=k_fold, seed=seed):
            weights, _ = ridge_regression(train_y, train_x, lambda_)

            loss_te = compute_mse(test_y, test_x, weights)
            y_pred = predict_labels(weights, test_x)
            accuracy = compute_accuracy(test_y, y_pred)
            f1 = compute_f1(test_y, y_pred)

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

def standardize(x):
    return (x-np.mean(x, axis=0)) / np.std(x, axis=0)


def batch_iter(tx, y, batch_size=None, num_batches=None, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        y = y[shuffle_indices]
        tx = tx[shuffle_indices]

    batch_size = batch_size or len(tx)
    batches = int(np.ceil(len(tx) // batch_size))
    num_batches = num_batches or batches

    for i in range(num_batches):
        yield tx[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]


def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    n = len(y)
    indexes = np.linspace(0, n-1, n, dtype=int)
    np.random.shuffle(indexes)
    split_i = int(n * ratio)
    return np.take(x, indexes[:split_i]), np.take(y, indexes[:split_i]), np.take(x, indexes[split_i:]), np.take(y, indexes[split_i:])


def logistic_regression(y, tx, initial_w = None, max_iters = 1000, gamma = 0.1, lr_decay = 0, verbose = False, batch_size=None, num_batches=None):
    return reg_logistic_regression(y, tx, lambda_=0, initial_w=initial_w, max_iters=max_iters, gamma=gamma, lr_decay=lr_decay, verbose=verbose, batch_size=batch_size, num_batches=num_batches)


def reg_logistic_regression(y, tx, lambda_ = 0.1, initial_w = None, max_iters = 1000, gamma = 0.1, lr_decay = 0, verbose = False, batch_size=None, num_batches=None):
    losses = []
    accuracies = []
    max_accuracy = 0
    best_epoch = None
    loss = None
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for tx_sample, y_sample in batch_iter(tx, y, batch_size=batch_size, num_batches=num_batches):
            y_pred = predict_log_reg(w, tx_sample)
            loss = compute_log_loss(y_sample, tx_sample, w)
            accuracy = compute_accuracy(y_sample, y_pred)
            w -= gamma * logistic_gradient(tx, y, w, lambda_)
            gamma *= (1. / (1. + lr_decay*i))
        
        losses.append(loss)
        accuracies.append(accuracy)

        if accuracies[i] > max_accuracy:
            max_accuracy = accuracies[i]
            best_epoch = i

        if verbose and i % 10 == 0:
            print("Iteration {}/{}".format(i, max_iters))
            print("Accuracy = {}%".format(accuracies[i]))
            print("Loss = {}".format(loss))
            print('\n')
        
    if verbose:
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(list(range(max_iters)), accuracies)
        axs[1].plot(list(range(max_iters)), losses)
    
    print("Best Accuracy : {}% reached at epoch {}".format(max_accuracy, best_epoch))
    
    return w, loss


def logistic_regression_cv(y, tx, lambdas=None, degrees=None, k_fold=5, max_iters=100, gamma=0.1, degree=3, cont_features=None, seed=1, verbose = False):
    lambdas = lambdas or np.logspace(-4, 0, 5)
    degrees = degrees or list(range(1, 4))
    best_lambda = None
    best_degree = None
    best_weights = None
    best_accuracy = None
    best_f1 = None
    min_loss = None
    cv_losses = []
    cv_accuracies = []
    cv_f1s = []

    y = y.reshape((-1, 1))

    for degree in degrees:
        tx_poly = build_poly(tx, degree, cont_features)
        for lambda_ in lambdas:
            losses = []
            accuracies = []
            f1s = []
            for train_x, train_y, test_x, test_y in kfold_cv_iter(y, tx_poly, k=k_fold, seed=seed):
                weights, _ = reg_logistic_regression(train_y, train_x, max_iters=max_iters, lambda_=lambda_, gamma=gamma)

                loss_te = compute_log_loss(test_y, test_x, weights)
                y_pred = predict_log_reg(weights, test_x)
                accuracy = compute_accuracy(test_y, y_pred)
                f1 = compute_f1(test_y, y_pred)

                losses.append(loss_te)
                accuracies.append(accuracy)
                f1s.append(f1)
            loss = np.array(losses).mean()
            accuracy = np.array(accuracies).mean()
            f1 = np.array(f1s).mean()
            cv_losses.append(loss)
            cv_accuracies.append(accuracy)
            cv_f1s.append(f1)

            if min_loss is None or loss < min_loss:
                best_lambda = lambda_
                best_degree = degree
                best_weights = weights
                best_accuracy = accuracy
                best_f1 = f1
                min_loss = loss
            
            if verbose:
                print("Lambda = {}".format(lambda_))
                print("Degree = {}".format(degree))
                print("Accuracy = {}%".format(accuracy))
                print("F1 Score = {}".format(f1))
                print("Loss = {}".format(loss))
                print('\n')
    
    # if verbose:
    #     fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    #     axs[0].plot(lambdas, cv_losses)
    #     axs[1].plot(lambdas, cv_accuracies)
    #     axs[2].plot(lambdas, cv_f1s)

    return best_weights, min_loss, best_lambda, best_degree, best_accuracy, best_f1

def logistic_gradient(x, y, w, lambda_=0):
    return (1 / len(y)) * (x.T @ (sigmoid(x @ w) - y)) + 2 * lambda_ * w


def sigmoid(x):
    x = np.clip(x, -20, 20)
    return np.exp(x) / (1 + np.exp(x))


def predict_log_reg(w, x):
    y_pred = sigmoid(x @ w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def compute_log_loss(y, x, w):
    probs = sigmoid(x @ w)
    return -(1 / len(y)) * np.sum(y * np.log(probs) + (1-y) * np.log(1-probs))


def switch_encoding(y):
    y = y.copy()
    y[y == -1] = 0
    return y


def add_bias(x):
    return np.c_[np.ones(x.shape[0]), x]


def convert_nans(x):
    x = x.copy()
    x[x == -999] = np.nan
    return x


def get_col_nan_ratio(x):
    return np.count_nonzero(np.isnan(x), axis=0) / len(x)


def transform_X(X, nan_cols, imputable_cols, encodable_cols):
    # Drop all columns with nan values
    tX = np.delete(X, nan_cols, axis=1)

    # Impute some columns
    medians = np.nanmedian(X[:, imputable_cols], axis=0)
    imputed_X = X[:, imputable_cols]
    imputed_X = np.where(np.isnan(imputed_X), np.repeat(medians.reshape((1, -1)), imputed_X.shape[0], axis=0), imputed_X)

    # Encode some columns
    encoded_X = X[:, encodable_cols]
    encoded_X = np.where(np.isnan(encoded_X), 0, 1)

    tX = np.hstack([tX, imputed_X])
    tX = standardize(tX)
    tX = add_bias(tX)

    # Get continous columns
    cont_features = list(range(tX.shape[1]))

    tX = np.hstack([tX, encoded_X])

    return tX, cont_features

def transform_y(y):
    if y is not None:
        y = switch_encoding(y)
        return y.reshape((-1, 1))
    return y


def preprocess(X_train, y_train, X_test, y_test=None, imputable_th=0.5, encodable_min_th=0.3, encodable_max_th=0.7):
    # Replace -999 values with NaN
    X_train = convert_nans(X_train)

    # Compute NaN ratio for each column and drop these columns
    col_nan_ratio = get_col_nan_ratio(X_train)
    nan_cols = (col_nan_ratio > 0)
    imputable_cols = (col_nan_ratio < imputable_th) & (col_nan_ratio > 0)
    encodable_cols = (col_nan_ratio > encodable_min_th) & (col_nan_ratio < encodable_max_th)
    
    # Transform train data
    tX_train, cont_features = transform_X(X_train, nan_cols=nan_cols, imputable_cols=imputable_cols, encodable_cols=encodable_cols)

    # Transform test data
    X_test = convert_nans(X_test)
    tX_test, _ = transform_X(X_test, nan_cols=nan_cols, imputable_cols=imputable_cols, encodable_cols=encodable_cols)

    # Transform labels
    ty_train = transform_y(y_train)
    ty_test = transform_y(y_test)

    return tX_train, ty_train, tX_test, ty_test, cont_features


def read_header(data_path):
    with open(data_path) as f:
        header = f.readline()
    # Skip ID and Prediction columns
    return header.strip().split(',')[2:]