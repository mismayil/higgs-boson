import numpy as np
from proj1_helpers import predict_labels
import matplotlib.pyplot as plt
from itertools import product

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

def build_poly(X, degree, cont_features=None):
    X_poly = []
    for x in X:
        x_poly = []
        for d in range(1, degree+1):
            x = x if cont_features is None else x[cont_features]
            x_poly.append(x**d)
        X_poly.append(np.hstack(x_poly))
    return np.array(X_poly)

def kfold_cv_iter(y, tx, k = 5, seed = 1):
    num_row = y.shape[0]
    fold_size = int(num_row / k)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = list(set(range(num_row)) - set(test_indices))
        yield tx[train_indices], y[train_indices], tx[test_indices], y[test_indices]

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