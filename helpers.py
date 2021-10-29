from typing import Generator, List, Tuple
import numpy as np
import csv


def load_csv_data(data_path: str, sub_sample: bool = False) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Loads data and returns y (class labels), tX (features) and ids (event ids)

    Args:
        data_path (str): Path to the data
        sub_sample (bool, optional): Whether to sub sample. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int]]: Label data, features data, ids
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_logistic(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Generates logistic class predictions given weights, and a test data matrix

    Args:
        w (np.ndarray): Weights
        x (np.ndarray): Input data

    Returns:
        np.ndarray: Predictions data
    """
    y_pred = sigmoid(x @ w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def predict_linear(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Generates linear class predictions given weights, and a test data matrix

    Args:
        w (np.ndarray): Weights
        x (np.ndarray): Input data

    Returns:
        np.ndarray: Predictions data
    """
    y_pred = x @ w
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred



def create_csv_submission(ids: List[int], y_pred: np.ndarray, name: str) -> None:
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Args: 
        ids (event ids associated with each prediction)
        y_pred (predicted class labels)
        name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score between ground truth and predictions

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        float: Accuracy as a percentage
    """
    y_true = y_true.reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    N = len(y_true)
    return ((y_true == y_pred).sum() / N) * 100


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1, neg_label: int = 0) -> float:
    """Compute F1 score between ground truth and predictions

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        pos_label (int, optional): Positive label value. Defaults to 1.
        neg_label (int, optional): Negative label value. Defaults to 0.

    Returns:
        float: F1 score
    """
    y_true = y_true.reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
    fp = ((y_pred == pos_label) & (y_true == neg_label)).sum()
    fn = ((y_pred == neg_label) & (y_true == pos_label)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def build_poly(tx: np.ndarray, degree: int, cont_features: List[int] = None, *args, **kwargs) -> np.ndarray:
    """Extend data with polynomial features of the given degree.
    If cont_features is specified, only expand those features,
    otherwise expand all features.

    Args:
        tx (np.ndarray): Data
        degree (int): Polynomial degree
        cont_features (List[int], optional): List of continuous features. Defaults to None.

    Returns:
        np.ndarray: Data extended with polynomial features
    """
    tx_poly = []
    for x in tx:
        x_poly = [x]
        for d in range(2, degree+1):
            x_sub = x if cont_features is None else x[list(cont_features)]
            x_poly.append(x_sub ** d)
        tx_poly.append(np.hstack(x_poly))
    return np.array(tx_poly)


def kfold_cv_iter(y: np.ndarray, tx: np.ndarray, k: int = 5, seed: float = 1) -> Generator:
    """K-fold cross validation. Split data into k parts and iterate through the folds

    Args:
        y (np.ndarray): Label data
        tx (np.ndarray): Features data
        k (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Seed for randomization. Defaults to 1.

    Yields:
        Generator: (x_train, y_train, x_test, y_test)
    """
    num_row = y.shape[0]
    fold_size = int(num_row / k)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = list(set(range(num_row)) - set(test_indices))
        yield tx[train_indices], y[train_indices], tx[test_indices], y[test_indices]


def standardize(x: np.ndarray) -> np.ndarray:
    """Standardize data

    Args:
        x (np.ndarray): Data

    Returns:
        np.ndarray: Standardized data
    """
    return (x-np.mean(x, axis=0)) / np.std(x, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid: (exp(x) / 1+ exp(x)) of the given array element-wise

    Args:
        x (np.ndarray): Array containing floats

    Returns:
        np.ndarray: Array after applying the sigmoid function element wise
    """
    # clipping to avoid overflow
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))


def batch_iter(tx: np.ndarray, y: np.ndarray, batch_size: int = None, num_batches: int = None, shuffle: bool = True) -> Generator:
    """Iterate through data in batches.

    Args:
        tx (np.ndarray): Features data
        y (np.ndarray): Labels data
        batch_size (int, optional): Batch size. Defaults to None (i.e. full batch)
        num_batches (int, optional): Number of batches to iterate through. Defaults to None (i.e. use all data)
        shuffle (bool, optional): Whether to shuffle the data before generating batches. Defaults to True.

    Yields:
        Generator: (tx_batch, y_batch)
    """
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


def split_data(tx: np.ndarray, y: np.ndarray, ratio: float = 0.8, seed: float = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and test sets specified by the ratio.

    Args:
        tx (np.ndarray): Features data
        y (np.ndarray): Labels data
        ratio (float, optional): Split ratio. Defaults to 0.8.
        seed (float, optional): Random seed. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (x_train, y_train, x_test, y_test)
    """
    np.random.seed(seed)
    n = len(y)
    indexes = np.linspace(0, n-1, n, dtype=int)
    np.random.shuffle(indexes)
    split_i = int(n * ratio)
    return np.take(tx, indexes[:split_i]), np.take(y, indexes[:split_i]), np.take(tx, indexes[split_i:]), np.take(y, indexes[split_i:])


def replace_values(x: np.ndarray, from_val: float, to_val: float) -> np.ndarray:
    """Replace instances of the source value with the target value

    Args:
        y (np.ndarray): Array of data
        from_val (float): Source value.
        to_val (float): Target value.

    Returns:
        np.ndarray: Array with replaced values
    """
    x = x.copy()
    x[x == from_val] = to_val
    return x


def add_bias(x: np.ndarray) -> np.ndarray:
    """Add bias column (column of 1's) to the data

    Args:
        x (np.ndarray): Data

    Returns:
        np.ndarray: Data with a bias column
    """
    return np.c_[np.ones(x.shape[0]), x]


def compute_nan_ratio(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute ratio of NaN values along the axis

    Args:
        x (np.ndarray): Data
        axis (int, optional): Axis to compute along. Defaults to 0.

    Returns:
        np.ndarray: Array of NaN ratios
    """
    return np.count_nonzero(np.isnan(x), axis=axis) / len(x)


def transform_X(X: np.ndarray, nan_cols: List[int], imputable_cols: List[int], encodable_cols: List[int]) -> Tuple[np.ndarray, List[int]]:
    """Transform features data

    Args:
        X (np.ndarray): Features data
        nan_cols (List[int]): List of columns that have NaN values
        imputable_cols (List[int]): List of columns that can be imputed
        encodable_cols (List[int]): List of columns that can be encoded

    Returns:
        Tuple[np.ndarray, List[int]]: Transformed data and the list of continuous features
    """
    # Compute number of nan values per row
    # nan_counts = np.sum(np.where(np.isnan(X), 1, 0), axis=1).reshape((-1, 1))

    # Drop all columns with nan values
    tX = np.delete(X, nan_cols, axis=1)

    # Impute some columns with nan values
    medians = np.nanmedian(X[:, imputable_cols], axis=0)
    imputed_X = X[:, imputable_cols]
    imputed_X = np.where(np.isnan(imputed_X), np.repeat(medians.reshape((1, -1)), imputed_X.shape[0], axis=0), imputed_X)

    # Encode some columns with nan values
    encoded_X = X[:, encodable_cols]
    encoded_X = np.where(np.isnan(encoded_X), 0, 1)

    tX = np.hstack([tX, imputed_X])
    tX = standardize(tX)
    tX = add_bias(tX)

    # Get continous columns (ignore bias column)
    cont_features = list(range(1, tX.shape[1]))

    tX = np.hstack([tX, encoded_X])

    return tX, cont_features

def transform_y(y: np.ndarray, switch_encoding: bool = False) -> np.ndarray:
    """Transform labels data

    Args:
        y (np.ndarray): Labels data
        switch_encoding (bool, optional): Whether to switch target encoding to (0, 1). Defaults to False.

    Returns:
        np.ndarray: Transformed labels data
    """
    if y is not None:
        if switch_encoding:
            y = replace_values(y, from_val=-1, to_val=0)
        return y.reshape((-1, 1))


def preprocess(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray = None,
               imputable_th: float = 0.3, encodable_th: float = 0.7,
               switch_encoding: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Preprocess training and test sets to prepare for training and prediction

    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test data
        y_test (np.ndarray, optional): Test labels. Defaults to None.
        imputable_th (float, optional): Imputable threshold for NaN values. 
                                        Columns that have ratio of nan values less than this will be imputed. Defaults to 0.5.
        encodable_th (float, optional): Encodable minimum threshold.
                                            Columns that have ratio of nan values less than this and greater than imputable_th will be encoded.
                                            Defaults to 0.7.
        switch_encoding (bool, optional): Whether to switch target encoding to (0, 1). Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]: (x_train, y_train, x_test, y_test, cont_features)
    """
    # Replace -999 values with NaN
    X_train = replace_values(X_train, from_val=-999, to_val=np.nan)

    # Compute NaN ratio for each column and derive imputable and encodable columns
    col_nan_ratio = compute_nan_ratio(X_train)
    nan_cols = (col_nan_ratio > 0)
    imputable_cols = (col_nan_ratio < imputable_th) & (col_nan_ratio > 0)
    encodable_cols = (col_nan_ratio > imputable_th) & (col_nan_ratio < encodable_th)
    
    # Transform train data
    tX_train, cont_features = transform_X(X_train, nan_cols=nan_cols, imputable_cols=imputable_cols, encodable_cols=encodable_cols)

    # Transform test data
    X_test = replace_values(X_test, from_val=-999, to_val=np.nan)
    tX_test, _ = transform_X(X_test, nan_cols=nan_cols, imputable_cols=imputable_cols, encodable_cols=encodable_cols)

    # Transform labels
    ty_train = transform_y(y_train, switch_encoding=switch_encoding)
    ty_test = transform_y(y_test, switch_encoding=switch_encoding)

    return tX_train, ty_train, tX_test, ty_test, cont_features


def read_feature_names(data_path: str) -> List[str]:
    """Read feature names from the data

    Args:
        data_path (str): Path to the data

    Returns:
        List[str]: List of feature names
    """
    with open(data_path) as f:
        header = f.readline()
    # Skip ID and Prediction columns
    return header.strip().split(',')[2:]
