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


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score between ground truth and predictions

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        float: F1 score
    """
    y_true = y_true.reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    labels = np.unique(y_true)
    pos_label = np.max(labels)
    neg_label = np.min(labels)
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
    return np.take(tx, indexes[:split_i], axis=0), np.take(y, indexes[:split_i], axis=0), np.take(tx, indexes[split_i:], axis=0), np.take(y, indexes[split_i:], axis=0)


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


def build_indicator_features(x: np.ndarray) -> np.ndarray:
    """Convert categorical feature into multiple indicator (dummy) features

    Args:
        x (np.ndarray): Categorical feature array

    Returns:
        np.ndarray: Array of indicator features
    """
    ind_features = []
    for v in np.unique(x):
        values = np.where(x == v, 1, 0)
        ind_features.append(values.reshape((-1, 1)))
    return np.hstack(ind_features)


def build_binned_features(X: np.ndarray, cols: List[int], num_bins: int = 3) -> np.ndarray:
    """Convert numeric features into binned categorical features

    Args:
        X (np.ndarray): Data with numerical features
        cols (List[int]): Numerical columns
        num_bins (int, optional): Number of bins. Defaults to 3.

    Returns:
        np.ndarray: Binned (indicator) feature data
    """
    binned_features = []
    for c in cols:
        bins = np.linspace(np.min(X[:, c]), np.max(X[:, c]), num_bins)
        digitized_col = np.digitize(X[:, c], bins)
        ind_features = build_indicator_features(digitized_col)
        binned_features.append(ind_features)
    return np.hstack(binned_features)


def build_nan_feature(X: np.ndarray) -> np.ndarray:
    """Build a feature for row count of nans

    Args:
        X (np.ndarray): Data

    Returns:
        np.ndarray: Array of nan counts for each row of X
    """
    return np.sum(np.where(np.isnan(X), 1, 0), axis=1).reshape((-1, 1))


def apply_log(X: np.ndarray) -> np.ndarray:
    """Apply log to all positive columns of X

    Args:
        X (np.ndarray): Feature data

    Returns:
        np.ndarray: Feature data with positive columns logged
    """
    tX = X.copy()
    pos_cols = np.all(tX > 0, axis=0)
    tX[:, pos_cols] = np.log(tX[:, pos_cols])
    return tX


def split_by_jet_num(data_path: str, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into 3 sets based on the value of PRI_jet_num feature

    Args:
        data_path (str): Data path to read the feature names
        X (np.ndarray): Feature data
        y (np.ndarray, optional): Labels data. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays for features and labels for each set respectively
    """
    features = read_feature_names(data_path)
    zero_jet_features = [f for f in features if f not in ['PRI_jet_num', 'PRI_tau_pt', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
                                                          'DER_prodeta_jet_jet', 'DER_pt_tot', 'DER_sum_pt', 'DER_lep_eta_centrality',
                                                          'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi',
                                                          'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi',
                                                          'PRI_jet_all_pt']]
    one_jet_features = [f for f in features if f not in ['PRI_jet_num', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet',
                                                         'DER_lep_eta_centrality', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi']]
    many_jet_features = [f for f in features if f not in ['PRI_jet_num']]

    zero_jet_mask = X[:, features.index('PRI_jet_num')] == 0
    one_jet_mask = X[:, features.index('PRI_jet_num')] == 1
    many_jet_mask = X[:, features.index('PRI_jet_num')] > 1

    X_zero, y_zero = X[zero_jet_mask][:, [features.index(f) for f in zero_jet_features]], y[zero_jet_mask] if y is not None else None
    X_one, y_one = X[one_jet_mask][:, [features.index(f) for f in one_jet_features]], y[one_jet_mask] if y is not None else None
    X_many, y_many = X[many_jet_mask][:, [features.index(f) for f in many_jet_features]], y[many_jet_mask] if y is not None else None

    return X_zero, y_zero, X_one, y_one, X_many, y_many


def remove_outliers(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers assuming a standardized input data.
    Removes rows with at least one value outside [-4,4] interval.

    Args:
        X (np.ndarray): Input features data
        y (np.ndarray): Input labels data

    Returns:
        Tuple[np.ndarray, np.ndarray]: Outlier-free data
    """
    non_outlier_mask = np.all((X < 4) & (X > -4), axis=1)
    return X[non_outlier_mask], y[non_outlier_mask]


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
    # Drop all columns with nan values
    tX = np.delete(X, nan_cols, axis=1)

    # Impute some columns with nan values
    medians = np.nanmedian(X[:, imputable_cols], axis=0)
    imputed_X = X[:, imputable_cols]
    imputed_X = np.where(np.isnan(imputed_X), np.repeat(medians.reshape((1, -1)), imputed_X.shape[0], axis=0), imputed_X)

    # Encode some columns with nan values
    encoded_X = X[:, encodable_cols]
    encoded_X = np.where(np.isnan(encoded_X), 1, 0)

    tX = np.hstack([tX, imputed_X])
    tX = apply_log(tX)
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
    """
    Preprocess training and test sets to prepare for training and prediction.
    This method performs several preprocessing steps:
        - Impute some columns based on the imputable threshold
        - Encode some columns based on the encodable threshold
        - Apply log transformation on positive columns
        - Standardize continuous features
        - Add a bias feature
        - Remove outliers

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

    tX_train, ty_train = remove_outliers(tX_train, ty_train)

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

def cross_polynomial_feature_expansion(tx: np.ndarray, degree: int = 1) -> np.ndarray:
    """Expand the data cross polynomially

    Args:
        tx (np.ndarray): Input data
        degree (int, optional): Polynomial degree. Defaults to 1.

    Returns:
        np.ndarray: Cross polynomially expanded data
    """
    tx_dlc = np.empty((tx.shape[0], 0))
    if degree < 1:
        return tx

    for i in range(1, tx.shape[1]):
        if ((i / tx.shape[1]) * 10) % 2 == 0:
            print("Cross Polynomial expansion : {}%".format((i / tx.shape[1]) * 100))
        for j in range(i, tx.shape[1]):
            for d in range(1, degree + 1):
                tx_dlc = np.c_[tx_dlc, (tx[:,i] * tx[:,j]) ** d]
        
    print("Cross Polynomial expansion of degree {} done : adding {}".format(degree,tx_dlc.shape))

    return tx_dlc


def simple_logarithmic_feature_expansion(tx: np.ndarray, degree: int = 1) -> np.ndarray:
    """Apply logarithmic feature expansion to the data

    Args:
        tx (np.ndarray): Input data
        degree (int, optional): Polynomial degree. Defaults to 1.

    Returns:
        np.ndarray: Logarithmically feature expanded data
    """
    tx_pos = tx - tx.min(axis=0)
    tx_dlc = np.empty((tx.shape[0], 0))
    if degree < 1:
        return tx_dlc

    for i in range(tx.shape[1]):
        for d in range(1,degree + 1):
            tx_dlc = np.c_[tx_dlc, np.power(np.log(1 + tx_pos[:,i]), d)]
    print("Logarithmic expansion of degree {} done : adding {}".format(degree, tx_dlc.shape))

    return tx_dlc