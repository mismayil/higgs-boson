from datetime import datetime
import numpy as np

from helpers import *
from implementations import *

DATA_TRAIN_PATH = 'data/train.csv' 
DATA_TEST_PATH  = 'data/test.csv'


def train_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                  max_iters: int = 3000, degree: int = 2, lambda_: float = 0.01,
                  gamma: float = 0.01, imputable_th: float = 0.5, encodable_th: float = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Train a Regularized Logistic Regression model on the data and predict on a test data

    Args:
        X_train (np.ndarray): Training features data
        y_train (np.ndarray): Training labels data
        X_test (np.ndarray): Test features data
        max_iters (int, optional): Max number of iterations. Defaults to 3000.
        degree (int, optional): Polynomial degree for feature expansion. Defaults to 2.
        lambda_ (float, optional): Regularization parameter. Defaults to 0.01.
        gamma (float, optional): Learning rate. Defaults to 0.01.
        imputable_th (float, optional): Threshold for imputing NaN values. Defaults to 0.5.
        encodable_th (float, optional): Threshold for encoding NaN values. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Predictions, weights and loss
    """
    print('Preprocessing data...')
    tX_train, ty_train, tX_test, _, cont_features = preprocess(X_train, y_train, X_test, imputable_th=imputable_th, encodable_th=encodable_th, switch_encoding=True)
    tX_train_poly = build_poly(tX_train, degree=degree, cont_features=cont_features)

    print('Running logistic regression model...')
    weights, loss = reg_logistic_regression(ty_train, tX_train_poly, max_iters=max_iters, lambda_=lambda_, gamma=gamma, verbose=False)

    print('Making predictions...')
    tX_test_poly = build_poly(tX_test, degree=degree, cont_features=cont_features)
    y_pred = predict_logistic(weights, tX_test_poly)
    y_pred = replace_values(y_pred, from_val=0, to_val=-1)

    # Report loss and the training metrics
    ty_train_pred = predict_logistic(weights, tX_train_poly)
    train_accuracy = compute_accuracy(ty_train, ty_train_pred)
    train_f1 = compute_f1(ty_train, ty_train_pred)
    print(f'\nFinal loss = {loss}')
    print(f'Training accuracy = {train_accuracy}')
    print(f'Training F1 score = {train_f1}\n')

    return y_pred, weights, loss


def run():
    """Run training and save predictions to csv file"""

    print('Loading train and test data...')
    y_train, X_train, _ = load_csv_data(DATA_TRAIN_PATH)
    _, X_test, ids_test = load_csv_data(DATA_TEST_PATH)

    
    print('Splitting the dataset into 3 datasets based on the PRI_jet_num values...')
    X_train_zero, y_train_zero, X_train_one, y_train_one, X_train_many, y_train_many = split_by_jet_num(DATA_TRAIN_PATH, X_train, y_train)
    X_test_zero, ids_test_zero, X_test_one, ids_test_one, X_test_many, ids_test_many = split_by_jet_num(DATA_TRAIN_PATH, X_test, ids_test)

    degree = 3
    lambda_ = 0.001
    gamma = 0.1
    max_iters = 3000
    imputable_th = 1
    encodable_th = 0

    print('Building a model for data with 0 jets...')
    y_pred_zero, weights_zero, loss_zero = train_predict(X_train_zero, y_train_zero, X_test_zero, max_iters=max_iters, gamma=gamma,
                                                         degree=degree, lambda_=lambda_, imputable_th=imputable_th, encodable_th=encodable_th)

    print('Building a model for data with 1 jet...')
    y_pred_one, weights_one, loss_one = train_predict(X_train_one, y_train_one, X_test_one, max_iters=max_iters, gamma=gamma,
                                                      degree=degree, lambda_=lambda_, imputable_th=imputable_th, encodable_th=encodable_th)

    print('Building a model for data with multiple jets...')
    y_pred_many, weights_many, loss_many = train_predict(X_train_many, y_train_many, X_test_many, max_iters=max_iters, gamma=gamma,
                                                         degree=degree, lambda_=lambda_, imputable_th=imputable_th, encodable_th=encodable_th)
                                                 
    # Prepare test submission file
    print('Preparing test submission file...')
    y_pred = np.vstack([y_pred_zero, y_pred_one, y_pred_many])
    ids_test = np.hstack([ids_test_zero, ids_test_one, ids_test_many])
    method = 'reg_logistic_regression'
    time = datetime.now().strftime('%Y%m%dH%H%M%S')
    OUTPUT_PATH = f'submission_{method}_{time}'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    
    print(f'Submission file {OUTPUT_PATH} successfully created.')


if __name__ == '__main__':
    run()