from datetime import datetime
import numpy as np

from helpers import *
from implementations import *

DATA_TRAIN_PATH = 'data/train.csv' 
DATA_TEST_PATH  = 'data/test.csv'

def run():
    # Load training and test data
    print('Loading training and test data...')
    y_train, X_train, _ = load_csv_data(DATA_TRAIN_PATH)
    _, X_test, ids_test = load_csv_data(DATA_TEST_PATH)

    # Preprocess data and do feature engineering
    print('Preprocessing data...')
    tX_train, ty_train, tX_test, ty_test, cont_features = preprocess(X_train, y_train, X_test,
                                                                     imputable_th=0.3, encodable_th=0.7,
                                                                     switch_encoding=True)
    tX_train_poly = build_poly(tX_train, degree=2, cont_features=cont_features)
    tX_test_poly = build_poly(tX_test, degree=2, cont_features=cont_features)

    # Run logistic regression model on the data
    print('Running logistic regression model...')
    weights, loss = reg_logistic_regression(ty_train, tX_train_poly, max_iters=1000, lambda_=0.01)

    # Report loss and the training metrics
    ty_train_pred = predict_logistic(weights, tX_train_poly)
    train_accuracy = compute_accuracy(ty_train, ty_train_pred)
    train_f1 = compute_f1(ty_train, ty_train_pred)
    print(f'\nFinal loss = {loss}')
    print(f'Training accuracy = {train_accuracy}')
    print(f'Training F1 score = {train_f1}\n')

    # Prepare test submission file
    print('Preparing test submission file...')
    method = 'reg_logistic_regression'
    time = datetime.now().strftime('%Y%m%dH%H%M%S')
    OUTPUT_PATH = f'submission_{method}_{time}'
    y_pred = predict_logistic(weights, tX_test_poly)
    y_pred = replace_values(y_pred, from_val=0, to_val=-1)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    
    print(f'Submission file {OUTPUT_PATH} successfully created.')


if __name__ == '__main__':
    run()