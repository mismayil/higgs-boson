from functools import partial
from typing import Tuple
from collections import namedtuple
import numpy as np
from proj1_helpers import predict_labels
from helpers import *
from itertools import product


Parameter = namedtuple('Parameter', ['name', 'value'])


def compute_error(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Given a data matrix tx, weight parameters w and dependent variable vector y, 
    computes the errors (residuals) of the linear regression model by taking the difference 
    between the true values (y) and the predicted values (tX@w)

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model

    Returns:
        np.ndarray: error vector
    """

    N = len(y)
    y = y.reshape((-1, 1))
    X = tx.reshape((N, -1))
    w = w.reshape((-1, 1))

    e = y - np.dot(X, w)
    return e


def compute_mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
     Given a data matrix tx, weight parameters w and dependent variable vector y, 
     computes the mean squared error of the linear regression model

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model

    Returns:
        float : The sum of the squared errors
    """

    N = len(y)
    e = compute_error(y, tx, w)
    return (1/(2*N)) * np.sum(e**2)


def least_squares_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Given a data matrix tx, weight parameters w and dependent variable vector y, 
    computes the gradient of the least squares linear regression model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model

    Returns:
        np.ndarray : The gradient with respect to the weight vector of the mean squared error.
    """

    return tx.T @ ((tx @ w) - y)


def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray = None,
                     max_iters: int = 100, gamma: float = 0.1, batch_size: int = None,
                     num_batches: int = None, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """ 
    Computes the weight parameters of the least squares linear regression using (mini-) batch gradient descent and
    returns the mean squared error of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        initial_w (np.ndarray, optional): Initial weight paramter to start the gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to None (i.e full batch gradient descent)
        num_batches (int, optional): Number of batches to sample. Defaults to None (i.e. uses all data)

    Returns:
        (np.ndarray, float): (weight parameters, mean squared error)
    """

    
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=num_batches):
            gradient = least_squares_gradient(y_batch, tx_batch, w)
            w -= gamma * gradient

    loss = compute_mse(y, tx, w) 
    
    return w, loss


def least_squares_SGD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray = None,
                      max_iters: int = 100, gamma: float = 0.1, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """ 
    Computes the weight parameters of the least squares linear regression using stochastic gradient descent
    with batch size of 1 and returns the mean squared error of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        initial_w (np.ndarray, optional): Initial weight paramter to start the stochastic gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.

    Returns:
        (np.ndarray, float): (weight parameters, mean squared error)
    """
    return least_squares_GD(y, tx, initial_w=initial_w, max_iters=max_iters, gamma=gamma, batch_size=1)


def least_squares(y: np.ndarray, tx: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """
    Computes the weight parameters of the least squares linear regression
    using the normal equations and returns the mean squared error of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)

    Returns:
        (np.ndarray, float): (weight parameters, mean squared error)
    """
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """ 
    Computes the weight parameters of the L2 regularized linear regression,
    also called ridge regression, using the normal equations.
    It also returns the mean squared error of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        lambda_ (float): The L2 regularization hyper-parameter (higher values incur higher regularization)

    Returns:
        (np.ndarray, float): (weight parameters , mean squared error)
    """
    N, D = tx.shape
    w = np.linalg.solve(np.dot(tx.T, tx) + 2 * lambda_ * N * np.eye(D), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    return w, loss


def compute_log_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """ 
    Given a data matrix tx, weight parameters w and dependent variable vector y, 
    compute the negative log-likelihood of the logistic regression model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model

    Returns:
        float: log likelihood of the logistic regression model
    """
    probs = sigmoid(tx @ w)
    return -(1 / len(y)) * np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

 
def logistic_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> np.ndarray:
    """
    Given a data matrix tx, weight parameters w and dependent variable vector y (and regularization parameter lambda_), 
    computes the gradient of the log likelihood of the (regularized) logistic regression model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model
        lambda_ (float, optional): The L2 regularization hyper-parameter. Defaults to 0, meaning no regularization.

    Returns:
        np.ndarray : The gradient with respect to the weight vector of the log likelihood of the model
    """
    return (1 / len(y)) * (tx.T @ (sigmoid(tx @ w) - y)) + 2 * lambda_ * w


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray = None,
                        max_iters: int = 100, gamma: float = 0.1,
                        batch_size: int = None, num_batches: int = None, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """ 
    Computes the weight parameters of the logistic regression using gradient descent with a custom batch size and
    returns it with the negative log-likelihood of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        initial_w (np.ndarray, optional): Initial weight paramter to start the stochastic gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to None (i.e full batch gradient descent)
        num_batches (int, optional): Number of batches to sample. Defaults to None (i.e. uses all data)

    Returns:
        (np.ndarray, float): (weight parameters , negative log-likelihood)
    """
    return reg_logistic_regression(y, tx, lambda_=0, initial_w=initial_w, max_iters=max_iters, gamma=gamma,
                                   batch_size=batch_size, num_batches=num_batches)


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, initial_w: np.ndarray = None,
                            max_iters: int = 100, gamma: float = 0.1,
                            batch_size: int = None, num_batches: int = None, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """ 
    Computes the weight parameters of the L2 regularized logistic regression using gradient descent with custom batch size
    and returns it with the negative log-likelihood of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        lambda_ (float): The L2 regularization hyper-parameter (higher values incur higher regularization)
        initial_w (np.ndarray, optional): Initial weight paramter to start the stochastic gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to None (i.e full batch gradient descent)
        num_batches (int, optional): Number of batches to sample. Defaults to None (i.e. uses all data)

    Returns:
        (np.ndarray, float): (weight parameters , negative log-likelihood)
    """
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=num_batches):
            gradient = logistic_gradient(y_batch, tx_batch, w, lambda_=lambda_)
            w = w - gamma * gradient
 
    loss = compute_log_loss(y, tx, w)   
    
    return w, loss


def cross_validate(y, tx, model_fn, loss_fn, predict_fn, k_fold = 5, seed = 1):
    accuracies = []
    losses = []
    f1_scores = []

    for tx_train, y_train, tx_test, y_test in kfold_cv_iter(y, tx, k=k_fold, seed=seed):
        w, _ = model_fn(y_train, tx_train)

        loss = loss_fn(y_test, tx_test, w)
        y_pred = predict_fn(w, tx_test)
        accuracy = compute_accuracy(y_test, y_pred)
        f1_score = compute_f1(y_test, y_pred)

        print(f'Accuracy={accuracy}')

        losses.append(loss)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)

    return w, np.mean(losses), np.mean(accuracies), np.mean(f1_scores)


def grid_search_cv(y, tx, model_fn, loss_fn, predict_fn, param_grid, transform_fn = None, scoring='loss', k_fold=5, seed=1):
    best_scoring_value = None
    best_params = None
    best_weights = None
    best_metrics = {
        'loss': None,
        'accuracy': None,
        'f1_score': None
    }
    parameter_space = []

    for param, values in param_grid.items():
        parameters = []

        if not isinstance(values, list) and not isinstance(values, np.ndarray):
            values = [values]

        for value in values:
            parameters.append(Parameter(name=param, value=value))

        parameter_space.append(parameters)
    
    for params in product(*parameter_space):
        params_dict = {param.name: param.value for param in params}
        model_fn_partial = partial(model_fn, **params_dict)
        transformed_tx = transform_fn(tx, **params_dict) if transform_fn else tx
        w, loss, accuracy, f1_score = cross_validate(y, transformed_tx, model_fn=model_fn_partial, loss_fn=loss_fn, predict_fn=predict_fn,
                                                     k_fold=k_fold, seed=seed)
        scoring_value = loss
        
        if scoring == 'accuracy':
            scoring_value = accuracy
        elif scoring == 'f1':
            scoring_value = f1_score
        
        if best_scoring_value is None or scoring_value < best_scoring_value:
            best_scoring_value = scoring_value
            best_params = params
            best_weights = w
            best_metrics['loss'] = loss
            best_metrics['accuracy'] = accuracy
            best_metrics['f1_score'] = f1_score

    
    return best_weights, best_metrics, {param.name: param.value for param in best_params}


# def poly_reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, initial_w: np.ndarray = None,
#                                  max_iters: int = 100, gamma: float = 0.1,
#                                  batch_size: int = None, num_batches: int = None,
#                                  degree: int = 1, cont_features: np.ndarray = None) -> Tuple[np.ndarray, float]:
#     tx_poly = build_poly(tx, degree, cont_features)
#     return reg_logistic_regression(y, tx_poly, lambda_=lambda_, initial_w=initial_w, max_iters=max_iters, gamma=gamma,
#                                    batch_size=batch_size, num_batches=num_batches)


def logistic_regression_cv(y, tx, param_grid, k_fold=5, seed=1):
    model_fn = reg_logistic_regression
    loss_fn = compute_log_loss
    predict_fn = predict_logistic
    transform_fn = lambda tx, degree, cont_features, *args, **kwargs: build_poly(tx, degree, cont_features)
    return grid_search_cv(y, tx, model_fn=model_fn, loss_fn=loss_fn, predict_fn=predict_fn, param_grid=param_grid, transform_fn=transform_fn)




