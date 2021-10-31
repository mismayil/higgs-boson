from functools import partial
from typing import Tuple, Callable, Dict, Any
from collections import namedtuple
import inspect
import numpy as np
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
    return (1/ (2 * N)) * np.sum(e ** 2)


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
    return (1 / len(y)) * (tx.T @ ((tx @ w) - y))


def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray = None,
                     max_iters: int = 100, gamma: float = 0.1, batch_size: int = None,
                     num_batches: int = None, verbose: bool = False, *args, **kwargs) -> Tuple[np.ndarray, float]:
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
        verbose (bool, optional): Whether to print accuracy and loss at each iteration. Defaults to False.

    Returns:
        (np.ndarray, float): (weight parameters, mean squared error)
    """
    w = initial_w if initial_w is not None else np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=num_batches):
            gradient = least_squares_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient

        if verbose and i % 10 == 0:
            y_pred = predict_linear(w, tx)
            accuracy = compute_accuracy(y, y_pred)
            loss = compute_log_loss(y, tx, w)
            print(f'Iteration = {i}, accuracy = {accuracy}, loss = {loss}')

    loss = compute_mse(y, tx, w)
    
    return w, loss


def least_squares_SGD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray = None,
                      max_iters: int = 100, gamma: float = 0.1, num_batches: int = None, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """ 
    Computes the weight parameters of the least squares linear regression using stochastic gradient descent
    with batch size of 1 and returns the mean squared error of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        initial_w (np.ndarray, optional): Initial weight paramter to start the stochastic gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.
        num_batches (int, optional): Number of batches to sample. Defaults to None (i.e. uses all data)

    Returns:
        (np.ndarray, float): (weight parameters, mean squared error)
    """
    return least_squares_GD(y, tx, initial_w=initial_w, max_iters=max_iters, gamma=gamma, batch_size=1, num_batches=num_batches)


def least_squares(y: np.ndarray, tx: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """
    Computes the weight parameters of the least squares linear regression
    using the normal equations and returns it with the mean squared error of the model.

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
                            batch_size: int = None, num_batches: int = None, verbose: bool = False, *args, **kwargs) -> Tuple[np.ndarray, float]:
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
        verbose (bool, optional): Whether to print accuracy and loss at each iteration. Defaults to False.

    Returns:
        (np.ndarray, float): (weight parameters , negative log-likelihood)
    """
    w = initial_w if initial_w is not None else np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=num_batches):
            gradient = logistic_gradient(y_batch, tx_batch, w, lambda_=lambda_)
            w = w - gamma * gradient
        
        if verbose and i % 10 == 0:
            y_pred = predict_logistic(w, tx)
            accuracy = compute_accuracy(y, y_pred)
            loss = compute_log_loss(y, tx, w)
            print(f'Iteration = {i}, accuracy = {accuracy}, loss = {loss}')

    loss = compute_log_loss(y, tx, w)   
    
    return w, loss


def cross_validate(y: np.ndarray, tx: np.ndarray, model_fn: Callable, loss_fn: Callable, predict_fn: Callable,
                   k_fold: int = 5, seed: float = 1) -> Tuple[np.ndarray, float, float, float]:
    """
    Run K-fold cross validation on the data and report the averaged loss, accuracy and F1 scores
    over K models.

    Args:
        y (np.ndarray): Labels data
        tx (np.ndarray): Features data
        model_fn (Callable): Modelling function
        loss_fn (Callable): Loss function
        predict_fn (Callable): Prediction function
        k_fold (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Random seed. Defaults to 1.

    Returns:
        Tuple[float, float, float]: Averaged loss, accuracy and F1 scores
    """
    accuracies = []
    losses = []
    f1_scores = []

    for tx_train, y_train, tx_test, y_test in kfold_cv_iter(y, tx, k=k_fold, seed=seed):
        w, _ = model_fn(y_train, tx_train)

        loss = loss_fn(y_test, tx_test, w)
        y_pred = predict_fn(w, tx_test)
        accuracy = compute_accuracy(y_test, y_pred)
        f1_score = compute_f1(y_test, y_pred)

        losses.append(loss)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)

    return np.mean(losses), np.mean(accuracies), np.mean(f1_scores)


def grid_search_cv(y: np.ndarray, tx: np.ndarray, model_fn: Callable, loss_fn: Callable, predict_fn: Callable,
                   param_grid: Dict[str, Any], transform_fn: Callable = None, scoring: str = 'loss',
                   k_fold: int = 5, seed: float = 1) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run a Grid Search with Cross validation on the data to tune the hyperparameters for the given model.

    Args:
        y (np.ndarray): Labels data
        tx (np.ndarray): Features data
        model_fn (Callable): Modelling function. Used to model the data
        loss_fn (Callable): Loss function. Used to compute the loss
        predict_fn (Callable): Prediction function. Used to predict on the data
        param_grid (Dict[str, Any]): Parameter space. Dictionary of <param, values> pairs.
        transform_fn (Callable, optional): Transformation function. Used to transform data before CV. Defaults to None.
        scoring (str, optional): Scoring criteria to choose the best parameter. Accepted values: ['loss', 'accuracy', 'f1'].
                                 Defaults to 'loss'.
        k_fold (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Random seed. Defaults to 1.

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Dict of metrics and dict of best parameters
    """
    best_scoring_value = None
    best_params = None
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
            # Convert other sequences to tuple to make the parameter accesible to be used as a dictionary key
            parameters.append(Parameter(name=param, value=value if np.isscalar(value) else tuple(value)))

        parameter_space.append(parameters)
    
    transformations = {}
    transform_params = list(inspect.signature(transform_fn).parameters.keys()) if transform_fn else None

    for params in product(*parameter_space):
        params_dict = {param.name: param.value for param in params}
        model_fn_partial = partial(model_fn, **params_dict)
        transformed_tx = tx

        if transform_fn:
            # Check if the transformation already exists with these parameters and avoid extra computation
            common_params = tuple([param for param in params if param.name in transform_params])
            transformed_tx = transformations.get(common_params)
            if transformed_tx is None:
                # Store transformations for later use
                transformed_tx = transform_fn(tx, **params_dict)
                transformations[common_params] = transformed_tx

        loss, accuracy, f1_score = cross_validate(y, transformed_tx, model_fn=model_fn_partial, loss_fn=loss_fn, predict_fn=predict_fn,
                                                  k_fold=k_fold, seed=seed)
        scoring_value = loss
        
        if scoring == 'accuracy':
            scoring_value = accuracy
        elif scoring == 'f1':
            scoring_value = f1_score
        
        if best_scoring_value is None or scoring_value < best_scoring_value:
            best_scoring_value = scoring_value
            best_params = params
            best_metrics['loss'] = loss
            best_metrics['accuracy'] = accuracy
            best_metrics['f1_score'] = f1_score

    
    return best_metrics, {param.name: param.value for param in best_params}


def logistic_regression_cv(y: np.ndarray, tx: np.ndarray, param_grid: Dict[str, Any],
                           k_fold: int = 5, seed: float = 1, transform: bool = True) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run logistic regression with grid search over cross validation

    Args:
        y (np.ndarray): Labels data
        tx (np.ndarray): Features data
        param_grid (Dict[str, Any]): Parameter space
        k_fold (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Random seed. Defaults to 1.
        transform (bool, optional): Whether to apply transformation. Defaults to True.

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Dict of metrics and best parameters
    """
    model_fn = reg_logistic_regression
    loss_fn = compute_log_loss
    predict_fn = predict_logistic
    transform_fn = build_poly if transform else None
    return grid_search_cv(y, tx, model_fn=model_fn, loss_fn=loss_fn, predict_fn=predict_fn,
                          param_grid=param_grid, transform_fn=transform_fn, k_fold=k_fold, seed=seed)


def ridge_regression_cv(y: np.ndarray, tx: np.ndarray, param_grid: Dict[str, Any],
                        k_fold: int = 5, seed: float = 1, transform: bool = True) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run ridge regression with grid search over cross validation

    Args:
        y (np.ndarray): Labels data
        tx (np.ndarray): Features data
        param_grid (Dict[str, Any]): Parameter space
        k_fold (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Random seed. Defaults to 1.
        transform (bool, optional): Whether to apply transformation. Defaults to True.

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Dict of metrics and best parameters
    """
    model_fn = ridge_regression
    loss_fn = compute_mse
    predict_fn = predict_linear
    transform_fn = build_poly if transform else None
    return grid_search_cv(y, tx, model_fn=model_fn, loss_fn=loss_fn, predict_fn=predict_fn,
                          param_grid=param_grid, transform_fn=transform_fn, k_fold=k_fold, seed=seed)


def least_squares_cv(y: np.ndarray, tx: np.ndarray, param_grid: Dict[str, Any],
                     k_fold: int = 5, seed: float = 1) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run least squares (normal equations) with grid search over cross validation

    Args:
        y (np.ndarray): Labels data
        tx (np.ndarray): Features data
        param_grid (Dict[str, Any]): Parameter space
        k_fold (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Random seed. Defaults to 1.

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Dict of metrics and best parameters
    """
    model_fn = least_squares
    loss_fn = compute_mse
    predict_fn = predict_linear
    return grid_search_cv(y, tx, model_fn=model_fn, loss_fn=loss_fn, predict_fn=predict_fn,
                          param_grid=param_grid, k_fold=k_fold, seed=seed)


def least_squares_GD_cv(y: np.ndarray, tx: np.ndarray, param_grid: Dict[str, Any],
                        k_fold: int = 5, seed: float = 1, transform: bool = True) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run least squares GD with grid search over cross validation

    Args:
        y (np.ndarray): Labels data
        tx (np.ndarray): Features data
        param_grid (Dict[str, Any]): Parameter space
        k_fold (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Random seed. Defaults to 1.
        transform (bool, optional): Whether to apply transformation. Defaults to True.

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Dict of metrics and best parameters
    """
    model_fn = least_squares_GD
    loss_fn = compute_mse
    predict_fn = predict_linear
    transform_fn = build_poly if transform else None
    return grid_search_cv(y, tx, model_fn=model_fn, loss_fn=loss_fn, predict_fn=predict_fn,
                          param_grid=param_grid, transform_fn=transform_fn, k_fold=k_fold, seed=seed)


def least_squares_SGD_cv(y: np.ndarray, tx: np.ndarray, param_grid: Dict[str, Any],
                         k_fold: int = 5, seed: float = 1, transform: bool = True) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run least squares SGD with grid search over cross validation

    Args:
        y (np.ndarray): Labels data
        tx (np.ndarray): Features data
        param_grid (Dict[str, Any]): Parameter space
        k_fold (int, optional): Number of folds. Defaults to 5.
        seed (float, optional): Random seed. Defaults to 1.
        transform (bool, optional): Whether to apply transformation. Defaults to True.

    Returns:
        Tuple[Dict[str, float], Dict[str, Any]]: Dict of metrics and best parameters
    """
    model_fn = least_squares_SGD
    loss_fn = compute_mse
    predict_fn = predict_linear
    transform_fn = build_poly if transform else None
    return grid_search_cv(y, tx, model_fn=model_fn, loss_fn=loss_fn, predict_fn=predict_fn,
                          param_grid=param_grid, transform_fn=transform_fn, k_fold=k_fold, seed=seed)