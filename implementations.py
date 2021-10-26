import numpy as np
from numpy.lib.function_base import gradient
from proj1_helpers import predict_labels
import matplotlib.pyplot as plt
from itertools import product
from helpers import *

def compute_error(y: np.ndarray, tx: np.ndarray, w: np.ndarray):
    """
    Given a data matrix tx, weight parameters w and dependent variable vector y, computes the errors (residuals) of the linear regression model
    by taking the difference between the true values (y) and the predicted values (tX@w)

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


def compute_mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray):
    """
     Given a data matrix tx, weight parameters w and dependent variable vector y, computes the mean squared error of the linear regression model

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


def least_squares_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray):
    """
     Given a data matrix tx, weight parameters w and dependent variable vector y, computes the gradient of the least squares linear regression model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model

    Returns:
        np.ndarray : The gradient with respect to the weight vector of the mean squared error.
    """

    return tx.T @ ((tx @ w) - y)




def least_squares_GD(y:np.ndarray, tx:np.ndarray, initial_w:np.ndarray = None, max_iters:int = 100, gamma: float = 0.1):
    """ 
    Computes the weight parameters of the least squares linear regression using gradient descent and returns the mean squared error of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        initial_w (np.ndarray, optional): Initial weight paramter to start the gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.

    Returns:
        (np.ndarray, float): (weight parameters , mean squared error)
    """

    
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        gradient = least_squares_gradient(y,tx,w)
        w -= gamma * gradient
    
        
    loss = compute_mse(y,tx,w)   
    
    return w, loss



def least_squares_SGD(y:np.ndarray, tx:np.ndarray, initial_w:np.ndarray = None, max_iters:int = 100, gamma: float = 0.1):
    """ 
    Computes the weight parameters of the least squares linear regression using stochastic gradient descent with batch size of 1 and returns the mean squared error of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        initial_w (np.ndarray, optional): Initial weight paramter to start the stochastic gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.

    Returns:
        (np.ndarray, float): (weight parameters , mean squared error)
    """
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            gradient = least_squares_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient
 
    loss = compute_mse(y,tx,w)   
    
    return w, loss




def least_squares(y: np.ndarray, tx: np.ndarray):
    """ Computes the weight parameters of the least squares linear regression using the normal equations and returns the mean squared error of the model

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)

    Returns:
        (np.ndarray, float): (weight parameters , mean squared error)
    """
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_:float):
    """ 
    Computes the weight parameters of the L2 regularized linear regression, also called ridge regression, using the normal equations.
    It also returns the mean squared error of the model

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


def sigmoid(x:np.ndarray):
    """"
    Computes the sigmoid: (exp(x) / 1+ exp(x)) of the given array element-wise

    Args:
        x (np.ndarray): array containing floats

    Returns:
        np.ndarray: array after applying the sigmoid function element wise
    """
    
    # clipping to avoid overflow
    x = np.clip(x, -20, 20)
    return np.exp(x) / (1 + np.exp(x))


def compute_log_loss( y: np.ndarray, tx: np.ndarray, w: np.ndarray):
    """ 
    Given a data matrix tx, weight parameters w and dependent variable vector y, compute the negative log-likelihood of the logistic regression model

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model

    Returns:
        float: log likelihood of the logistic regression model
    """
    probs = sigmoid(x @ w)
    return -(1 / len(y)) * np.sum(y * np.log(probs) + (1-y) * np.log(1-probs))


 
def logistic_gradient( y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_:float=0):
    """
    Given a data matrix tx, weight parameters w and dependent variable vector y (and regularization parameter lamba_), 
    computes the gradient of the log likelihood of the (regularized)  logistic regression model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        w (np.ndarray): The weight parameters of the linear model
        lambda_ (float, optional): The L2 regularization hyper-parameter. Defaults to 0, meaning no regularization.

    Returns:
        np.ndarray : The gradient with respect to the weight vector of the log likelihood of the model
    """

    
    return (1 / len(y)) * (x.T @ (sigmoid(x @ w) - y)) + 2 * lambda_ * w





def logistic_regression(y:np.ndarray, tx:np.ndarray, initial_w:np.ndarray = None, max_iters:int = 100, gamma: float = 0.1):
    """ 
    Computes the weight parameters of the logistic regression using stochastic gradient descent with batch size of 1 and returns the negative log-likelihood of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        initial_w (np.ndarray, optional): Initial weight paramter to start the stochastic gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.

    Returns:
        (np.ndarray, float): (weight parameters , negative log-likelhiood)
    """
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            gradient = logistic_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient
 
    loss = compute_log_loss(y,tx,w)   
    
    return w, loss


def reg_logistic_regression(y:np.ndarray, tx:np.ndarray, lambda_:float, initial_w:np.ndarray = None, max_iters:int = 100, gamma: float = 0.1):
    """ 
    Computes the weight parameters of the L2 regularized logistic regression using stochastic gradient descent with batch size of 1 and returns the negative log-likelihood of the model.

    Args:
        y (np.ndarray): The dependent variable y
        tx (np.ndarray): The data matrix (a row represents one observation of the features)
        lambda_ (float): The L2 regularization hyper-parameter (higher values incur higher regularization)
        initial_w (np.ndarray, optional): Initial weight paramter to start the stochastic gradient descent. If None, initialized randomly
        max_iters (int, optional): Number of iterations. Defaults to 100.
        gamma (float, optional): Fixed step-size for the gradient descent. Defaults to 0.1.

    Returns:
        (np.ndarray, float): (weight parameters , negative log-likelhiood)
    """
    w = initial_w or np.random.rand(tx.shape[1], 1)
    y = y.reshape((-1, 1))

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            gradient = logistic_gradient(y_batch, tx_batch, w,lambda_=lambda_)
            w = w - gamma * gradient
 
    loss = compute_log_loss(y,tx,w)   
    
    return w, loss







