import numpy as np
import copy

from planar_utils import sigmoid


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- dictionary containing parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost given equation
    """
    m = Y.shape[1]
    logprob = (np.multiply(Y, np.log(A2)) +
               np.multiply((1 - Y), np.log(1 - A2)))
    cost = np.sum(logprob) / - m
    cost = float(np.squeeze(cost))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing gradients with respect to different parameters
    """

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)
    dW2 = dW2 / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    g1_Z1_ = (1 - np.power(A1, 2))
    _dZ1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(_dZ1, g1_Z1_)

    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- dictionary containing parameters
        grads -- dictionary containing gradients

        Returns:
        parameters -- dictionary containing updated parameters
    """
    W1 = copy.deepcopy(parameters["W1"])
    W2 = copy.deepcopy(parameters["W2"])
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 = W1 - (learning_rate * dW1)
    W2 = W2 - (learning_rate * dW2)
    b1 = b1 - (learning_rate * db1)
    b2 = b2 - (learning_rate * db2)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def nn_model(X, Y, n_h, iterations=1000, print_cost=False):
    """
       Arguments:
       X -- dataset of shape (2, number of examples)
       Y -- labels of shape (1, number of examples)
       n_h -- size of the hidden layer
       iterations -- Number of iterations in gradient descent loop
       print_cost -- if True, print the cost every 1000 iterations

       Returns:
       parameters -- parameters learnt by the model.
   """
    n_x, _, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h=n_h, n_y=n_y)

    # gradient descent loop

    for i in range(0, iterations):
        A2, cache = forward_propagation(X, parameters=parameters)

        cost = compute_cost(A2, Y)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions  model (red: 0 / blue: 1)
    """

    A2, cache = forward_propagation(X, parameters=parameters)
    predictions = A2 > 0.5

    return predictions
