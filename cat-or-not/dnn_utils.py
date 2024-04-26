import copy
import numpy as np
import h5py
from activations import relu, relu_backward, sigmoid, sigmoid_backward


# for a two layer (L2) network
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return parameters


# for L layers
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in network

    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}

    L = len(layer_dims)

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i],
                                                   layer_dims[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    Yhat -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    # parameters contains W,b hence the half
    L = len(parameters) // 2

    # 0 is the input layer
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    Yhat, cache = linear_activation_forward(
        A, parameters['W' + str(L)], parameters["b" + str(L)], 'sigmoid')
    caches.append(cache)

    return Yhat, caches


def compute_cost(Yhat, Y):
    """
    Implements the cost function
    Arguments:
    Yhat -- probability vector corresponding to the label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    Returns:
    cost -- cross-entropy cost
   """
    m = Y.shape[1]
    cost = np.multiply(Y, np.log(Yhat))
    cost += np.multiply(1 - Y, np.log(1 - Yhat))
    cost = np.sum(cost) / - m
    cost = np.squeeze(cost)

    return cost


# dx -> dL/dx

def linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)
    dW /= m

    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) stored for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(Yhat, Y, caches):
    """
    Implements the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    Yhat -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """

    grads = {}
    L = len(caches)
    m = Y.shape[1]
    Y = Y.reshape(Yhat.shape)

    # derivative of cost with respect to AL
    dYhat = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))

    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[L]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
        dYhat, current_cache, "sigmoid")
    grads["dA"+str(L - 1)] = dA_prev_temp
    grads["db"+str(L)] = db_temp
    grads["dW"+str(L)] = dW_temp

    for l in reversed(range(L - 1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA"+str(l + 1)], current_cache, "relu"
        )

        grads["dA"+str(l - 1)] = dA_prev_temp
        grads["db"+str(l)] = db_temp
        grads["dW"+str(l)] = dW_temp

    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing parameters 
    grads -- python dictionary containing gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """

    L = len(params) // 2

    parameters = copy.deepcopy(params)

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - \
            (learning_rate * grads["dW"+str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - \
            (learning_rate * grads["db"+str(l + 1)])

    return parameters


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def predict(X, y, parameters):
    """
    Predicts the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    y -- numpy array containing true labels
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    # print("X = ", X.shape)
    # print("y = ", y.shape)

    print(parameters)

    m = X.shape[1]

    L = len(parameters) // 2

    p = np.zeros((1, m))

    probs, caches = L_model_forward(X, parameters)

    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    accuracy = np.sum((p == y) / m)

    print(f"Accuracy is {str(accuracy)}")

    return p
