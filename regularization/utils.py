from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import scipy.io


def initialize_parameters_zeros(layer_dims: list[int]):

    L = len(layer_dims)

    parameters = {}

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def initialize_parameters_random(layer_dims: list[int]):

    L = len(layer_dims)

    parameters = {}
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def initialize_parameters_he(layer_dims: list[int]):

    L = len(layer_dims)

    parameters = {}

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l - 1]) * np.sqrt(2/layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def sigmoid(x: np.ndarray) -> np.ndarray:

    s = 1 / (1 + np.exp(-x))
    return s


def relu(x: np.ndarray) -> np.ndarray:

    s = np.maximum(0, x)
    return s


def forward_prop(X: np.ndarray, parameter: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:

    W1 = parameter["W1"]
    b1 = parameter["b1"]
    W2 = parameter["W2"]
    b2 = parameter["b2"]
    W3 = parameter["W3"]
    b3 = parameter["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    return a3, cache


def backprop(X: np.ndarray, Y: np.ndarray, cache: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:

    m = X.shape[1]

    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W1.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    return {
        "dz3": dz3,
        "dW3": dW3,
        "db3": db3,
        "da2": da2,
        "dz2": dz2,
        "dW2": dW2,
        "db2": db2,
        "da1": da1,
        "dz1": dz1,
        "dW1": dW1,
        "db1": db1,
    }


def backprop_regularized(X, Y, cache, lambd) -> dict[str, np.ndarray]:
    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dW3 = 1/m * np.dot(dZ3, A2.T) + (lambd / m) * W3

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd / m) * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd / m) * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def update_params(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], learning_rate: float) -> dict[str, np.ndarray]:

    L = len(params) // 2

    for l in range(1, L):
        params["W" + str(l)] = params["W" + str(l)] - \
            (learning_rate * grads["dW" + str(l)])

        params["b" + str(l)] = params["b" + str(l)] - \
            (grads["db" + str(l)] * learning_rate)

    return params


def compute_loss(a3: np.ndarray, Y: np.ndarray):

    m = Y.shape[1]
    logprobs = np.multiply(np.log(a3), Y) + \
        np.multiply(- np.log(1 - a3), 1 - Y)

    loss = 1. / m * np.sum(logprobs)

    return loss


def compute_loss_with_regularization(a3: np.ndarray, parameters: dict[str, np.ndarray], Y: np.ndarray, lambd: float):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    ce_loss = compute_loss(a3, Y)

    regularized_loss = (lambd / (2 * m)) * \
        sum([np.sum(np.square(W)) for W in [W1, W2, W3]])

    loss = ce_loss + regularized_loss

    return loss


def load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x_orig = train_set_x_orig.reshape(
        train_set_x_orig.shape[0], -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_orig/255
    test_set_x = test_set_x_orig/255

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int_)

    # Forward propagation
    a3, caches = forward_prop(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

    return p


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_prop(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1],
                c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :],
                c=train_Y, s=40, cmap=plt.cm.Spectral)

    return train_X, train_Y, test_X, test_Y
