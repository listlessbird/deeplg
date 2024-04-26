import numpy as np
import matplotlib.pyplot as plt
from dnn_utils import *
from dnn_app_utils_v3 import predict as predict_v2

n_x = 12288
n_h = 7
n_y = 1

layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075


def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=True):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    grads = {}
    costs = []
    for i in range(0, num_iterations):
        # forward prop with relu and another one with sigmoid

        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        # get cost
        cost = compute_cost(A2, Y)

        # backprop for grads

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        grads['dA1'], grads['dW2'], grads['db2'] = linear_activation_backward(
            dA2, cache2, 'sigmoid')

        grads['dA0'], grads['dW1'], grads['db1'] = linear_activation_backward(
            grads['dA1'], cache1, 'relu')

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
            print(costs)

    return parameters, costs


def plot_costs(costs):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.xlabel('Iterations (Per Hundered)')
    plt.ylabel('Cost')
    plt.show()


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

parameters, costs = two_layer_model(
    train_x, train_y, (n_x, n_h, n_y), num_iterations=2500, print_cost=False)

# parameters = {
#     'W1': np.random.randn(7, 12288) * 0.01,  # (7, 12288)
#     'b1': np.zeros((7, 1)),                 # (7, 1)
#     'W2': np.random.randn(1, 7) * 0.01,     # (1, 7)
#     'b2': np.zeros((1, 1))                  # (1, 1)
# }

# plot_costs(costs)

predictions_test = predict(test_x, test_y, parameters)
