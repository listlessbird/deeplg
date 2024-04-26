import numpy as np
import matplotlib.pyplot as plt

import dnn_utils


# 4 layer model

layer_dims = [12288, 20, 7, 5, 1]


def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []

    parameters = dnn_utils.initialize_parameters_deep(layer_dims)

    for i in range(0, num_iterations):

        Yhat, caches = dnn_utils.L_model_forward(X, parameters)

        cost = dnn_utils.compute_cost(Yhat, Y)

        grads = dnn_utils.L_model_backward(Yhat, Y, caches)

        parameters = dnn_utils.update_parameters(
            parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


def plot_costs(costs):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.xlabel('Iterations (Per Hundered)')
    plt.ylabel('Cost')
    plt.show()


train_x_orig, train_y, test_x_orig, test_y, classes = dnn_utils.load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0


parameters, costs = L_layer_model(
    train_x, train_y, layer_dims, num_iterations=5500, print_cost=True)


plot_costs(costs)

pred_test = dnn_utils.predict(test_x, test_y, parameters)
