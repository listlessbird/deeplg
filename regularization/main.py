from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from utils import backprop, backprop_regularized, compute_loss_with_regularization, forward_prop, compute_loss, initialize_parameters_he, initialize_parameters_random, initialize_parameters_zeros, load_2D_dataset, plot_decision_boundary, predict_dec, update_params, predict

train_X, train_Y, test_X, test_Y = load_2D_dataset()


def model(X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.01, iterations=1500, print_cost=True, initialization='he', lambd=0.0, keep_prob=1):
    m = X.shape[1]

    grads = {}
    costs = []

    layer_dims = [X.shape[0], 10, 5, 1]

    if initialization == 'zero':
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layer_dims)

    for i in range(iterations):

        a3, cache = forward_prop(X, parameter=parameters)

        if lambd == 0:
            cost = compute_loss(a3, Y)
        else:
            cost = compute_loss_with_regularization(a3, parameters, Y, lambd)

        if lambd == 0.0 and keep_prob == 1:
            grads = backprop(X, Y, cache=cache)
        elif lambd > 0.0:
            grads = backprop_regularized(X, Y, cache, lambd)

        parameters = update_params(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


params = model(train_X, train_Y, lambd=0.7, iterations=30000)
print("On train set")
preds_train = predict(train_X, train_Y, params)
print("On test")
preds_test = predict(test_X, test_Y, params)

plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)
