import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from helpers import nn_model, predict


X, Y = load_planar_dataset()


shape_X = X.shape
shape_Y = Y.shape

# parameters = nn_model(X, Y, n_h=4, iterations=1000, print_cost=True)

# plot_decision_boundary(lambda x: predict(parameters, X=x.T), X, Y)
# plt.title("Decision boundary for hidden layer of size" + str(4))

# predictions = predict(parameters, X)


# print("Accuracy: %d" % accuracy)


plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5]


for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    print("Now running for hidden layer of size: %d" % (n_h))
    parameters = nn_model(X, Y, n_h, iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    # eget true positives and negatives

    tp = np.dot(Y, predictions.T)
    tn = np.dot(1 - Y, 1 - predictions.T)

    accuracy = (tp + tn) / float(Y.size)
    accuracy = float(accuracy) * 100
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()
