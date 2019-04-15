import random

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_partial(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_partial(x):
    return 1 - np.tanh(x) ** 2


def sigmoid_quadratic(target, prediction, net):
    # print("sigmoid(net)", sigmoid(net))
    return (target - prediction) * sigmoid(net) * (1 - sigmoid(net))


def tanh_quadratic(target, prediction, net):
    return (target - prediction) * tanh_partial(net)


def sigmoid_entropy(target, prediction, net):
    return target / prediction * sigmoid_partial(net) + (1 - target) / (1 - prediction) * sigmoid_partial(net)


def tanh_entropy(target, prediction, net):
    return target / prediction * tanh_partial(net) + (1 - target) / (1 - prediction) * tanh_partial(net)


class Neuron:
    def __init__(self, learning_rate, weights, act_f="sigmoid", error_cost_type="quadratic"):
        """
        Initialization of the neuron
        :param learning_rate: constant value describing the learning rate needed for the weights' update
        :param act_f: Activation function e.g. sigmoid, tanh
        :param error_cost_type: error function (without regularisation) can be either quadratic or cross entropy
        """
        if act_f == "sigmoid":
            self.activation_function = sigmoid
        elif act_f == "tanh":
            self.activation_function = np.tanh

        self.learning_rate = learning_rate

        if error_cost_type == "quadratic":
            if act_f == "sigmoid":
                self.error_cost = sigmoid_quadratic
            elif act_f == "tanh":
                self.error_cost = tanh_quadratic
        elif error_cost_type == "cross_entropy":
            if act_f == "sigmoid":
                self.error_cost = sigmoid_entropy
            elif act_f == "tanh":
                self.error_cost = tanh_entropy
        self.weights_matrix = weights

    def predict(self, input):
        """
        Returns the output for the Neuron
        :param x: Input matrix
        :param w: Weights matrix
        :return: output value
        """
        net = np.dot(input, self.weights_matrix)
        return self.activation_function(net)

    def train(self, x, target, iterations):
        x_size = x.shape[1]

        MSE_output = []
        MSE = 0
        for iteration in range(iterations):
            MSE = 0
            print("Epoch ---> ", iteration)
            for i, iter_x in enumerate(x):
                yp = self.predict(iter_x)

                if yp != target[i]:
                    net = np.dot(self.weights_matrix, iter_x)
                    error = self.error_cost(target[i], yp, net)
                    self.weights_matrix += self.learning_rate * error

                MSE += (yp - target[i])**2

            MSE_output.append(MSE/x_size)
        return MSE_output


def first_question():
    train_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs = np.array([[0, 1, 1, 0]]).T
    weights = np.ones(train_inputs.shape[1])

    neuron = Neuron(0.1, weights)
    MSE = neuron.train(train_inputs, train_outputs, 100)

    plt.plot(np.arange(0, len(MSE), 1), MSE)
    plt.show()


def second_question():
    # Question 2
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = (iris["target"] == 2).astype(np.int).T  # 1 if Iris-Virginica, else 0

    # Initialization
    weights = np.array([random.random() for _ in range(X.shape[1])])
    print(weights)
    # weights = np.ones(X.shape[1])
    neuron = Neuron(0.1, weights)
    MSE = neuron.train(X, y, 100)

    print("Predict", neuron.predict(np.array([[5.1, 3.5]])))

    plt.plot(np.arange(0, len(MSE), 1), MSE)
    plt.ylabel("MSE")
    plt.xlabel("epcoch")
    plt.show()

    # This is how you would do it in sklearn
    # from sklearn.linear_model import LogisticRegression
    #
    # log_reg = LogisticRegression()
    # log_reg.fit(X, y)
    # print("Size y:", len(y))
    # print("Predict:", log_reg.predict(X[:len(y), :]))


if __name__ == "__main__":
    first_question()
    # second_question()