import random

import math
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


# ================ ACTIVATION FUNCTIONS ================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear(x):
    return x


# ================ ERROR COST FUNCTIONS ================
def linear_quadratic(labels, prediction):
    # calculates these formulas by hand
    return labels - prediction


def sigmoid_quadratic(labels, prediction):
    # calculates these formulas by hand
    return (labels - prediction) * prediction * (1 - prediction)


def sigmoid_entropy(labels, prediction):
    # calculates these formulas by hand
    return labels * (1 - prediction) - (1 - labels) * prediction


#
# ================ NEURON IMPLEMENTATION ================
#

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
        elif act_f == "linear":
            self.activation_function = linear

        self.learning_rate = learning_rate

        if error_cost_type == "quadratic":
            if act_f == "sigmoid":
                self.error_cost = sigmoid_quadratic
            elif act_f == "linear":
                self.error_cost = linear_quadratic

        elif error_cost_type == "cross_entropy":
            if act_f == "sigmoid":
                self.error_cost = sigmoid_entropy

        self.weights_matrix = weights

    def predict(self, input):
        """
        Returns the output for the Neuron
        :param x: Input matrix
        :param w: Weights matrix
        :return: output value
        """

        net = np.dot(input, self.weights_matrix[1:]) + self.weights_matrix[0]
        return self.activation_function(net)

    def cost_function(self, features, labels):
        # FIXME: This is where the evaluations will be perform. Right now is the cross entropy error this needs to be ignored
        # observations = len(labels)

        predictions = self.predict(features)

        # Take the error when label=1
        # class1_cost = labels * (1 - predictions)

        # Take the error when label=0
        # class2_cost = (1 - labels) * predictions

        # Take the sum of both costs
        # cost = class1_cost - class2_cost

        # Take the average cost
        # cost = cost.sum()
        # print(cost)
        # return(cost)

        classifications = classifier(predictions)

        return -self.error_cost(labels, predictions[1:]).sum(), accuracy(classifications, labels)

    # def update_weights(self, features, labels):
    #     N = len(features)
    #
    #     # 1 - Get Predictions
    #     predictions = self.predict(features)
    #
    #     gradient = self.error_cost(labels, predictions) * features.T
    #     gradient = gradient.sum()
    #     # 3 Take the average cost derivative for each feature
    #     gradient /= N
    #
    #     # 4 - Multiply the gradient by our learning rate
    #     gradient *= self.learning_rate
    #
    #     # 5 - Subtract from our weights to minimize cost
    #     self.weights_matrix -= gradient

    def update_weights(self, input, answer):
        """
        Updates the weights matrix
        :param input: the input being considered
        :param answer: the labeled answer for th einput
        """
        prediction = self.predict(input)
        gradient = self.error_cost(answer, prediction)
        gradient *= self.learning_rate

        self.weights_matrix[1:] += gradient * input
        self.weights_matrix[0] += gradient

    def train(self, features, labels, iters):
        """
        Trains the neuron by updating the weights matrix for the considered number of epochs
        :param features: The features given to the neural network
        :param labels: The corresponding labels
        :param iters: The number of iterations (epochs)
        """
        cost_history = []
        acc_history = []

        for i in range(iters):
            # print("Epoch -> ", i)
            for input, answer in zip(features, labels):
                self.update_weights(input, answer)

            # Calculate error for auditing purposes
            cost, acc = self.cost_function(features, labels)
            cost_history.append(cost)
            acc_history.append(acc)

        return cost_history, acc_history


def classifier(predictions):
    def decision_boundary(prob):
        return 1 if prob > .5 else 0

    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    decision_boundary = np.vectorize(decision_boundary)
    return decision_boundary(predictions).flatten()


def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels[1:] - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

# def first_question():
#     epochs = 500
#     X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
#     y = np.array([[0, 1, 1, 0]])
#
#     # Initialization of Weights
#     weights = np.array([random.random() for _ in range(X.shape[1])])
#
#     # Compare cost functions
#     # # Neuron with h = sigmoid and cost functions -> quadratic
#     neuron = Neuron(0.001, weights)
#     cost = neuron.train(X, y, epochs)
#
#     # # Neuron with cost function -> entropy
#     neuron = Neuron(0.001, weights, error_cost_type="cross_entropy")
#     cost2 = neuron.train(X, y, epochs)
#
#     plt.figure()
#     plt.plot(np.arange(0, len(cost), 1), cost)
#     plt.plot(np.arange(0, len(cost2), 1), cost2)
#     plt.savefig("plots/cost_functions.png")
#
#     # Compare Initializations
#     # # Neuron with h = sigmoid and cost functions -> quadratic
#     neuron = Neuron(0.1, weights)
#     cost = neuron.train(X, y, epochs)
#
#     weights = np.ones(X.shape[1])
#     neuron = Neuron(0.1, weights)
#     cost2 = neuron.train(X, y, epochs)
#
#     plt.figure()
#     plt.plot(np.arange(0, len(cost), 1), cost)
#     plt.plot(np.arange(0, len(cost2), 1), cost2)
#     plt.savefig("plots/initialization.png")


def second_question():
    def validation_set(X, y, percentage):
        """
        Creates the validation set from the dataset
        :param X: features
        :param y: labels
        :param percentage: percentage of the validation set from the training set
        :return: training set features and labels
        """
        total = y.shape[0]
        size = total * percentage
        start = random.randint(1, total - size)
        end = int(start + size)
        # print("start", start)
        # print("end", end)
        test_set_y = y[start:end]
        test_set_X = X[start:end]
        # print("input", X.shape)
        # print("answer", y.shape)
        # print("test size", test_set_y.shape)
        # print("test set", test_set_y)
        # print("test size", test_set_X.shape)
        # print("test set", test_set_X)

        X = np.concatenate((X[:start], X[end:]), axis=0)
        y = np.concatenate((y[:start], y[end:]), axis=0)
        # print(X.shape)
        # print(size)
        return test_set_X, test_set_y, X, y

    epochs = 300

    # ================ DATA SET ================
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
    test_X, test_y, X, y = validation_set(X, y, 0.1)

    bias = np.ones(X.shape[1])
    X = np.vstack((bias, X))

    # ================ INITIALIZATION ================
    # Initialization of Weights
    # weights = np.ones(X.shape[1]+1)
    weights = np.array([random.random() for _ in range(X.shape[1] + 1)])
    # print(weights)

    # ================ EVALUATION ================
    # Activation function: linear    Error function: Quadratic
    neuron = Neuron(0.01, weights, act_f="linear")
    cost, acc = neuron.train(X, y, epochs)
    probabilities = neuron.predict(X).flatten()
    classifications = classifier(probabilities)
    our_acc = accuracy(classifications, y.flatten())
    print(our_acc)

    # Activation function: sigmoid   Error function: Quadratic
    neuron = Neuron(0.01, weights)
    cost2, acc2 = neuron.train(X, y, epochs)
    probabilities = neuron.predict(X).flatten()
    classifications = classifier(probabilities)
    our_acc = accuracy(classifications, y.flatten())
    print(our_acc)

    # Activation function: sigmoid   Error function: Cross Enropy
    neuron = Neuron(0.01, weights, error_cost_type="cross_entropy")
    cost3, acc3 = neuron.train(X, y, epochs)
    probabilities = neuron.predict(X).flatten()
    classifications = classifier(probabilities)
    our_acc = accuracy(classifications, y.flatten())
    print(our_acc)

    plt.figure()
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(cost), 1), cost)

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(acc), 1), acc)

    plt.figure()
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(cost2), 1), cost2)

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(acc2), 1), acc2)

    plt.figure()
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(cost3), 1), cost3)

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(acc3), 1), acc3)
    # plt.savefig("iris.png")

    # plt.figure()
    # plt.plot(np.arange(0, len(cost), 1), cost)
    # # plt.plot(np.arange(0, len(cost2), 1), cost2)
    # # plt.plot(np.arange(0, len(cost3), 1), cost3)
    # plt.savefig("plots/iris.png")


if __name__ == "__main__":
    # first_question()
    second_question()
