import random

import math
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

chance = 0.85
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
    return ((labels - prediction) / (prediction - prediction * prediction)) * prediction * (1 - prediction)

#
# ================ NEURON IMPLEMENTATION ================
#


class Neuron:

    def __init__(self, learning_rate, weights, act_f = "sigmoid", error_cost_type = "quadratic"):
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
        predictions = self.predict(features)
        classifications = classifier(predictions)
        return self.error_cost(labels, predictions[1:]).sum(), accuracy(classifications, labels)

    def update_weights(self, input, answer):

        def decision_boundary(prob):
            return 1 if prob > chance else 0

        """
        Updates the weights matrix
        :param input: the input being considered
        :param answer: the labeled answer for th einput
        """
        prediction = self.predict(input)
        # if decision_boundary(prediction) != answer:
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
            for input, answer in zip(features, labels):
                self.update_weights(input, answer)

            # Calculate error for auditing purposes
            cost, acc = self.cost_function(features, labels)
            
            cost_history.append(cost)
            acc_history.append(acc)

        return cost_history, acc_history


def classifier(predictions):

    def decision_boundary(prob):
        return 1 if prob > chance else 0

    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    decision_boundary = np.vectorize(decision_boundary)
    return decision_boundary(predictions).flatten()


def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels[1:] - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


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
        test_set_y = y[start:end]
        test_set_X = X[start:end]

        X = np.concatenate((X[:start], X[end:]), axis = 0)
        y = np.concatenate((y[:start], y[end:]), axis = 0)

        return test_set_X, test_set_y, X, y

    epochs = 200
    learning_rate = 0.001

    # ================ DATA SET ================
    iris = datasets.load_iris()
    X = iris.data[:, :2]   # we only take the first two features.
    y = (iris["target"] == 2).astype(np.int)   # 1 if Iris-Virginica, else 0

    # Shuffle Data
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Get Validation Set
    test_X, test_y, X, y = validation_set(X, y, 0.3)

    # Add Bias
    bias = np.ones(X.shape[1])
    X = np.vstack((bias, X))
    bias = np.ones(test_X.shape[1])
    test_X = np.vstack((bias, test_X))

    # ================ INITIALIZATION ================
    # Initialization of Weights
    weights = np.ones(X.shape[1] + 1)
    # weights = np.array([random.random() for _ in range(X.shape[1] + 1)])

    # ================ TRAINING ================

    # Activation function: linear    Error function: Quadratic
    neuron = Neuron(learning_rate, weights, act_f = "linear")
    cost, acc = neuron.train(X, y, epochs)

    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(cost), 1), cost)
    # plt.plot(np.arange(0, len(cost_t), 1), cost_t)
    plt.legend(['Train'], loc = 'upper left')
    # plt.savefig("plots/loss_linear")
    plt.show()

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(acc), 1), acc)
    # plt.plot(np.arange(0, len(acc_t), 1), acc_t)
    plt.legend(['Train'], loc = 'upper left')
    # plt.savefig("plots/acc_linear")
    plt.show()

    probabilities = neuron.predict(test_X).flatten()
    classifications = classifier(probabilities)
    our_acc = accuracy(classifications, test_y.flatten())
    print("Accuracy Linear Quadratic Model:", our_acc)

    # Activation function: sigmoid   Error function: Quadratic
    neuron = Neuron(learning_rate, weights)
    cost2, acc2 = neuron.train(X, y, epochs)

    probabilities = neuron.predict(test_X).flatten()
    classifications = classifier(probabilities)
    our_acc = accuracy(classifications, test_y.flatten())
    print("Accuracy Sigma Quadratic Model:", our_acc)

    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(cost2), 1), cost2)
    # plt.plot(np.arange(0, len(cost2_t), 1), cost2_t)
    plt.legend(['Train'], loc = 'upper left')
    # plt.savefig("plots/loss_sig_quadratic")
    plt.show()

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(acc2), 1), acc2)
    # plt.plot(np.arange(0, len(acc2_t), 1), acc2_t)
    plt.legend(['Train'], loc = 'upper left')
    # plt.savefig("plots/acc_sig_quadratic")
    plt.show()
    
    # Activation function: sigmoid   Error function: Cross Enropy
    neuron = Neuron(learning_rate, weights, error_cost_type = "cross_entropy")
    cost3, acc3 = neuron.train(X, y, epochs)

    probabilities = neuron.predict(test_X).flatten()
    classifications = classifier(probabilities)
    our_acc = accuracy(classifications, test_y.flatten())
    print("Accuracy Sigma Cross Entropy Model:", our_acc)
    # cost3_t, acc3_t = neuron.train(test_X, test_y, epochs)
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(cost3), 1), cost3)
    # plt.plot(np.arange(0, len(cost3_t), 1), cost3_t)
    plt.legend(['Train'], loc = 'upper left')
   # plt.savefig("plots/loss_sig_cross")
    plt.show()

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(0, len(acc3), 1), acc3)
    # plt.plot(np.arange(0, len(acc3_t), 1), acc3_t)
    plt.legend(['Train'], loc = 'upper left')
    # plt.savefig("plots/acc_sig_cross")
    plt.show()


if __name__ == "__main__":
    # first_question()
    second_question()
