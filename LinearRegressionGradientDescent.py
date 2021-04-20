import numpy as np


class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self, learning_rate=0.01):
        predicted = self.features.dot(self.coeff)
        print("Predicted array")
        print(predicted)
        s = self.features.T.dot(predicted - self.target)
        print("S is")
        print(s)
        gradient = (1. / len(self.features)) * s
        print("Gradient is")
        print(gradient)
        print("Coefficients in descent step")
        print(self.coeff)
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate=0.01, num_iterations=100):
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        print("Coefficients")
        print(self.coeff)
        return self.coeff, self.mse_history

    def fit(self, features, target):
        self.features = features.copy(deep=True)
        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        self.features = self.features.to_numpy()
        self.target = target.to_numpy().reshape(-1, 1)
