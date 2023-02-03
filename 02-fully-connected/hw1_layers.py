import numpy as np


class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        self.Y = self.X @ self.W + self.b

        return self.Y

    def backward(self, dLdy):
        self.dLdW = self.X.T @ dLdy
        self.dLdb = dLdy.sum(axis=0)

        dLdx = dLdy @ self.W.T

        return dLdx

    def step(self, learning_rate):
        self.W -= learning_rate * self.dLdW
        self.b -= learning_rate * self.dLdb


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, X):
        Y = 1 / (1 + np.exp(-X))
        self.Y = Y

        return self.Y

    def backward(self, dLdy):
        return dLdy * self.Y * (1 - self.Y)

    def step(self, learning_rate):
        pass


class NLLLoss:
    def __init__(self):
        pass

    def forward(self, X, y):
        log_softmax = X - self.__log_sum_exp(X).reshape(-1, 1)
        self.log_softmax = log_softmax

        self.y = y
        nll = -log_softmax[np.arange(y.size), y]

        return nll.mean()

    def backward(self):
        self.one_hot_y = np.zeros(self.log_softmax.shape)
        self.one_hot_y[np.arange(self.y.size), self.y] = 1

        self.dLdx = np.exp(self.log_softmax) - self.one_hot_y
        return self.dLdx / self.one_hot_y.shape[0]

    def __log_sum_exp(self, X):
        c = np.max(X, axis=1).reshape(-1, 1)
        return np.log(np.exp(X - c).sum(axis=1)) + c.flatten()


class NeuralNetwork:
    def __init__(self, modules):
        self.layers = list(modules)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def backward(self, dLdy):
        for layer in self.layers[::-1]:
            dLdy = layer.backward(dLdy)

    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)
