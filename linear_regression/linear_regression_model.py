import numpy as np


class LinearRegression:

    @staticmethod
    def loss(h, y):
        sq_error = (h - y) ** 2
        n = len(y)
        return 1.0 / (2 * n) * sq_error.sum()

    def predict(self, x):
        x = (x.copy() - x.mean()) / x.std()
        x = np.c_[np.ones(x.shape[0]), x]
        return np.dot(x, self._W)

    def _gradient_descent_step(self, x, targets, lr):
        predictions = np.dot(x, self._W)

        error = predictions - targets
        gradient = np.dot(x.T, error) / len(x)

        self._W -= lr * gradient

    def fit(self, x, y, n_iter=100000, lr=0.01):
        x = (x - x.mean()) / x.std()
        x = np.c_[np.ones(x.shape[0]), x]

        self._W = np.zeros(x.shape[1])

        self._cost_history = []
        self._w_history = [self._W]

        for i in range(n_iter):
            prediction = np.dot(x, self._W)
            cost = self.loss(prediction, y)

            self._cost_history.append(cost)

            self._gradient_descent_step(x, y, lr)

            self._w_history.append(self._W.copy())

        return self
