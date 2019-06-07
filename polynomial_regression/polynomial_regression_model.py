import numpy as np


class PolynomialRegression:

    def __init__(self, model_order=1):
        if model_order < 1:
            raise Exception('Model order can\'t be lower then 1')
        elif model_order > 1:
            self._order = model_order + 1
        else:
            self._order = model_order

    def loss(self, x, y):
        y_estimate = np.dot(x, self._w)
        error = (y_estimate - y.flatten())

        n = len(y)

        return 1.0 / (2 * n) * np.sum(np.power(error, 2))

    def process_data(self, x):
        x /= np.max(x, axis=0)

        if self._order > 1:
            x = np.power(x, range(self._order))
        else:
            x = np.c_[np.ones(x.shape[0]), x]

        return x

    def predict(self, x):

        x = self.process_data(x)

        return np.dot(x, self._w)

    def _gradient_descent_step(self, x, y, lr):
        y_estimate = np.dot(x, self._w)
        error = (y_estimate - y.flatten())

        n = len(x)

        gradient = (1.0 / n) * np.dot(x.T, error)

        self._w -= lr * gradient

    def fit(self, x, y, lr=0.05, tolerance=1e-8):
        x = self.process_data(x)

        self._w = np.zeros(x.shape[1])

        self._cost_history = []
        self._w_history = [self._w]

        epochs = 0
        while True:
            cost = self.loss(x, y)

            self._cost_history.append(cost)

            self._gradient_descent_step(x, y, lr)

            self._w_history.append(self._w.copy())

            epochs += 1

            if epochs % 100 == 0:
                new_cost = self.loss(x, y)
                print("Epoch: %d - Cost: %.4f" % (epochs, new_cost))

                # Stopping Condition
                if abs(new_cost - cost) < tolerance:
                    print("Converged.")
                    break

        return self
