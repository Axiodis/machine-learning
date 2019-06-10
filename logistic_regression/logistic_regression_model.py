import numpy as np


class LogisticRegression:

    def process_data(self, x):
        x /= np.max(x, axis=0)
        x = np.c_[np.ones((x.shape[0], 1)), x]
        return x

    def sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def probability(self, x):
        # Probability after passing through sigmoid
        return self.sigmoid(np.dot(x, self._w))

    def loss(self, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        prob = self.probability(x)
        total_cost = -(1 / m) * np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        return total_cost

    def _gradient_descent_step(self, x, y, lr):
        m = x.shape[0]
        prob = self.probability(x)
        gradient = (1 / m) * np.dot(x.T, prob - y)

        self._w -= lr * gradient

    def fit(self, x, y, lr=0.05, tolerance=1e-8):
        x = self.process_data(x)
        y = y[:, np.newaxis]
        self._w = np.zeros((x.shape[1], 1))

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

    def predict(self, x):
        x = self.process_data(x)
        return self.probability(x)

    def accuracy(self, x, actual_classes, probability_threshold=0.5):
        predicted_classes = (self.predict(x) >= probability_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100
