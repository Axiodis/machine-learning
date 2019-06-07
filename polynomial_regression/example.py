import numpy as np
import matplotlib.pyplot as plt

from polynomial_regression.polynomial_regression_model import PolynomialRegression

model_order = 6

data_x_org = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x_org) + 0.1 * np.power(data_x_org, 2) + 0.5 * np.random.randn(100, 1)

regression = PolynomialRegression(6)
regression.fit(data_x_org, data_y)
y_estimate = regression.predict(data_x_org)

plt.scatter(data_x_org, data_y, s=32)
plt.plot(data_x_org, y_estimate, color='red')
plt.show()
