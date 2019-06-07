import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from linear_regression.linear_regression_model import LinearRegression

plt.interactive(True)

data = pd.read_csv('house_prices_train.csv')

data.sample(frac=1)

data_train, data_test = np.split(data, [int(0.7 * len(data))])

test_x = data_test['GrLivArea'].values
test_y = data_test['SalePrice'].values

train_x = data_train['GrLivArea'].values
train_y = data_train['SalePrice'].values

clf = LinearRegression()
clf.fit(train_x, train_y, n_iter=1000, lr=0.01)

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(clf._cost_history)
plt.show()

predicted = clf.predict(test_x)

plt.title('Sale price to area')
plt.xlabel('GrLivArea')
plt.ylabel('Sale Price')
plt.scatter(test_x, test_y, s=32, alpha=0.5)
plt.plot(test_x, predicted)
plt.show()
