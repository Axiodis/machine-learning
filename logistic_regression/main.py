import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from logistic_regression.logistic_regression_model import LogisticRegression


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


if __name__ == "__main__":
    data = load_data("marks.txt", None)

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    admitted = data.loc[y == 1]
    not_admitted = data.loc[y == 0]

    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    plt.legend()
    plt.show()

    logistic_reg = LogisticRegression()
    logistic_reg.fit(x_train, y_train)

    acc = logistic_reg.accuracy(x_test, y_test)
    print(acc)
