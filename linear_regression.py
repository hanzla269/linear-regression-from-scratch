import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LinearRegression:

    def __init__(self, epochs=10000, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):

            y_pred = np.dot(X, self.weight) + self.bias

            error = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weight) + self.bias

    def mse(self, X, y):
        prediction = self.predict(X)
        return np.mean((y - prediction) ** 2)

    def mae(self, X, y):
        prediction = self.predict(X)
        return np.mean(np.abs(y - prediction))

    def r2(self, X, y):

        prediction = self.predict(X)

        ss_res = np.sum((y - prediction) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)
