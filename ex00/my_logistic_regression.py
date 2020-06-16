import numpy as np


class MyLogisticRegression():
    """ My personnal logistic regression to classify things """

    def __init__(self, theta, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.theta = np.array(
            theta, copy=True).reshape(-1, 1).astype("float64")

    def fit_(self, x, y):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if (x.size == 0 or y.size == 0 or self.theta.size == 0
            or x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]
                or x.shape[1] + 1 != self.theta.shape[0] or y.shape[1] != 1):
            return None

        x_prime = np.c_[np.ones(x.shape[0]), x]
        for _ in range(self.n_cycle):
            h = 1 / (1 + np.exp(-x_prime @ self.theta))
            nabla = (x_prime.T @ (h - y)) / y.shape[0]
            self.theta = self.theta - self.alpha * nabla

        return self.theta

    def predict_(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if (x.size == 0 or self.theta.size == 0
                or x.ndim != 2 or self.theta.ndim != 2
                or x.shape[1] + 1 != self.theta.shape[0]
                or self.theta.shape[1] != 1):
            return None

        x_prime = np.c_[np.ones(x.shape[0]), x]
        return 1 / (1 + np.exp(-x_prime @ self.theta))

    def cost_(self, x, y, eps=1e-15):
        # Using one dimensional array to use dot product with np.dot
        # (np.dot use matmul with two dimensional array)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        y_hat = self.predict_(x).flatten()

        if (y.size == 0 or y.ndim != 1
            or y_hat is None
                or y.shape != y_hat.shape):
            return None

        return -(y.dot(np.log(y_hat + eps)) + (1 - y).dot(np.log(1 - y_hat + eps))) / y.shape[0]
