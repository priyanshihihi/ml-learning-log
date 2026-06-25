# Linear Regression from scratch — Week 1
# Implementing gradient descent without sklearn

import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Quick test
if __name__ == "__main__":
    X = np.array([[1],[2],[3],[4],[5]], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    model = LinearRegression(lr=0.01, epochs=1000)
    model.fit(X, y)
    print("Prediction for 6:", model.predict([[6]]))  # Should be ~12