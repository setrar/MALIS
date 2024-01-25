import numpy as np

# using alpha to represent lambda to avoid naming conflicts
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        # Add an intercept term to the feature matrix X
        n_samples = X.shape[0]
        X_intercept = np.ones((n_samples, 1))
        X = np.hstack((X_intercept, X))

        # Create the penalty matrix for Ridge Regression
        n_features = X.shape[1]
        A = self.alpha * np.eye(n_features)
        A[0, 0] = 0  # No regularization for the intercept term

        # Solve the linear equation (X^T * X + alpha * I) * w = X^T * y for weights
        self.weights = np.linalg.solve(X.T @ X + A, X.T @ y)

    def predict(self, X):
        n_samples = X.shape[0]
        X_intercept = np.ones((n_samples, 1))
        X = np.hstack((X_intercept, X))
        return X.dot(self.weights)