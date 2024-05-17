import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, alpha=0.01, num_iterations=1000):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descent(self, X, y):
        m = len(y)
        for _ in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.alpha * gradient

    def train(self, X, y):
        # Add intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        num_features = X.shape[1]
        self.theta = np.zeros(num_features)

        unique_classes = np.unique(y)
        models = {}
        for cls in unique_classes:
            binary_y = np.where(y == cls, 1, 0)
            self.gradient_descent(X, binary_y)
            models[cls] = self.theta.copy()

        return models


def main():
    # Load training dataset
    train_data = pd.read_csv("dataset_train.csv")

    # Preprocess data (handle missing values, encode categorical variables, etc.)
    # (to be implemented)

    # Split data into features (X) and target variable (y)
    X_train = train_data.drop(columns=["Index", "Hogwarts House"]).values
    y_train = train_data["Hogwarts House"].values

    # Initialize and train logistic regression model
    model = LogisticRegression()
    trained_models = model.train(X_train, y_train)

    # Save trained models to a file
    np.save("trained_models.npy", trained_models)


if __name__ == "__main__":
    main()
