import numpy as np
import pandas as pd


class LogisticRegressionPredictor:
    def __init__(self, models):
        self.models = models

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        # Add intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        probabilities = np.zeros((X.shape[0], len(self.models)))

        for i, (cls, theta) in enumerate(self.models.items()):
            z = np.dot(X, theta)
            probabilities[:, i] = self.sigmoid(z)

        predictions = [
            max(self.models, key=lambda cls: p[i])
            for i, p in enumerate(probabilities)
        ]
        return predictions


def main():
    # Load test dataset
    test_data = pd.read_csv("dataset_test.csv")

    # Preprocess test data (handle missing values, encode categorical variables, etc.)
    # ...

    # Load trained models
    trained_models = np.load("trained_models.npy", allow_pickle=True).item()

    # Initialize predictor
    predictor = LogisticRegressionPredictor(trained_models)

    # Predict classes
    X_test = test_data.drop(columns=["Index"]).values
    predictions = predictor.predict(X_test)

    # Write predictions to output file
    with open("houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for i, prediction in enumerate(predictions):
            f.write(f"{i},{prediction}\n")


if __name__ == "__main__":
    main()
