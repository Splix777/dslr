import numpy as np
import pandas as pd
import json
import logging
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class LogisticRegressionPredictor:
    def __init__(self, weights_file: str, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.weights = self.load_weights(weights_file)
        self.scaler = None
        self.features_scaled = None
        self.houses = list(self.weights.keys())
        self.features = None

    def load_weights(self, file_path: str):
        """
        Load the weights from a JSON file.

        Parameters:
        file_path (str): Path to the JSON file.

        Returns:
        dict: Loaded weights.
        """
        try:
            with open(file_path, "r") as f:
                weights = json.load(f)
            logging.info(f"Weights loaded from {file_path}")
            return weights
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            logging.error(f"Error loading weights: {str(e)}")
            raise e

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the dataset from a CSV file.

        Parameters:
        file_path (str): Path to the CSV file.

        Returns:
        pd.DataFrame: Loaded dataset.
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise e

    def preprocess_data(self, data: pd.DataFrame) -> None:
        """
        Preprocess the test data: handle missing values and scale features.

        Parameters:
        data (pd.DataFrame): Test data.
        """
        try:
            data["Best Hand"] = (
                data["Best Hand"].map({"Right": 0, "Left": 1}).astype(float)
            )
            self.features = data.select_dtypes(
                include=["float64"]
            ).columns.tolist()

            # Remove Howarts House from the features list
            self.features.remove("Hogwarts House")

            data[self.features] = data[self.features].apply(
                lambda col: col.fillna(col.mean())
            )

            self.scaler = StandardScaler()
            self.features_scaled = self.scaler.fit_transform(
                data[self.features]
            )
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise e

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict the Hogwarts house for the test data.

        Parameters:
        data (pd.DataFrame): Test data.

        Returns:
        np.ndarray: Predicted houses.
        """
        self.preprocess_data(data)
        logging.info(f"{self.preprocess_data}")
        predictions = {}

        for house in self.houses:
            weights = np.array(self.weights[house])
            linear_output = self.calculate_linear_output(weights)
            predictions[house] = self.sigmoid_activation(linear_output)

        predicted_classes = np.argmax(
            np.array(list(predictions.values())), axis=0
        )
        predicted_houses = [self.houses[idx] for idx in predicted_classes]
        return predicted_houses

    def calculate_linear_output(self, weights):
        """
        Calculate the linear output of the model.

        Parameters:
        weights (np.ndarray): Model weights.

        Returns:
        np.ndarray: Linear output.
        """
        return np.dot(self.features_scaled, weights)

    @staticmethod
    def sigmoid_activation(linear_output):
        """
        Apply the sigmoid activation function.

        Parameters:
        linear_output (np.ndarray): Linear output.

        Returns:
        np.ndarray: Sigmoid activation output.
        """
        return 1 / (1 + np.exp(-linear_output))

    def evaluate_model(self, true_labels, predicted_labels):
        """
        Evaluate the model's performance using various metrics.

        Parameters:
        true_labels (np.ndarray): True labels.
        predicted_labels (np.ndarray): Predicted labels.
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(
            true_labels, predicted_labels, average="weighted"
        )
        recall = recall_score(
            true_labels, predicted_labels, average="weighted"
        )
        f1 = f1_score(true_labels, predicted_labels, average="weighted")
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")


# Example usage
if __name__ == "__main__":
    if os.path.exists("prediction.log"):
        os.remove("prediction.log")

    logging.basicConfig(
        filename="prediction.log", level=logging.INFO, format="%(message)s"
    )

    predictor = LogisticRegressionPredictor(weights_file="weights.json")
    test_data = predictor.load_data("dataset_test.csv")

    predicted_labels = predictor.predict(test_data)

    # Adding the predicted labels to the test data and saving the results
    test_data["Hogwarts House"] = predicted_labels
    test_data.to_csv("predicted_test_data.csv", index=False)
    logging.info(f"Predicted labels saved to predicted_test_data.csv")

    true_csv = pd.read_csv("dataset_truth.csv")
    true_labels = true_csv["Hogwarts House"].values

    if true_labels is not None:
        predictor.evaluate_model(true_labels, predicted_labels)
