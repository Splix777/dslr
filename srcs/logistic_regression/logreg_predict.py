from typing import List, Any

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

    @staticmethod
    def load_weights(file_path: str):
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

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
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

    def restore_modified_features(self, data: pd.DataFrame) -> None:
        """
        Restore the modified features in the test data.

        Parameters:
        data (pd.DataFrame): Test data.
        """
        data[self.features] = self.scaler.inverse_transform(
            self.features_scaled
        )
        data["Best Hand"] = data["Best Hand"].map({0: "Right", 1: "Left"})

    def predict(self, data: pd.DataFrame) -> list[list[Any] | Any]:
        """
        Predict the Hogwarts house for the test data.

        Parameters:
        data (pd.DataFrame): Test data.

        Returns:
        np.ndarray: Predicted houses.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        self.preprocess_data(data)
        logging.info(f"{self.preprocess_data}")
        predictions = {}

        for house in self.houses:
            weights = np.array(self.weights[house]["weights"])
            bias = self.weights[house]["bias"]
            linear_output = self.calculate_linear_output(weights, bias)
            predictions[house] = self.sigmoid_activation(linear_output)

        predicted_classes = np.argmax(
            np.array(list(predictions.values())), axis=0
        )
        predicted_houses = [self.houses[idx] for idx in predicted_classes]
        self.restore_modified_features(data)
        return predicted_houses

    def calculate_linear_output(self, weights, bias):
        """
        Calculate the linear output of the model.

        Parameters:
        weights (np.ndarray): Model weights.
        bias (float): Model bias.

        Returns:
        np.ndarray: Linear output.
        """
        return np.dot(self.features_scaled, weights) + bias

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

    def evaluate_model(
        self, true_labels: np.ndarray, predicted_labels: np.ndarray, output_file: str
    ) -> None:
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

        with open(output_file, "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}")


def get_project_base_path() -> str:
    """Get the base path of the project."""
    current_file = os.path.abspath(__file__)
    root_path = os.path.abspath(os.path.join(current_file, "..", "..", ".."))
    return root_path


if __name__ == "__main__":
    if os.path.exists("logs/prediction.log"):
        os.remove("logs/prediction.log")

    logging.basicConfig(
        filename="logs/prediction.log",
        level=logging.INFO,
        format="%(message)s",
    )
    base_dir = get_project_base_path()
    logreg_dir = os.path.join(base_dir, "outputs/logistic_regression")
    csv_dir = os.path.join(base_dir, "csv_files")
    weights_file = os.path.join(logreg_dir, "weights.json")
    csv_test_file = os.path.join(csv_dir, "dataset_test.csv")

    predictor = LogisticRegressionPredictor(weights_file)
    test_data = predictor.load_data(csv_test_file)

    predicted_labels = predictor.predict(test_data)

    # Adding the predicted labels to the test data and saving the results
    test_data["Hogwarts House"] = predicted_labels
    results_csv = os.path.join(logreg_dir, "dataset_truth.csv")
    test_data.to_csv(results_csv, index=False)

    true_csv_file = os.path.join(csv_dir, "sample_truth.csv")

    true_csv = pd.read_csv(true_csv_file)
    true_labels = true_csv["Hogwarts House"].values

    evaluation_file = os.path.join(logreg_dir, "evaluation.txt")
    predictor.evaluate_model(true_labels, np.array(predicted_labels), evaluation_file)
