import numpy as np
import pandas as pd
import json
import logging
import os

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded dataset.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise e


class LogisticRegressionTrainer:
    def __init__(
        self,
        learning_rate: float = 0.001,
        num_iterations: int = 1000,
        epsilon: float = 1e-5,
    ):
        """
        Initialize the logistic regression trainer.

        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for training.
        epsilon (float): Small value to avoid log(0).
        """
        self.costs = None
        self.scaler = None
        self.data = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.weights = {}
        self.file_path = None
        self.features_scaled = None
        self.houses = None
        self.features = None

    def load_data(self, file_path: str) -> None:
        """
        Load the dataset from a CSV file.

        Parameters:
        file_path (str): Path to the CSV file.

        Returns:
        pd.DataFrame: Loaded dataset.
        """
        try:
            self.file_path = file_path
            self.data = pd.read_csv(file_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise e

    def preprocess_data(self) -> None:
        try:

            self.houses = self.data["Hogwarts House"].unique()
            self.data["Best Hand"] = (
                self.data["Best Hand"]
                .map({"Right": 0, "Left": 1})
                .astype(float)
            )
            self.features = self.data.select_dtypes(
                include=["float64"]
            ).columns.tolist()

            self.data[self.features] = self.data[self.features].apply(
                lambda col: col.fillna(col.mean())
            )

            self.scaler = StandardScaler()
            self.features_scaled = self.scaler.fit_transform(
                self.data[self.features]
            )
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise e

    def train_one_vs_all(self) -> None:
        self.preprocess_data()
        self.costs = {}
        target_labels = self.data["Hogwarts House"].values

        for house in self.houses:
            binary_target = (target_labels == house).astype(int)
            weights = np.zeros(self.features_scaled.shape[1])

            for iteration in range(self.num_iterations):
                linear_output = self.calculate_linear_output(weights)
                predictions = self.sigmoid_activation(linear_output)
                predictions = self.ensure_valid_predictions(predictions)
                gradient = self.calculate_gradient(
                    predictions, binary_target, weights
                )
                if iteration % 100 == 0:
                    cost = self.calculate_cost(binary_target, predictions)
                    self.costs[house] = cost
                    logging.info(
                        f"House: {house}, Iteration: {iteration}, Cost: {cost}"
                    )
                weights = self.update_weights(weights, gradient)

            self.weights[house] = weights.tolist()

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

    def ensure_valid_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Ensure predictions are within valid range to prevent numerical instability.

        Parameters:
        predictions (np.ndarray): Predictions.

        Returns:
        np.ndarray: Predictions clipped within a valid range.
        """
        return np.clip(predictions, self.epsilon, 1 - self.epsilon)

    def calculate_gradient(self, predictions, target, weights):
        """
        Calculate the gradient of the cost function.

        Parameters:
        predictions (np.ndarray): Model predictions.
        target (np.ndarray): Target labels.
        weights (np.ndarray): Model weights.

        Returns:
        np.ndarray: Gradient of the cost function.
        """
        return (
            np.dot(self.features_scaled.T, (predictions - target))
            / target.size
        )

    def update_weights(self, weights, gradient):
        """
        Update model weights using gradient descent.

        Parameters:
        weights (np.ndarray): Current model weights.
        gradient (np.ndarray): Gradient of the cost function.

        Returns:
        np.ndarray: Updated model weights.
        """
        return weights - self.learning_rate * gradient

    @staticmethod
    def calculate_cost(target, predictions):
        """
        Calculate the cost function.

        Parameters:
        target (np.ndarray): Target labels.
        predictions (np.ndarray): Model predictions.

        Returns:
        float: Cost value.
        """
        return -np.mean(
            target * np.log(predictions)
            + (1 - target) * np.log(1 - predictions)
        )

    def save_weights(self, file_path: str) -> None:
        """
        Save the weights to a JSON file.

        Parameters:
        file_path (str): Path to the JSON file.
        """
        try:
            with open(file_path, "w") as f:
                json.dump(self.weights, f)
                logging.info(f"Weights saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving weights: {str(e)}")
            raise e

    def plot_s_curve(self):
        palette = sns.color_palette("husl", len(self.houses))
        fig, axes = plt.subplots(
            len(self.houses),
            len(self.features),
            figsize=((len(self.features) * 12), (len(self.houses) * 8)),
        )
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        pbar = tqdm(total=len(self.houses) * len(self.features))
        pbar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}"

        for row, house in enumerate(self.houses):
            self.data[house] = (self.data["Hogwarts House"] == house).astype(
                int
            )
            logging.info(f"House: {self.data[house]}")
            for col, feature in enumerate(self.features):
                logging.info(f"Feature: {feature}")
                ax = axes[row, col] if len(self.houses) > 1 else axes[col]
                sns.set(style="whitegrid")
                color = palette[row]
                sns.regplot(
                    x=feature,
                    y=house,
                    data=self.data,
                    label=f"{house} vs {feature}",
                    ax=ax,
                    color=color,
                    logistic=True,
                )
                ax.scatter(
                    self.data[feature],
                    self.data[house],
                    alpha=0.5,
                    label="Data points",
                    color=color,
                )
                ax.set_xlabel(feature)
                ax.set_ylabel(f"Probability of Belonging to {house}")
                ax.set_title(f"Logistic Regression of {feature} on {house}")
                ax.legend()
                pbar.update(1)

        plt.tight_layout()
        save_name = self.file_path.split(".")[0] + ".png"
        plt.savefig(save_name)
        plt.show()


# Example usage
if __name__ == "__main__":
    if os.path.exists("training.log"):
        os.remove("training.log")
    trainer = LogisticRegressionTrainer(
        learning_rate=0.001, num_iterations=10_000, epsilon=1e-5
    )
    logging.basicConfig(
        filename="training.log", level=logging.INFO, format="%(message)s"
    )
    trainer.load_data("dataset_train.csv")
    trainer.train_one_vs_all()
    trainer.save_weights("weights.json")
    # trainer.plot_s_curve()
