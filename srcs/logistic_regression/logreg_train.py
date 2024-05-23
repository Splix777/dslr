import json
import logging
import os
import multiprocessing
import pickle
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
        Num_iterations (int): Number of iterations for training.
        Epsilon (float): Small value to avoid log(0).
        """
        self.project_path = get_project_base_path()
        self.output_dir = os.path.join(
            self.project_path, "outputs/logistic_regression"
        )
        self.data = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.scaler = None
        self.epsilon = epsilon
        self.weights = {}
        self.csv_path = None
        self.features_scaled = None
        self.target = None
        self.features = None
        self.costs = None
        self.weight_history = None
        self.bias_history = None
        self.target_labels = None
        logging.info(
            f"Initialized Logistic Regression Trainer with learning rate: "
            f"{learning_rate}, num_iterations: {num_iterations}, "
            f"epsilon: {epsilon}"
        )

    def load_data(self, file_path: str) -> None:
        """
        Load the dataset from a CSV file.

        Parameters:
        file_path (str): Path to the CSV file.

        Returns:
        pd.DataFrame: Loaded dataset.
        """
        try:
            self.csv_path = file_path
            self.data = pd.read_csv(file_path)
            logging.info(f"Data loaded from {file_path}")
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise e

    def __preprocess_data(self) -> None:
        """
        Preprocess the data before training the model.
        """
        try:
            # Maps the "Best Hand" column to a binary value
            self.data["Best Hand"] = (
                self.data["Best Hand"]
                .map({"Right": 0, "Left": 1})
                .astype(float)
            )
            # Select only the float columns as features
            self.features = self.data.select_dtypes(
                include=["float64"]
            ).columns.tolist()
            # Fill missing values with the mean of the column to avoid NaN
            self.data[self.features] = self.data[self.features].apply(
                lambda col: col.fillna(col.mean())
            )
            self.scaler = StandardScaler()
            # Scale features to have a mean of 0 and a standard deviation of 1
            self.features_scaled = self.scaler.fit_transform(
                self.data[self.features]
            )
            # Save Scaler to Pickle file
            scale_file = os.path.join(self.output_dir, "scaler.pkl")
            self.__save_scaler(scale_file)
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise e

    def __initialize_variables(self):
        """
        Initialize the variables required for training the model.
        """
        self.target = self.data["Hogwarts House"].unique()
        self.target_labels = self.data["Hogwarts House"].values
        self.weight_history = {house: [] for house in self.target}
        self.bias_history = {house: [] for house in self.target}
        self.costs = {house: [] for house in self.target}
        logging.info(f"Target labels: {self.target}")
        logging.info(f"Features: {self.features}")

    def train_one_vs_all(self) -> None:
        """
        Train the logistic regression model for each house.
        """
        logging.info("Training one vs all logistic regression model")
        self.__preprocess_data()
        self.__initialize_variables()
        self.__logistic_regression()
        weights_json = os.path.join(self.output_dir, "weight_base.json")
        self.__save_weights(weights_json)

    def worker(self, house: str) -> Dict[str, Any]:
        """
        Wrapper function to call the gradient descent function with multiprocessing.

        Parameters:
        house (str): House to train the model for.

        Returns:
        Dict[str, Any]: Results of the gradient descent function.
        """
        return self.__gradient_descent(house)

    def __logistic_regression(self):
        """
        Train the logistic regression model for each house.
        """
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            results = p.map(self.worker, list(self.target))

        for result in results:
            self.weights[result["house"]] = {
                "weights": result["weights"],
                "bias": result["bias"],
            }
            self.weight_history[result["house"]] = result["weight_history"]
            self.bias_history[result["house"]] = result["bias_history"]
            self.costs[result["house"]] = result["costs"]

    def __gradient_descent(self, house: str) -> Dict[str, Any]:
        """
        Perform gradient descent to train the logistic regression model.

        Parameters:
        house (str): House to train the model for.
        """
        binary_labels = (self.target_labels == house).astype(int)
        weights = np.zeros(self.features_scaled.shape[1])
        weight_history = []
        bias_history = []
        costs = []
        bias = 0
        pbar = tqdm(
            total=self.num_iterations,
            desc="Training model",
            colour="green",
        )

        for _ in range(self.num_iterations):
            linear_output = self.__calculate_linear_output(weights, bias)
            initial_p = self.__sigmoid_activation(linear_output)
            validated_p = self.__ensure_valid_predictions(initial_p)
            gradient_weights, gradient_bias = self.__calculate_gradient(
                validated_p, binary_labels
            )
            weights, bias = self.__update_weights(
                weights, bias, gradient_weights, gradient_bias
            )
            weight_history.append(weights.copy())
            bias_history.append(bias)
            costs.append(self.__calculate_cost(binary_labels, validated_p))
            pbar.update(1)

            if len(costs) > 1 and abs(costs[-1] - costs[-2]) < self.epsilon:
                logging.info(f"Training complete for {house} at iteration {_}")
                break

        pbar.set_description(f"Training model for {house}", refresh=True)
        pbar.close()

        return {
            "weights": weights.tolist(),
            "bias": bias,
            "house": house,
            "costs": costs,
            "weight_history": weight_history,
            "bias_history": bias_history,
        }

    def __calculate_linear_output(
        self, weights: np.ndarray, bias: float
    ) -> np.ndarray:
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
    def __sigmoid_activation(linear_output):
        """
        Apply the sigmoid activation function.

        Parameters:
        linear_output (np.ndarray): Linear output.

        Returns:
        np.ndarray: Sigmoid activation output.
        """
        return 1 / (1 + np.exp(-linear_output))

    def __ensure_valid_predictions(
        self, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Ensure predictions are within valid range to prevent numerical instability.

        Parameters:
        predictions (np.ndarray): Predictions.

        Returns:
        np.ndarray: Predictions clipped within a valid range.
        """
        return np.clip(predictions, self.epsilon, 1 - self.epsilon)

    def __calculate_gradient(
        self,
        predictions: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float | Any, Any]:
        """
        Calculate the gradient of the cost function.

        Parameters:
        predictions (np.ndarray): Model predictions.
        Target (np.ndarray): Target labels.

        Returns:
        np.ndarray: Gradient of the cost function.
        """
        error = predictions - target
        gradient_weights = np.dot(self.features_scaled.T, error) / len(target)
        gradient_bias = np.mean(error)
        return gradient_weights, gradient_bias

    def __update_weights(
        self,
        weights: np.ndarray,
        bias: float,
        gradient_weights: np.ndarray,
        gradient_bias: float,
    ) -> tuple[np.ndarray, float]:
        """
        Update the weights using gradient descent.

        Parameters:
        weights (np.ndarray): Model weights.
        Bias (float): Model bias.
        Gradient_weights (np.ndarray): Gradient of the weights.
        Gradient_bias (float): Gradient of the bias.

        Returns:
        Tuple[np.ndarray, float]: Updated weights and bias.
        """
        weights -= self.learning_rate * gradient_weights
        bias -= self.learning_rate * gradient_bias
        return weights, bias

    @staticmethod
    def __calculate_cost(target, predictions) -> float:
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

    def __save_weights(self, file_path: str) -> None:
        """
        Save the weights to a JSON file.

        Parameters:
        file_path (str): Path to the JSON file.
        """
        try:
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(self.weights, f)
                logging.info(f"Weights saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving weights: {str(e)}")
            raise e

    def __save_scaler(self, file_path: str) -> None:
        """
        Save the scaler to a Pickle file.

        Parameters:
        file_path (str): Path to the Pickle file.
        """
        try:
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self.scaler, f)
                logging.info(f"Scaler saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving scaler: {str(e)}")
            raise e

    def __reg_plot(self) -> None:
        palette = sns.color_palette("husl", len(self.target))
        fig, axes = plt.subplots(
            len(self.target),
            len(self.features),
            figsize=((len(self.features) * 12), (len(self.target) * 8)),
        )
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        pbar = tqdm(total=len(self.target) * len(self.features))
        pbar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}"
        pbar.set_description("Plotting regression plots")

        for row, house in enumerate(self.target):
            self.data[house] = (self.data["Hogwarts House"] == house).astype(
                int
            )
            logging.info(f"House: {self.data[house]}")
            for col, feature in enumerate(self.features):
                logging.info(f"Feature: {feature}")
                ax = axes[row, col] if len(self.target) > 1 else axes[col]
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
        save_name = os.path.join(self.output_dir, "reg_plot.png")
        plt.savefig(save_name)
        pbar.close()

    def __reg_plot_custom(self) -> None:
        palette = sns.color_palette("husl", len(self.target))
        fig, axes = plt.subplots(
            len(self.target),
            len(self.features),
            figsize=((len(self.features) * 12), (len(self.target) * 8)),
        )
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        pbar = tqdm(total=len(self.target) * len(self.features))
        pbar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}"
        pbar.set_description("Plotting regression plots")

        for row, house in enumerate(self.target):
            self.data[house] = (self.data["Hogwarts House"] == house).astype(
                int
            )
            logging.info(f"House: {self.data[house]}")
            for col, feature in enumerate(self.features):
                logging.info(f"Feature: {feature}")
                ax = axes[row, col] if len(self.target) > 1 else axes[col]
                sns.set(style="whitegrid")
                color = palette[row]
                linear_output = np.dot(
                    self.features_scaled, self.weights[house]["weights"]
                )
                predictions = self.__sigmoid_activation(linear_output)
                logging.info(f"Predictions: {predictions}")
                sns.regplot(
                    x=feature,
                    y=predictions,
                    data=self.data,
                    label=f"{house} vs {feature}",
                    ax=ax,
                    color=color,
                    truncate=True,
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
        save_name = os.path.join(self.output_dir, "reg_plot_custom.png")
        plt.savefig(save_name)
        pbar.close()

    def __plot_cost_progress(self) -> None:
        fig, axes = plt.subplots(
            len(self.target),
            1,
            figsize=(20, 40),
        )
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("Cost Function Progress for Each House", fontsize=20)

        for i, house in enumerate(self.target):
            axes[i].plot(
                range(len(self.costs[house])),
                self.costs[house],
                label=f"{house} Cost",
            )
            axes[i].set_title(f"{house} Cost Progress")
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel("Cost")
            axes[i].legend()
            axes[i].grid(True)

        save_name = os.path.join(self.output_dir, "cost_progress.png")
        plt.savefig(save_name)

    def __plot_heatmap(self) -> None:
        fig, axes = plt.subplots(
            len(self.target),
            1,
            figsize=(20, 40),
        )
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("Weights Heatmap for Each House", fontsize=20)

        logging.info(f"Weight history: {self.weight_history}")

        for i, house in enumerate(self.target):
            sns.heatmap(
                np.array(self.weight_history[house]),
                ax=axes[i],
                cmap="coolwarm",
                cbar_kws={"label": "Weight Value"},
            )
            axes[i].set_title(f"{house} Weights Heatmap")
            axes[i].set_ylabel("Iteration")
            axes[i].set_xticklabels(self.features, rotation=45)
            axes[i].grid(True)

        save_name = os.path.join(self.output_dir, "weights_heatmap.png")
        plt.savefig(save_name)

    def __plot_weights_bias(self) -> None:
        fig, axes = plt.subplots(
            len(self.target),
            1,
            figsize=(20, 30),
        )
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("Weights and Bias for Each House", fontsize=20)

        for i, house in enumerate(self.target):
            for j in range(len(self.features)):
                axes[i].plot(
                    range(len(self.weight_history[house])),
                    np.array(self.weight_history[house])[:, j],
                    label=f"{self.features[j]} Weight",
                )
            axes[i].plot(
                range(len(self.bias_history[house])),
                np.array(self.bias_history[house]),
                label="Bias",
            )
            axes[i].set_title(f"{house} Weights and Bias")
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel("Value")
            axes[i].legend()
            axes[i].grid(True)

        save_name = os.path.join(self.output_dir, "weights_bias.png")
        plt.savefig(save_name)

    def __bar_plot_averages(self) -> None:
        colors = sns.color_palette("husl", len(self.target))
        fig, axes = plt.subplots(len(self.target), 1, figsize=(20, 40))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("Average Weights and Bias for Each House", fontsize=20)

        for i, house in enumerate(self.target):
            axes[i].bar(
                self.features,
                self.weights[house]["weights"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.2,
            )
            axes[i].bar(
                "Bias",
                (self.bias_history[house][-1]),
                color="black",
                alpha=0.7,
                edgecolor="black",
                linewidth=1.2,
            )
            axes[i].set_title(f"{house} Average Weights")
            axes[i].grid(True)
            axes[i].set_xlabel("Feature")
            axes[i].set_ylabel("Weight")

        save_name = os.path.join(self.output_dir, "average_weights.png")
        plt.savefig(save_name)


def get_project_base_path() -> str:
    """Get the base path of the project."""
    current_file = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_file, "..", "..", ".."))


if __name__ == "__main__":
    if os.path.exists("training.log"):
        os.remove("training.log")
    logging.basicConfig(
        filename="training.log", level=logging.INFO, format="%(message)s"
    )

    trainer = LogisticRegressionTrainer(
        learning_rate=0.01, num_iterations=5_000, epsilon=1e-5
    )

    base_path = get_project_base_path()
    dataset_path = os.path.join(base_path, "csv_files/dataset_train.csv")

    trainer.load_data(dataset_path)
    trainer.train_one_vs_all()
