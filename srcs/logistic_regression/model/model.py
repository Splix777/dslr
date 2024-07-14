import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from srcs.logistic_regression.model.history import History


class Model:
    def __init__(self, target: str, **kwargs):
        self.target: str = target
        self.learning_rate: float = kwargs.get("learning_rate", 0.01)
        self.epochs: int = kwargs.get("epochs", 1000)
        self.batch_size: int = kwargs.get("batch_size", 32)

        self.epsilon: float = kwargs.get("epsilon", 1e-8)
        self.patience: int = kwargs.get("patience", 5)
        self.scaler = StandardScaler()

        self.history = History()

        self.weights = None
        self.bias = None

    def __str__(self):
        return f"Model Target {self.target}"

    def __repr__(self):
        return self.__str__()

    def __call__(self, X: np.ndarray, y: np.ndarray):
        return self.fit(X, y)

    def _initialize_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Initialize the model weights and bias.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                The scaled input features and target values.
        """
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        X = self.scaler.fit_transform(X)
        return X, y

    def _batch_generator(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Generate batches of data for training.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                The input features and target values for the batch.
        """
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i:i + self.batch_size], y[i:i + self.batch_size]

    @staticmethod
    def _shuffle_data(X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Shuffle the input features and target values.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                The shuffled input features and target values.
        """
        rng = np.random.default_rng()
        indices = rng.permutation(len(X))
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given dataset.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            None
        """
        X, y = self._initialize_model(X, y)

        pbar = tqdm(range(self.epochs), desc=f"Training {self.target}")
        best_loss = np.inf
        best_epoch = 0
        for epoch in range(self.epochs):
            X, y = self._shuffle_data(X, y)
            mloss = []
            for X_batch, y_batch in self._batch_generator(X, y):
                loss, accuracy = self._train_batch(X_batch, y_batch)
                mloss.append(loss)
                pbar.set_postfix(loss=np.mean(mloss), accuracy=accuracy * 100)

            self.history.update(self.weights.copy(), self.bias, np.mean(mloss))

            if np.mean(mloss) < best_loss:
                best_loss = np.mean(mloss)
                best_epoch = epoch

            if self.early_stopping(loss=np.mean(mloss)):
                self.weights = self.history.weights_history[best_epoch]
                self.bias = self.history.bias_history[best_epoch]
                pbar.set_postfix({
                    "Early Stopping at Epoch": epoch,
                    "Best Loss": best_loss,
                })
                break

            pbar.update(1)

        pbar.close()

    def call(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Make predictions on the given dataset and compute the loss.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            tuple[np.ndarray, float, float]:
                The predicted target values, loss, and accuracy.
        """
        y_pred = self._predict(X)
        loss = self._compute_loss(y, y_pred)
        accuracy = np.mean(np.where(y_pred >= 0.5, 1, 0) == y)
        return y_pred, loss, accuracy

    @staticmethod
    def backward(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Compute the gradients for the model parameters.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            tuple[np.ndarray, float]:
                The gradients for the weights and bias.
        """
        gradient_weights = np.dot(X.T, y_pred - y) / len(y)
        gradient_bias = np.mean(y_pred - y)
        return gradient_weights, gradient_bias

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make predictions on the given dataset.

        Args:
            X (pd.DataFrame | np.ndarray): Input features.

        Returns:
            np.ndarray: The predicted target values.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained")
        if self.scaler:
            X = self.scaler.transform(X)
        return self._predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model on the given dataset using
        common evaluation metrics.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        y_pred = self.predict(X)

        # Convert predictions back to binary class labels if necessary
        y_pred_class = np.where(y_pred >= 0.5, 1, 0)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y, y_pred_class)
        precision = precision_score(y, y_pred_class)
        recall = recall_score(y, y_pred_class)
        f1 = f1_score(y, y_pred_class)

        return {
            "accuracy": accuracy * 100,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def _train_batch(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Train the model on a batch of data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            tuple[np.ndarray, float]:
                The loss and accuracy for the batch.
        """
        y_pred, loss, accuracy = self.call(X, y)
        grad_weight, grad_bia = self.backward(X, y, y_pred)

        self.weights -= self.learning_rate * grad_weight
        self.bias -= self.learning_rate * grad_bia

        return loss, accuracy

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the given dataset.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: The predicted target values.
        """
        return self._sigmoid(z=self._z(X))

    def _z(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear combination of the input features
        and model weights.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: The linear combination of the input features.
        """
        return np.dot(X, self.weights) + self.bias

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid activation function.

        Args:
            z (np.ndarray): The linear combination of
                the input features.

        Returns:
            np.ndarray: The sigmoid activation function.
        """
        return np.clip(1 / (1 + np.exp(-z)), self.epsilon, 1 - self.epsilon)

    @staticmethod
    def _compute_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.

        Args:
            y (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: The binary cross-entropy loss.
        """
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def early_stopping(self, loss: float) -> bool:
        """
        Check if early stopping criteria is met.

        Args:
            loss (float): The current loss value.

        Returns:
            bool: Whether to stop training or not.
        """
        if len(self.history.loss_history) > self.patience:
            return loss > np.min(self.history.loss_history[-self.patience:])
        return False
