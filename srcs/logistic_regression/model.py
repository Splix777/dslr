import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Model:
    target: str
    learning_rate: float
    epochs: int

    epsilon: float = 1e-5
    patience: int = 10

    weights_history: list[np.ndarray] = field(default_factory=list)
    bias_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)

    weights = None
    bias = None

    def __str__(self):
        return f"Model Target {self.target}"

    def __repr__(self):
        return self.__str__()

    def __call__(self, X: np.ndarray, y: np.ndarray):
        return self.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        y = np.where(y == self.target, 1, 0)

        for epoch in range(self.epochs):
            # Forward pass
            y_pred, loss = self.call(X, y)

            # Backward pass
            gradient_weights, gradient_bias = self.backward(X, y, y_pred)

            # Update weights and bias
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            self.update_history(
                weights=self.weights,
                bias=self.bias,
                loss=loss
            )

            if self._early_stop():
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self._sigmoid(z=self._get_z(X))
        return np.where(y_pred > 0.5, 1, 0)

    def call(self, X: np.ndarray, y: np.ndarray):
        y_pred = self._sigmoid(z=self._get_z(X))
        loss = self._compute_loss(y, y_pred)
        return y_pred, loss

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        gradient_weights = np.dot(X.T, y_pred - y) / len(y)
        gradient_bias = np.mean(y_pred - y)

        return gradient_weights, gradient_bias

    def update_history(self, weights: np.ndarray, bias: float, loss: float):
        self.weights_history.append(weights)
        self.bias_history.append(bias)
        self.loss_history.append(loss)

    def _get_z(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return z

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return np.clip(1 / (1 + np.exp(-z)), self.epsilon, 1 - self.epsilon)

    @staticmethod
    def _compute_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def _early_stop(self) -> bool:
        if len(self.loss_history) < self.patience:
            return False

        loss = abs(self.loss_history[-1] - self.loss_history[-self.patience])

        if loss < self.epsilon:
            return True

        return np.all(np.diff(self.loss_history[-self.patience:]) > self.epsilon)


def _preprocess_data(data: pd.DataFrame):
    data["Best Hand"] = data["Best Hand"].map({'Right': 1.0, 'Left': 0.0})

    data.drop(columns=["Index"], inplace=True)

    y = data["Hogwarts House"]

    features = data.select_dtypes(include=['float64']).columns.tolist()
    X = data[features]
    X = X.apply(lambda col: col.fillna(col.mean()))
    X_s = StandardScaler().fit_transform(X)

    return train_test_split(X_s, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    csv = '../../csv_files/dataset_train.csv'
    with open(csv, 'r') as f:
        data = pd.read_csv(f)

    X_train, X_test, y_train, y_test = _preprocess_data(data)

    model = Model(
        target="Ravenclaw",
        learning_rate=0.001,
        epochs=5_000
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test = np.where(y_test == "Ravenclaw", 1, 0)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Model: {model}")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
