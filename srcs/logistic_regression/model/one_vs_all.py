import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from srcs.logistic_regression.model.model import Model
from srcs.logistic_regression.utils.plotter import Plotter
from srcs.logistic_regression.utils.decorators import error_handler


class OneVsAll:
    def __init__(self, csv_path: str, **kwargs):
        self.data = self.load_data(csv_path)
        if model_path := kwargs.get("model_path", None):
            self.model = self.load_model(model_path)
        self.save_path = kwargs.get("save_path", None)
        self.plot = kwargs.get("plot", False)

    @staticmethod
    @error_handler(handle_exceptions=(FileNotFoundError, PermissionError))
    def load_model(path: str | None) -> dict[str, Model]:
        """
        Load a model from a pickle file.

        Args:
            path (str): The path to the pickle file.

        Returns:
            dict[str, Model]: The loaded model.
        """
        return None if path is None else pickle.load(open(path, "rb"))

    @error_handler(handle_exceptions=(FileNotFoundError, PermissionError))
    def save_model(self, model: dict[str, Model]) -> None:
        """
        Save a model to a pickle file.

        Args:
            model (dict[str, Model]): The model to save.

        Raises:
            ValueError: If no save path is provided.
        """
        if self.save_path is None:
            raise ValueError("No save path provided")

        os.makedirs(self.save_path, exist_ok=True)
        file = f"{self.save_path}/model.pkl"
        pickle.dump(model, open(file, "wb"))

    @staticmethod
    @error_handler(handle_exceptions=(FileNotFoundError, PermissionError))
    def load_data(csv_path: str) -> pd.DataFrame:
        """
        Load the data from a csv file.

        Args:
            csv_path (str): The path to the csv file.

        Returns:
            pd.DataFrame: The loaded data.
        """
        return pd.read_csv(csv_path)

    def train_model(self) -> dict[str, dict[str, float]]:
        """
        Train the model on the data.

        Returns:
            dict[str, dict[str, float]]:
                The evaluation metrics for each house.
        """
        X, y = self._process_data(self.data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.model = {}
        evals = {}
        for house in self.data["Hogwarts House"].unique():
            y_train_house = np.where(y_train == house, 1, 0)
            y_test_house = np.where(y_test == house, 1, 0)
            sub_model = Model(
                target=house,
                learning_rate=0.01,
                epochs=1_000,
                batch_size=32,
                patience=15
            )
            sub_model.fit(X_train, y_train_house)
            evals[house] = sub_model.evaluate(X_test, y_test_house)
            self.model[house] = sub_model

        self.save_model(self.model)
        if self.plot:
            self.plot_data()

        return evals

    def predict(self) -> tuple[list[str], int, float]:
        """
        Make predictions on the data.

        Returns:
            list[str]: The predicted houses.
            int: The total number of correct predictions
            float: The accuracy of the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        X, y = self._process_data(self.data)
        pred = {house: model.predict(X) for house, model in self.model.items()}

        pred_classes = np.argmax(np.array(list(pred.values())), axis=0)
        pred_names = [list(pred.keys())[idx] for idx in pred_classes]

        total_correct = np.sum(pred_names == y)
        accuracy = total_correct / len(y)

        return pred_names, total_correct, accuracy

    @staticmethod
    @error_handler(handle_exceptions=(KeyError, ValueError, TypeError))
    def _process_data(data: pd.DataFrame) -> tuple:
        """
        Preprocess the data before training the model.

        Args:
            data (pd.DataFrame): The data to preprocess.

        Returns:
            tuple: The preprocessed features and target.
        """
        data["Best Hand"] = data["Best Hand"].map({'Right': 1.0, 'Left': 0.0})
        data.drop(columns=["Index"], inplace=True)

        features = data.select_dtypes(include=['float64']).columns.tolist()
        X = data[features]
        X_filled = X.apply(lambda col: col.fillna(col.mean()))
        y = data["Hogwarts House"].values

        return X_filled, y

    def plot_data(self) -> None:
        """
        Plot the data.
        """
        plotter = Plotter(self.save_path)
        for house, model in self.model.items():
            plotter.plot_history(model.history, house)
        plotter.plot_sigmoid(self.data.copy())
