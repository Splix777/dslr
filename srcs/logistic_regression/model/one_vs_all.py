import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from srcs.logistic_regression.model.model import Model
from srcs.logistic_regression.utils.decorators import error_handler


class OneVsAll:
    def __init__(self, csv_path: str, **kwargs):
        self.data = self.load_data(csv_path)
        if model_path := kwargs.get("model_path", None):
            self.model = self.load_model(model_path)
        self.save_path = kwargs.get("save_path", None)

    @staticmethod
    @error_handler(handle_exceptions=(FileNotFoundError, PermissionError))
    def load_model(path: str | None):
        return None if path is None else pickle.load(open(path, "rb"))

    @error_handler(handle_exceptions=(FileNotFoundError, PermissionError))
    def save_model(self, model: dict[str, Model]):
        if self.save_path is None:
            raise ValueError("No save path provided")

        file = f"{self.save_path}/model.pkl"
        pickle.dump(model, open(file, "wb"))

    @staticmethod
    @error_handler(handle_exceptions=(FileNotFoundError, PermissionError))
    def load_data(csv_path: str):
        return pd.read_csv(csv_path)

    def train_model(self):
        X_train, X_test, y_train, y_test = self._process_train_data(self.data)

        houses = self.data["Hogwarts House"].unique()
        self.model = {}
        evals = {}
        for house in houses:
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
        return evals

    def predict(self):
        if self.model is None:
            raise ValueError("Model not trained yet")
        print(self.model)
        X = self._process_predict_data(self.data)
        pred = {house: model.predict(X) for house, model in self.model.items()}

        pred_classes = np.argmax(np.array(list(pred.values())), axis=0)
        return [list(pred.keys())[idx] for idx in pred_classes]

    @staticmethod
    @error_handler(handle_exceptions=(KeyError, ValueError, TypeError))
    def _process_train_data(data: pd.DataFrame):
        """
        Preprocess the data before training the model.
        """
        data["Best Hand"] = data["Best Hand"].map({'Right': 1.0, 'Left': 0.0})
        data.drop(columns=["Index"], inplace=True)

        features = data.select_dtypes(include=['float64']).columns.tolist()
        X = data[features]
        X_filled = X.apply(lambda col: col.fillna(col.mean()))
        y = data["Hogwarts House"].values

        return train_test_split(X_filled, y, test_size=0.2, random_state=42)

    @staticmethod
    @error_handler(handle_exceptions=(KeyError, ValueError, TypeError))
    def _process_predict_data(data: pd.DataFrame):
        """
        Preprocess the data before making predictions.
        """
        data["Best Hand"] = data["Best Hand"].map({'Right': 1.0, 'Left': 0.0})
        data.drop(columns=["Index", "Hogwarts House"], inplace=True)

        features = data.select_dtypes(include=['float64']).columns.tolist()
        X = data[features]
        return X.apply(lambda col: col.fillna(col.mean()))
