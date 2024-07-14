import pandas as pd
import numpy as np

from srcs.logistic_regression.model.one_vs_all import OneVsAll


class LogRegPredict:
    def __init__(self, csv_path: str, model_path: str, **kwargs):
        self.csv_path = csv_path
        self.one_vs_all = OneVsAll(csv_path, model_path=model_path, **kwargs)

    def predict(self) -> tuple[int, float]:
        """
        Predict the Hogwarts House of the students in the
        provided CSV file and save the predictions in a new CSV file

        Returns:
            int: The number of correct predictions
            float: The accuracy of the model
        """
        predictions, num_correct, accuracy = self.one_vs_all.predict()

        df = pd.DataFrame({
                "Index": np.arange(0, len(predictions)),
                "Hogwarts House": predictions
            })

        csv_dir = self.csv_path.split("/")[:-1]
        csv_dir.append("predictions.csv")
        save_path = "/".join(csv_dir)
        df.to_csv(save_path, index=False)

        return num_correct, accuracy
