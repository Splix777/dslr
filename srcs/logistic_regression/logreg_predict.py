import pandas as pd
import numpy as np

from srcs.logistic_regression.model.one_vs_all import OneVsAll


class LogRegPredict:
    def __init__(self, csv_path: str, model_path: str, **kwargs):
        self.csv_path = csv_path
        self.one_vs_all = OneVsAll(csv_path, model_path=model_path, **kwargs)

    def predict(self):
        """
        Predict the Hogwarts House of the students in the
        provided CSV file and save the predictions in a new CSV file

        """
        predictions = self.one_vs_all.predict()

        df = pd.DataFrame({
                "Index": np.arange(0, len(predictions)),
                "Hogwarts House": predictions
            })

        csv_dir = self.csv_path.split("/")[:-1]
        csv_dir.append("predictions.csv")
        save_path = "/".join(csv_dir)
        df.to_csv(save_path, index=False)
