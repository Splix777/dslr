from srcs.logistic_regression.model.one_vs_all import OneVsAll


class LogRegTrain:
    def __init__(self, csv_path: str, save_path: str, **kwargs):
        self.one_vs_all = OneVsAll(csv_path, save_path=save_path, **kwargs)

    def train(self):
        """
        Train the model using the data provided by the user
        """
        return self.one_vs_all.train_model()
