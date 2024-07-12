from srcs.logistic_regression.model.one_vs_all import OneVsAll
from srcs.logistic_regression.utils.plotter import Plotter


class LogRegTrain:
    def __init__(self, csv_path: str, save_path: str, **kwargs):
        self.one_vs_all = OneVsAll(csv_path, save_path=save_path, **kwargs)
        self.plot = kwargs.get("plot", False)
        if self.plot:
            self.plotter = Plotter()

    def train(self):
        evals = self.one_vs_all.train_model()
        if self.plot:
            self.plot_results()
        return evals

    def plot_results(self):
        models = self.one_vs_all.model
        for model in models.values():
            self.plotter.plot_history(model.history, model.target)


if __name__ == "__main__":
    csv_path = "../../csv_files/dataset_train.csv"
    model_save = "../../models"
    logreg = LogRegTrain(csv_path, model_save)
    logreg.train()
