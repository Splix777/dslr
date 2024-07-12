import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        pass

    def plot_history(self, history, name):
        self.plot(history.loss_history, name, " Loss")
        self.plot(history.bias_history, name, " Bias")
        self.plot(history.weights_history, name, " Weights")

    @staticmethod
    def plot(history, name, title):
        plt.plot(range(len(history)), history)
        plt.title(f"{name}{title}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()
