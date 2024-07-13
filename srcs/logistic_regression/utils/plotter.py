import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy import ndarray, where

from srcs.logistic_regression.model.history import History


class Plotter:
    def __init__(self, save_path: str):
        self.save_path = save_path
        os.makedirs(name=f"{self.save_path}/plots", exist_ok=True)

    def plot_history(self, history: History, name: str) -> None:
        """
        Plot the history of the loss, bias and weights for each model.

        Args:
            history (History): The history of the loss,
                bias and weights for each model.
            name (str): The name of the model.

        Returns:
            None
        """
        self.plot(history.loss_history, name, " Loss")
        self.plot(history.bias_history, name, " Bias")
        self.plot(history.weights_history, name, " Weights")

    def plot(self, hist: list[float] | list[ndarray], name: str, title: str):
        """
        Plot the history of the loss, bias and weights for each model.

        Args:
            hist (list[float] | list[ndarray]): The history to plot.
            name (str): The name of the model.
            title (str): The title of the plot.

        Returns:
            None
        """
        plt.plot(range(len(hist)), hist)
        plt.title(f"{name}{title}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(f"{self.save_path}/plots/{name}_{title}.png")
        plt.close()

    def plot_sigmoid(self, data: pd.DataFrame):
        """
        Plot the sigmoid function for each model.

        Args:
            data (pd.DataFrame): The data to plot.

        Returns:
            None
        """
        houses = data["Hogwarts House"].unique()
        features = data.select_dtypes(include=['float64']).columns.tolist()

        palette = sns.color_palette(palette="husl", n_colors=len(houses))
        fig, axes = plt.subplots(
            nrows=len(houses),
            ncols=len(features),
            figsize=((len(features) * 12), (len(houses) * 8)),
        )
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        pbar = tqdm(total=len(houses) * len(features))
        pbar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}"
        pbar.set_description("Plotting regression plots")

        for row, house in enumerate(houses):
            data[house] = where(data["Hogwarts House"] == house, 1, 0)
            for col, feature in enumerate(features):
                ax = axes[row, col] if len(houses) > 1 else axes[col]
                sns.set(style="whitegrid")
                color = palette[row]
                sns.regplot(
                    x=feature,
                    y=house,
                    data=data,
                    label=f"{house} vs {feature}",
                    ax=ax,
                    color=color,
                    logistic=True,
                )
                ax.scatter(
                    data[feature],
                    data[house],
                    alpha=0.5,
                    label="Data points",
                    color=color,
                )
                ax.set_xlabel(feature)
                ax.set_ylabel(f"Probability of Belonging to {house}")
                ax.set_title(f"Logistic Regression of {feature} on {house}")
                ax.legend()
                pbar.update(1)

        pbar.close()
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/plots/logistic_regression.png")
        plt.close()
