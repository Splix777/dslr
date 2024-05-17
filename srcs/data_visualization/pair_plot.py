import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_csv_file(csv_file_path: str) -> pd.DataFrame:
    """
    Read the dataset from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    try:
        return pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        exit(1)


def house_colors(house: str) -> dict[str, str]:
    """
    Return the color associated with the Hogwarts house.
    Keeping in mind we're using seaborn, we can use the color names
    directly.

    Args:
        house (str): Hogwarts house name.

    Returns:
        str: Color associated with the house.
    """
    colors = {
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
    }
    return colors[house]


def courses_list(df: pd.DataFrame) -> list:
    """
    Return the list of courses in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        list: List of course names.
    """
    # Get numerical features
    return (
        df.select_dtypes(include=["float64", "int64"])
        .columns.drop("Index")
        .tolist()
    )


def create_pair_plot_seaborn(df: pd.DataFrame) -> None:
    """
    Create a pair plot of the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        None
    """
    if (
        "Hogwarts House" not in df.columns
        or df["Hogwarts House"].dtype != "object"
    ):
        raise ValueError(
            "Column 'Hogwarts House' not found or not of type 'object'."
        )

    courses = courses_list(df)
    house_colors_list = [
        house_colors(house) for house in df["Hogwarts House"].unique()
    ]
    sns.pairplot(
        df,
        hue="Hogwarts House",
        palette=house_colors_list,
        vars=courses,
    )
    plt.show()


def create_scatter_plot_matrix_pandas(df: pd.DataFrame) -> None:
    """
    Create a scatter plot matrix of the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        None
    """
    if (
        "Hogwarts House" not in df.columns
        or df["Hogwarts House"].dtype != "object"
    ):
        raise ValueError(
            "Column 'Hogwarts House' not found or not of type 'object'."
        )
    courses = courses_list(df)
    df_filtered = df.dropna(subset=courses)
    colors = [house_colors(house) for house in df_filtered["Hogwarts House"]]

    pd.plotting.scatter_matrix(
        df_filtered[courses],
        alpha=0.7,
        figsize=(30, 30),
        diagonal="hist",
        grid=True,
        s=20,
        c=colors,
    )
    plt.show()


def pair_plot_hist(
    ax: plt.Axes, x: pd.Series, house_labels: pd.Series
) -> None:
    """
    Plot histograms for each segment of the data.

    Args:
        ax (plt.Axes): Axes object for plotting.
        x (pd.Series): Data to plot histograms for.
        house_labels (pd.Series): Labels indicating the house for each data point.

    Returns:
        None
    """
    colors = {
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
        "Slytherin": "green",
    }
    for house, color in colors.items():
        mask = house_labels == house
        ax.hist(x[mask].dropna().values, alpha=0.5, color=color, label=house)


def pair_plot_scatter(
    ax: plt.Axes, x: pd.Series, y: pd.Series, house_labels: pd.Series
) -> None:
    """
    Plot scatter plots for each segment of the data.

    Args:
        ax (plt.Axes): Axes object for plotting.
        x (pd.Series): Data for x-axis.
        y (pd.Series): Data for y-axis.
        house_labels (pd.Series): Labels indicating the house for each data point.

    Returns:
        None
    """
    colors = {
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
        "Slytherin": "green",
    }
    for house, color in colors.items():
        mask = house_labels == house
        ax.scatter(x[mask], y[mask], s=10, color=color, alpha=0.5, label=house)


def custom_pair_plot(df: pd.DataFrame) -> None:
    """
    Create a pair plot of the dataset with custom features.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        None
    """
    legend = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    house_labels = df["Hogwarts House"]
    numeric_df = (
        df.select_dtypes(include=["float64", "int64"])
        .drop("Index", axis=1)
        .dropna()
    )
    features = numeric_df.columns
    size = numeric_df.shape[1]

    fig, ax = plt.subplots(nrows=size, ncols=size, figsize=(30, 30))
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    for row, col in itertools.product(range(size), range(size)):
        x = numeric_df.iloc[:, col]
        y = numeric_df.iloc[:, row]

        if col == row:
            # pair_plot_hist(ax[row, col], x)
            pair_plot_hist(ax[row, col], x, house_labels)
        else:
            pair_plot_scatter(ax[row, col], x, y, house_labels)

        if col == 0:
            ax[row, col].set_ylabel(
                features[row].replace(" ", "\n"),
            )
        else:
            ax[row, col].tick_params(labelleft=False)

        if row == size - 1:
            ax[row, col].set_xlabel(features[col].replace(" ", "\n"))
        else:
            ax[row, col].tick_params(labelbottom=False)

        ax[row, col].spines["right"].set_visible(False)
        ax[row, col].spines["top"].set_visible(False)

    plt.legend(
        legend,
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1, 0.5),
        fontsize=12,
    )
    plt.suptitle("Pair Plot of Hogwarts Houses", fontsize=20)
    plt.show()


if __name__ == "__main__":
    csv_path = "/home/splix/Desktop/dslr/csv_files/"
    csv_file = "dataset_train.csv"
    dataset = read_csv_file(csv_path + csv_file)
    create_pair_plot_seaborn(dataset)
    create_scatter_plot_matrix_pandas(dataset)
    custom_pair_plot(dataset)
