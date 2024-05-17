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


if __name__ == "__main__":
    csv_path = "/home/splix/Desktop/dslr/csv_files/"
    csv_file = "dataset_train.csv"
    dataset = read_csv_file(csv_path + csv_file)
    create_pair_plot_seaborn(dataset)
    create_scatter_plot_matrix_pandas(dataset)
