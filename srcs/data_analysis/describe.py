import pandas as pd

from srcs.data_analysis.statistics_class import StatisticsClass


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Describe the input DataFrame mimicking the
    behavior of the pandas describe method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame

    Returns
    -------
    pd.DataFrame
        The description of the input DataFrame
    """
    description = pd.DataFrame()
    numeric_list = df.select_dtypes(include=["float64", "int64"]).columns

    for column in numeric_list:
        stats = StatisticsClass(df[column].dropna().values)
        description[column] = [
            stats.count,
            stats.mean,
            stats.std,
            stats.min_value,
            stats.percentile_25,
            stats.percentile_50,
            stats.percentile_75,
            stats.max_value,
        ]
    description.index = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    return description


def print_descriptions(csv_path: str) -> None:
    """
    Print the description of the input CSV file.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        None
    """
    dataframe = pd.read_csv(csv_path)
    print("Dataframe description: ")
    print(dataframe.describe())
    print('-' * 50)
    print("Custom description: ")
    print(describe_dataframe(dataframe))


def describe(csv_path: str = None) -> None:
    """
    Describe the input CSV file.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        None
    """
    if not csv_path:
        csv_path = input("Please indicate the path to the CSV file: ")
    try:
        print_descriptions(csv_path=csv_path)
    except FileNotFoundError:
        print("File not found. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    describe()
