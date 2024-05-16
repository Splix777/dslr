import pandas as pd
from statistics_class import StatisticsClass


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


if __name__ == "__main__":
    dataframe = pd.read_csv("/home/splix/Desktop/dslr/csv_files/dataset_train.csv")
    print(dataframe.describe())

    # Save the output to a CSV file
    description_df = describe_dataframe(dataframe)
    print(description_df)
    description_df.to_csv("/home/splix/Desktop/dslr/csv_files/description.csv")