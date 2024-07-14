from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def find_similar_features(
    df: pd.DataFrame, corr_threshold: float = 0.5
) -> List[Tuple[str, str]]:
    """
    Identify pairs of features with high correlation coefficients.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        corr_threshold (float): Threshold for identifying
        high correlations.

    Returns:
        List[Tuple[str, str]]: List of tuples containing
            pairs of feature names with high correlations.
    """
    if not any(df.select_dtypes(include=["float64", "int64"])):
        raise ValueError("No numerical columns found in the DataFrame.")

    try:
        num_features = df.select_dtypes(include=["float64", "int64"]).columns
        correlations = df[num_features].corr()
    except Exception as e:
        raise ValueError(
            f"Error calculating correlation matrix: {str(e)}"
        ) from e

    if not 0 <= corr_threshold <= 1:
        raise ValueError("Correlation threshold must be between 0 and 1.")

    return [
        (correlations.columns[i], correlations.columns[j])
        for i in range(len(correlations.columns))
        for j in range(i + 1, len(correlations.columns))
        if corr_threshold <= abs(correlations.iloc[i, j]) < 1
    ]


def create_scatter_plot(
    ax: plt.Axes, df: pd.DataFrame, feature1: str, feature2: str
) -> None:
    """
    Create a scatter plot for a pair of features.

    Args:
        ax (plt.Axes): Matplotlib Axes object to plot on.
        df (pd.DataFrame): DataFrame containing the dataset.
        feature1 (str): Name of the first feature.
        feature2 (str): Name of the second feature.
    """
    if feature1 not in df.columns or feature2 not in df.columns:
        raise ValueError("Specified features not found in the DataFrame.")

    if (
        "Hogwarts House" not in df.columns
        or df["Hogwarts House"].dtype != "object"
    ):
        raise ValueError(
            "Invalid or missing 'Hogwarts House' column in the DataFrame."
        )

    for house in df["Hogwarts House"].unique():
        house_data = df[df["Hogwarts House"] == house]
        if len(house_data) > 0:
            ax.scatter(house_data[feature1], house_data[feature2], label=house)

    ax.set_title(f"Scatter Plot of {feature1} vs {feature2}", fontsize=10)
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.legend(title="Hogwarts House")


def annotate_plot(
    ax: plt.Axes, df: pd.DataFrame, feature1: str, feature2: str
) -> None:
    """
    Annotate the scatter plot with the correlation percentage.

    Args:
        ax (plt.Axes): Matplotlib Axes object to annotate.
        df (pd.DataFrame): DataFrame containing the dataset.
        feature1 (str): Name of the first feature.
        feature2 (str): Name of the second feature.
    """
    if feature1 not in df.columns or feature2 not in df.columns:
        raise ValueError("Specified features not found in the DataFrame.")

    try:
        correlation = df[[feature1, feature2]].corr().iloc[0, 1]
    except Exception as e:
        raise ValueError(
            f"Error calculating correlation coefficient: {str(e)}"
        ) from e

    correlation_percentage = abs(correlation) * 100
    ax.annotate(
        f"{correlation_percentage:.2f}%",
        xy=(0.5, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        color="red",
    )


def plot_scatter_plots(
    df: pd.DataFrame, features: List[Tuple[str, str]], threshold: float
) -> None:
    """
    Generate scatter plots for pairs of similar features.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        features (List[Tuple[str, str]]): List of tuples
            containing pairs of feature names with high correlations.
        threshold (float): Correlation threshold used to
            identify similar features.
    """
    if df.empty or not features:
        raise ValueError(
            "Empty DataFrame or no features provided for plotting."
        )

    filtered_features = [
        (f1, f2)
        for f1, f2 in features
        if f1 in df.columns and f2 in df.columns
    ]
    if not filtered_features:
        raise ValueError("No valid pairs of features found in the DataFrame.")

    num_plots = len(filtered_features)
    if num_plots == 0:
        print("No pairs of similar features found.")
        return

    num_cols = min(3, num_plots)
    num_rows = -(-num_plots // num_cols)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows)
    )

    if num_plots == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"Pairs of similar features with correlation coefficient "
            f">= {threshold}",
            fontsize=16,
        )

    for idx, (feature1, feature2) in enumerate(filtered_features):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row][col] if num_plots > 1 else axes[0]

        create_scatter_plot(ax, df, feature1, feature2)
        annotate_plot(ax, df, feature1, feature2)

    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(
            axes[i // num_cols][i % num_cols] if num_plots > 1 else axes[0]
        )

    plt.tight_layout()
    plt.show()


def print_correlation_percentages(
    df: pd.DataFrame, features: List[Tuple[str, str]]
) -> None:
    """
    Calculate and print correlation coefficients as
    percentages for each pair of similar features.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        features (List[Tuple[str, str]]): List of tuples
            containing pairs of feature names with high correlations.
    """
    for feature1, feature2 in features:
        if feature1 not in df.columns or feature2 not in df.columns:
            print(
                f"Features {feature1} and {feature2} "
                f"not found in the DataFrame."
            )
            continue

        try:
            correlation = df[[feature1, feature2]].corr().iloc[0, 1]
            correlation_percentage = abs(correlation) * 100
            print(f"{feature1} and {feature2}: {correlation_percentage:.2f}%")
        except Exception as e:
            print(
                f"Error calculating correlation for "
                f"{feature1} and {feature2}: {str(e)}"
            )


def make_scatter_plot(csv_path: str = None) -> None:
    """
    Create scatter plots for pairs of similar features in a dataset.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        None
    """
    dataset = pd.read_csv(csv_path)
    correlation_threshold = 0.70
    similar_features = find_similar_features(dataset, correlation_threshold)

    print(
        f"Pairs of similar features with correlation coefficient >= "
        f"{correlation_threshold}:"
    )

    plot_scatter_plots(dataset, similar_features, correlation_threshold)
    print_correlation_percentages(dataset, similar_features)


def main(csv_path: str = None) -> None:
    """
    Create scatter plots for pairs of similar features in a dataset.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        None
    """
    if not csv_path:
        csv_path = input("Please indicate the path to the CSV file: ")
    try:
        make_scatter_plot(csv_path)
    except FileNotFoundError:
        print("File not found. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
