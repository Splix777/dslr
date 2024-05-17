import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple


def read_csv_file(csv_file_path: str) -> pd.DataFrame:
    """
    Read the dataset from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        exit(1)


def find_similar_features(df: pd.DataFrame, corr_threshold: float = 0.5) -> List[Tuple[str, str]]:
    """
    Identify pairs of features with high correlation coefficients.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        corr_threshold (float): Threshold for identifying high correlations (default is 0.5).

    Returns:
        List[Tuple[str, str]]: List of tuples containing pairs of feature names with high correlations.
    """
    # Select numerical columns
    num_features = df.select_dtypes(include=['float64', 'int64']).columns
    # Calculate correlation matrix
    correlations = df[num_features].corr()

    # Find pairs of features with high correlation coefficients
    corr_features = [(correlations.columns[i], correlations.columns[j])
                     for i in range(len(correlations.columns))
                     for j in range(i + 1, len(correlations.columns))
                     if corr_threshold <= abs(correlations.iloc[i, j]) < 1]

    return corr_features


def create_scatter_plot(ax: plt.Axes, df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """
    Create a scatter plot for a pair of features.

    Args:
        ax (plt.Axes): Matplotlib Axes object to plot on.
        df (pd.DataFrame): DataFrame containing the dataset.
        feature1 (str): Name of the first feature.
        feature2 (str): Name of the second feature.
    """
    # Plot data for each house
    for house in df['Hogwarts House'].unique():
        house_data = df[df['Hogwarts House'] == house]
        ax.scatter(house_data[feature1], house_data[feature2], label=house)
    ax.set_title(f'Scatter Plot of {feature1} vs {feature2}', fontsize=10)
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.legend(title='Hogwarts House')


def annotate_plot(ax: plt.Axes, df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """
    Annotate the scatter plot with the correlation percentage.

    Args:
        ax (plt.Axes): Matplotlib Axes object to annotate.
        df (pd.DataFrame): DataFrame containing the dataset.
        feature1 (str): Name of the first feature.
        feature2 (str): Name of the second feature.
    """
    # Calculate the correlation coefficient
    correlation = df[[feature1, feature2]].corr().iloc[0, 1]
    correlation_percentage = abs(correlation) * 100
    # Annotate plot with the correlation percentage
    ax.annotate(f'{correlation_percentage:.2f}%', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',
                fontsize=10, color='red')


def plot_scatter_plots(df: pd.DataFrame, features: List[Tuple[str, str]], threshold: float) -> None:
    """
    Generate scatter plots for pairs of similar features.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        features (List[Tuple[str, str]]): List of tuples containing pairs of feature names with high correlations.
        threshold (float): Correlation threshold used to identify similar features.
    """
    num_plots = len(features)
    if num_plots == 0:
        print("No pairs of similar features found.")
        return

    # Determine subplot grid dimensions
    num_cols = min(3, num_plots)  # Number of columns in the subplot grid, maximum 3
    num_rows = -(-num_plots // num_cols)  # Ceiling division to get number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # If there's only one plot, axes would be a single object, not an array
    if num_plots == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_2d(axes)
        fig.suptitle(f'Pairs of similar features with correlation coefficient >= {threshold}', fontsize=16)

    for idx, (feature1, feature2) in enumerate(features):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row][col] if num_plots > 1 else axes[0]

        create_scatter_plot(ax, df, feature1, feature2)
        annotate_plot(ax, df, feature1, feature2)

    # Hide any empty subplots
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols][i % num_cols] if num_plots > 1 else axes[0])

    plt.tight_layout()
    plt.show()


def print_correlation_percentages(df: pd.DataFrame, features: List[Tuple[str, str]]) -> None:
    """
    Calculate and print correlation coefficients as percentages for each pair of similar features.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        features (List[Tuple[str, str]]): List of tuples containing pairs of feature names with high correlations.
    """
    for feature1, feature2 in features:
        correlation = df[[feature1, feature2]].corr().iloc[0, 1]
        correlation_percentage = abs(correlation) * 100
        print(f"{feature1} and {feature2}: {correlation_percentage:.2f}%")


if __name__ == '__main__':
    csv_path = '/home/splix/Desktop/dslr/csv_files/'
    csv_file = 'dataset_train.csv'
    dataset = read_csv_file(csv_path + csv_file)

    correlation_threshold = 0.75
    similar_features = find_similar_features(dataset, correlation_threshold)

    print(f"Pairs of similar features with correlation coefficient >= {correlation_threshold}:")

    plot_scatter_plots(dataset, similar_features, correlation_threshold)
    print_correlation_percentages(dataset, similar_features)
