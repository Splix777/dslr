import matplotlib.pyplot as plt
import pandas as pd


def house_colors(house: str) -> str:
    """
    Return the color associated with the Hogwarts house.

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
    return colors.get(house)


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
        .drop("Index", axis=1)
        .columns.tolist()
    )


def plot_course_distributions(df: pd.DataFrame) -> None:
    """
    Plot the histogram distribution of the courses for
    each Hogwarts house.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        None
    """
    courses = courses_list(df)

    fig, axes = plt.subplots(
        nrows=len(courses), ncols=1, figsize=(12, 8 * len(courses))
    )

    for i, course in enumerate(courses):
        ax = axes[i]
        for house in df["Hogwarts House"].unique():
            house_data = df[
                (df["Hogwarts House"] == house) & (df[course].notna())
            ][course]
            ax.hist(
                house_data,
                bins=20,
                alpha=0.5,
                label=house,
                color=house_colors(house),
            )
        ax.set_title(f"Histogram of {course} scores by Hogwarts House")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.show()


def print_homogeneous_course(df: pd.DataFrame) -> None:
    """
    Print the course with the most homogeneous score
    distribution between all four Hogwarts houses.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        None
    """
    courses = courses_list(df)
    min_std = float("inf")
    homogeneous_course = None

    for course in courses:
        print(f"Course: {course}")
        std_sum = 0
        for house in df["Hogwarts House"].unique():
            house_data = df[
                (df["Hogwarts House"] == house) & (df[course].notna())
            ][course]
            std = house_data.std()
            std_sum += std
            print(f"  {house}: {std}")
        avg_std = std_sum / len(df["Hogwarts House"].unique())
        if avg_std < min_std:
            min_std = avg_std
            homogeneous_course = course
        print()

    print(
        f"The course with the most homogeneous "
        f"score distribution is: {homogeneous_course}"
    )


if __name__ == "__main__":
    csv_path = "/home/splix/Desktop/dslr/csv_files/"
    csv_file = "dataset_train.csv"

    try:
        dataset = pd.read_csv(csv_path + csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        exit(1)

    plot_course_distributions(dataset)
    print_homogeneous_course(dataset)
