import pandas as pd
import matplotlib.pyplot as plt


def house_colors(house: str) -> str:
    """
    Return the color associated with the Hogwarts house.
    """
    colors = {
        'Ravenclaw': 'blue',
        'Slytherin': 'green',
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow'
    }
    return colors.get(house, 'black')


def courses_list(df: pd.DataFrame) -> list:
    """
    Return the list of courses in the DataFrame.
    """
    return df.columns[6:].tolist()


def plot_course_distributions(df: pd.DataFrame) -> None:
    """
    Plot the histogram distribution of the courses for each Hogwarts house.
    """
    courses = courses_list(df)

    fig, axes = plt.subplots(nrows=len(courses), ncols=1, figsize=(12, 8 * len(courses)))

    for i, course in enumerate(courses):
        ax = axes[i]
        for house in df['Hogwarts House'].unique():
            house_data = df[(df['Hogwarts House'] == house) & (df[course].notna())][course]
            ax.hist(house_data, bins=20, alpha=0.5, label=house, color=house_colors(house))
        ax.set_title(f'Histogram of {course} scores by Hogwarts House')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.show()


def print_homogeneous_course(df: pd.DataFrame) -> None:
    """
    Print the course with the most homogeneous score distribution between all four Hogwarts houses.
    """
    courses = courses_list(df)
    min_std = float('inf')
    homogeneous_course = None

    for course in courses:
        print(f"Course: {course}")
        std_sum = 0
        for house in df['Hogwarts House'].unique():
            house_data = df[(df['Hogwarts House'] == house) & (df[course].notna())][course]
            std = house_data.std()
            std_sum += std
            print(f"  {house}: {std}")
        avg_std = std_sum / len(df['Hogwarts House'].unique())
        if avg_std < min_std:
            min_std = avg_std
            homogeneous_course = course
        print()

    print(f"The course with the most homogeneous score distribution is: {homogeneous_course}")


if __name__ == '__main__':
    # csv_path = '/home/splix/Desktop/dslr/csv_files/'
    # csv_file = 'dataset_train.csv'
    #
    # try:
    #     df = pd.read_csv(csv_path + csv_file)
    # except FileNotFoundError:
    #     print(f"Error: File '{csv_file}' not found.")
    #     exit(1)
    # except Exception as e:
    #     print(f"An error occurred while reading the CSV file: {e}")
    #     exit(1)
    #
    # plot_course_distributions(df)
    # print_homogeneous_course(df)

    # Read the CSV file
    csv_path = '/home/splix/Desktop/dslr/csv_files/'
    csv_file = 'dataset_train.csv'
    df = pd.read_csv(csv_path + csv_file)

    # Select the course for the histograms
    course = 'Arithmancy'

    # Define house colors
    house_colors = {
        'Ravenclaw': 'blue',
        'Slytherin': 'green',
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow'
    }

    # Plot histograms for each house
    plt.figure(figsize=(10, 6))
    for house, color in house_colors.items():
        house_data = df.loc[df['Hogwarts House'] == house, course].dropna()
        plt.hist(house_data, bins=30, alpha=0.5, color=color, label=house)

    plt.title(f'Histogram of {course} scores by Hogwarts House')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()