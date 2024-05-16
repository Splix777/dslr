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
    return colors[house]








# Load the dataset
df = pd.read_csv("/home/splix/Desktop/dslr/csv_files/dataset_train.csv")

# Exclude non-numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64'])
print("Numeric columns:", numeric_columns.columns)

# Iterate over each numeric column (course)
homogeneous_courses = []
for column in numeric_columns.columns:
    # Calculate the distribution of scores for each Hogwarts house
    house_scores = {}
    for house in df['Hogwarts House'].unique():
        house_scores[house] = df[df['Hogwarts House'] == house][column].dropna().values

    # Determine if the score distribution is homogeneous
    score_variances = [pd.Series(scores).var() for scores in house_scores.values()]
    is_homogeneous = all(
        abs(variance - sum(score_variances) / len(score_variances)) < 1e-5 for variance in score_variances)

    if is_homogeneous:
        homogeneous_courses.append(column)
        # Plot histograms for homogeneous courses
        plt.figure(figsize=(8, 6))
        for house, scores in house_scores.items():
            plt.hist(scores, bins=20, alpha=0.5, label=house)
        plt.title(f'Histogram of {column}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

# Identify the homogeneous courses
print("Homogeneous courses:", homogeneous_courses)
