<h1 align="center">dslr</h1>


<div style="text-align:center;">
    <img src="images/harry-potter-4077473_1280.png" alt="Harry Potter" width="25%">
</div>


## Data Science & Logistic Regression

### Objectives

This project introduces you to new tools in your exploration of Machine Learning, focusing on Logistic Regression:

- **Logistic Regression Implementation:** Implement a linear classification model called Logistic Regression, building upon your understanding of linear regression.
- **Machine Learning Toolkit Creation:** Develop a personal machine learning toolkit as you progress through this project.

**Note:** While the title mentions "Data Science," the scope is intentionally limited. We'll focus on specific foundational concepts relevant for data exploration before applying them to Machine Learning algorithms.

---

## Summary

- **Data Exploration and Visualization:** Learn techniques for reading datasets, visualizing them in various ways, and selecting/cleaning irrelevant information.
- **Logistic Regression for Classification:** Train a Logistic Regression model to solve classification problems.

---

## General Instructions

- **Language Choice:** You can use any programming language you prefer. However, we recommend choosing one with libraries that simplify data plotting and statistical calculations.
- **Avoiding Cheating:** Using pre-built functions that handle heavy lifting (e.g., Pandas' `describe`) is considered cheating. Aim to understand the core concepts.

---

# Mandatory Part

## V.1 Data Analysis

Professor McGonagall has assigned you the task of creating a program called `describe.py`. This program takes a dataset as input and displays information for all numerical features, resembling the following example:

![Data Analysis Example](images/data_analysis_eg.png)

---

## Understanding the Task

- **Program Name:** Develop a program named `describe.py` that accepts a dataset as input.
- **Output Format:** The program's output should strictly match the provided example, presenting information for all numerical features.

---

## Implementation Options

### The Easy Way (Using Pandas)

#### `describe.py`

- **Pandas Library:** Utilize the `describe()` function from the Pandas library, which efficiently provides descriptive statistics for numerical data.
- **Steps:**
  1. Read the dataset from a CSV file using Pandas.
  2. Use the `describe()` method to compute descriptive statistics.
  3. Print the results in the desired format.

### The Hard Way (Not So Hard)

#### `statistics_class.py`

- **Statistics Class:** This Python class, `StatisticsClass`, calculates various descriptive statistics for a dataset, including count, mean, standard deviation, minimum value, percentiles (25th, 50th, and 75th), and maximum value.
- **Importance of `@property` Decorator:**
  - The `@property` decorator creates methods that function like attributes, promoting code organization and readability.
- **Steps:**
  1. Implement the `StatisticsClass` to calculate descriptive statistics dynamically.
  2. Utilize properties like `count`, `mean`, `std`, etc., for accessing computed statistics.

#### `describe.py`

- **Script Functionality:** This script reads data from a CSV file and calculates descriptive statistics for numerical features using the `StatisticsClass`.
- **Steps:**
  1. Read the dataset from a CSV file.
  2. Create instances of `StatisticsClass` for each numerical feature.
  3. Compute descriptive statistics for each feature.
  4. Print descriptive statistics for each feature in the desired format.

#### `test_describe.py`

- **Testing Script:** This script tests the `statistics_class.py` script by comparing its output with the expected output from the Pandas `describe()` method.
- **Steps:**
  1. Create a test dataset and calculate results using `StatisticsClass`.
  2. Compare the results with the expected output from Pandas' `describe()` method using the `unittest` library.

---

## V.2 Data Visualization
Data visualization is a powerful tool for a data scientist. It allows you to make insights
and develop an intuition of what your data looks like. Visualizing your data also allows
you to detect defects or anomalies.

---

## V.2.1 Histogram

Professor Flitwick has asked you to create a program called `histogram.py`. This program
takes a dataset as input and displays a histogram for each numerical feature, resembling
the following example:

![Histogram Example](images/histogram_sample.png)

## Understanding Histograms

Histograms are graphical representations of the distribution of data. They provide a visual summary of the frequency or count of data points falling within certain ranges, often called bins or intervals. Histograms are commonly used in statistics to understand the underlying distribution of a dataset.

### X-Axis (Horizontal Axis)

The x-axis of a histogram represents the range of values for the variable being measured or observed. Each interval or bin on the x-axis corresponds to a range of values, and data points falling within that range contribute to the count or frequency displayed on the y-axis.

### Y-Axis (Vertical Axis)

The y-axis of a histogram represents the frequency or count of data points falling within each interval or bin on the x-axis. It shows how many data points fall into each range or category, providing a visual representation of the distribution of the data.

### Example:

Consider a histogram of exam scores for students in a class. The x-axis would represent the range of possible scores, such as 0-10, 10-20, 20-30, and so on. The y-axis would represent the frequency or count of students who scored within each score range.

#### Interpretation:

- If the histogram is skewed to the right, it indicates that most students scored lower on the exam.
- If the histogram is skewed to the left, it indicates that most students scored higher on the exam.
- If the histogram is symmetric, it indicates a balanced distribution of scores.

In the context of the provided histograms:

- **X-Axis:** Represents the range of scores for the selected course (e.g., "Arithmancy").
- **Y-Axis:** Represents the frequency or count of students from each Hogwarts house falling within each score range.
  
This visualization allows us to observe the distribution of scores across different houses, providing insights into how students from each house perform in the selected course.


---

## Histogram Analysis

Professor Sprout has requested a program called `histogram.py` to visualize the distribution of scores for different Hogwarts courses across all four houses. This script utilizes the Pandas and Matplotlib libraries to accomplish this task.

### Script Functionality

#### `plot_course_distributions` Function:

- **Objective:** Plot the histogram distribution of the courses for each Hogwarts house.
- **How It Works:**
    1. Iterate over a list of Hogwarts courses.
    2. For each course, iterate over each Hogwarts house and retrieve the corresponding scores.
    3. Plot a histogram of the scores for each house with a unique color.
    4. Add a title, labels, and legend to each plot.
    5. Display the plots using Matplotlib.


## **Which Hogwarts course has a homogeneous score distribution between all four houses?**

### ELI5
- In simpler terms, we're looking for a course where students from all four houses have similar scores. This means that the scores are not significantly different between the houses, indicating a more balanced distribution.

<details>
<summary><strong>Click to reveal the answer</strong></summary>

### Methodology

To determine the course with a homogeneous score distribution, we analyzed the standard deviation (SD) for each course across all four houses. A low standard deviation suggests that the scores are tightly clustered around the mean, indicating a more homogeneous distribution.

### Result Analysis

From the provided results, it's evident that the course "Care of Magical Creatures" exhibits relatively low standard deviations across all four houses:

- **Ravenclaw:** 0.9736
- **Slytherin:** 0.9374
- **Gryffindor:** 0.9884
- **Hufflepuff:** 0.9762

These values indicate that the scores for "Care of Magical Creatures" are similar across all four houses, suggesting a homogeneous score distribution. Therefore, "Care of Magical Creatures" emerges as the course with a homogeneous score distribution between all four houses.

### Plot Analysis

  <p align="center">
    <img src="images/plots/histogram_analysis.png" alt="Histogram Analysis" width="75%">
  </p>
</details>

