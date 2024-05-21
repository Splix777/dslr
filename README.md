<h1 align="center">dslr</h1>


<div align="center">
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

## Table of Contents

1. [Mandatory Part](#mandatory-part)
   - [V.1 Data Analysis](#v1-data-analysis)
   - [V.2 Data Visualization](#v2-data-visualization)
     - [V.2.1 Histogram](#v21-histogram)
     - [V.2.2 Scatter Plot](#v22-scatter-plot)
     - [V.2.3 Pair Plot](#v23-pair-plot)
   - [V.3 Logistic Regression](#v3-logistic-regression)
     - [Part 1: Training the Model](#script-functionality-part-1-training)



---

# Mandatory Part

## V.1 Data Analysis

Professor McGonagall has assigned you the task of creating a program called `describe.py`. This program takes a dataset as input and displays information for all numerical features, resembling the following example:

<p align="center">
<img src="images/data_analysis_eg.png" alt="Pair Plot Example" width="">
</p>

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

Check out the `description.csv` file where you can see the output of the `describe.py` script.

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

<p align="center">
<img src="images/histogram_sample.png" alt="Pair Plot Example" width="50%">
</p>

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


---

## V.2.2 Scatter Plot

Professor Snape has assigned you the task of creating a program called `scatter_plot.py`. This program takes a dataset as input and displays a scatter plot for two numerical features, resembling the following example:

<p align="center">
<img src="images/scatter_eg.png" alt="Pair Plot Example" width="50%">
</p>

## Understanding Scatter Plots

Scatter plots are graphical representations of the relationship between two numerical variables. They display individual data points as dots on a two-dimensional plane, with one variable on the x-axis and the other on the y-axis. Scatter plots are useful for visualizing patterns, trends, and correlations between variables.

### X-Axis (Horizontal Axis)

The x-axis of a scatter plot represents one numerical variable or feature from the dataset. Each data point's x-coordinate corresponds to the value of this variable for that data point.

### Y-Axis (Vertical Axis)

The y-axis of a scatter plot represents another numerical variable or feature from the dataset. Each data point's y-coordinate corresponds to the value of this variable for that data point.

### Example:

Consider a scatter plot of students' exam scores in two different subjects. The x-axis could represent the scores in Subject A, while the y-axis represents the scores in Subject B. Each data point on the scatter plot would correspond to a student's scores in both subjects.

#### Interpretation:

- If the data points form a clear pattern or trend (e.g., a line), it indicates a relationship between the two variables.
- If the data points are scattered randomly, it suggests no apparent relationship between the variables.
- If the data points form a curve or cluster, it may indicate a non-linear relationship between the variables.
- If the data points show a positive slope, it suggests a positive correlation between the variables.
- If the data points show a negative slope, it suggests a negative correlation between the variables.
- If the data points are evenly distributed, it suggests no correlation between the variables.
- If the data points form a cluster or group, it may indicate subgroups within the data.

In the context of the provided scatter plots:

- **X-Axis:** Represents one numerical feature from the dataset.
- **Y-Axis:** Represents another numerical feature from the dataset.

This visualization allows us to observe the relationship between two numerical features, identifying patterns, trends, or correlations between them.

---

## Scatter Plot Analysis

Professor Snape has requested a program called `scatter_plot.py` to visualize the relationship between two numerical features in the dataset. This script utilizes the Pandas and Matplotlib libraries to accomplish this task.

### Script Functionality

#### `find_similar_features` Function:

- **Objective:** Find two numerical features with similar distributions for a scatter plot given a correlation threshold.
- **How It Works:**
  1. Selects numerical features from the dataset.
  2. Using .corr() method, calculates the correlation between all numerical features.
  3. Using list comprehension, filters out features with a correlation greater than the threshold.

#### `plot_scatter_plots` Function:

- **Objective:** Plot scatter plots for pairs of numerical features with similar distributions.
- **How It Works:**
  1. Iterates over pairs of numerical features with similar distributions.
  2. Plots a scatter plot for each pair with a unique color.
  3. Adds a title, labels, and legend to each plot.
  4. Displays the plots using Matplotlib.

### **Which pair of features has the most similar distribution?**

### ELI5
- We're looking for two features that have a similar distribution of values. This means that the data points in the scatter plot are closely clustered together, indicating a strong relationship between the features.
- In simpler terms, we're searching for two features where the data points form a clear pattern or trend, suggesting a correlation between them.

<details>
<summary><strong>Click to reveal the answer</strong></summary>

### Methodology

To identify the pair of features with the most similar distribution, we analyzed the scatter plots for each pair of features with a correlation greater than the threshold. We visually inspected the scatter plots to determine which pair exhibited the most similar distribution of values.
Additionally, we have a function `print_correlation_percentages` that prints the correlation percentages for each pair of features. It allows us to more accurately identify the pair with the most similar distribution.

### Result Analysis

From the provided scatter plots, the pair of features "Astronomy" and "Defense Against the Dark Arts" exhibits the most similar distribution. The scatter plot for these features shows a clear linear relationship, with data points forming a distinct pattern or trend. This pattern suggests a strong correlation between the "Astronomy" and "Defense Against the Dark Arts" scores, indicating that students who perform well in one subject tend to perform well in the other.

- **Pairs of similar features with correlation coefficient >= 0.70:**
- **Astronomy and Defense Against the Dark Arts:** 100.00%
- **Herbology and Charms: 74.66%**
- **Muggle Studies and Charms: 84.76%**
- **History of Magic and Transfiguration: 84.92%**
- **History of Magic and Flying: 89.63%**
- **Transfiguration and Flying: 87.37%**

These results suggest that the pair "Astronomy" and "Defense Against the Dark Arts" has the most similar distribution, with a correlation coefficient of 100.00%.

### Plot Analysis

  <p align="center">
    <img src="images/plots/scatter_plot_analysis.png" alt="Scatter Plot Analysis" width="75%">
  </p>

</details>

## V.2.3 Pair Plot

Professor Trelawney has assigned you the task of creating a program called `pair_plot.py`. This program takes a dataset as input and displays a pair plot for all numerical features, resembling the following example:

## Understanding Pair Plots

Pair plots are graphical representations of the relationships between pairs of numerical variables in a dataset. They display scatter plots for each pair of numerical features, allowing you to visualize the interactions and correlations between variables. Pair plots are useful for identifying patterns, trends, and relationships in multivariate datasets.

### Pair Plot Matrix

A pair plot matrix consists of a grid of scatter plots, where each cell represents the relationship between two numerical features. The diagonal cells typically display histograms or kernel density plots for each feature, while the off-diagonal cells show scatter plots for pairs of features.

### Example:

Consider a pair plot matrix of exam scores for students in different subjects. The diagonal cells would display histograms or density plots of scores in each subject, while the off-diagonal cells would show scatter plots of scores between pairs of subjects.

#### Interpretation:

- The scatter plots in the off-diagonal cells reveal the relationships between pairs of features (e.g., positive correlation, negative correlation, no correlation).
- The histograms in the diagonal cells display the distribution of values for each feature, providing insights into the data's spread and shape.
- Patterns, trends, or clusters in the scatter plots can indicate relationships or dependencies between variables.
- Outliers or anomalies in the scatter plots may suggest data points that deviate significantly from the overall pattern.
- Correlations between variables can be visually assessed based on the scatter plot patterns (e.g., linear, non-linear, clustered).
- Pair plots help identify potential interactions or dependencies between variables, guiding further analysis or modeling.
- Pair plots are particularly useful for exploring multivariate datasets and understanding the relationships between multiple features simultaneously.

In the context of the provided pair plots:

- **Pair Plot Matrix:** Displays scatter plots for pairs of numerical features in the dataset.
- **Diagonal Cells:** Show histograms or density plots for each numerical feature.
- **Off-Diagonal Cells:** Show scatter plots for pairs of numerical features.

This visualization allows us to explore the relationships between numerical features, identify patterns, and detect correlations in the dataset.

We can quickly see `Defense Against the Dark Arts` and `Astronomy` have a strong correlation.

On the other hand `Care of Magical Creatures` and `Charms` have a weak correlation.

Looking at the histograms, we can quickly see how the four houses are distributed in each course. Reviewing V2.1 Histograms, we can see that `Care of Magical Creatures` has a homogeneous score distribution between all four houses.

---

## Pair Plot Analysis

Professor Trelawney has requested a program called `pair_plot.py` to visualize the relationships between pairs of numerical features in the dataset. This script utilizes the Seaborn library to create a pair plot matrix, displaying scatter plots for all numerical features.

### Script Functionality

**For the purpose of comparing my custom pair plot I also provided plots generated by Seaborn (Pair Plot) and Pandas (Scatter Matrix).**

#### `custom_pair_plot` Function:

- **Objective:** Create a custom pair plot matrix using Matplotlib.
- **How It Works:**
  1. Iterates over pairs of numerical features in the dataset.
  2. Plots a scatter plot for each pair with a unique color.
  3. Adds a title, labels, and legend to each plot.
  4. Displays the plots using Matplotlib.

#### `seaborn_pair_plot` Function:

- **Objective:** Create a pair plot matrix using Seaborn.
- **How It Works:**
  1. Utilizes the Seaborn library to create a pair plot matrix.
  2. Displays scatter plots for pairs of numerical features.
  3. Includes histograms or density plots on the diagonal cells.
  4. Provides additional functionalities like hue (coloring by a categorical variable) and markers.

#### `pandas_scatter_matrix` Function:

- **Objective:** Create a pair plot matrix using Pandas.
- **How It Works:**
  1. Utilizes the Pandas library to create a scatter matrix.
  2. Displays scatter plots for pairs of numerical features.
  3. Includes histograms on the diagonal cells.
  4. Provides additional functionalities like color, marker, and figure size.

### **From this visualization, what features are you going to use for your logistic regression?**

### ELI5
- We're looking for features that exhibit clear patterns or relationships in the pair plot matrix. These features should show a distinct trend or correlation between them, indicating a potential predictive relationship for logistic regression.
- In simpler terms, we're selecting features that have a strong visual relationship in the pair plot matrix, suggesting that they may influence the outcome we want to predict.
- Features that show a clear linear or non-linear relationship in the scatter plots are ideal candidates for logistic regression.
- We're focusing on features that exhibit patterns or trends that can help predict the target variable accurately.

<details>

<summary><strong>Click to reveal the answer</strong></summary>

### Methodology

<p align="center">
<img src="images/plots/pair_plot.png" alt="Pair Plot Analysis" width="75%">
</p>

### Result Analysis

From the pair plot matrix, the following features exhibit clear patterns or relationships that can be used for logistic regression:

1. **Astronomy and Defense Against the Dark Arts:** These features show a strong positive linear relationship, indicating that students who perform well in Astronomy also tend to excel in Defense Against the Dark Arts. This relationship suggests that scores in Astronomy could be a good predictor for scores in Defense Against the Dark Arts.
2. **Astronomy and Herbology:** Shows a strong distinction between houses in terms of scores, indicating that Astronomy and Herbology scores may influence each other. This could help us predict a student house based on their performance in Astronomy and Herbology.
3. **Arithmacy and Care of Magical Creatures:** These features dont show a clear pattern or relationship, suggesting that they may not be ideal predictors for logistic regression. The scatter plot indicates a weak correlation between Arithmacy and Care of Magical Creatures scores, making them less suitable for predicting outcomes.

Based on the pair plot matrix, we would consider using Astronomy, Defense Against the Dark Arts, and Herbology as features for logistic regression due to their clear relationships and patterns in the scatter plots.

</details>

---

<p align="center">
    <img src="images/sorting_hat.webp" width="15%">
</p>

<p align="center">
    <img src="outputs/logistic_regression/reg_plot.png">
</p>

## V.3 Logistic Regression

You arrive at the last part: code your Magic Hat. To do this, you have to perform a multi-classifier using a logistic regression one-vs-all.

- Professor Dumbledore has asked you to create a program called `logreg_train.py`. This program takes a dataset as input and trains a logistic regression model to predict the Hogwarts house of a student based on their scores in various courses.

- The Magic Hat program (`logreg_predict.py`) will predict the Hogwarts house of a student based on their scores in various courses using the trained logistic regression model.

### Understanding Logistic Regression

Logistic regression is a classification algorithm used to predict the probability of a binary outcome based on one or more independent variables. It is commonly used for binary classification tasks, where the target variable has two classes (e.g., 0 or 1, True or False).

**One-vs-All (OvA) Strategy:**

- A multi-class logistic regression model can be trained using the one-vs-all (OvA) strategy, also known as one-vs-rest (OvR). In this approach, a separate binary logistic regression model is trained for each class, treating it as the positive class and all other classes as the negative class. The model predicts the probability of each class independently, and the class with the highest probability is assigned as the predicted class.

- Trained multi-class classifiers are able to predict the Hogwarts house of a student based on their scores in various courses. The logistic regression model is trained using the OvA strategy, allowing it to classify students into one of the four houses: Gryffindor, Hufflepuff, Ravenclaw, or Slytherin.

---

### Script Functionality Part 1: Training

The idea of One-vs-All is to transform a multi-class problem into multiple binary classification problems. For each class (e.g., Gryffindor), you create a binary classifier that distinguishes between "class vs. all other classes". So, you end up with as many binary classifiers as there are classes. Each classifier is trained to predict whether an instance belongs to Its class or not.

#### Features (Independent Variables):** The scores of students in various courses.
  
- In this case, the features are the courses scores (float64). Each column in your data (Defense Against the Dark Arts, Muggle Studies, and so on) represents a single feature.
- The OvA model will use the values in these columns (each student's score in each course) to learn how to classify students into different Hogwarts houses.
- Scaling: It's essential to scale the features to ensure consistent scaling across different courses. This step helps the model learn effectively without being biased by the scale of the features.
- We used fit_transform() to scale the features in the training dataset.
  - The scaler is fitted on the training data to learn the scaling parameters (mean and standard deviation).
  - Let's say we have a feature with values ranging from 0 to 100 and another feature with values ranging from 0 to 1000. The model might give more importance to the feature with higher values, which is not desirable. Scaling helps to avoid this issue.
  - After scaling, the features have a mean of 0 and a standard deviation of 1, ensuring consistent scaling across all courses.
  - This way the model can learn from the features effectively without being biased by the scale of the data.

#### Target (Dependent Variable): The Hogwarts house of the student.

- In OvA logistic regression, you typically have multiple target variables, one for each class you're trying to predict. Here, the target variable is the Hogwarts house (Gryffindor, Ravenclaw, Hufflepuff, Slytherin).
  - Due to the OvA strategy, you'll have four target variables, one for each house.
  - Each target variable is binary (0 or 1), indicating whether the student belongs to that house or not.
  - The logistic regression model will learn to predict the probability of a student belonging to each house based on their course scores.

#### `logreg_train.py`

- **Objective:** Train a logistic regression model to predict the Hogwarts house of a student based on their scores in various courses.


- **How It Works:**
  1. Load the training dataset from a CSV file.
  2. Split the dataset into features (course scores) and targets (Hogwarts houses).
  3. Using a scaler, normalize the features to ensure consistent scaling.
  4. Train a logistic regression model using the OvA strategy to predict the student's house.
     1. For each house (Gryffindor, Hufflepuff, Ravenclaw, Slytherin):
        1. Create a binary label variable for the specific house (1 if the student belongs to that house, zero otherwise).
        2. Initialize weights and bias for the logistic regression model. (In our case, we always start with 0)
        3. **`Gradient Descent` (The core of the logistic regression model):**
           1. Calculate Linear Output: The Formula: `z = w * x + b` (where `w` is the weight, `x` is the feature, and `b` is the bias).
           2. Apply the Sigmoid Function: The Formula: `sigmoid(z) = 1 / (1 + exp(-z))`. This function maps the linear output to a probability between 0 and 1.
           3. Ensure Valid Predictions: Clip the predicted probabilities to avoid numerical instability: `np.clip(p, epsilon, 1 - epsilon)`.
              * This step ensures that the predicted probabilities are within a valid range (between 0 and 1) to prevent overflow or underflow issues.
              * 'p' is the predicted probability, and 'epsilon' is a small value (e.g., 1e-15) to prevent extreme values.
              * The clip function ensures that the predicted probabilities are within a valid range, preventing potential errors during training.
           4. Calculate Gradient: The Formula: `gradient = np.dot(x.T, (p - target)) / len(target)`. This step computes the gradient of the logistic loss function.
              * 'gradient' is the gradient of the loss function, 'x' is the feature matrix, 'p' is the predicted probability, 'target' is the binary target variable, and 'len(target)' is the number of target variables.
              * The gradient represents the direction and magnitude of the steepest ascent in the loss function, guiding the model towards the optimal weights and bias.
           5. Calculate Cost: The Formula: `cost = -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))`. This step computes the logistic loss cost.
              * 'cost' is the logistic loss cost, 'target' is the binary target variable, 'p' is the predicted probability, and 'np.log' is the natural logarithm.
              * The cost function measures the error between the predicted and actual values, penalizing incorrect predictions.
              * The cost is calculated using the binary cross-entropy loss formula, which quantifies the difference between the predicted and actual probabilities.
           6. Update Weights and Bias: The Formula: `w -= learning_rate * gradient`. This step updates the weights and bias using the gradient descent algorithm.
              * 'w' is the weight vector, 'learning_rate' is the step size for updating the weights, and 'gradient' is the gradient of the loss function.
              * The weights are updated in the direction that minimizes the cost function, moving the model closer to the optimal solution.
              * The learning rate controls the step size of the weight updates, influencing the convergence speed and stability of the model.
           7. Repeat the Gradient Descent Steps: Iterate over the training data multiple times (epochs) to update the weights and bias iteratively.
     2. Store the learned weights and bias for each house in a dictionary.
        * `{"Ravenclaw": [-0.0009868341294897878, 0.06962188652678399, -0.5579166563484348, 0.39875190335458915, 0.5628056489665395, 0.35134153700367265, 0.9846571492335171, 0.7965639194620214, 0.06198095424575081, 0.08187381240200624, 0.04589124210735973, 0.018121847976973252, 0.8481914997775947, 0.033243022124068544]}`
        * Each house has a corresponding list of weights for each feature selected.
        * The weights represent the importance of each feature in predicting the student's house.




## Fair Use Disclaimer

This repository may contain screenshots or excerpts from copyrighted materials. These materials are used under the principles of fair use for purposes such as commentary, criticism, news reporting, teaching, scholarship, or research. Fair use is a use permitted by copyright statute that might otherwise be infringing.

If you are the owner of the copyrighted material and have any concerns about its use here, please contact us so we can address the issue promptly.

- The use of copyrighted material in this repository is not intended to infringe upon the rights of the copyright holder.
- This use is non-commercial and intended for educational and informational purposes only.
- All rights to the original materials remain with their respective copyright holders.

For more information on fair use,
you can visit the [U.S. Copyright Office](https://www.copyright.gov/fair-use/) website.
