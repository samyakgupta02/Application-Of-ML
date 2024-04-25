import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Task 1: Basic DataFrame Operations
# I. Display the first 5 rows, check for missing values
print("Task 1 Results:")
print("First 5 rows of the dataset:")
print(iris_df.head())

# Check for missing values (Iris dataset)
print("\nHandling missing values:")
print(iris_df.isnull().sum())

# Get a summary of the dataset
print("\nSummary of the dataset:")
print(iris_df.describe())

# III. Select a subset of columns using label-based and position-based indexing
subset_label = iris_df[['sepal length (cm)', 'sepal width (cm)']]
subset_position = iris_df.iloc[:, [0, 1]]

# Create a new DataFrame by filtering rows based on a condition
filtered_df = iris_df[iris_df['petal length (cm)'] > 3.5]

# Task 2: Data Cleaning and Preprocessing
# I. Identify missing values and handle them (not necessary for the Iris dataset)

# II. Create a new column and convert a categorical variable into numerical representation
iris_df['sepal_area'] = iris_df['sepal length (cm)'] * iris_df['sepal width (cm)']
iris_df = pd.get_dummies(iris_df, columns=['target'], prefix='species')

# III. Group the data by a specific column and apply aggregation functions
grouped_data = iris_df.groupby('species_2.0')
aggregated_results = grouped_data.agg({'sepal_area': ['sum', 'mean', 'count']})
print("\nTask 2 Results:")
print("Aggregated Results:")
print(aggregated_results)

# Task 3: Load two different datasets (I'm assuming two Iris datasets for illustration purposes)
iris_df2 = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
merged_inner = pd.merge(iris_df, iris_df2, on='target', how='inner')
merged_outer = pd.merge(iris_df, iris_df2, on='target', how='outer')
merged_left = pd.merge(iris_df, iris_df2, on='target', how='left')
merged_right = pd.merge(iris_df, iris_df2, on='target', how='right')

# Task 4: Visualization
# I. Create a bar plot, line plot, and scatter plot
iris_df.plot(kind='bar', y=['sepal length (cm)', 'sepal width (cm)'], title='Bar Plot')
plt.show()

iris_df.plot(kind='line', y=['petal length (cm)', 'petal width (cm)'], title='Line Plot')
plt.show()

sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=iris_df)
plt.title('Scatter Plot')
plt.show()

# II. Visualize the correlation matrix
correlation_matrix = iris_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# III. Create histograms and box plots
iris_df.hist(column='petal length (cm)', bins=20)
plt.title('Histogram')
plt.show()

sns.boxplot(x='target', y='sepal length (cm)', data=iris_df)
plt.title('Box Plot')
plt.show()
