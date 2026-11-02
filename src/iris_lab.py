import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

iris_df = pd.read_csv('.//data//iris.csv')

print(iris_df.sample(10))
iris_df.info()

# drop Virginica samples - then sample data to verifiy virginica was dropped.
iris_df = iris_df[iris_df["iris_type"] != "Iris-virginica"]
print(iris_df["iris_type"].unique())



# check for imbalanced classes

iris_df["iris_type"].value_counts().plot(kind="bar")
plt.title("Class Distribution of Iris Species")
plt.xlabel("Iris Species")
plt.ylabel("Count")
plt.savefig(".//output/iris_class_distribution_bar.png")
plt.close()

# sepal length vs sepal width scatter plot
plt.figure(figsize=(8, 6))

for flower in iris_df["iris_type"].unique():
    subset = iris_df[iris_df["iris_type"] == flower]

    plt.scatter(
        subset["sepal_length"], 
        subset["sepal_width"], 
        label=flower, 
        alpha=0.7
    )

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.savefig(".//output/iris_sepal_scatter.png")
plt.close()


#petal length vs petal width scatter plot
plt.figure(figsize=(8, 6))

for flower in iris_df["iris_type"].unique():
    subset = iris_df[iris_df["iris_type"] == flower]

    plt.scatter(
        subset["petal_length"], 
        subset["petal_width"], 
        label=flower, 
        alpha=0.7
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.savefig(".//output/iris_petal_scatter.png")
plt.close()

#EDA Q's
# How Balanced are the two classes?
# The classes are balanced, with 50 samples of each, after removing Iris-virginica.

# Which feature pair show the best separation?
# Petal length and petal width show the best separation between the two classes, as seen in the scatter plot.

# Does the data appear linearly separable?
# Yes. The petal feature suggest that the classes can be separated by a linear boundary.

# What challenges might a simple model face?
# The classes are well separated. A simple model like perceptron or logistic regression should perform well.