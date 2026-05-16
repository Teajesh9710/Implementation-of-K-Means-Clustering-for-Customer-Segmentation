# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Load the dataset and separate input and output variables.

#### 2.Split the data into training and testing sets.

#### 3.Train the linear regression model using the training data.

#### 4.Predict the output for the test data and evaluate the results. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: TEAJESH R
RegisterNumber: 212225240167
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customer.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker='X')

# Labels
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()
```

## Output:

<img width="818" height="589" alt="image" src="https://github.com/user-attachments/assets/2ecdc298-b20b-49f6-beed-6fdc52b76327" />

  
## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
